"""
Knowledge Graph + Optional Vector Index Builder
------------------------------------------------

Reads: outputs/chunks.json
Builds:
    - Knowledge Graph (NetworkX)
    - Optional FAISS Vector Index

Control:
    SKIP_VECTOR = True  -> Only KG
    SKIP_VECTOR = False -> KG + FAISS

Extracted Relationships:
    - function_calls
    - function_uses_variable
    - function_uses_struct
    - struct_uses_struct
    - struct_uses_variable
"""

import gc
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import networkx as nx
import faiss
from tree_sitter import Parser, Node


import logging

logger = logging.getLogger(__name__)

from .parse_codebase import C_LANGUAGE


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_JSON = BASE_DIR / "outputs" / "chunks.json"
OUTPUT_DIR = BASE_DIR / "outputs"

KG_PKL = OUTPUT_DIR / "knowledge_graph.pkl"
FAISS_INDEX_PATH = OUTPUT_DIR / "faiss_index.index"
FAISS_METADATA_PATH = OUTPUT_DIR / "faiss_metadata.json"

SKIP_VECTOR = True   #  Set False to enable embeddings

EMBEDDING_MODEL = "jinaai/jina-code-embeddings-0.5b"
MAX_CHUNK_LENGTH = 16000
BATCH_SIZE = 4


# ============================================================
# UTILITIES
# ============================================================

C_KEYWORDS = {
    "if", "else", "for", "while", "do", "switch", "case", "break",
    "continue", "return", "goto", "typedef", "struct", "union",
    "enum", "const", "static", "extern", "inline", "void",
    "int", "char", "short", "long", "float", "double",
    "unsigned", "signed", "sizeof", "NULL", "true", "false",
    "bool", "malloc", "free", "calloc", "realloc", "assert"
}


def get_node_text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode(
        "utf-8", errors="replace"
    )


def _first_identifier(node: Node, source_bytes: bytes):
    if node.type == "identifier":
        return get_node_text(node, source_bytes).strip()
    for child in node.children:
        result = _first_identifier(child, source_bytes)
        if result:
            return result
    return None


def _get_type_name(spec_node: Node, source_bytes: bytes):
    name_node = spec_node.child_by_field_name("name")
    if name_node:
        return get_node_text(name_node, source_bytes).strip()

    for child in spec_node.children:
        if child.type in ("identifier", "type_identifier"):
            return get_node_text(child, source_bytes).strip()

    return None


# ============================================================
# RELATIONSHIP EXTRACTION
# ============================================================

def extract_relationships(text: str, entity_name: str, parser: Parser):
    source_bytes = text.encode("utf-8", errors="replace")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    calls = set()
    type_refs = set()
    identifiers = set()

    def visit(node: Node):

        # Function calls
        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node:
                name = _first_identifier(func_node, source_bytes)
                if name and name != entity_name and name not in C_KEYWORDS:
                    calls.add(name)

        # Composite types
        elif node.type in ("struct_specifier", "union_specifier", "enum_specifier"):
            name = _get_type_name(node, source_bytes)
            if name:
                type_refs.add(name)

        # Identifiers (variables)
        elif node.type == "identifier":
            name = get_node_text(node, source_bytes).strip()
            if (
                name
                and name != entity_name
                and name not in C_KEYWORDS
                and len(name) > 1
            ):
                identifiers.add(name)

        for child in node.children:
            visit(child)

    visit(root)
    return calls, type_refs, identifiers


def build_name_index(chunks):
    index = defaultdict(list)
    for c in chunks:
        name = c.get("name")
        if name:
            index[name].append((c["node_id"], c["entity_type"].lower()))
    return dict(index)


def extract_all_relationships(chunks, parser, name_index):
    edges = []
    chunk_by_id = {c["node_id"]: c for c in chunks}
    composite_types = ("struct", "union", "enum")

    for c in chunks:
        nid = c["node_id"]
        name = c.get("name", "")
        entity_type = c.get("entity_type", "").lower()
        text = c.get("text", "")

        calls, type_refs, identifiers = extract_relationships(
            text, name, parser
        )

        # FUNCTION
        if entity_type == "function":

            for callee in calls:
                for tid, etype in name_index.get(callee, []):
                    if etype == "function":
                        edges.append((nid, "function_calls", tid))

            for tname in type_refs:
                for tid, etype in name_index.get(tname, []):
                    if etype in composite_types:
                        edges.append((nid, "function_uses_struct", tid))

            for ident in identifiers:
                for tid, etype in name_index.get(ident, []):
                    if etype == "variable":
                        tgt_chunk = chunk_by_id.get(tid, {})
                        if tgt_chunk.get("is_global", True):
                            edges.append((nid, "function_uses_variable", tid))

        # STRUCT / UNION / ENUM
        elif entity_type in composite_types:

            for tname in type_refs:
                for tid, etype in name_index.get(tname, []):
                    if etype in composite_types:
                        edges.append((nid, "struct_uses_struct", tid))

            for ident in identifiers:
                for tid, etype in name_index.get(ident, []):
                    if etype == "variable":
                        edges.append((nid, "struct_uses_variable", tid))

    return edges


# ============================================================
# KG BUILD
# ============================================================

def build_kg(chunks, edges):
    G = nx.DiGraph()

    for c in chunks:
        G.add_node(
            c["node_id"],
            name=c.get("name", ""),
            entity_type=c.get("entity_type", ""),
            file_path=c.get("file_path", ""),
            line_start=c.get("line_start", 0),
            line_end=c.get("line_end", 0),
        )

    for src, rel, tgt in edges:
        G.add_edge(src, tgt, relationship_type=rel)

    return G


# ============================================================
# VECTOR INDEX (OPTIONAL)
# ============================================================

def build_faiss_index(chunks):
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        # print("Install: pip install langchain-huggingface")
        return None, {}

    # print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    index = None
    metadata = {}
    vector_counter = 0

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = []

        for c in batch:
            t = c.get("text", "")
            if len(t) > MAX_CHUNK_LENGTH:
                t = t[:MAX_CHUNK_LENGTH]
            texts.append(t)

        vectors = embeddings.embed_documents(texts)
        arr = np.array(vectors, dtype="float32")

        if index is None:
            dim = arr.shape[1]
            index = faiss.IndexHNSWFlat(dim, 64)

        index.add(arr)

        for j, c in enumerate(batch):
            metadata[str(vector_counter)] = {
                "node_id": c["node_id"],
                "name": c.get("name", ""),
                "entity_type": c.get("entity_type", ""),
            }
            vector_counter += 1

        gc.collect()

    return index, metadata


# ============================================================
# MAIN
# ============================================================

def main():

    # print("=" * 60)
    # print("KG Builder (Relationships + Optional Vectors)")
    # print("=" * 60)

    if not CHUNKS_JSON.exists():
        # print("chunks.json not found")
        return

    with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # print(f"Loaded {len(chunks)} chunks")

    parser = Parser(C_LANGUAGE)
    name_index = build_name_index(chunks)

    # print("Extracting relationships...")
    edges = extract_all_relationships(chunks, parser, name_index)

    rel_counts = defaultdict(int)
    for _, r, _ in edges:
        rel_counts[r] += 1

    # print(f"Total edges: {len(edges)}")
    for r, c in sorted(rel_counts.items(), key=lambda x: -x[1]):
        pass  # print(f"  {r}: {c}")
        

    kg = build_kg(chunks, edges)
    # print(f"Graph: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(KG_PKL, "wb") as f:
        pickle.dump(kg, f)

    # print(f"Saved KG : {KG_PKL}")

    # -------------------------------
    # VECTOR PART
    # -------------------------------
    if not SKIP_VECTOR:
        # print("Building FAISS index...")
        index, metadata = build_faiss_index(chunks)

        if index:
            faiss.write_index(index, str(FAISS_INDEX_PATH))
            with open(FAISS_METADATA_PATH, "w") as f:
                json.dump(metadata, f, indent=2)

            # print(f"Saved FAISS : {FAISS_INDEX_PATH}")
            # print(f"Saved metadata : {FAISS_METADATA_PATH}")
        else:
            
            pass  # print("Vector index failed or skipped.")
    else:
        pass  # print("Vector creation skipped (SKIP_VECTOR=True)")
        

    # print("=" * 60)
    # print("Done.")
    # print("=" * 60)


if __name__ == "__main__":
    main()
