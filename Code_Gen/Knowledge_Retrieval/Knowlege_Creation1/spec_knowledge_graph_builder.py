"""
NetworkX-based spec knowledge graph builder for KG-only pipeline.

Design goals:
1) Reuse strong relationship patterns from existing Knowldge_creations logic
2) Avoid embeddings/FAISS entirely
3) Produce robust graph structure for agentic retrieval
"""

from __future__ import annotations

import json
import pickle
import re
import os
import time
from datetime import datetime, timedelta, timezone
IST = timezone(timedelta(hours=5, minutes=30))
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import numpy as np
import faiss
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI

load_dotenv()


class SpecKnowledgeGraphBuilder:
    def __init__(self, doc_id: str) -> None:
        self.doc_id = doc_id
        self.llm = self._build_llm()
        self.embeddings = self._build_embeddings()

    @staticmethod
    def _build_llm() -> AzureChatOpenAI | None:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        deployment = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o-mini")

        if not api_key or not endpoint:
            return None
        return AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            azure_deployment=deployment,
            temperature=0.1,
            timeout=90,
            max_retries=2,
        )

    @staticmethod
    def _build_embeddings() -> HuggingFaceEmbeddings:
        # Use bge-base as requested.
        return HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    @staticmethod
    def _load_chunks(chunks_path: str) -> List[Dict[str, Any]]:
        p = Path(chunks_path)
        if not p.exists():
            raise FileNotFoundError(f"chunks.json not found: {chunks_path}")
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Expected chunks.json to contain a JSON array.")
        return data

    @staticmethod
    def _numeric_sort_key(section_id: str) -> Tuple[int, List[int]]:
        try:
            parts = [int(x) for x in section_id.split(".")]
            return (len(parts), parts)
        except Exception:
            return (999999, [999999])

    def _extract_hierarchical_relationships(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Tuple[str, str, str]]:
        rels: Set[Tuple[str, str, str]] = set()

        for chunk in chunks:
            sid = chunk.get("section_id")
            meta = chunk.get("metadata", {}) or {}
            parent = meta.get("parent_section_id")
            children = meta.get("child_section_ids", []) or []

            if not sid:
                continue

            if parent:
                rels.add((sid, "CHILD_OF", parent))
                rels.add((parent, "PARENT_OF", sid))

            for child in children:
                if child:
                    rels.add((sid, "PARENT_OF", child))
                    rels.add((child, "CHILD_OF", sid))

        return list(rels)

    def _extract_explicit_references(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Tuple[str, str, str]]:
        rels: Set[Tuple[str, str, str]] = set()
        valid_section_ids = {c.get("section_id") for c in chunks if c.get("section_id")}

        reference_patterns = [
            r"section\s+(\d+(?:\.\d+)+)",
            r"see\s+(\d+(?:\.\d+)+)",
            r"as\s+defined\s+in\s+(\d+(?:\.\d+)+)",
            r"defined\s+in\s+(\d+(?:\.\d+)+)",
            r"according\s+to\s+(\d+(?:\.\d+)+)",
            r"per\s+(\d+(?:\.\d+)+)",
        ]

        for chunk in chunks:
            source_id = chunk.get("section_id")
            if not source_id:
                continue
            content = chunk.get("content", "") or ""

            for pat in reference_patterns:
                for m in re.finditer(pat, content, flags=re.IGNORECASE):
                    target_id = m.group(1)
                    if target_id in valid_section_ids and target_id != source_id:
                        rels.add((source_id, "REFERENCES", target_id))
                        rels.add((target_id, "REFERENCED_BY", source_id))

        return list(rels)

    def _extract_same_parent_sibling_relationships(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Tuple[str, str, str]]:
        rels: Set[Tuple[str, str, str]] = set()
        by_parent: Dict[str, List[str]] = {}

        for c in chunks:
            sid = c.get("section_id")
            parent = (c.get("metadata", {}) or {}).get("parent_section_id")
            if sid and parent:
                by_parent.setdefault(parent, []).append(sid)

        for _, child_ids in by_parent.items():
            ordered = sorted(set(child_ids), key=self._numeric_sort_key)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    a, b = ordered[i], ordered[j]
                    rels.add((a, "SIBLING_OF", b))
                    rels.add((b, "SIBLING_OF", a))

        return list(rels)

    def _create_faiss_index(
        self, chunks: List[Dict[str, Any]]
    ) -> Tuple[faiss.Index, Dict[int, str]]:
        chunk_texts = [c.get("content", "") or "" for c in chunks]
        embeddings_list = self.embeddings.embed_documents(chunk_texts)
        embeddings_array = np.array(embeddings_list).astype("float32")

        dimension = embeddings_array.shape[1]
        index = faiss.IndexHNSWFlat(dimension, 64)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 50
        index.add(embeddings_array)
        id_to_section = {i: chunks[i].get("section_id", "") for i in range(len(chunks))}
        return index, id_to_section

    @staticmethod
    def _extract_json_array(text: str) -> List[Dict[str, Any]]:
        text = (text or "").strip()
        if not text:
            return []
        try:
            data = json.loads(text)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            m = re.search(r"\[[\s\S]*\]", text)
            if not m:
                return []
            data = json.loads(m.group(0))
            return data if isinstance(data, list) else []

    def _extract_semantic_relationships_llm(
        self,
        chunks: List[Dict[str, Any]],
        top_k_candidates: int = 10,
        final_k: int = 5,
        max_chunks_for_semantic: int = 350,
    ) -> List[Tuple[str, str, str]]:
        if self.llm is None:
            return []

        print("[Semantic] Building FAISS index for candidate selection...")
        idx_start = time.time()
        faiss_index, id_to_section = self._create_faiss_index(chunks)
        print(f"[Semantic] FAISS ready: {faiss_index.ntotal} vectors in {time.time() - idx_start:.1f}s")

        by_id = {c.get("section_id"): c for c in chunks if c.get("section_id")}
        source_chunks = chunks[:max_chunks_for_semantic]
        rels: Set[Tuple[str, str, str]] = set()
        llm_calls = 0

        for i, source in enumerate(source_chunks, start=1):
            source_id = source.get("section_id")
            if not source_id:
                continue
            source_content_full = source.get("content", "") or ""
            source_meta = source.get("metadata", {}) or {}
            source_parent = source_meta.get("parent_section_id")
            source_children = set(source_meta.get("child_section_ids", []) or [])

            if not source_content_full.strip():
                continue

            # Same technique as earlier pipeline: embed query chunk -> FAISS top_k -> filter -> keep final_k.
            try:
                q_emb = self.embeddings.embed_query(source_content_full)
            except Exception:
                continue
            q_arr = np.array([q_emb]).astype("float32")
            distances, indices = faiss_index.search(q_arr, top_k_candidates)
            raw_candidate_ids = [
                id_to_section.get(int(idx), "")
                for idx in (indices[0] if len(indices) else [])
                if int(idx) >= 0
            ]
            candidate_ids = []
            for cid in raw_candidate_ids:
                if not cid or cid == source_id:
                    continue
                c_parent = (by_id.get(cid, {}).get("metadata", {}) or {}).get("parent_section_id")
                if cid == source_parent or c_parent == source_id or cid in source_children:
                    continue
                candidate_ids.append(cid)
            # Preserve order while de-duplicating.
            seen = set()
            candidate_ids = [x for x in candidate_ids if not (x in seen or seen.add(x))]
            candidate_ids = candidate_ids[:final_k]
            if not candidate_ids:
                continue

            source_title = source.get("section_title", "")
            source_content = source_content_full[:3000]
            candidate_block = []
            for cid in candidate_ids:
                c = by_id.get(cid)
                if not c:
                    continue
                candidate_block.append(
                    {
                        "section_id": cid,
                        "title": c.get("section_title", ""),
                        "content_preview": (c.get("content", "") or "")[:600],
                    }
                )

            prompt = f"""
Analyze semantic relationships for a 3GPP spec section.
Return ONLY JSON array. No markdown.

Source:
- section_id: {source_id}
- title: {source_title}
- content: {source_content}

Candidate sections:
{json.dumps(candidate_block, ensure_ascii=False, indent=2)}


Identify semantic relationships between the source section and candidate sections. Consider:
- DEPENDS_ON: Source section depends on concepts/definitions from target
- USES: Source section uses mechanisms/procedures defined in target
- DEFINES: Source section defines terms/concepts used in target
- RELATED_TO: General semantic relationship
- PREREQUISITE_FOR: Source section is prerequisite knowledge for target

Return ONLY a valid JSON array of relationships. Each relationship should have:
- "target_section_id": section ID
- "relationship_type": one of DEPENDS_ON, USES, DEFINES, RELATED_TO, PREREQUISITE_FOR
- "confidence": "high" or "medium" or "low"

Return format:
[
  {{
    "target_section_id": "x.y.z",
    "relationship_type": "DEPENDS_ON",
    "confidence": "high|medium|low"
  }}
]
If no valid relation, return [].
"""
            try:
                llm_calls += 1
                print(
                    f"[Semantic][{i}/{len(source_chunks)}] "
                    f"LLM call #{llm_calls} for section {source_id} with {len(candidate_ids)} candidates..."
                )
                call_start = time.time()
                resp = self.llm.invoke(prompt)
                call_time = time.time() - call_start
                print(f"[Semantic][{source_id}] LLM response in {call_time:.1f}s")
                raw = resp.content if hasattr(resp, "content") else str(resp)
                items = self._extract_json_array(raw)
            except Exception:
                continue

            for item in items:
                if not isinstance(item, dict):
                    continue
                target = str(item.get("target_section_id", "")).strip()
                rel_type = str(item.get("relationship_type", "")).strip().upper()
                conf = str(item.get("confidence", "medium")).strip().lower()
                if target not in candidate_ids:
                    continue
                if rel_type not in {"DEPENDS_ON", "USES", "DEFINES", "RELATED_TO", "PREREQUISITE_FOR"}:
                    continue
                if conf not in {"high", "medium"}:
                    # Keep precision high; drop low-confidence edges.
                    continue

                rels.add((source_id, rel_type, target))
                # Symmetric/complementary reverse edges for stronger traversal.
                if rel_type == "DEPENDS_ON":
                    rels.add((target, "PREREQUISITE_FOR", source_id))
                elif rel_type == "PREREQUISITE_FOR":
                    rels.add((target, "DEPENDS_ON", source_id))
                elif rel_type == "DEFINES":
                    rels.add((target, "USES", source_id))
                elif rel_type == "USES":
                    rels.add((target, "DEFINES", source_id))
                elif rel_type == "RELATED_TO":
                    rels.add((target, "RELATED_TO", source_id))

        print(f"[Semantic] Completed semantic extraction. LLM calls: {llm_calls}, semantic edges: {len(rels)}")
        return list(rels)

    def build_graph(
        self,
        chunks: List[Dict[str, Any]],
        include_semantic_relations: bool = True,
        semantic_top_k_candidates: int = 10,
        semantic_final_k: int = 5,
        semantic_max_chunks: int = 350,
    ) -> nx.DiGraph:
        graph = nx.DiGraph()

        # Add nodes.
        for chunk in chunks:
            sid = chunk.get("section_id")
            if not sid:
                continue
            meta = chunk.get("metadata", {}) or {}
            # Avoid duplicate kwargs when metadata already contains these fields.
            safe_meta = {
                k: v
                for k, v in meta.items()
                if k not in {"doc_id", "section_id", "section_title", "content"}
            }
            graph.add_node(
                sid,
                doc_id=self.doc_id,
                section_id=sid,
                section_title=chunk.get("section_title", ""),
                content=chunk.get("content", ""),
                **safe_meta,
            )

        # Build relationships.
        rels = []
        rels.extend(self._extract_hierarchical_relationships(chunks))
        rels.extend(self._extract_explicit_references(chunks))
        rels.extend(self._extract_same_parent_sibling_relationships(chunks))
        if include_semantic_relations:
            rels.extend(
                self._extract_semantic_relationships_llm(
                    chunks,
                    top_k_candidates=semantic_top_k_candidates,
                    final_k=semantic_final_k,
                    max_chunks_for_semantic=semantic_max_chunks,
                )
            )

        # Add edges with relationship aggregation.
        for source, rel_type, target in rels:
            if not graph.has_node(source) or not graph.has_node(target):
                continue
            if graph.has_edge(source, target):
                existing = graph[source][target].get("relationship_types", [])
                if not isinstance(existing, list):
                    existing = [existing] if existing else []
                if rel_type not in existing:
                    existing.append(rel_type)
                graph[source][target]["relationship_types"] = existing
            else:
                graph.add_edge(source, target, relationship_types=[rel_type])

        return graph

    @staticmethod
    def save_graph(graph: nx.DiGraph, output_path: str) -> str:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(graph, f)
        return str(p)

    @staticmethod
    def save_graph_summary(graph: nx.DiGraph, output_path: str) -> str:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        edge_type_counts: Dict[str, int] = {}
        for _, _, attrs in graph.edges(data=True):
            rels = attrs.get("relationship_types", []) or []
            for r in rels:
                edge_type_counts[r] = edge_type_counts.get(r, 0) + 1

        summary = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "edge_type_counts": edge_type_counts,
        }
        p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return str(p)

    @staticmethod
    def save_graph_adjacency_json(
        graph: nx.DiGraph,
        output_path: str,
        doc_id: str = "",
    ) -> str:
        """
        Serialize a compact, human-readable view: per section_id, list neighbors
        with edge direction and relationship_types (same semantics as the .pkl).
        """
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        resolved_doc_id = doc_id
        if graph.number_of_nodes() > 0:
            first = next(iter(graph.nodes()))
            attrs = graph.nodes[first]
            if isinstance(attrs, dict) and attrs.get("doc_id"):
                resolved_doc_id = str(attrs["doc_id"])

        sections: Dict[str, Dict[str, Any]] = {}
        for node in sorted(graph.nodes(), key=lambda x: str(x)):
            node_s = str(node)
            outgoing: List[Dict[str, Any]] = []
            for succ in graph.successors(node):
                attrs = graph.get_edge_data(node, succ) or {}
                rels = attrs.get("relationship_types", []) or []
                if not isinstance(rels, list):
                    rels = [rels] if rels else []
                outgoing.append({"to": str(succ), "relationship_types": list(rels)})
            incoming: List[Dict[str, Any]] = []
            for pred in graph.predecessors(node):
                attrs = graph.get_edge_data(pred, node) or {}
                rels = attrs.get("relationship_types", []) or []
                if not isinstance(rels, list):
                    rels = [rels] if rels else []
                incoming.append({"from": str(pred), "relationship_types": list(rels)})
            sections[node_s] = {"outgoing": outgoing, "incoming": incoming}

        payload = {
            "doc_id": resolved_doc_id,
            "generated_at": datetime.now(IST).isoformat(),
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "sections": sections,
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(p)


if __name__ == "__main__":
    # Static config (no args)
    DOC_ID = "ts_138401v180600p"
    CHUNKS_PATH = "./spec_chunks/ts_138401v180600p/chunks.json"
    OUTPUT_GRAPH_PATH = "./spec_chunks/ts_138401v180600p/KnowledgeGraph/knowledge_graph.pkl"
    OUTPUT_SUMMARY_PATH = "./spec_chunks/ts_138401v180600p/KnowledgeGraph/graph_summary.json"
    OUTPUT_ADJACENCY_JSON_PATH = (
        "./spec_chunks/ts_138401v180600p/KnowledgeGraph/knowledge_graph_adjacency.json"
    )
    INCLUDE_SEMANTIC_RELATIONS = True
    SEMANTIC_TOP_K_CANDIDATES = 10
    SEMANTIC_FINAL_K = 5
    SEMANTIC_MAX_CHUNKS = 350

    print("=" * 72)
    print("KG-Only Spec Knowledge Graph Builder")
    print("=" * 72)
    print(f"Entry point : {Path(__file__).name}")
    print(f"doc_id      : {DOC_ID}")
    print(f"chunks_path : {CHUNKS_PATH}")
    print(f"semantic    : {INCLUDE_SEMANTIC_RELATIONS}")
    print("-" * 72)

    builder = SpecKnowledgeGraphBuilder(doc_id=DOC_ID)
    chunks = builder._load_chunks(CHUNKS_PATH)
    graph = builder.build_graph(
        chunks,
        include_semantic_relations=INCLUDE_SEMANTIC_RELATIONS,
        semantic_top_k_candidates=SEMANTIC_TOP_K_CANDIDATES,
        semantic_final_k=SEMANTIC_FINAL_K,
        semantic_max_chunks=SEMANTIC_MAX_CHUNKS,
    )

    graph_path = builder.save_graph(graph, OUTPUT_GRAPH_PATH)
    summary_path = builder.save_graph_summary(graph, OUTPUT_SUMMARY_PATH)
    adjacency_path = builder.save_graph_adjacency_json(
        graph, OUTPUT_ADJACENCY_JSON_PATH, doc_id=DOC_ID
    )

    print(f"Nodes        : {graph.number_of_nodes()}")
    print(f"Edges        : {graph.number_of_edges()}")
    print(f"Graph saved  : {graph_path}")
    print(f"Summary saved: {summary_path}")
    print(f"Adjacency JSON: {adjacency_path}")
    print("=" * 72)

