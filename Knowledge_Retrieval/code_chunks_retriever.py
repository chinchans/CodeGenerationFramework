import os
import json
import pickle
import numpy as np
import faiss
import networkx as nx
from typing import List, Dict, Any, Optional
from collections import defaultdict
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime


class SemanticGraphRAG:
    def __init__(self, faiss_index_path: str, faiss_metadata_path: str, kg_path: str, feature_name: str = ""):
        self.faiss_index_path = faiss_index_path
        self.faiss_metadata_path = faiss_metadata_path
        self.kg_path = kg_path
        self.feature_name = feature_name
        
        self.faiss_index = None
        self.metadata = None
        self.embeddings = None
        self.kg: Optional[nx.DiGraph] = None
        self.nodeid_to_chunkid = {}

        self._load_vector_db()
        self._load_kg()
        self._build_node_lookup()

    def _load_vector_db(self):
        # print(f" Loading FAISS index: {self.faiss_index_path}")
        # FAISS expects a C string path; str() ensures Path or other path-like types work (e.g. when run via MCP from another directory)
        self.faiss_index = faiss.read_index(str(self.faiss_index_path))
        
        with open(self.faiss_metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # print(" Initializing Jina Embeddings...")
        self.embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-code-embeddings-0.5b")
        # print(f" Vector DB Ready: {self.faiss_index.ntotal} vectors")

    def _load_kg(self):
        if not os.path.exists(self.kg_path):
            # print(f" KG not found at {self.kg_path}")
            return
        with open(self.kg_path, "rb") as f:
            self.kg = pickle.load(f)
        # print(f" KG Loaded: {self.kg.number_of_nodes()} nodes")

    def _build_node_lookup(self):
        if self.metadata:
            for idx, meta in self.metadata.items():
                node_id = meta.get("node_id")
                if node_id:
                    self.nodeid_to_chunkid[node_id] = str(idx)

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding]).astype("float32")
        query_vector_normalized = self._normalize_vector(query_vector[0])
        query_array = np.array([query_vector_normalized]).astype("float32")

        k = min(top_k * 2, self.faiss_index.ntotal)
        distances, indices = self.faiss_index.search(query_array, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0: continue
            meta = self.metadata.get(str(idx))
            if not meta: continue

            distance = float(distances[0][i])
            # Standard Cosine Similarity from Distance
            cosine_similarity = max(0.0, 1.0 - (distance ** 2 / 2.0)) if distance < 2.0 else 1.0 / (1.0 + distance)

            results.append({
                "chunk_id": str(idx),
                "node_id": meta.get("node_id"),
                "chunk_type": meta.get("chunk_type"),
                "cosine_score": max(0.0, min(1.0, cosine_similarity)),
                "metadata": meta
            })
        results.sort(key=lambda x: x["cosine_score"], reverse=True)
        return results[:top_k]

    def traverse_from_multiple(self, start_nodes: List[str], depth: int, rel_filters: List[str], direction: str) -> Dict[str, int]:
        if self.kg is None: return {}
        G = self.kg if direction == "out" else self.kg.reverse(copy=False)
        
        visited = {node: 0 for node in start_nodes if node in G}
        frontier = set(visited.keys())

        for d in range(1, depth + 1):
            next_frontier = set()
            for node in frontier:
                for neighbor in G.successors(node):
                    rel_type = G[node][neighbor].get("relationship_type")
                    if rel_filters and rel_type not in rel_filters:
                        continue
                    if neighbor not in visited:
                        visited[neighbor] = d
                        next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier: break
        return visited

    def retrieve(self, query: str, top_k=10, kg_depth=2, rel_filters=None, direction="out"):
        # print(f" Searching: {query}")
        semantic_results = self.semantic_search(query, top_k)
        seed_nodes = [r["node_id"] for r in semantic_results if r.get("node_id")]
        
        depth_map = self.traverse_from_multiple(seed_nodes, kg_depth, rel_filters, direction)
        return semantic_results, depth_map, seed_nodes

    def build_output_json(self, query: str, semantic_results: List[Dict], depth_map: Dict[str, int], direction: str) -> Dict[str, Any]:
        output = {
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "user_query": query,
                "feature_name": self.feature_name,
                "num_semantic_chunks": len(semantic_results),
                "total_related_nodes": len(depth_map)
            },
            "semantic_chunks": [],
            "expanded_chunks": {}
        }

        # 1. Process Semantic Chunks
        for rank, chunk in enumerate(semantic_results, start=1):
            meta = chunk.get("metadata", {})
            output["semantic_chunks"].append({
                "chunk_id": chunk.get("chunk_id"),
                "node_id": chunk.get("node_id"),
                "chunk_type": chunk.get("chunk_type"),
                "chunk_text": meta.get("chunk_text", ""),
                "cosine_score": chunk.get("cosine_score"),
                "rank": rank,
                "metadata": meta
            })

        # 2. Process Expanded Chunks (Depth Grouping)
        grouped = defaultdict(list)
        for node_id, d in depth_map.items():
            if d > 0: grouped[d].append(node_id)

        for d in sorted(grouped.keys()):
            depth_key = f"depth_{d}"
            output["expanded_chunks"][depth_key] = []
            
            for node in grouped[d]:
                node_data = self.kg.nodes.get(node, {})
                
                # Predecessor search for relationship context
                source_node, rel_type = None, None
                for pred in self.kg.predecessors(node):
                    if pred in depth_map and depth_map[pred] == d - 1:
                        source_node = pred
                        rel_type = self.kg[pred][node].get("relationship_type")
                        break

                output["expanded_chunks"][depth_key].append({
                    "chunk_id": self.nodeid_to_chunkid.get(node),
                    "node_id": node,
                    "chunk_type": node_data.get("chunk_type") or (node.split(":")[0] if ":" in node else "unknown"),
                    "relationship_info": {
                        "source_node": source_node,
                        "relationship_type": rel_type,
                        "direction": direction
                    },
                    "metadata": {
                        "name": node_data.get("name"),
                        "file_path": node_data.get("file_path"),
                        "line_range": [node_data.get("line_start"), node_data.get("line_end")]
                    }
                })
        return output