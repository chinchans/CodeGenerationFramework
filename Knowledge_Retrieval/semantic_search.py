"""
Simple Semantic Search: FAISS-based semantic search for user queries
- Loads FAISS index and metadata
- Performs semantic search using embeddings
- Reranks results with LLM for optimal retrieval
- Returns ranked results with similarity scores
"""

import json
import os
import pickle
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv


import logging

logger = logging.getLogger(__name__)

load_dotenv()

# # Configuration
# KG_FILE = "./resources/ts_138401v180600p/KnowledgeGraph/knowledge_graph.pkl"
# FAISS_INDEX_FILE = "./resources/ts_138401v180600p/vector_db/faiss_index.index"
# FAISS_METADATA_FILE = "./resources/ts_138401v180600p/vector_db/faiss_metadata.json"

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o-mini")

# Reranking configuration
CONFIG = {
    "use_llm_reranking": True,  # Enable LLM reranking
    "rerank_top_k": 20,  # Top K to rerank with LLM
}

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"
)


class SemanticSearcher:
    """Simple semantic search using FAISS vector database"""
    
    def __init__(self, 
                 kg_file: str,
                 faiss_index_file: str,
                 faiss_metadata_file: str):
        """
        Initialize Semantic Searcher
        
        Args:
            kg_file: Path to knowledge graph pickle file
            faiss_index_file: Path to FAISS index file
            faiss_metadata_file: Path to FAISS metadata JSON file
        """
        # print(" Initializing Semantic Searcher...")
        # print("-" * 60)
        
        # Load knowledge graph to get chunk data
        # print(" Loading knowledge graph...")
        if not os.path.exists(kg_file):
            raise FileNotFoundError(f"Knowledge graph not found: {kg_file}")
        with open(kg_file, 'rb') as f:
            self.kg = pickle.load(f)
        # print(f"    Loaded graph: {self.kg.number_of_nodes()} nodes, {self.kg.number_of_edges()} edges")
        
        # Load FAISS index
        # print(" Loading FAISS index...")
        faiss_path = str(faiss_index_file)
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
        # FAISS expects a C string path; str() ensures Path or other path-like types work (e.g. when run via MCP from another directory)
        self.faiss_index = faiss.read_index(faiss_path)
        # print(f"    Loaded FAISS index: {self.faiss_index.ntotal} vectors")
        
        # Load metadata
        # print(" Loading metadata...")
        if not os.path.exists(faiss_metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {faiss_metadata_file}")
        with open(faiss_metadata_file, 'r') as f:
            self.metadata = json.load(f)
        self.id_to_section = self.metadata.get("id_to_section", {})
        self.section_to_chunk = self.metadata.get("section_to_chunk", {})
        # print(f"    Loaded metadata: {len(self.id_to_section)} mappings")
        
        # Build chunk dictionary from knowledge graph
        # print("Building chunk dictionary...")
        self.all_chunks = {}
        for section_id in self.kg.nodes():
            node_data = self.kg.nodes[section_id]
            self.all_chunks[section_id] = {
                "section_id": section_id,
                "section_title": node_data.get("section_title", ""),
                "content": node_data.get("content", ""),
                "metadata": node_data
            }
        # print(f"    Built chunk dictionary: {len(self.all_chunks)} chunks")
        
        # Initialize LLM for reranking (if enabled)
        if CONFIG["use_llm_reranking"]:
            # print(" Initializing Azure OpenAI LLM for reranking...")
            if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
                # print(f"     Azure OpenAI credentials not found")
                # print(f"     Reranking will be disabled")
                self.llm = None
            else:
                try:
                    self.llm = AzureChatOpenAI(
                        api_key=AZURE_OPENAI_API_KEY,
                        api_version=AZURE_OPENAI_API_VERSION,
                        azure_endpoint=AZURE_OPENAI_ENDPOINT,
                        azure_deployment=AZURE_DEPLOYMENT_NAME,
                        temperature=0.3,  # Lower temperature for consistent ranking
                    )
                    # print(f"   LLM loaded: {AZURE_DEPLOYMENT_NAME}")
                except Exception as e:
                    # print(f"     Failed to load LLM: {str(e)}")
                    # print(f"     Reranking will be disabled")
                    self.llm = None
        else:
            self.llm = None
        
        # print(" Initialization complete!")
        # print()
    
    def search(self, query: str, top_k: int = 10, rerank: bool = True) -> Dict[str, Any]:
        """
        Perform semantic search on user query with optional reranking
        
        Args:
            query: User query string
            top_k: Number of results to return
            rerank: Whether to rerank results with LLM
            
        Returns:
            Dictionary with 'original_results' and 'final_results' (after reranking and expansion)
        """
        # print(f" Semantic Search for query: '{query}'")
        # print("-" * 60)
        
        start_time = time.time()
        
        # Generate query embedding
        # print(f"    Generating query embedding...")
        query_embedding = embeddings.embed_query(query)
        query_embedding_array = np.array([query_embedding]).astype('float32')
        
        # Normalize query embedding for cosine similarity
        query_norm = np.linalg.norm(query_embedding_array)
        if query_norm > 0:
            query_embedding_array = query_embedding_array / query_norm
        
        # Search FAISS (returns L2 distances)
        # print(f"    Searching FAISS index (top_k={top_k})...")
        # Get more results for reranking if enabled
        search_k = CONFIG["rerank_top_k"] if rerank and CONFIG["use_llm_reranking"] else top_k * 2
        k = min(search_k, self.faiss_index.ntotal)
        distances, indices = self.faiss_index.search(query_embedding_array, k)
        
        # Process results
        results = []
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.id_to_section):
                continue
            
            section_id = self.id_to_section.get(str(idx))
            if not section_id or section_id not in self.all_chunks:
                continue
            
            # Get chunk
            chunk = self.all_chunks[section_id].copy()
            
            # Convert L2 distance to similarity score (approximate)
            # For normalized vectors: cosine_sim ≈ 1 - (L2_distance^2 / 2)
            similarity_score = max(0.0, 1.0 - (distance / 2.0))
            
            chunk["semantic_score"] = round(similarity_score, 4)
            chunk["rank"] = i + 1
            
            results.append(chunk)
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x["semantic_score"], reverse=True)
        
        # Store original results before reranking
        original_results = results.copy()
        
        # Rerank with LLM if enabled
        reranked_results_list = None
        if rerank and CONFIG["use_llm_reranking"] and self.llm is not None:
            # Take top rerank_top_k for reranking
            rerank_candidates = results[:CONFIG["rerank_top_k"]]
            reranked_results_list = self.rerank_with_llm(query, rerank_candidates)
            # Remaining chunks that weren't reranked keep their original semantic scores
            remaining_chunks = results[CONFIG["rerank_top_k"]:]
            # Combine: reranked first, then remaining
            results = reranked_results_list + remaining_chunks
        
        # Take top 10 reranked chunks for KG expansion
        top_10_reranked = reranked_results_list[:10] if reranked_results_list else results[:10]
        
        # Expand top 10 reranked chunks with Knowledge Graph
        expanded_chunks_list = []
        expanded_by_depth = {}
        if top_10_reranked:
            # print(f"    Expanding top 10 reranked chunks using Knowledge Graph...")
            expanded_results = self.expand_chunks_with_kg(top_10_reranked, depth=2)
            # Get expanded chunks organized by depth
            expanded_chunks_list = expanded_results.get("all_expanded", [])
            expanded_by_depth = expanded_results.get("by_depth", {})
        
        # Final results: original chunks first, then expanded chunks
        final_results = results + expanded_chunks_list
        
        elapsed = time.time() - start_time
        
        # print(f"    Semantic search completed in {elapsed:.3f}s")
        # print(f"      Found {len(final_results)} results")
        original_count = len([c for c in final_results if not c.get("from_graph_expansion", False)])
        expanded_count = len([c for c in final_results if c.get("from_graph_expansion", False)])
        # print(f"      - Original chunks: {original_count}")
        # print(f"      - Expanded chunks: {expanded_count}")
        if final_results:
            if 'llm_rerank_score' in final_results[0]:
                pass  # print(f"      Top LLM rerank score: {final_results[0]['llm_rerank_score']:.4f}")
                
            pass  # print(f"      Top semantic score: {final_results[0]['semantic_score']:.4f}")
        # print()
        
        return {
            "original_results": original_results,
            "reranked_results": reranked_results_list if reranked_results_list else original_results[:CONFIG["rerank_top_k"]],
            "expanded_chunks": expanded_chunks_list,
            "expanded_by_depth": expanded_by_depth,
            "final_results": final_results
        }
    
    def rerank_with_llm(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank chunks using LLM
        
        Args:
            query: User query string
            chunks: List of chunks to rerank
            
        Returns:
            Reranked list of chunks with LLM scores
        """
        if not CONFIG["use_llm_reranking"] or self.llm is None:
            return chunks
        
        if not chunks:
            return chunks
        
        # print(f"    Reranking {len(chunks)} chunks using LLM...")
        
        start_time = time.time()
        
        # Prepare chunks information for LLM
        chunks_info = []
        for i, chunk in enumerate(chunks):
            section_id = chunk.get("section_id", "")
            section_title = chunk.get("section_title", "")
            content = chunk.get("content", "")[:1000]  # Limit content length
            
            chunks_info.append({
                "index": i,
                "section_id": section_id,
                "section_title": section_title,
                "content_preview": content
            })
        
        # Create prompt for LLM reranking
        chunks_text = "\n\n".join([
            f"[{i+1}] Section ID: {info['section_id']}\n"
            f"Title: {info['section_title']}\n"
            f"Content: {info['content_preview'][:500]}..."
            for i, info in enumerate(chunks_info)
        ])
        
        prompt = f"""You are a ranking expert. Rank the following document sections based on their relevance to the user query.

User Query: {query}

Document Sections:
{chunks_text}

Rank these sections from most relevant (1) to least relevant ({len(chunks)}). Consider:
- How well the section title and content match the query
- The specificity and depth of information related to the query
- The completeness of information for answering the query

Return ONLY a valid JSON array with the ranking. Each item should have:
- "rank": integer from 1 to {len(chunks)} (1 = most relevant)
- "section_id": the section ID
- "relevance_score": float between 0.0 and 1.0 (1.0 = most relevant)
- "reasoning": brief explanation (1-2 sentences)

Format:
[
  {{"rank": 1, "section_id": "5.1.2", "relevance_score": 0.95, "reasoning": "Most relevant because..."}},
  {{"rank": 2, "section_id": "5.3.1", "relevance_score": 0.85, "reasoning": "Relevant because..."}},
  ...
]

Ensure all {len(chunks)} sections are ranked and each rank (1-{len(chunks)}) appears exactly once."""
        
        try:
            # Call LLM
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            
            # Clean JSON response
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            ranking_data = json.loads(response_text)
            
            if not isinstance(ranking_data, list):
                raise ValueError("Expected list from LLM")
            
            # Create mapping from section_id to ranking info
            ranking_map = {}
            for item in ranking_data:
                section_id = item.get("section_id")
                if section_id:
                    ranking_map[section_id] = {
                        "rank": item.get("rank", len(chunks)),
                        "relevance_score": item.get("relevance_score", 0.0),
                        "reasoning": item.get("reasoning", "")
                    }
            
            # Rerank chunks based on LLM ranking
            reranked_chunks = []
            for chunk in chunks:
                section_id = chunk.get("section_id")
                chunk_copy = chunk.copy()
                
                if section_id in ranking_map:
                    rank_info = ranking_map[section_id]
                    chunk_copy["llm_rerank_score"] = round(rank_info["relevance_score"], 4)
                    chunk_copy["llm_rank"] = rank_info["rank"]
                    chunk_copy["llm_reasoning"] = rank_info["reasoning"]
                else:
                    # If section not in ranking, assign default values
                    chunk_copy["llm_rerank_score"] = 0.0
                    chunk_copy["llm_rank"] = len(chunks) + 1
                    chunk_copy["llm_reasoning"] = "Not ranked by LLM"
                
                reranked_chunks.append(chunk_copy)
            
            # Sort by LLM rank (ascending, 1 = best)
            reranked_chunks.sort(key=lambda x: x.get("llm_rank", len(chunks) + 1))
            
            # Update ranks
            for i, chunk in enumerate(reranked_chunks):
                chunk["rank"] = i + 1
            
            elapsed = time.time() - start_time
            
            # print(f"      LLM reranking completed in {elapsed:.3f}s")
            if reranked_chunks:
                pass  # print(f"         Top LLM rerank score: {reranked_chunks[0].get('llm_rerank_score', 0):.4f}")
                
            
            return reranked_chunks
            
        except json.JSONDecodeError as e:
            # print(f"        JSON parse error in LLM response: {str(e)[:100]}")
            # print(f"        Returning original ranking")
            return chunks
        except Exception as e:
            # print(f"        LLM reranking failed: {str(e)[:100]}")
            # print(f"        Returning original ranking")
            return chunks
    
    def _expand_graph(self, initial_section_ids: set, depth: int = 2) -> Dict[str, Dict[str, Any]]:
        """
        Expand graph by traversing relationships and track expansion source, depth, and paths
        
        Args:
            initial_section_ids: Set of starting section IDs
            depth: Maximum traversal depth
            
        Returns:
            Dictionary mapping expanded section_id -> {
                'source': original_source_section_id,
                'relationship_types': list of relationship types,
                'direct_source': immediate parent in expansion path,
                'depth_level': depth at which this node was found,
                'expansion_path': list of section_ids from source to this node
            }
        """
        expansion_map = {}  # expanded_section_id -> expansion info
        all_section_ids = set(initial_section_ids)
        current_level = set(initial_section_ids)
        
        # Track depth for each section
        section_depth = {sid: 0 for sid in initial_section_ids}
        
        # Helper function to build expansion path
        def build_expansion_path(section_id: str) -> List[str]:
            """Build path from original source to current section"""
            if section_id in initial_section_ids:
                return [section_id]
            if section_id in expansion_map:
                direct_source = expansion_map[section_id].get('direct_source')
                if direct_source:
                    path = build_expansion_path(direct_source)
                    path.append(section_id)
                    return path
            return [section_id]  # Fallback
        
        for level in range(1, depth + 1):
            next_level = set()
            
            for section_id in current_level:
                if not self.kg.has_node(section_id):
                    continue
                
                # Get all neighbors (both successors and predecessors)
                successors = list(self.kg.successors(section_id))
                predecessors = list(self.kg.predecessors(section_id))
                
                # Add neighbors to next level
                for neighbor in successors + predecessors:
                    if neighbor not in all_section_ids:
                        next_level.add(neighbor)
                        all_section_ids.add(neighbor)
                        section_depth[neighbor] = level
                        
                        # Build expansion path
                        expansion_path = build_expansion_path(section_id)
                        expansion_path.append(neighbor)
                        
                        # Get relationship types from KG
                        relationship_types = []
                        if self.kg.has_node(section_id) and self.kg.has_node(neighbor):
                            # Check edge from current to neighbor
                            if self.kg.has_edge(section_id, neighbor):
                                edge_data = self.kg[section_id][neighbor]
                                rel_types = edge_data.get("relationship_types", [])
                                if isinstance(rel_types, list):
                                    relationship_types.extend(rel_types)
                                elif rel_types:
                                    relationship_types.append(rel_types)
                            
                            # Check edge from neighbor to current (reverse direction)
                            if self.kg.has_edge(neighbor, section_id):
                                edge_data = self.kg[neighbor][section_id]
                                rel_types = edge_data.get("relationship_types", [])
                                if isinstance(rel_types, list):
                                    relationship_types.extend(rel_types)
                                elif rel_types:
                                    relationship_types.append(rel_types)
                        
                        # Remove duplicates
                        relationship_types = list(set(relationship_types)) if relationship_types else ["RELATED_TO"]
                        
                        # Find original source (first in path)
                        original_source = expansion_path[0] if expansion_path else section_id
                        
                        expansion_map[neighbor] = {
                            'source': original_source,
                            'relationship_types': relationship_types,
                            'direct_source': section_id,  # Immediate parent in expansion
                            'depth_level': level,
                            'expansion_path': expansion_path  # Full path from source
                        }
            
            if not next_level:
                break  # No more nodes to expand
            
            current_level = next_level
        
        return expansion_map
    
    def expand_chunks_with_kg(self, chunks: List[Dict[str, Any]], depth: int = 2) -> Dict[str, List[Dict[str, Any]]]:
        """
        Expand chunks using Knowledge Graph relationships, organized by depth
        
        Args:
            chunks: List of initial chunks
            depth: Maximum depth for graph traversal
            
        Returns:
            Dictionary with keys 'all_expanded' (all expanded chunks) and 'by_depth' (organized by depth level)
        """
        if not chunks:
            return {"all_expanded": [], "by_depth": {}}
        
        # print(f"    Expanding {len(chunks)} chunks using Knowledge Graph (depth={depth})...")
        
        # Get initial section IDs
        initial_section_ids = {chunk["section_id"] for chunk in chunks if "section_id" in chunk}
        
        # Expand using graph and get expansion map
        expansion_map = self._expand_graph(initial_section_ids, depth=depth)
        
        # Create a set of existing section IDs to track which chunks are new
        existing_section_ids = initial_section_ids
        
        # Create a dictionary to store chunks by section_id for quick lookup
        chunks_dict = {chunk["section_id"]: chunk for chunk in chunks}
        
        # Organize expanded chunks by depth
        expanded_by_depth = {}
        all_expanded_chunks = []
        new_chunks_count = 0
        
        for section_id, expansion_info in expansion_map.items():
            if section_id not in existing_section_ids:
                # This is a new chunk from graph expansion
                if section_id in self.all_chunks:
                    new_chunk = self.all_chunks[section_id].copy()
                    # Mark as expanded chunk
                    new_chunk["from_graph_expansion"] = True
                    
                    depth_level = expansion_info.get('depth_level', 1)
                    expansion_path = expansion_info.get('expansion_path', [])
                    direct_source = expansion_info.get('direct_source')
                    relationship_types = expansion_info.get('relationship_types', ["RELATED_TO"])
                    
                    # Build expanded_chunk_from structure
                    if len(expansion_path) > 1:
                        # Multi-depth: build path
                        path_list = []
                        for idx, path_section_id in enumerate(expansion_path[:-1]):  # Exclude current node
                            # Determine depth: 0 for initial chunks, or from expansion_map
                            if path_section_id in initial_section_ids:
                                path_depth = 0
                            else:
                                path_depth = expansion_map.get(path_section_id, {}).get('depth_level', idx)
                            
                            path_chunk = chunks_dict.get(path_section_id)
                            if path_chunk:
                                path_list.append({
                                    "section_id": path_section_id,
                                    "section_title": path_chunk.get("section_title", path_section_id),
                                    "depth": path_depth
                                })
                            else:
                                # If not in chunks_dict, try to get from all_chunks
                                if path_section_id in self.all_chunks:
                                    path_list.append({
                                        "section_id": path_section_id,
                                        "section_title": self.all_chunks[path_section_id].get("section_title", path_section_id),
                                        "depth": path_depth
                                    })
                        
                        # Immediate source (direct parent)
                        immediate_source_chunk = chunks_dict.get(direct_source) if direct_source else None
                        if not immediate_source_chunk and direct_source in self.all_chunks:
                            immediate_source_chunk = {"section_id": direct_source, "section_title": self.all_chunks[direct_source].get("section_title", direct_source)}
                        
                        new_chunk["expanded_chunk_from"] = {
                            "path": path_list,
                            "immediate_source": {
                                "section_id": direct_source,
                                "section_title": immediate_source_chunk.get("section_title", direct_source) if immediate_source_chunk else direct_source
                            } if direct_source else None
                        }
                    else:
                        # Single depth: just direct source
                        source_chunk = chunks_dict.get(direct_source) if direct_source else None
                        if not source_chunk and direct_source in self.all_chunks:
                            source_chunk = {"section_id": direct_source, "section_title": self.all_chunks[direct_source].get("section_title", direct_source)}
                        
                        new_chunk["expanded_chunk_from"] = {
                            "section_id": direct_source,
                            "section_title": source_chunk.get("section_title", direct_source) if source_chunk else direct_source
                        } if direct_source else None
                    
                    # Store depth level and relationship types
                    new_chunk["depth_level"] = depth_level
                    new_chunk["expansion_relation"] = relationship_types
                    
                    # Set default scores to 0 (these are not from search)
                    new_chunk["semantic_score"] = 0.0
                    new_chunk["search_methods"] = ["graph_expansion"]
                    
                    # Organize by depth
                    depth_key = f"depth_{depth_level}"
                    if depth_key not in expanded_by_depth:
                        expanded_by_depth[depth_key] = []
                    expanded_by_depth[depth_key].append(new_chunk)
                    all_expanded_chunks.append(new_chunk)
                    new_chunks_count += 1
        
        # print(f"       Expanded to {len(all_expanded_chunks)} total chunks ({new_chunks_count} new from KG)")
        for depth_key in sorted(expanded_by_depth.keys()):
            pass  # print(f"         - {depth_key}: {len(expanded_by_depth[depth_key])} chunks")
            
        
        return {
            "all_expanded": all_expanded_chunks,
            "by_depth": expanded_by_depth
        }


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def main():
    """Main function for example usage"""
    # Example query
    query = "Implement F1AP UE Context Setup Request Message Procedure of Inter-gNB-DU LTM Handover"
    
    # Initialize searcher
    searcher = SemanticSearcher()
    
    # Perform semantic search
    search_results = searcher.search(query, top_k=20)
    
    original_results = search_results["original_results"]
    reranked_results = search_results["reranked_results"]
    expanded_chunks = search_results["expanded_chunks"]
    final_results = search_results["final_results"]
    
    # Display original chunks (before reranking)
    # print("=" * 60)
    # print(" Original Chunks (Before Reranking)")
    # print("=" * 60)
    # print(f"   Total original chunks: {len(original_results)}")
    # print(f"   Showing top 10 chunks\n")
    
    for i, chunk in enumerate(original_results[:10], 1):
        pass  # print(f"{i}.  Section: {chunk['section_id']}")
        # print(f"   Title: {chunk.get('section_title', 'N/A')}")
        # print(f"   Semantic Score: {chunk.get('semantic_score', 0):.4f}")
        # print(f"   Content preview: {chunk.get('content', '')[:150]}...")
        # print()
        
    
    if len(original_results) > 10:
        pass  # print(f"... and {len(original_results) - 10} more original chunks (not shown)")
        # print()
        
    
    # Display top 10 reranked chunks
    # print("=" * 60)
    # print(" Top 10 Reranked Chunks (After LLM Reranking)")
    # print("=" * 60)
    # print(f"   Total reranked chunks: {len(reranked_results)}")
    # print(f"   Showing top {min(10, len(reranked_results))} chunks\n")
    
    for i, chunk in enumerate(reranked_results[:10], 1):
        pass  # print(f"{i}.  Section: {chunk['section_id']}")
        # print(f"   Title: {chunk.get('section_title', 'N/A')}")
        
        # if 'llm_rerank_score' in chunk:
        #     print(f"   LLM Rerank Score: {chunk.get('llm_rerank_score', 0):.4f} (Rank: {chunk.get('llm_rank', 'N/A')})")
        #     if 'llm_reasoning' in chunk:
        #         print(f"   LLM Reasoning: {chunk.get('llm_reasoning', '')[:100]}")
                
        # print(f"   Semantic Score: {chunk.get('semantic_score', 0):.4f}")
        # print(f"   Content preview: {chunk.get('content', '')[:150]}...")
        # print()
    
    if len(reranked_results) > 10:
        pass  # print(f"... and {len(reranked_results) - 10} more reranked chunks (not shown)")
        # print()
        
    
    # Display expanded chunks with relationship information
    # print("=" * 60)
    # print(" Expanded Chunks from Knowledge Graph")
    # print("=" * 60)
    # print(f"   Total expanded chunks: {len(expanded_chunks)}")
    # print(f"   Showing top 10 chunks\n")
    
    if expanded_chunks:
        top_10_expanded = expanded_chunks[:10]
        for i, chunk in enumerate(top_10_expanded, 1):
            expansion_source = chunk.get("expansion_source", "Unknown")
            relationship_types = chunk.get("relationship_types", ["RELATED_TO"])
            
            pass  # print(f"{i}.  Section: {chunk['section_id']}")
            # print(f"   Title: {chunk.get('section_title', 'N/A')}")
            # print(f"    Expanded from: {expansion_source}")
            # print(f"    Relationship types: {', '.join(relationship_types)}")
            # print(f"   Semantic Score: {chunk.get('semantic_score', 0):.4f}")
            # print(f"   Content preview: {chunk.get('content', '')[:150]}...")
            # print()
        
        if len(expanded_chunks) > 10:
            pass  # print(f"... and {len(expanded_chunks) - 10} more expanded chunks (not shown)")
            # print()
            
    else:
        pass  # print("   No expanded chunks found.\n")
        
    
    # Save all chunks to JSON file
    # print("=" * 60)
    # print(" Saving all chunks to JSON file...")
    # print("=" * 60)
    
    # Create output directory if it doesn't exist
    output_dir = "./outputs/retrieved_chunks"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize query for filename
    safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
    safe_query = safe_query.replace(' ', '_')
    filename = f"all_chunks_{safe_query}_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)
    
    # Prepare data to save
    output_data = {
        "query": query,
        "timestamp": timestamp,
        "total_chunks": len(final_results),
        "original_chunks_count": len(original_results),
        "reranked_chunks_count": len(reranked_results),
        "expanded_chunks_count": len(expanded_chunks),
        "chunks": final_results
    }
    
    # Convert numpy types to Python native types for JSON serialization
    output_data = convert_numpy_types(output_data)
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # print(f"    All chunks saved to: {output_path}")
    # print(f"      Total chunks: {len(final_results)}")
    # print()
    
    # print(" Process completed!")


if __name__ == "__main__":
    main()
