"""
Multi-Source Chunk Retriever
- Retrieves chunks from multiple knowledge sources using SemanticSearcher
- Stores all chunks from all sources
- Selects top 60-80% chunks from each source
- Handles multiple knowledge sources configuration
"""

import json
import os
import sys
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from datetime import datetime
import numpy as np


# ----------------------------------------------------------
# Add project root to Python path
# ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from semantic_search import SemanticSearcher, convert_numpy_types

load_dotenv()

# Google Gemini Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# print(f"GOOGLE_API_KEY: {GOOGLE_API_KEY}")
if not GOOGLE_API_KEY:
    raise ValueError(
        "Google API key not found. Please set GOOGLE_API_KEY environment variable."
    )
# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    api_key=GOOGLE_API_KEY,
    model="gemini-2.5-flash",
    temperature=0.4,
    top_p=0.4
)


class MultiSourceChunkRetriever:
    """
    Multi-source chunk retriever that retrieves chunks from multiple knowledge sources
    """
    
    def __init__(self, 
                 knowledge_sources: List[Dict[str, str]],
                 top_chunk_percentage: float = 0.7,
                 template_file: str = None):
        """
        Initialize Multi-Source Chunk Retriever
        
        Args:
            knowledge_sources: List of dicts with keys:
                - source_id: Unique identifier for the source (e.g., "ts_138473v180600p")
                - kg_file: Path to knowledge graph file
                - faiss_index_file: Path to FAISS index file
                - faiss_metadata_file: Path to FAISS metadata file
            top_chunk_percentage: Percentage of top chunks to keep (0.6-0.8, default 0.7 = 70%)
        """
        # print(" Initializing Multi-Source Chunk Retriever...")
        # print("-" * 60)
        
        self.knowledge_sources = knowledge_sources
        self.top_chunk_percentage = max(0.7, min(0.8, top_chunk_percentage))  # Clamp between 60-80%
        # Load template
        if os.path.exists(template_file):
            with open(template_file, 'r') as f:
                self.template = json.load(f)
        else:
            self.template = {}
            # print(f"     Template file not found: {template_file}")
        

        # Initialize SemanticSearcher for each source
        self.searchers = {}
        for source_config in knowledge_sources:
            source_id = source_config["source_id"]
            # print(f" Initializing searcher for source: {source_id}")
            
            try:
                searcher = SemanticSearcher(
                    kg_file=source_config["kg_file"],
                    faiss_index_file=source_config["faiss_index_file"],
                    faiss_metadata_file=source_config["faiss_metadata_file"]
                )
                self.searchers[source_id] = searcher
                # print(f"    Source {source_id} initialized")
            except Exception as e:
                # print(f"     Failed to initialize source {source_id}: {str(e)}")
                continue
        
        # print(f"Multi-Source Retriever initialized with {len(self.searchers)} sources")
        # print(f"   Top chunk percentage: {self.top_chunk_percentage*100:.0f}%")
        # print()
    
    def retrieve_all_chunks(self, 
                          query: str, 
                          top_k_per_source: int = 50,
                          rerank: bool = True) -> Dict[str, Any]:
        """
        Retrieve all chunks from all knowledge sources
        
        Args:
            query: User query string
            top_k_per_source: Number of chunks to retrieve per source (before filtering)
            rerank: Whether to use LLM reranking
            
        Returns:
            Dictionary with:
                - all_chunks_by_source: Dict[source_id, List[chunks]]
                - top_chunks_by_source: Dict[source_id, List[chunks]]
                - all_chunks_combined: List of all chunks (with source metadata)
                - top_chunks_combined: List of top chunks (with source metadata)
        """
        # print("=" * 60)
        # print(" Phase 1: Retrieving All Chunks from All Sources")
        # print("=" * 60)
        # print(f"Query: {query}")
        # print(f"Sources: {list(self.searchers.keys())}")
        # print(f"Top K per source: {top_k_per_source}")
        # print()
        
        all_chunks_by_source = {}
        all_chunks_combined = []
        
        # Store semantic, reranked, and expanded chunks separately
        semantic_chunks_by_source = {}
        reranked_chunks_by_source = {}
        expanded_chunks_by_source = {}
        expanded_by_depth_by_source = {}
        
        # Retrieve from each source
        for source_id, searcher in self.searchers.items():
            # print(f" Retrieving from source: {source_id}")
            # print("-" * 60)
            
            try:
                # Perform semantic search with reranking
                search_results = searcher.search(query, top_k=top_k_per_source, rerank=rerank)
                
                # Get separate results
                original_results = search_results.get("original_results", [])
                reranked_results = search_results.get("reranked_results", [])
                expanded_chunks = search_results.get("expanded_chunks", [])
                expanded_by_depth = search_results.get("expanded_by_depth", {})
                final_results = search_results.get("final_results", [])
                
                # Add source metadata to each chunk type
                semantic_chunks = []
                for chunk in original_results:
                    chunk_copy = chunk.copy()
                    chunk_copy["knowledge_source"] = source_id
                    chunk_copy["source_id"] = source_id
                    chunk_copy = convert_numpy_types(chunk_copy)
                    semantic_chunks.append(chunk_copy)
                
                reranked_chunks = []
                for chunk in reranked_results:
                    chunk_copy = chunk.copy()
                    chunk_copy["knowledge_source"] = source_id
                    chunk_copy["source_id"] = source_id
                    chunk_copy = convert_numpy_types(chunk_copy)
                    reranked_chunks.append(chunk_copy)
                
                expanded_chunks_with_source = []
                for chunk in expanded_chunks:
                    chunk_copy = chunk.copy()
                    chunk_copy["knowledge_source"] = source_id
                    chunk_copy["source_id"] = source_id
                    chunk_copy = convert_numpy_types(chunk_copy)
                    expanded_chunks_with_source.append(chunk_copy)
                
                # Store expanded by depth with source metadata
                expanded_by_depth_with_source = {}
                for depth_key, depth_chunks in expanded_by_depth.items():
                    expanded_by_depth_with_source[depth_key] = []
                    for chunk in depth_chunks:
                        chunk_copy = chunk.copy()
                        chunk_copy["knowledge_source"] = source_id
                        chunk_copy["source_id"] = source_id
                        chunk_copy = convert_numpy_types(chunk_copy)
                        expanded_by_depth_with_source[depth_key].append(chunk_copy)
                
                # Add source metadata to final results
                for chunk in final_results:
                    chunk_copy = chunk.copy()
                    chunk_copy["knowledge_source"] = source_id
                    chunk_copy["source_id"] = source_id
                    chunk_copy = convert_numpy_types(chunk_copy)
                    all_chunks_combined.append(chunk_copy)
                
                # Store by source
                all_chunks_by_source[source_id] = final_results
                semantic_chunks_by_source[source_id] = semantic_chunks
                reranked_chunks_by_source[source_id] = reranked_chunks
                expanded_chunks_by_source[source_id] = expanded_chunks_with_source
                expanded_by_depth_by_source[source_id] = expanded_by_depth_with_source
                
                # print(f"    Retrieved {len(final_results)} chunks from {source_id}")
                # print(f"      - Semantic: {len(semantic_chunks)}")
                # print(f"      - Reranked: {len(reranked_chunks)}")
                # print(f"      - Expanded: {len(expanded_chunks_with_source)}")
                
            except Exception as e:
                # print(f"     Error retrieving from {source_id}: {str(e)}")
                all_chunks_by_source[source_id] = []
                semantic_chunks_by_source[source_id] = []
                reranked_chunks_by_source[source_id] = []
                expanded_chunks_by_source[source_id] = []
                expanded_by_depth_by_source[source_id] = {}
        
        # print()
        # print(f" Total chunks retrieved: {len(all_chunks_combined)}")
        # print(f"   Breakdown by source:")
        for source_id, chunks in all_chunks_by_source.items():
            pass  # print(f"      - {source_id}: {len(chunks)} chunks")
            
        # print()
        
        # Select top chunks from each source
        top_chunks_by_source = {}
        top_chunks_combined = []
        
        # print("=" * 60)
        # print(f" Phase 2: Selecting Top {self.top_chunk_percentage*100:.0f}% Chunks")
        # print("=" * 60)
        
        for source_id, chunks in all_chunks_by_source.items():
            if not chunks:
                top_chunks_by_source[source_id] = []
                continue
            
            # Calculate how many chunks to keep
            num_to_keep = max(1, int(len(chunks) * self.top_chunk_percentage))
            
            # Sort chunks by score (prefer llm_rerank_score, fallback to semantic_score)
            def get_sort_score(chunk):
                # Prefer LLM rerank score if available
                if 'llm_rerank_score' in chunk and chunk.get('llm_rerank_score', 0) > 0:
                    return chunk.get('llm_rerank_score', 0)
                # Fallback to semantic score
                return chunk.get('semantic_score', 0)
            
            sorted_chunks = sorted(chunks, key=get_sort_score, reverse=True)
            top_chunks = sorted_chunks[:num_to_keep]
            
            # Add source metadata
            for chunk in top_chunks:
                chunk_copy = chunk.copy()
                chunk_copy["knowledge_source"] = source_id
                chunk_copy["source_id"] = source_id
                chunk_copy = convert_numpy_types(chunk_copy)
                top_chunks_combined.append(chunk_copy)
            
            top_chunks_by_source[source_id] = top_chunks
            
            # print(f"   {source_id}: {len(chunks)} : {len(top_chunks)} chunks (top {self.top_chunk_percentage*100:.0f}%)")
        
        # print()
        # print(f"Top chunks selected: {len(top_chunks_combined)}")
        # print()
        
        return {
            "all_chunks_by_source": all_chunks_by_source,
            "top_chunks_by_source": top_chunks_by_source,
            "all_chunks_combined": all_chunks_combined,
            "top_chunks_combined": top_chunks_combined,
            "semantic_chunks_by_source": semantic_chunks_by_source,
            "reranked_chunks_by_source": reranked_chunks_by_source,
            "expanded_chunks_by_source": expanded_chunks_by_source,
            "expanded_by_depth_by_source": expanded_by_depth_by_source,
            "query": query,
            "retrieval_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
    
    def save_all_chunks(self, 
                       retrieval_results: Dict[str, Any],
                       output_dir: str = "./backend/resources") -> str:
        """
        Save all retrieved chunks to JSON file in the new format
        
        Args:
            retrieval_results: Results from retrieve_all_chunks
            output_dir: Directory to save output (default: backend/resources per user preference)
            
        Returns:
            Path to saved JSON file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Helper function to format semantic chunk
        def format_semantic_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "section_id": chunk.get("section_id", ""),
                "section_title": chunk.get("section_title", ""),
                "chunk_text": chunk.get("content", ""),
                "metadata": {
                    "knowledge_source": chunk.get("knowledge_source", ""),
                    "rank": chunk.get("rank", 0),
                    "semantic_score": chunk.get("semantic_score", 0.0),
                    **{k: v for k, v in chunk.get("metadata", {}).items() if k not in ["content"]}
                }
            }
        
        # Helper function to format reranked chunk
        def format_reranked_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "section_id": chunk.get("section_id", ""),
                "section_title": chunk.get("section_title", ""),
                "chunk_text": chunk.get("content", ""),
                "score_by_llm": chunk.get("llm_rerank_score", 0.0),
                "score_by_cosine_or_semantic_search": chunk.get("semantic_score", 0.0),
                "metadata": {
                    "knowledge_source": chunk.get("knowledge_source", ""),
                    "llm_rank": chunk.get("llm_rank", 0),
                    "semantic_rank": chunk.get("rank", 0),
                    "llm_reasoning": chunk.get("llm_reasoning", ""),
                    **{k: v for k, v in chunk.get("metadata", {}).items() if k not in ["content"]}
                }
            }
        
        # Helper function to format expanded chunk
        def format_expanded_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
            expanded_from = chunk.get("expanded_chunk_from", {})
            
            return {
                "section_id": chunk.get("section_id", ""),
                "section_title": chunk.get("section_title", ""),
                "chunk_text": chunk.get("content", ""),
                "expanded_chunk_from": expanded_from,
                "depth_level": chunk.get("depth_level", 1),
                "expansion_relation": chunk.get("expansion_relation", []),
                "metadata": {
                    "knowledge_source": chunk.get("knowledge_source", ""),
                    **{k: v for k, v in chunk.get("metadata", {}).items() if k not in ["content"]}
                }
            }
        
        # Collect all chunks from all sources
        all_semantic_chunks = []
        all_reranked_chunks = []
        all_expanded_by_depth = {}
        
        semantic_chunks_by_source = retrieval_results.get("semantic_chunks_by_source", {})
        reranked_chunks_by_source = retrieval_results.get("reranked_chunks_by_source", {})
        expanded_by_depth_by_source = retrieval_results.get("expanded_by_depth_by_source", {})
        
        # Collect semantic chunks
        for source_id, chunks in semantic_chunks_by_source.items():
            all_semantic_chunks.extend([format_semantic_chunk(chunk) for chunk in chunks])
        
        # Collect reranked chunks
        for source_id, chunks in reranked_chunks_by_source.items():
            all_reranked_chunks.extend([format_reranked_chunk(chunk) for chunk in chunks])
        
        # Collect expanded chunks organized by depth
        max_depth = 0
        for source_id, depth_dict in expanded_by_depth_by_source.items():
            for depth_key, chunks in depth_dict.items():
                depth_num = int(depth_key.split("_")[-1]) if depth_key.startswith("depth_") else 1
                max_depth = max(max_depth, depth_num)
                
                if depth_key not in all_expanded_by_depth:
                    all_expanded_by_depth[depth_key] = []
                
                all_expanded_by_depth[depth_key].extend([format_expanded_chunk(chunk) for chunk in chunks])
        
        # Prepare output data
        output_data = {
            "query": retrieval_results["query"],
            "timestamp": retrieval_results["retrieval_timestamp"],
            "retrieval_metadata": {
                "total_semantic_chunks": len(all_semantic_chunks),
                "total_reranked_chunks": len(all_reranked_chunks),
                "total_expanded_chunks": sum(len(chunks) for chunks in all_expanded_by_depth.values()),
                "max_expansion_depth": max_depth,
                "sources": list(semantic_chunks_by_source.keys())
            },
            "chunks": {
                "semantic_search": all_semantic_chunks,
                "reranked": all_reranked_chunks,
                "expanded": all_expanded_by_depth if all_expanded_by_depth else {}
            }
        }
        
        # Generate filename
        timestamp = retrieval_results["retrieval_timestamp"]
        safe_query = "".join(c for c in retrieval_results["query"] if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
        safe_query = safe_query.replace(' ', '_')
        filename = f"all_chunks_{safe_query}_{timestamp}.json"
        output_path = os.path.join(output_dir, filename)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # print(f" Saved all chunks to: {output_path}")
        # print(f"   - Semantic chunks: {len(all_semantic_chunks)}")
        # print(f"   - Reranked chunks: {len(all_reranked_chunks)}")
        # print(f"   - Expanded chunks: {sum(len(chunks) for chunks in all_expanded_by_depth.values())}")
        for depth_key in sorted(all_expanded_by_depth.keys()):
            pass  # print(f"      - {depth_key}: {len(all_expanded_by_depth[depth_key])} chunks")
            
        # print()
        
        return output_path

    def _extract_message_ie_name(self, query: str) -> Optional[str]:
        """
        Extract message IE name from query (generic, protocol-agnostic)
        
        Examples:
        - "UE CONTEXT SETUP REQUEST" -> "UEContextSetupRequestIEs"
        - "RRC CONNECTION REJECT" -> "RRCConnectionRejectIEs"
        - "INITIAL UE MESSAGE" -> "InitialUeMessageIEs"
        
        Args:
            query: User query string
            
        Returns:
            Message IE name in camelCase format (e.g., "UEContextSetupRequestIEs"), or None
        """
        # Pattern 1: Look for message names with common protocol keywords
        message_keywords = r'(?:REQUEST|RESPONSE|FAILURE|REJECT|SETUP|RELEASE|MODIFICATION|MESSAGE|COMMAND|INDICATION|NOTIFICATION)'
        pattern1 = rf'([A-Z][A-Z\s]*(?:{message_keywords}))'
        match = re.search(pattern1, query, re.IGNORECASE)
        
        if match:
            message_name = match.group(1).strip()
            # Convert to camelCase + "IEs"
            words = message_name.split()
            camel_case = ''.join(word.capitalize() for word in words)
            # Remove common suffixes if they appear at the end, then add "IEs"
            camel_case = re.sub(r'(Request|Response|Failure|Reject|Setup|Release|Modification|Message|Command|Indication|Notification)$', '', camel_case, flags=re.IGNORECASE)
            camel_case = camel_case + 'IEs'
            return camel_case
        
        # Pattern 2: Look for quoted message names (e.g., "UE Context Setup Request")
        quoted_match = re.search(r'["\']([^"\']+?)["\']', query)
        if quoted_match:
            message_name = quoted_match.group(1).strip()
            words = message_name.split()
            if len(words) >= 2:  # Only if it looks like a message name
                camel_case = ''.join(word.capitalize() for word in words)
                camel_case = re.sub(r'(Request|Response|Failure|Reject|Setup|Release|Modification|Message|Command|Indication|Notification)$', '', camel_case, flags=re.IGNORECASE)
                camel_case = camel_case + 'IEs'
                return camel_case
        
        # Pattern 3: Look for capitalized phrases that might be message names
        fallback_match = re.search(r'\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){1,4})\b', query)
        if fallback_match:
            message_name = fallback_match.group(1).strip()
            words = message_name.split()
            if len(words) >= 2 and len(words) <= 6:  # Reasonable message name length
                camel_case = ''.join(word.capitalize() for word in words)
                camel_case = camel_case + 'IEs'
                return camel_case
        
        return None

    def discover_main_ie_definition(self,
                                   query: str,
                                   existing_chunks: List[Dict[str, Any]],
                                   all_searchers: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Discover and retrieve main IE definition chunks across all sources (generic, protocol-agnostic)
        
        This method searches for IE definitions using generic patterns that work across
        different protocols (F1AP, RRC, NGAP, etc.) and procedures.
        
        Args:
            query: User query string
            existing_chunks: List of chunks already retrieved (for deduplication)
            all_searchers: Dictionary mapping source_id to SemanticSearcher instances
            
        Returns:
            List of new chunks containing IE definitions
        """
        # print("   Extracting message IE name from query...")
        message_ie_name = self._extract_message_ie_name(query)
        
        if not message_ie_name:
            # print("     Could not extract message IE name from query, skipping IE discovery")
            return []
        
        # print(f"    Message IE name: {message_ie_name}")
        
        # Generate multiple generic search queries (protocol-agnostic)
        search_queries = [
            f"{message_ie_name} PROTOCOL-IES",      # Generic PROTOCOL-IES pattern
            f"{message_ie_name} ::=",               # ASN.1 definition pattern
            f"{message_ie_name} definition",        # Natural language pattern
            message_ie_name,                        # Just the IE name
        ]
        
        # print(f"    Trying {len(search_queries)} search query patterns across {len(all_searchers)} sources...")
        
        # Track existing chunk IDs for deduplication
        existing_ids = {chunk.get('section_id') for chunk in existing_chunks if chunk.get('section_id')}
        new_chunks = []
        chunks_found_by_query = {}
        
        # Search across all sources with all query variations
        for source_id, searcher in all_searchers.items():
            for search_query in search_queries:
                try:
                    results = searcher.search(search_query, top_k=5, rerank=False)
                    found_chunks = results.get('final_results', [])
                    
                    for chunk in found_chunks:
                        chunk_id = chunk.get('section_id')
                        if chunk_id and chunk_id not in existing_ids:
                            # Add metadata
                            chunk['knowledge_source'] = source_id
                            chunk['source_id'] = source_id
                            chunk['discovered_via'] = 'ie_search'
                            chunk['ie_search_query'] = search_query
                            chunk['message_ie_name'] = message_ie_name
                            
                            new_chunks.append(chunk)
                            existing_ids.add(chunk_id)
                            
                            # Track which query found this chunk
                            if search_query not in chunks_found_by_query:
                                chunks_found_by_query[search_query] = []
                            chunks_found_by_query[search_query].append(chunk_id)
                            
                except Exception as e:
                    # Continue with next query/source on error
                    continue
        
        # Print summary
        if new_chunks:
            pass  # print(f"    Found {len(new_chunks)} new IE definition chunks")
            for query, chunk_ids in chunks_found_by_query.items():
                pass  # print(f"      - Query '{query}': {len(chunk_ids)} chunks")
                
        else:
            pass  # print(f"     No new IE definition chunks found")
            
        
        # print()
        
        return new_chunks
    
    def expand_ies_agentically(self,
                              query: str,
                              existing_chunks: List[Dict[str, Any]],
                              all_searchers: Dict[str, Any],
                              max_iterations: int = 2) -> List[Dict[str, Any]]:
        """
        Agentically expand IE definitions iteratively (using patterns from retriever_agent.py)
        
        Uses LLM to analyze context, decide actions, and execute searches recursively
        across all sources until sufficient IEs are found.
        
        Args:
            query: User query string
            existing_chunks: List of chunks already retrieved
            all_searchers: Dictionary mapping source_id to SemanticSearcher instances
            max_iterations: Maximum number of expansion iterations
            
        Returns:
            List of newly discovered IE definition chunks
        """
        # print("    Starting agentic IE expansion (using retriever_agent.py patterns)...")
        # print(f"   Current context: {len(existing_chunks)} chunks")
        # print(f"    Max iterations: {max_iterations}")
        
        all_new_chunks = []
        current_chunks = existing_chunks.copy()
        existing_ids = {chunk.get('section_id') for chunk in existing_chunks if chunk.get('section_id')}
        visited_references = set()
        
        for iteration in range(1, max_iterations + 1):
            # print(f"\n    Iteration {iteration}/{max_iterations}")
            # print("-" * 60)
            
            # Step 1: LLM analyzes current context (using retriever_agent.py pattern)
            analysis = self._llm_analyze_context_ie_expansion(query, current_chunks, iteration)
            # print(f"    Analysis: {analysis.get('summary', 'N/A')[:100]}...")
            
            # Step 2: LLM decides actions (using retriever_agent.py pattern)
            decisions = self._llm_decide_actions_ie_expansion(analysis, query, iteration, max_iterations, visited_references)
            
            if decisions.get('should_stop', False):
                # print(f"    LLM decided to stop: {decisions.get('reason', 'N/A')}")
                break
            
            # Step 3: Execute actions (using retriever_agent.py pattern, adapted for multi-source)
            new_chunks = self._execute_actions_ie_expansion(
                decisions, query, analysis, current_chunks, all_searchers, visited_references
            )
            
            if not new_chunks:
                # print(f"     No new chunks found, stopping")
                break
            
            # print(f"    Found {len(new_chunks)} new chunks (Total: {len(current_chunks) + len(new_chunks)})")
            
            # Add to current context for next iteration
            current_chunks.extend(new_chunks)
            all_new_chunks.extend(new_chunks)
            
            # Update visited references
            for action in decisions.get('actions', []):
                target = action.get('target', '')
                if target:
                    visited_references.add(target)
        
        # print(f"\n    Agentic expansion complete: {len(all_new_chunks)} new IE chunks found")
        # print()
        
        return all_new_chunks
        
    def _identify_source_chunks_for_action(self,
                                         action: Dict[str, Any],
                                         analysis: Dict[str, Any],
                                         context_chunks: List[Dict[str, Any]],
                                         target: str) -> List[str]:
        """
        Identify which specific context chunks triggered this action (same as retriever_agent.py)
        
        Args:
            action: Action dictionary
            analysis: LLM analysis that led to this action
            context_chunks: Current context chunks
            target: Target entity/reference for the action
            
        Returns:
            List of section_ids of source chunks that triggered this action
        """
        source_chunk_ids = []
        target_lower = target.lower()
        search_query = action.get('search_query', target).lower()
        
        # Method 1: Check if chunks directly mention the target or search query
        for chunk in context_chunks:
            chunk_id = chunk.get('section_id')
            if not chunk_id:
                continue
            
            # Combine content and title for searching
            content = (chunk.get('content', '') + ' ' + chunk.get('section_title', '')).lower()
            
            # Check if chunk mentions the target or search query
            if target_lower in content or search_query in content:
                if chunk_id not in source_chunk_ids:
                    source_chunk_ids.append(chunk_id)
        
        # Method 2: Check missing_info from analysis for references
        for missing_item in analysis.get('missing_info', []):
            entity = missing_item.get('entity', '').lower()
            reference_text = missing_item.get('reference_text', '').lower()
            
            # Check if this missing item relates to our target
            if target_lower in entity or entity in target_lower or \
               target_lower in reference_text or search_query in reference_text:
                # Find chunks that mention this entity/reference
                for chunk in context_chunks:
                    chunk_id = chunk.get('section_id')
                    if not chunk_id:
                        continue
                    content = (chunk.get('content', '') + ' ' + chunk.get('section_title', '')).lower()
                    # Check if chunk mentions the entity or reference text
                    if entity in content or (reference_text and reference_text in content):
                        if chunk_id not in source_chunk_ids:
                            source_chunk_ids.append(chunk_id)
        
        # Method 3: Check references_found from analysis
        for ref in analysis.get('references_found', []):
            ref_value = ref.get('value', '').lower()
            ref_context = ref.get('context', '').lower()
            
            # Check if this reference relates to our target
            if target_lower in ref_value or ref_value in target_lower or \
               target_lower in ref_context or search_query in ref_value:
                # Find chunks that mention this reference
                for chunk in context_chunks:
                    chunk_id = chunk.get('section_id')
                    if not chunk_id:
                        continue
                    content = (chunk.get('content', '') + ' ' + chunk.get('section_title', '')).lower()
                    if ref_value in content or ref_context in content:
                        if chunk_id not in source_chunk_ids:
                            source_chunk_ids.append(chunk_id)
        
        # Method 4: If no specific source found, use top-ranked chunks as sources
        # (this happens when the action is based on general analysis rather than specific references)
        if not source_chunk_ids:
            # Use top-ranked chunks as sources
            sorted_chunks = sorted(
                context_chunks,
                key=lambda x: x.get('rank', x.get('cross_encoder_score', x.get('combined_score', x.get('semantic_score', 0)))),
                reverse=True
            )
            # Take top 3-5 chunks as sources
            source_chunk_ids = [
                c.get('section_id') for c in sorted_chunks[:5]
                if c.get('section_id')
            ]
        
        return source_chunk_ids
     
    def _build_expansion_relationship(self,
                                    action_type: str,
                                    method: str,
                                    target: str,
                                    kg_relations: List[str],
                                    search_query: str) -> str:
        """
        Build a human-readable description of the expansion relationship (same as retriever_agent.py)
        
        Args:
            action_type: Type of action (vector_search, kg_query, section_expand)
            method: Method used (semantic, kg, both)
            target: Target entity/reference
            kg_relations: Knowledge Graph relation types (if applicable)
            search_query: Search query used (if applicable)
            
        Returns:
            String describing the expansion relationship
        """
        if action_type == 'kg_query' or method in ['kg', 'both']:
            if kg_relations:
                relations_str = ', '.join(kg_relations)
                return f"KG relation: {relations_str} (from entity: {target})"
            else:
                return f"KG relation: general (from entity: {target})"
        elif action_type == 'vector_search' or action_type == 'section_expand':
            if search_query and search_query != target:
                return f"Semantic search: '{search_query}' (target: {target})"
            else:
                return f"Semantic search: '{target}'"
        else:
            return f"Expansion: {action_type} for '{target}'"

    def _query_kg_for_entity_multi_source(self, 
                            entity_name: str,
                            relation_types: List[str],
                            source_chunk_ids: List[str] = None,
                            expansion_relationship: str = None,
                            expansion_reason: str = None,
                            all_searchers: Dict[str, Any] = None,
                            context_chunks: List[Dict[str, Any]] = None,
                            max_kg_neighbors: int = 5) -> List[Dict[str, Any]]:
        """
        Query Knowledge Graph for related entities across all sources (adapted from retriever_agent.py for multi-source)
        
        Strategy:
        1. First, find chunks that mention the entity (semantic search) across all sources
        2. Then, use KG to find related chunks via graph edges in each source
        3. Return chunks that are related via KG relationships
        
        Args:
            entity_name: Entity name to query
            relation_types: List of KG relation types to filter by
            source_chunk_ids: Source chunk IDs that triggered this query
            expansion_relationship: Description of expansion relationship
            expansion_reason: Reason for expansion
            all_searchers: Dictionary of all searchers (source_id -> SemanticSearcher)
            context_chunks: Current context chunks for deduplication
            max_kg_neighbors: Maximum number of KG neighbors to retrieve per entity (default: 5)
            
        Returns:
            List of chunks found via KG queries
        """
        chunks = []
        context_section_ids = {chunk.get('section_id') for chunk in context_chunks if chunk.get('section_id')}
        
        # Step 1: Find chunks that mention the entity (using semantic search across all sources)
        entity_section_ids_by_source = {}
        for source_id, searcher in all_searchers.items():
            try:
                results = searcher.search(entity_name, top_k=10, rerank=False)
                found_chunks = results.get('final_results', [])
                entity_section_ids = {chunk.get('section_id') for chunk in found_chunks if chunk.get('section_id')}
                entity_section_ids_by_source[source_id] = entity_section_ids
            except Exception as e:
                continue
        
        if not entity_section_ids_by_source:
            return chunks
        
        # Step 2: Expand via Knowledge Graph from these section IDs in each source
        for source_id, searcher in all_searchers.items():
            if not hasattr(searcher, 'kg') or not searcher.kg:
                continue
            
            entity_section_ids = entity_section_ids_by_source.get(source_id, set())
            if not entity_section_ids:
                continue
            
            related_section_ids = set()
            
            for section_id in entity_section_ids:
                if not searcher.kg.has_node(section_id):
                    continue
                
                # Get neighbors (both successors and predecessors) - limit to max_kg_neighbors
                successors = list(searcher.kg.successors(section_id))
                predecessors = list(searcher.kg.predecessors(section_id))
                
                # Limit neighbors to reduce expansion breadth
                total_neighbors = successors + predecessors
                neighbors = total_neighbors[:max_kg_neighbors]
                
                for neighbor in neighbors:
                    if neighbor not in entity_section_ids and neighbor not in context_section_ids:
                        # Check edge type if relation_types specified
                        if relation_types:
                            edge_data = searcher.kg.get_edge_data(section_id, neighbor) or searcher.kg.get_edge_data(neighbor, section_id)
                            if edge_data:
                                # Handle both single relation type and list of relationship_types
                                edge_type = edge_data.get('type', '')
                                rel_types = edge_data.get('relationship_types', [])
                                if isinstance(rel_types, str):
                                    rel_types = [rel_types]
                                
                                # Check if any relation type matches
                                if edge_type in relation_types or any(rt in relation_types for rt in rel_types):
                                    related_section_ids.add(neighbor)
                        else:
                            # Include all neighbors if no relation type filter
                            related_section_ids.add(neighbor)
            
            # Step 3: Get chunks for related section IDs from this source
            for section_id in related_section_ids:
                if section_id in searcher.all_chunks:
                    chunk = searcher.all_chunks[section_id].copy()
                    chunk['discovered_via_kg'] = True
                    chunk['kg_entity'] = entity_name
                    chunk['knowledge_source'] = source_id
                    chunk['source_id'] = source_id
                    
                    # Add source tracking metadata
                    if source_chunk_ids is not None:
                        chunk['source_chunk_ids'] = source_chunk_ids
                    else:
                        # If no specific source provided, use entity chunks as sources
                        chunk['source_chunk_ids'] = list(entity_section_ids)
                    
                    if expansion_relationship:
                        chunk['expansion_relationship'] = expansion_relationship
                    else:
                        # Build default relationship
                        if relation_types:
                            relations_str = ', '.join(relation_types)
                            chunk['expansion_relationship'] = f"KG relation: {relations_str} (from entity: {entity_name})"
                        else:
                            chunk['expansion_relationship'] = f"KG relation: general (from entity: {entity_name})"
                    
                    if expansion_reason:
                        chunk['expansion_reason'] = expansion_reason
                    
                    chunk['expansion_method'] = 'knowledge_graph'
                    chunk['expansion_target'] = entity_name
                    if relation_types:
                        chunk['kg_relation_types'] = relation_types
                    
                    chunks.append(chunk)
        
        return chunks

    def _deduplicate_chunks(self, 
                           new_chunks: List[Dict[str, Any]],
                           existing_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate chunks based on section_id (same as retriever_agent.py)
        """
        existing_ids = {chunk.get('section_id') for chunk in existing_chunks if chunk.get('section_id')}
        
        unique_chunks = []
        for chunk in new_chunks:
            chunk_id = chunk.get('section_id')
            if chunk_id and chunk_id not in existing_ids:
                unique_chunks.append(chunk)
                existing_ids.add(chunk_id)
        
        return unique_chunks

    def _execute_actions_ie_expansion(self, 
                                     decisions: Dict[str, Any],
                                     query: str,
                                     analysis: Dict[str, Any],
                                     context_chunks: List[Dict[str, Any]],
                                     all_searchers: Dict[str, Any],
                                     visited_references: Set[str]) -> List[Dict[str, Any]]:
        """
        Execute the actions decided by LLM (using retriever_agent.py pattern, adapted for multi-source)
        """
        new_chunks = []
        
        for action in decisions.get('actions', []):
            action_type = action.get('type', '')
            target = action.get('target', '')
            method = action.get('method', 'semantic')
            search_query = action.get('search_query', target)
            kg_relations = action.get('kg_relations', [])
            priority = action.get('priority', 'medium')
            expected_result = action.get('expected_result', '')
            
            # Skip if already visited
            if target in visited_references:
                continue
            
            visited_references.add(target)
            
            # print(f"    Action: {action_type} - {target} (method: {method})")
            
            # Identify which source chunks triggered this action
            source_chunk_ids = self._identify_source_chunks_for_action(
                action, analysis, context_chunks, target
            )
            
            # Build expansion relationship description
            expansion_relationship = self._build_expansion_relationship(
                action_type, method, target, kg_relations, search_query
            )
            
            if action_type == 'vector_search' or action_type == 'section_expand':
                # Semantic search across all sources
                if method in ['semantic', 'both']:
                    for source_id, searcher in all_searchers.items():
                        try:
                            results = searcher.search(search_query, top_k=5, rerank=False)
                            found_chunks = results.get('final_results', [])
                            
                            # Add source tracking metadata
                            for chunk in found_chunks:
                                chunk['source_chunk_ids'] = source_chunk_ids
                                chunk['expansion_relationship'] = expansion_relationship
                                chunk['expansion_reason'] = decisions.get('reason', 'N/A')
                                chunk['expansion_method'] = 'semantic_search'
                                chunk['expansion_target'] = target
                                chunk['expansion_search_query'] = search_query
                                chunk['knowledge_source'] = source_id
                                chunk['source_id'] = source_id
                            new_chunks.extend(found_chunks)
                        except Exception as e:
                            continue
            
            if action_type == 'kg_query' or method in ['kg', 'both']:
                # Knowledge Graph query across all sources
                kg_chunks = self._query_kg_for_entity_multi_source(
                    target, kg_relations, source_chunk_ids, expansion_relationship, 
                    decisions.get('reason', 'N/A'), all_searchers, context_chunks
                )
                new_chunks.extend(kg_chunks)
        
        # Deduplicate chunks
        new_chunks = self._deduplicate_chunks(new_chunks, context_chunks)
        
        # Add common metadata for all expanded chunks
        for chunk in new_chunks:
            chunk['discovered_via'] = 'agentic_expansion'
            # Ensure source_chunk_ids is always a list
            if 'source_chunk_ids' not in chunk:
                chunk['source_chunk_ids'] = []
            # Ensure expansion_relationship is always present
            if 'expansion_relationship' not in chunk:
                chunk['expansion_relationship'] = 'Unknown expansion'
        
        return new_chunks
    
    def _llm_decide_actions_ie_expansion(self, 
                                        analysis: Dict[str, Any],
                                        query: str,
                                        iteration: int,
                                        max_iterations: int,
                                        visited_references: Set[str]) -> Dict[str, Any]:
        """
        LLM decides what actions to take based on analysis (using retriever_agent.py pattern)
        """
        prompt = f"""
            You are an expert system analyst. Based on the analysis, decide what actions to take next.

            ANALYSIS:
            {json.dumps(analysis, indent=2)}

            CURRENT ITERATION: {iteration}/{max_iterations}
            VISITED REFERENCES: {list(visited_references)[:10]}...

            DECISION TASK:
            1. Should we continue exploring? (should_stop: true/false)
            - Stop if: All critical information found, or unlikely to find more, or max iterations reached
            - **CRITICAL**: Do NOT stop if the MAIN message IE definition is missing
            
            2. What specific actions should we take?
            - **HIGHEST PRIORITY**: Search for MAIN message IE definition (pattern: "[MessageName]IEs PROTOCOL-IES ::=")
            - Search vector DB for specific references (semantic search)
            - Query Knowledge Graph for entity relations
            - Expand specific sections

            3. For each action, decide the method:
            - "kg": Use Knowledge Graph (for entity names, functions, structs that are likely in KG)
            - "semantic": Use semantic search (for section numbers, natural language references, definitions, MAIN IE definitions)
            - "both": Use both methods for comprehensive coverage
            - **For MAIN IE definition**: Use "semantic" method with search query like "[MessageName]IEs PROTOCOL-IES" or "[MessageName]IEs PROTOCOL-IES ::="

            4. Prioritize actions based on:
            - **HIGHEST**: MAIN message IE definition (if missing)
            - Criticality for template filling
            - Likelihood of finding the information
            - Already explored paths (avoid redundancy)

            Return ONLY valid JSON (no markdown, no code blocks):
            {{
                "should_stop": false,
                "reason": "Still missing ASN.1 structures for 3 IEs",
                "actions": [
                    {{
                        "type": "vector_search|kg_query|section_expand",
                        "target": "[IE or entity name] ASN.1 definition",
                        "method": "semantic|kg|both",
                        "search_query": "[Search query string]",
                        "kg_relations": ["STRUCT_DEFINITION", "IE_DEFINITION"],
                        "priority": "high|medium|low",
                        "expected_result": "ASN.1 type definition"
                    }}
                ],
                "skip_actions": [
                    {{
                        "target": "Section 8.1",
                        "reason": "Already retrieved in initial search"
                    }}
                ]
        }}"""

        try:
            response = llm.invoke(prompt)
            response_text = response.content.strip()
            
            # Clean JSON response
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            decisions = json.loads(response_text)
            return decisions
        except Exception as e:
            # print(f"     Error in LLM decision: {e}")
            return {
                "should_stop": True,
                "reason": f"Decision error: {e}",
                "actions": [],
                "skip_actions": []
            }
    
    def _has_asn1_structure(self, chunk: Dict[str, Any]) -> bool:
        """
        Check if chunk contains ASN.1 structure/definition
        
        Args:
            chunk: Chunk dictionary
            
        Returns:
            True if chunk contains ASN.1 patterns
        """
        content = chunk.get('content', '')
        title = chunk.get('section_title', '')
        text = f"{title} {content}".upper()
        
        # ASN.1 patterns
        asn1_patterns = [
            r'INTEGER\s*\(',           # INTEGER (0..4095)
            r'OCTET\s+STRING',          # OCTET STRING
            r'BIT\s+STRING',            # BIT STRING
            r'CHOICE\s*\{',             # CHOICE {
            r'SEQUENCE\s*\{',           # SEQUENCE {
            r'ENUMERATED\s*\{',         # ENUMERATED {
            r'::=\s*',                  # ::= (definition)
            r'OPTIONAL',                # OPTIONAL
            r'MANDATORY',               # MANDATORY
            r'SIZE\s*\(',               # SIZE (1..32)
            r'PROTOCOL-IES',            # PROTOCOL-IES
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in asn1_patterns)
    
    def _summarize_chunks_for_analysis(self, chunks: List[Dict[str, Any]], max_chunks: int = 20) -> str:
        """
        Create a summary of chunks for LLM analysis (focused on IE-related content)
        
        Args:
            chunks: List of chunk dictionaries
            max_chunks: Maximum number of chunks to include in summary
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            section_id = chunk.get('section_id', 'Unknown')
            section_title = chunk.get('section_title', '')
            content_preview = chunk.get('content', '')[:300]
            source_id = chunk.get('knowledge_source', chunk.get('source_id', ''))
            
            # Check for ASN.1 patterns
            has_structure = self._has_asn1_structure(chunk)
            structure_indicator = " [HAS ASN.1]" if has_structure else ""
            
            source_label = f" [{source_id}]" if source_id else ""
            
            summary_parts.append(
                f"{i}. {section_id}: {section_title}{structure_indicator}{source_label}\n"
                f"   {content_preview}...\n"
            )
        
        if len(chunks) > max_chunks:
            summary_parts.append(f"\n... and {len(chunks) - max_chunks} more chunks")
        
        return "\n".join(summary_parts)   

    def _llm_analyze_context_ie_expansion(self, 
                                        query: str, 
                                        chunks: List[Dict[str, Any]], 
                                        iteration: int) -> Dict[str, Any]:
        """
        LLM analyzes current context and identifies what's missing (using retriever_agent.py pattern)
        """
        # Summarize chunks
        chunk_summary = self._summarize_chunks_for_analysis(chunks, max_chunks=20)
        
        prompt = f"""
            You are an expert system analyst. Analyze the current context and identify what information is missing or needs expansion to fill the template.

            USER QUERY: {query}

            CURRENT ITERATION: {iteration}

            CURRENT CONTEXT SUMMARY:
            {chunk_summary}

            TEMPLATE STRUCTURE:
            {json.dumps(self.template, indent=2)[:1000]}...

            TASK:
            1. **CRITICAL: Check for MAIN MESSAGE IE DEFINITION**:
            - For the query "{query}", identify the main message name from the query
            - Check if the MAIN message IE definition is present in the context (pattern: "[MessageName]IEs PROTOCOL-IES ::= {{ ... }}")
            - If the MAIN IE definition is NOT found, this is HIGHEST PRIORITY - it must be retrieved first
            - The MAIN IE definition contains all child IEs for the message and is essential for correct extraction

            2. Identify missing information needed to fill the template:
            - Which Information Elements (IEs) need ASN.1 structures?
            - Which sections are referenced but not retrieved?
            - Which entities have relations in Knowledge Graph that should be explored?
            - What other critical information is missing?

            3. Identify what's already sufficient:
            - Which fields can be filled with current context?
            - Which references don't need expansion?

            4. Extract references from chunks:
            - Section references (e.g., "5.3.3.1", "section 8.1")
            - IE/Entity references (e.g., IE names, entity names mentioned in the query)
            - Type references (e.g., ASN.1 type names)
            - Cross-specification references (e.g., "TS 38.331")
            - **MAIN IE definition references** (e.g., "[MessageName]IEs", "[MessageName]IEs PROTOCOL-IES")

            5. Prioritize what needs expansion:
            - **HIGHEST PRIORITY**: MAIN message IE definition (if missing)
            - Critical missing information (high priority)
            - Nice-to-have information (low priority)

            Return ONLY valid JSON (no markdown, no code blocks):
            {{
                "summary": "Brief summary of analysis",
                "missing_info": [
                    {{
                        "type": "main_ie_definition|ie_structure|section_reference|kg_relation|type_definition",
                        "entity": "[MessageName]IEs PROTOCOL-IES",
                        "reference_text": "Exact text from chunks that mentions this",
                        "reason": "MAIN message IE definition is required to identify all child IEs",
                        "priority": "high|medium|low"
                    }}
                ],
                "sufficient_info": [
                    {{
                        "field": "Feature_Name",
                        "reason": "Already found in chunks"
                    }}
                ],
                "references_found": [
                    {{
                        "type": "section|ie|type|spec",
                        "value": "5.3.3.1",
                        "context": "Mentioned in chunk about..."
                    }}
                ],
                "recommendations": [
                    "Search for ASN.1 definition of [IE name from query]",
                    "Explore semantic search for [entity name from query]",
                    "Explore KG relations for [entity name from query]"
                ]
        }}"""

        try:
            response = llm.invoke(prompt)
            response_text = response.content.strip()
            
            # Clean JSON response
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            analysis = json.loads(response_text)
            return analysis
        except Exception as e:
            # print(f"     Error in LLM analysis: {e}")
            return {
                "summary": "Analysis failed",
                "missing_info": [],
                "sufficient_info": [],
                "references_found": [],
                "recommendations": []
            }
    
    def save_final_chunks(
                        self,
                        query: str,
                        top_chunks: List[Dict[str, Any]],
                        ie_chunks: List[Dict[str, Any]],
                        expanded_ie_chunks: List[Dict[str, Any]],
                        final_chunks: List[Dict[str, Any]],
                        output_dir: str) -> str:
        """
        Save final chunks after IE expansion to JSON file
        
        Args:
            query: User query string
            top_chunks: Initial top chunks from retrieval
            ie_chunks: Chunks discovered in IE discovery phase
            expanded_ie_chunks: Chunks discovered during agentic IE expansion
            final_chunks: Combined final chunks
            output_dir: Output directory path
            
        Returns:
            Path to saved JSON file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data for saving
        output_data = {
            "query": query,
            "timestamp": timestamp,
            "summary": {
                "initial_chunks_count": len(top_chunks),
                "ie_discovery_chunks_count": len(ie_chunks),
                "expanded_ie_chunks_count": len(expanded_ie_chunks),
                "final_chunks_count": len(final_chunks),
                "chunks_by_phase": {
                    "initial_retrieval": len(top_chunks),
                    "ie_discovery": len(ie_chunks),
                    "ie_expansion": len(expanded_ie_chunks),
                    "total_final": len(final_chunks)
                }
            },
            "chunks_by_phase": {
                "initial_chunks": [convert_numpy_types(chunk.copy()) for chunk in top_chunks],
                "ie_discovery_chunks": [convert_numpy_types(chunk.copy()) for chunk in ie_chunks],
                "expanded_ie_chunks": [convert_numpy_types(chunk.copy()) for chunk in expanded_ie_chunks]
            },
            "final_chunks_combined": [convert_numpy_types(chunk.copy()) for chunk in final_chunks]
        }
        
        # Generate filename
        safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
        safe_query = safe_query.replace(' ', '_')
        filename = f"final_chunks_{safe_query}_{timestamp}.json"
        output_path = os.path.join(output_dir, filename)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # print(f" Saved final chunks to: {output_path}")
        # print(f"   - Initial chunks: {len(top_chunks)}")
        # print(f"   - IE discovery chunks: {len(ie_chunks)}")
        # print(f"   - Expanded IE chunks: {len(expanded_ie_chunks)}")
        # print(f"   - Total final chunks: {len(final_chunks)}")
        # print()
        
        return output_path
