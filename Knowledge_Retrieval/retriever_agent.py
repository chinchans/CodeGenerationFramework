import os
import json
import sys
import logging
from pathlib import Path
# from agentic_template_filler import AgenticTemplateFiller
import time


# ----------------------------------------------------------
# Add project root to Python path
# ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

from .code_chunks_retriever import SemanticGraphRAG
from .multi_source_retriever import MultiSourceChunkRetriever

def codeChunkRetrieverAgent(state,query,CODE_KNOWLEDGE_PATHS):

    # print("------------Code Retriever Agent----------")
    # print(state)

    # return state
    # 1. CENTRAL CONFIGURATION
    feature_name = state.get("message_names")[0]
    CHUNKS_OUTPUT_PATH =  f"./outputs/code_chunks/{feature_name}_chunks.json"
    PATHS = {
        "faiss_index": CODE_KNOWLEDGE_PATHS.get("faiss_index_file"),
        "faiss_meta": CODE_KNOWLEDGE_PATHS.get("faiss_metadata_file"),
        "kg_path": CODE_KNOWLEDGE_PATHS.get("kg_file"),
        "feature_name": state.get("message_names")[0],
        "output_file": CHUNKS_OUTPUT_PATH
    }

    # print("Before Semantic Graph RAG")
    # 2. INITIALIZE RETRIEVER
    retriever = SemanticGraphRAG(
        faiss_index_path=PATHS["faiss_index"],
        faiss_metadata_path=PATHS["faiss_meta"],
        kg_path=PATHS["kg_path"],
        feature_name=PATHS["feature_name"]
    )
    # print("After Semantic Graph RAG")
    # 3. SET SEARCH PARAMETERS
    
    # We pass function_calls and function_uses_struct here
    REL_FILTERS = ["function_calls", "function_uses_struct"] 

    # print("Before Retrieve")
    # 4. EXECUTE PIPELINE
    semantic_res, d_map, seeds = retriever.retrieve(
        query=query,
        top_k=10,
        kg_depth=2,
        rel_filters=REL_FILTERS,
        direction="out"
    )
    # print("After Retrieve")
    
    # 5. BUILD JSON AND SAVE
    final_json = retriever.build_output_json(query, semantic_res, d_map, direction="out")

    # Storing in the global state
    state['code_artifacts_context'] = {
        "semantic_chunks": final_json.get("semantic_chunks", []),
        "expanded_chunks": final_json.get("expanded_chunks", {}),
        "metadata": final_json.get("metadata", {})
    }
    
    os.makedirs(os.path.dirname(PATHS["output_file"]), exist_ok=True)
    with open(PATHS["output_file"], "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2)


    state['code_artifacts_chunks_path'] = PATHS['output_file']
    # print("Code Chunks Retrieval completed. Results: %s", PATHS['output_file'])

    return state


def specChunkRetrieverAgent(state, query,INITIAL_RETRIEVAL_SOURCES,template_path):
    
    # TEMPLATE_FILE = "../inputs/Template.json"
    # OUTPUT_DIR = "../outputs/filled_templates"
    OUTPUT_DIR_CHUNKS = "./outputs/spec_chunks"

    
    # Knowledge Sources Configuration
    # ==============================================================================
    # Define knowledge sources for INITIAL RETRIEVAL.
    #
    # Each source must have:
    # - source_id: Unique identifier (e.g., "ts_138473v180600p")
    # - official_spec_id: Official specification identifier (e.g., "TS 38.473 v18.6.0", "TS 38.401 v18.6.0")
    # - kg_file: Path to knowledge graph pickle file
    # - faiss_index_file: Path to FAISS index file
    # - faiss_metadata_file: Path to FAISS metadata JSON file
    #
    # INITIAL_RETRIEVAL_SOURCES: Sources used for initial chunk retrieval and IE discovery/expansion
    # ==============================================================================

    # Sources for INITIAL chunk retrieval
    # INITIAL_RETRIEVAL_SOURCES = [
    #     {
    #         "source_id": "ts_138473v180600p",
    #         "official_spec_id": "TS 38.473 v18.6.0",  # Official spec identifier for reference mapping
    #         "kg_file": "../resources/Specs_Knowledge/ts_138473v180600p/KnowledgeGraph/knowledge_graph.pkl",
    #         "faiss_index_file": "../resources/Specs_Knowledge/ts_138473v180600p/vector_db/faiss_index.index",
    #         "faiss_metadata_file": "../resources/Specs_Knowledge/ts_138473v180600p/vector_db/faiss_metadata.json"
    #     },
    #     {
    #         "source_id": "ts_138401v180600p",
    #         "official_spec_id": "TS 38.401 v18.6.0",  # Official spec identifier for reference mapping
    #         "kg_file": "../resources/Specs_Knowledge/ts_138401v180600p/KnowledgeGraph/knowledge_graph.pkl",
    #         "faiss_index_file": "../resources/Specs_Knowledge/ts_138401v180600p/vector_db/faiss_index.index",
    #         "faiss_metadata_file": "../resources/Specs_Knowledge/ts_138401v180600p/vector_db/faiss_metadata.json"
    #     }
    # ]

    
    # Retrieval Configuration
    TOP_CHUNK_PERCENTAGE = 0.7  # 70% of top chunks from each source
    TOP_K_PER_SOURCE = 50  # Number of chunks to retrieve per source (before filtering)
    USE_RERANKING = True  # Use LLM reranking

    # print("=" * 60)
    # print("Multi-Source Retrieval v0")
    # print("=" * 60)
    # print("Query: %s", query)
    # print("Knowledge Sources: %d", len(INITIAL_RETRIEVAL_SOURCES))
    for source in INITIAL_RETRIEVAL_SOURCES:
        pass  # print("   - %s", source['source_id'])
    
    start_time = time.time()
    
    try:
        # Step 1: Initialize Multi-Source Retriever (uses INITIAL_RETRIEVAL_SOURCES)
        # print("=" * 60)
        # print("Step 1: Initializing Multi-Source Retriever")
        # print("=" * 60)
        multi_source_retriever = MultiSourceChunkRetriever(
            knowledge_sources=INITIAL_RETRIEVAL_SOURCES,  # Uses initial retrieval sources
            top_chunk_percentage=TOP_CHUNK_PERCENTAGE,
            template_file = template_path
        )
        
        # Step 2: Retrieve all chunks from all sources
        # print("=" * 60)
        # print("Step 2: Retrieving All Chunks from All Sources")
        # print("=" * 60)
        retrieval_results = multi_source_retriever.retrieve_all_chunks(
            query=query,
            top_k_per_source=TOP_K_PER_SOURCE,
            rerank=USE_RERANKING
        )
        
        # Step 3: Save all chunks to JSON
        # print("=" * 60)
        # print("Step 3: Saving All Chunks")
        # print("=" * 60)
        chunks_file_path = multi_source_retriever.save_all_chunks(
            retrieval_results=retrieval_results,
            output_dir=OUTPUT_DIR_CHUNKS
        )
        
        # Step 4: Discover Main IE Definitions (Generic, Protocol-Agnostic)
        # print("=" * 60)
        # print("Step 4: Discovering Main IE Definitions")
        # print("=" * 60)
        
        top_chunks = retrieval_results["top_chunks_combined"]
        
        
        # Use searchers from initial retrieval only
        all_searchers = multi_source_retriever.searchers
        
        # Discover IE definitions across all sources
        ie_chunks = multi_source_retriever.discover_main_ie_definition(
            query=query,
            existing_chunks=top_chunks,
            all_searchers=all_searchers
        )
        
        # Combine top chunks with initial IE definitions
        chunks_before_expansion = top_chunks + ie_chunks
        
        # Step 5: Agentic Iterative IE Expansion (Recursive)
        # print("=" * 60)
        # print("Step 5: Agentic Iterative IE Expansion")
        # print("=" * 60)
        
        # Expand IEs agentically (similar to retriever_agent.py)
        expanded_ie_chunks = multi_source_retriever.expand_ies_agentically(
            query=query,
            existing_chunks=chunks_before_expansion,
            all_searchers=all_searchers,
            max_iterations=1
        )
        
        # Combine all chunks: initial + initial IE definitions + expanded IE definitions
        final_chunks = chunks_before_expansion + expanded_ie_chunks
        # print("Final context: %d chunks (initial: %d, IE defs: %d, expanded IE: %d)",len(final_chunks), len(top_chunks), len(ie_chunks), len(expanded_ie_chunks))

        state['specs_context'] = final_chunks
        
        state['specs_chunks_path'] = chunks_file_path
        # print("Specs chunks path: %s", state['specs_chunks_path'])
        
        # Step 5.5: Save Final Chunks (after IE expansion)
        # print("=" * 60)
        # print("Step 5.5: Saving Final Chunks (After IE Expansion)")
        # print("=" * 60)
        # final_chunks_file_path = multi_source_retriever.save_final_chunks(
        #     query=query,
        #     top_chunks=top_chunks,
        #     ie_chunks=ie_chunks,
        #     expanded_ie_chunks=expanded_ie_chunks,
        #     final_chunks=final_chunks,
        #     output_dir=OUTPUT_DIR_CHUNKS
        # )
        
        # Summary
        total_time = time.time() - start_time
        
        # print("=" * 60)
        # print("Multi-Source Retrieval completed successfully")
        # print("Statistics: total_time=%.1fs, sources=%d, all_chunks=%d, top_chunks=%d, ie_chunks=%d, expanded_ie=%d, final=%d, chunks_file=%s",total_time, len(INITIAL_RETRIEVAL_SOURCES), len(retrieval_results['all_chunks_combined']),len(top_chunks), len(ie_chunks), len(expanded_ie_chunks), len(final_chunks), chunks_file_path)
        # print("=" * 60)

        return state
        
    except Exception as e:
        # logger.exception("Error: %s", str(e))
        raise



def retrieverAgent(state):
    # print("Retriever Agent")
    message_name = state.get("message_names")[0]
    user_intent = state.get("messages")[0].content

    # print("Message Name: %s, User Query: %s", message_name, user_intent)

    


    # ----------------------Specs Content Retrieval-----------------------
    ALL_RETRIEVAL_SOURCES = state.get("specs_retrieval_sources")
    template_path = state.get("selected_template_path")

    INITIAL_RETRIEVAL_SOURCES = []
    required_docs = ["ts_138473v180400p","ts_138401v180600p"]

    for source in ALL_RETRIEVAL_SOURCES:
        if source.get("source_id") in required_docs:
            INITIAL_RETRIEVAL_SOURCES.append(source)

    # print("Before Specification Chunk Retriever Agent")
    state = specChunkRetrieverAgent(state, message_name,INITIAL_RETRIEVAL_SOURCES,template_path)

    # print("After Specification Chunk Retriever Agent")
    # print("Spec Retriever Agent : ")
    # # print(state['specs_context'][:10])
    # # print(len(state['specs_context']))
    # # print(type(state['specs_context']))
    # print(state['specs_chunks_path'])
    # # print({k: v for k, v in vars(state).items() if k != "specs_context"})


    
    # ----------------------Code Artifacts Retrieval-----------------------
    CODE_KNOWLEDGE_PATHS = state['code_retrieval_sources']
    # print("Before Code Chunk Retriever Agent")
    state = codeChunkRetrieverAgent(state,message_name,CODE_KNOWLEDGE_PATHS)
    # print("After Code Chunk Retriever Agent")
    return state

