import sys
import json
from pathlib import Path
import time

# ----------------------------------------------------------
# Add project root to Python path
# ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Knowledge_Retrieval
# App root (Code_Generation_v0) - for resources that live next to gateway when run from project dir
APP_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from .create_spec_vector_kg import CreateSpecVectorKg
from .code_knowledge.parse_codebase import main as parse_main
from .code_knowledge.extract_chunks import run_extraction
from .code_knowledge.build_kg_vector_new import main as kg_main

from pathlib import Path

# Get the parent directory of the current script
parent_dir = Path(__file__).resolve().parent

import logging

logger = logging.getLogger(__name__)

# Main Execution Pipeline
def specKnowledgeCreatorForEachSpec(DOC_ID,SPEC_PATH,SPEC_NUM,RUN_KNOWLEDGE_CREATE=False):
    # print("Doc id :",DOC_ID)
    # print("Spec Path :",SPEC_PATH)

    
    # Configuration - use APP_ROOT so paths work when MCP runs from a different directory
    KG_FILE = str(APP_ROOT / f"resources/Specs_Knowledge/{DOC_ID}/KnowledgeGraph/knowledge_graph.pkl")
    FAISS_INDEX_FILE = str(APP_ROOT / f"resources/Specs_Knowledge/{DOC_ID}/vector_db/faiss_index.index")
    FAISS_METADATA_FILE = str(APP_ROOT / f"resources/Specs_Knowledge/{DOC_ID}/vector_db/faiss_metadata.json")
    


    if RUN_KNOWLEDGE_CREATE:
        # print("=" * 60)
        # print(f"GraphRAG v1: Knowledge Graph + FAISS for {DOC_ID} spec")
        # print("=" * 60)
        # print()


        specVectorKgCreator = CreateSpecVectorKg()

        pipeline_start = time.time()

        # Step 1: Load and parse PDF
        # print(" Step 1: Loading and parsing PDF...")
        # print("-" * 60)
        sections = specVectorKgCreator.load_and_parse_pdf(SPEC_PATH, DOC_ID)
        # print()

        # Step 2: Extract deepest-level chunks
        # print(" Step 2: Extracting deepest-level chunks...")
        # print("-" * 60)
        chunks = specVectorKgCreator.extract_deepest_chunks(sections)
        # print()

        # Step 3: Extract hierarchical relationships
        # print(" Step 3: Extracting hierarchical relationships...")
        # print("-" * 60)
        hierarchical_rels = specVectorKgCreator.extract_hierarchical_relationships(chunks, sections)
        # print()

        # Step 4: Extract explicit references
        # print(" Step 4: Extracting explicit section references...")
        # print("-" * 60)
        explicit_rels = specVectorKgCreator.extract_explicit_references(chunks)
        # print()

        # Step 5: Create FAISS index (needed for semantic relationship extraction)
        # print(" Step 5: Creating FAISS vector index...")
        # print("-" * 60)
        faiss_index, id_to_section, embeddings_array = specVectorKgCreator.create_faiss_index(chunks)
        # print()

        # Step 6: Extract semantic relationships (using FAISS for candidate selection)
        # print(" Step 6: Extracting semantic relationships (this may take a while)...")
        # print("-" * 60)
        semantic_rels = specVectorKgCreator.extract_semantic_relationships(
            chunks,
            faiss_index,
            id_to_section,
            top_k=10,      # Get top 30 similar from FAISS
            final_k=5,    # Keep top 20 after filtering
            batch_size=2   # Process 5 chunks at a time
        )
        # print()

        # Step 7: Combine all relationships
        # print(" Step 7: Combining all relationships...")
        # print("-" * 60)
        all_relationships = hierarchical_rels + explicit_rels + semantic_rels
        # print(f"    Total relationships: {len(all_relationships)}")
        # print(f"      - Hierarchical: {len(hierarchical_rels)}")
        # print(f"      - Explicit references: {len(explicit_rels)}")
        # print(f"      - Semantic: {len(semantic_rels)}")
        # print()

        # Step 8: Build knowledge graph
        # print(" Step 8: Building knowledge graph...")
        # print("-" * 60)
        kg = specVectorKgCreator.build_knowledge_graph(chunks, all_relationships)
        # print()

        # Step 9: Save knowledge graph
        # print(" Step 9: Saving knowledge graph...")
        # print("-" * 60)
        specVectorKgCreator.save_kg(kg_file_path=KG_FILE,graph=kg)
        # print()

        # Step 10: Save FAISS index
        # print(" Step 10: Saving FAISS index...")
        # print("-" * 60)
        specVectorKgCreator.save_faiss_index(faiss_index, id_to_section, chunks,FAISS_INDEX_FILE,FAISS_METADATA_FILE)
        # print()

        total_time = time.time() - pipeline_start

        # print("=" * 60)
        # print(" Pipeline completed successfully!")
        # print("=" * 60)
        # print(f" Final Statistics:")
        # print(f"   - Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        # print(f"   - Knowledge Graph: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")
        # print(f"   - FAISS Index: {faiss_index.ntotal} vectors")
        # print(f"   - Total relationships: {len(all_relationships)}")
        # print(f" Output Files:")
        # print(f"   - {KG_FILE}")
        # print(f"   - {FAISS_INDEX_FILE}")
        # print(f"   - {FAISS_METADATA_FILE}")
        # print("=" * 60)

    spec_knowledge_data = {
        "source_id":DOC_ID,
        "official_spec_id":SPEC_NUM,
        "kg_file":KG_FILE,
        "faiss_index_file":FAISS_INDEX_FILE,
        "faiss_metadata_file":FAISS_METADATA_FILE

    }

    return spec_knowledge_data

    # state


def specKnowledgeCreator(state):
    # print("Knowledge Creator Agent")
    # print(state['session_id'])
    # print("----------------Specs Path----------------\n")
    # print(state['specifications'])
    

    spec_knowledge_paths = []
    for spec in state.get('specifications'):
        spec_path = specKnowledgeCreatorForEachSpec(DOC_ID=spec.get('doc_id'),SPEC_PATH=spec.get('downloaded_pdf_path'),SPEC_NUM=spec.get("spec_number"),RUN_KNOWLEDGE_CREATE=False)
        spec_knowledge_paths.append(spec_path)

    state['specs_retrieval_sources'] = spec_knowledge_paths

    return state


def codeKnowledgeCreator(state,RUN_KNOWLEDGE_CREATE=False):   


    # Codebase Configuration - resources live at app root (Code_Generation_v0), not under Knowledge_Retrieval
    KG_FILE = str(APP_ROOT / "resources/Codebase_Knowledge/OAI/KnowledgeGraph/knowledge_graph.pkl")
    FAISS_INDEX_FILE = str(APP_ROOT / "resources/Codebase_Knowledge/OAI/vector_db/faiss_index.index")
    FAISS_METADATA_FILE = str(APP_ROOT / "resources/Codebase_Knowledge/OAI/vector_db/faiss_metadata.json")

    if RUN_KNOWLEDGE_CREATE:
        # print("=" * 60)
        # print("codeKGv1 — Full pipeline (parse -> chunks -> KG + FAISS)")
        # print("=" * 60)

        # Step 1: Parse codebase (prints file summary)
        # print("\n[1/3] Parsing codebase...")
        parse_main()

        # Step 2: Extract chunks
        # print("\n[2/3] Extracting chunks...")
        chunks_path = "./outputs" / "chunks.json"
        run_extraction(output_json=chunks_path)

        # Step 3: Build KG + FAISS
        # print("\n[3/3] Building knowledge graph and FAISS vector DB...")
        kg_main()

        # print("\n" + "=" * 60)
        # print("Pipeline finished. Outputs in: outputs/")
        # print("  - chunks.json")
        # print("  - knowledge_graph.pkl")
        # print("  - faiss_index.index")
        # print("  - faiss_metadata.json")
        # print("=" * 60)


    state['code_retrieval_sources'] = {
        "codebase_name": "openairinterface5g-develop",
        "target_dirs": ["openair1", "openair2", "openair3", "common"],
        "kg_file": KG_FILE,
        "faiss_index_file": FAISS_INDEX_FILE,
        "faiss_metadata_file": FAISS_METADATA_FILE
    }  

    return state



def createKnowledge(state):
    # Specification Knowledge Creation
    # print("Before Specification Knowledge Creation")
    state = specKnowledgeCreator(state)
    # print("After Specification Knowledge Creation")

    # Codebase Knowledge Creation
    state = codeKnowledgeCreator(state)


    return state
