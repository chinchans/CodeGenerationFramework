import logging
import json
from pathlib import Path
from typing import Any, Dict

from .code_knowledge.parse_codebase import main as parse_main
from .code_knowledge.extract_chunks import run_extraction
from .code_knowledge.build_kg_vector_new import main as kg_main
from .spec_ingestion_chunking import SpecIngestionChunker, save_ingestion_outputs
from .spec_knowledge_graph_builder import SpecKnowledgeGraphBuilder
from .toc_parser import parse_toc_sections, save_toc_sections


logger = logging.getLogger(__name__)

MODULE_DIR = Path(__file__).resolve().parent
# Align canonical KG paths with what `Code_Gen/pipeline.py` expects.
# - pipeline.py checks: `Code_Gen/resources/...`
# - This module previously copied to workspace-root `resources/...`,
#   which caused the "missing graphs remain" error.
CODE_GEN_ROOT = MODULE_DIR.parent.parent  # .../Code_Gen
WORKSPACE_ROOT = MODULE_DIR.parent.parent.parent  # repo root

SPEC_KG_ROOT = CODE_GEN_ROOT / "resources" / "Spec_knowledge"
CODE_KG_ROOT = CODE_GEN_ROOT / "resources" / "Code_knowledge" / "OAI"
CODE_KNOWLEDGE_DIR = MODULE_DIR / "code_knowledge"
DEFAULT_FEATURE_INPUT_JSON = (
    WORKSPACE_ROOT / "Inter-gNB-DU_LTM_handover_procedure_20260323_093447.json"
)


def _as_doc_id(doc_id: str | None, spec_path: str | None) -> str:
    if doc_id:
        return str(doc_id).strip()
    if spec_path:
        return Path(spec_path).stem
    raise ValueError("Either DOC_ID or SPEC_PATH must be provided.")


def _spec_kg_paths(doc_id: str) -> Dict[str, str]:
    base = SPEC_KG_ROOT / doc_id
    return {
        "kg_file": str(base / "KnowledgeGraph" / "knowledge_graph.pkl"),
        "faiss_index_file": str(base / "vector_db" / "faiss_index.index"),
        "faiss_metadata_file": str(base / "vector_db" / "faiss_metadata.json"),
    }


def _spec_resource_exists(doc_id: str) -> bool:
    paths = _spec_kg_paths(doc_id)
    # For spec retrieval in your new flow, KG file is the required source of truth.
    return Path(paths["kg_file"]).exists()


def _load_feature_input_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Feature input JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _specifications_from_feature_payload(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    specs = payload.get("specs", []) or []
    if not isinstance(specs, list):
        raise ValueError("Feature input JSON must contain a list 'specs'.")
    normalized: list[Dict[str, Any]] = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        normalized.append(
            {
                "doc_id": spec.get("doc_id"),
                "downloaded_pdf_path": spec.get("downloaded_pdf_path"),
                "spec_number": spec.get("spec_number"),
            }
        )
    return normalized


def _ensure_state_from_feature_input(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("specifications"):
        return state

    feature_input_path = state.get("feature_validation_input_path")
    json_path = Path(feature_input_path) if feature_input_path else DEFAULT_FEATURE_INPUT_JSON
    payload = _load_feature_input_json(json_path)
    state["specifications"] = _specifications_from_feature_payload(payload)
    state["intent"] = state.get("intent") or payload.get("intent", "")
    state["feature_validation_input_path"] = str(json_path)
    return state


def _create_spec_knowledge(doc_id: str, spec_pdf_path: str) -> None:
    pdf_path = Path(spec_pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Spec PDF not found for {doc_id}: {pdf_path}")

    output_root = MODULE_DIR / "spec_chunks"
    output_root.mkdir(parents=True, exist_ok=True)

    # Stable defaults copied from your local spec pipeline usage.
    skip_start = 9
    skip_from = None
    toc_page_overrides = {
        "ts_138401v180600p": (4, 8),
        "ts_138473v180600p": (4, 21),
    }
    strict_toc_only = False
    include_semantic_relations = True
    semantic_top_k_candidates = 10
    semantic_final_k = 5
    semantic_max_chunks = 350

    override = toc_page_overrides.get(doc_id)
    toc_start_page = override[0] if override else None
    toc_end_page = override[1] if override else None

    toc_entries = parse_toc_sections(
        str(pdf_path),
        toc_start_page=toc_start_page,
        toc_end_page=toc_end_page,
        strict_toc_only=strict_toc_only,
    )
    toc_sections_path = save_toc_sections(toc_entries, str(output_root), doc_id)

    chunker = SpecIngestionChunker(doc_id=doc_id)
    result = chunker.run(
        pdf_path=str(pdf_path),
        skip_start=skip_start,
        skip_from=skip_from,
        toc_sections_path=toc_sections_path,
    )
    paths = save_ingestion_outputs(
        doc_id=doc_id,
        sections=result["sections"],
        chunks=result["chunks"],
        output_root=str(output_root),
    )

    builder = SpecKnowledgeGraphBuilder(doc_id=doc_id)
    chunks = builder._load_chunks(paths["chunks_path"])
    graph = builder.build_graph(
        chunks,
        include_semantic_relations=include_semantic_relations,
        semantic_top_k_candidates=semantic_top_k_candidates,
        semantic_final_k=semantic_final_k,
        semantic_max_chunks=semantic_max_chunks,
    )

    graph_dir = output_root / doc_id / "KnowledgeGraph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    source_graph_path = graph_dir / "knowledge_graph.pkl"
    source_summary_path = graph_dir / "graph_summary.json"
    source_adj_path = graph_dir / "knowledge_graph_adjacency.json"
    builder.save_graph(graph, str(source_graph_path))
    builder.save_graph_summary(graph, str(source_summary_path))
    builder.save_graph_adjacency_json(graph, str(source_adj_path), doc_id=doc_id)

    # Copy to canonical retrieval location expected by downstream modules.
    target_base = SPEC_KG_ROOT / doc_id / "KnowledgeGraph"
    target_base.mkdir(parents=True, exist_ok=True)
    target_graph_path = target_base / "knowledge_graph.pkl"
    target_summary_path = target_base / "graph_summary.json"
    target_adj_path = target_base / "knowledge_graph_adjacency.json"
    target_graph_path.write_bytes(source_graph_path.read_bytes())
    target_summary_path.write_bytes(source_summary_path.read_bytes())
    target_adj_path.write_bytes(source_adj_path.read_bytes())

    # Mirror full spec artifacts (not only KG files) to canonical retrieval location,
    # so Code_Gen/resources/Spec_knowledge/<doc_id>/ looks like KG_Only_Pipeline outputs.
    target_doc_dir = SPEC_KG_ROOT / doc_id
    target_doc_dir.mkdir(parents=True, exist_ok=True)

    # Ingestion outputs
    src_chunks = Path(paths["chunks_path"])
    src_sections = Path(paths["sections_path"])
    if src_chunks.exists():
        (target_doc_dir / "chunks.json").write_bytes(src_chunks.read_bytes())
    if src_sections.exists():
        (target_doc_dir / "sections.json").write_bytes(src_sections.read_bytes())

    # TOC outputs (tree + flat)
    src_toc = Path(toc_sections_path)
    src_toc_flat = src_toc.with_name("toc_sections_flat.json")
    if src_toc.exists():
        (target_doc_dir / "toc_sections.json").write_bytes(src_toc.read_bytes())
    if src_toc_flat.exists():
        (target_doc_dir / "toc_sections_flat.json").write_bytes(src_toc_flat.read_bytes())


def specKnowledgeCreatorForEachSpec(
    DOC_ID,
    SPEC_PATH,
    SPEC_NUM,
    RUN_KNOWLEDGE_CREATE=False,
):
    doc_id = _as_doc_id(DOC_ID, SPEC_PATH)
    spec_paths = _spec_kg_paths(doc_id)

    should_create = RUN_KNOWLEDGE_CREATE or (not _spec_resource_exists(doc_id))
    if should_create:
        _create_spec_knowledge(doc_id=doc_id, spec_pdf_path=SPEC_PATH)

    spec_knowledge_data: Dict[str, Any] = {
        "source_id": doc_id,
        "official_spec_id": SPEC_NUM,
        "kg_file": spec_paths["kg_file"],
        "faiss_index_file": spec_paths["faiss_index_file"],
        "faiss_metadata_file": spec_paths["faiss_metadata_file"],
    }
    return spec_knowledge_data


def specKnowledgeCreator(state):
    # print("Knowledge Creator Agent")
    # print(state['session_id'])
    # print("----------------Specs Path----------------\n")
    # print(state['specifications'])
    

    state = _ensure_state_from_feature_input(state)
    spec_knowledge_paths = []
    for spec in state.get("specifications", []):
        spec_path = specKnowledgeCreatorForEachSpec(
            DOC_ID=spec.get("doc_id"),
            SPEC_PATH=spec.get("downloaded_pdf_path"),
            SPEC_NUM=spec.get("spec_number"),
            RUN_KNOWLEDGE_CREATE=False,
        )
        spec_knowledge_paths.append(spec_path)

    state["specs_retrieval_sources"] = spec_knowledge_paths

    return state


def codeKnowledgeCreator(state, RUN_KNOWLEDGE_CREATE=False):
    kg_file = str(CODE_KG_ROOT / "KnowledgeGraph" / "knowledge_graph.pkl")
    faiss_index_file = str(CODE_KG_ROOT / "vector_db" / "faiss_index.index")
    faiss_metadata_file = str(CODE_KG_ROOT / "vector_db" / "faiss_metadata.json")

    if RUN_KNOWLEDGE_CREATE:
        parse_main()

        chunks_path = CODE_KNOWLEDGE_DIR / "outputs" / "chunks.json"
        run_extraction(output_json=chunks_path)

        kg_main()

        # Mirror generated outputs to the canonical resources path used by retriever.
        generated_kg = CODE_KNOWLEDGE_DIR / "outputs" / "knowledge_graph.pkl"
        generated_faiss = CODE_KNOWLEDGE_DIR / "outputs" / "faiss_index.index"
        generated_meta = CODE_KNOWLEDGE_DIR / "outputs" / "faiss_metadata.json"
        (CODE_KG_ROOT / "KnowledgeGraph").mkdir(parents=True, exist_ok=True)
        (CODE_KG_ROOT / "vector_db").mkdir(parents=True, exist_ok=True)
        if generated_kg.exists():
            Path(kg_file).write_bytes(generated_kg.read_bytes())
        if generated_faiss.exists():
            Path(faiss_index_file).write_bytes(generated_faiss.read_bytes())
        if generated_meta.exists():
            Path(faiss_metadata_file).write_bytes(generated_meta.read_bytes())

    state["code_retrieval_sources"] = {
        "codebase_name": "openairinterface5g-develop",
        "target_dirs": ["openair1", "openair2", "openair3", "common"],
        "kg_file": kg_file,
        "faiss_index_file": faiss_index_file,
        "faiss_metadata_file": faiss_metadata_file,
    }

    return state



def createKnowledge(state):
    state = _ensure_state_from_feature_input(state)
    state = specKnowledgeCreator(state)
    state = codeKnowledgeCreator(state)
    return state
