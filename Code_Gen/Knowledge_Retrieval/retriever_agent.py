import os
import json
import sys
import logging
import re
from pathlib import Path
# from agentic_template_filler import AgenticTemplateFiller
import time
from datetime import datetime, timedelta, timezone


# ----------------------------------------------------------
# Add project root to Python path
# ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

from .code_chunks_retriever import SemanticGraphRAG
from .spec_agentic_ie_retrieval_phase import run_agentic_ie_retrieval_phase

IST = timezone(timedelta(hours=5, minutes=30))

def _safe_filename_stem(text: str, max_len: int = 80) -> str:
    """
    Convert arbitrary text (message/intent) to a filesystem-safe filename stem.
    """
    s = str(text or "").strip()
    # Remove invalid Windows filename characters + control chars.
    s = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", s)
    # Collapse whitespace and keep it compact.
    s = re.sub(r"\s+", "_", s).strip("._-")
    if not s:
        s = "feature"
    return s[:max_len]

def codeChunkRetrieverAgent(state,query,CODE_KNOWLEDGE_PATHS):

    # print("------------Code Retriever Agent----------")
    # print(state)

    # return state
    message_names = state.get("message_names") or ["feature"]
    run_ts = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
    code_chunks_dir = PROJECT_ROOT / "outputs" / "code_chunks"
    os.makedirs(code_chunks_dir, exist_ok=True)

    base_paths = {
        "faiss_index": CODE_KNOWLEDGE_PATHS.get("faiss_index_file"),
        "faiss_meta": CODE_KNOWLEDGE_PATHS.get("faiss_metadata_file"),
        "kg_path": CODE_KNOWLEDGE_PATHS.get("kg_file"),
    }

    # We pass function_calls and function_uses_struct here
    rel_filters = ["function_calls", "function_uses_struct"]

    all_semantic_chunks = []
    all_expanded_chunks = {}
    per_message_outputs = []

    for message_name in message_names:
        retriever = SemanticGraphRAG(
            faiss_index_path=base_paths["faiss_index"],
            faiss_metadata_path=base_paths["faiss_meta"],
            kg_path=base_paths["kg_path"],
            feature_name=message_name,
        )
        message_query = f"{query}\nTarget message: {message_name}"
        semantic_res, d_map, seeds = retriever.retrieve(
            query=message_query,
            top_k=10,
            kg_depth=2,
            rel_filters=rel_filters,
            direction="out",
        )
        final_json = retriever.build_output_json(message_query, semantic_res, d_map, direction="out")

        for chunk in final_json.get("semantic_chunks", []):
            chunk["target_message_name"] = message_name
            all_semantic_chunks.append(chunk)

        for depth_key, chunks in (final_json.get("expanded_chunks", {}) or {}).items():
            all_expanded_chunks.setdefault(depth_key, [])
            for chunk in chunks:
                chunk["target_message_name"] = message_name
                all_expanded_chunks[depth_key].append(chunk)

        safe_message_name = _safe_filename_stem(message_name)
        message_file = code_chunks_dir / f"{safe_message_name}_code_chunks_{run_ts}.json"
        with open(message_file, "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2)
        per_message_outputs.append(str(message_file))

    combined_json = {
        "metadata": {
            "timestamp": datetime.now(IST).isoformat(),
            "user_query": query,
            "target_messages": message_names,
            "num_messages": len(message_names),
            "num_semantic_chunks": len(all_semantic_chunks),
        },
        "semantic_chunks": all_semantic_chunks,
        "expanded_chunks": all_expanded_chunks,
        "per_message_outputs": per_message_outputs,
    }
    combined_file = code_chunks_dir / f"all_messages_code_chunks_{run_ts}.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(combined_json, f, indent=2)

    # Storing in the global state
    state['code_artifacts_context'] = {
        "semantic_chunks": combined_json.get("semantic_chunks", []),
        "expanded_chunks": combined_json.get("expanded_chunks", {}),
        "metadata": combined_json.get("metadata", {})
    }
    state['code_artifacts_chunks_path'] = str(combined_file)
    state['code_artifacts_chunks_paths_by_message'] = per_message_outputs
    return state


def specChunkRetrieverAgent(state, query,INITIAL_RETRIEVAL_SOURCES,template_path):
    repo_root = PROJECT_ROOT
    feature_json_path = state.get("feature_validation_input_path")
    if feature_json_path:
        feature_json_path = Path(feature_json_path)
    else:
        feature_json_path = repo_root / "Inter-gNB-DU_LTM_handover_procedure_20260323_093447.json"

    if not feature_json_path.exists():
        raise FileNotFoundError(f"Feature JSON not found: {feature_json_path}")

    if template_path:
        template_file = Path(template_path)
    else:
        template_file = repo_root / "inputs" / "Template.json"
    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found: {template_file}")

    kg_base_dir = repo_root / "resources" / "Spec_knowledge"

    payload = run_agentic_ie_retrieval_phase(
        feature_json_path=feature_json_path,
        template_path=template_file,
        kg_base_dir=kg_base_dir,
        max_depth_kg_expand=2,
        llm_iexpand_max_depth=2,
        llm_iexpand_max_nodes=60,
    )

    final_context = payload.get("final_context", []) if isinstance(payload, dict) else []
    state["specs_context"] = final_context

    output_dir = repo_root / "outputs" / "spec_chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"agentic_ie_retrieval_context_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.json"
    latest_path = output_dir / "agentic_ie_retrieval_context.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    state["specs_chunks_path"] = str(out_path)
    state["specs_retrieval_payload"] = payload

    return state



def retrieverAgent(state):
    # print("Retriever Agent")
    message_name = state.get("message_names")[0]
    user_intent = (
        state.get("intent")
        or state.get("messages")[0].content
    )

    # print("Message Name: %s, User Query: %s", message_name, user_intent)

    


    # ----------------------Specs Content Retrieval-----------------------
    ALL_RETRIEVAL_SOURCES = state.get("specs_retrieval_sources")
    template_path = state.get("selected_template_path")

    # print("Before Specification Chunk Retriever Agent")
    state = specChunkRetrieverAgent(state, user_intent, ALL_RETRIEVAL_SOURCES, template_path)

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
    state = codeChunkRetrieverAgent(state, user_intent, CODE_KNOWLEDGE_PATHS)
    # print("After Code Chunk Retriever Agent")
    return state

