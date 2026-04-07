from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .Knowledge_Retrieval.spec_agentic_ie_retrieval_phase import (
    run_agentic_ie_retrieval_phase,
)
from .Knowledge_Retrieval.Knowlege_Creation1.knowledge_creator_agent import (
    specKnowledgeCreatorForEachSpec,
    codeKnowledgeCreator,
)
from .Knowledge_Retrieval.retriever_agent import codeChunkRetrieverAgent
from .Template_Orchestrator.code_template_filler import CodeTemplateFiller
from .Template_Orchestrator.prompt_generator import promptGenerationAgent
from .Template_Orchestrator.spec_retrieval_context_adapter import (
    agentic_ie_retrieval_to_template_filler_inputs,
)
from .Template_Orchestrator.spec_template_filler import SpecTemplateFiller
from .Template_Orchestrator.template_filler_agent import llm as template_orchestrator_llm
from .Self_Learning.self_learning_agent import validate_template_with_mapping_rules
from .state import CodeGenState
from .sqlite_state_store import SqliteStateStore


@dataclass
class Message:
    content: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _code_gen_root() -> Path:
    return Path(__file__).resolve().parent


IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist() -> datetime:
    return datetime.now(IST)


def _session_db_path(repo_root: Path) -> Path:
    return repo_root / "outputs" / "session_state.sqlite"


def _get_state_store(repo_root: Path) -> SqliteStateStore:
    return SqliteStateStore(_session_db_path(repo_root))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _extract_doc_id(spec: Dict[str, Any]) -> str:
    doc_id = str(spec.get("doc_id", "")).strip()
    if doc_id:
        return doc_id
    pdf_path = str(spec.get("downloaded_pdf_path", "")).strip()
    if not pdf_path:
        raise ValueError("Spec entry must include either 'doc_id' or 'downloaded_pdf_path'.")
    return Path(pdf_path).stem


def _resolve_template_path(feature_payload: Dict[str, Any], repo_root: Path) -> Path:
    template_path = (
        feature_payload.get("template", {}) or {}
    ).get("template_path", "")
    if template_path and Path(template_path).exists():
        return Path(template_path)

    # If the incoming JSON points to a template path from another machine,
    # fall back to a local template by name (repo-local `inputs/`).
    template_name = (feature_payload.get("template", {}) or {}).get("template_name", "")
    if template_name:
        local_by_name = repo_root / "inputs" / str(template_name)
        if local_by_name.exists():
            return local_by_name

    # Legacy fallback (kept for backward compatibility), but fail fast with
    # a clearer error if missing.
    fallback = repo_root / "inputs" / "Template_common.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        "Template JSON not found. Tried:\n"
        f"- template.template_path (from feature JSON): {template_path}\n"
        f"- template.template_name (repo inputs/): {template_name}\n"
        f"- fallback: {fallback}\n"
        "Fix: ensure `template.template_name` matches a file under `inputs/`."
    )


def _spec_kg_missing(spec: Dict[str, Any], repo_root: Path) -> bool:
    doc_id = _extract_doc_id(spec)
    kg_path = (
        _code_gen_root()
        / "resources"
        / "Spec_knowledge"
        / doc_id
        / "KnowledgeGraph"
        / "knowledge_graph.pkl"
    )
    return not kg_path.exists()


def _ensure_spec_knowledge_sources(
    feature_payload: Dict[str, Any], feature_json_path: Path, repo_root: Path
) -> Dict[str, Any]:
    specs = feature_payload.get("specs", []) or []
    if not specs:
        raise ValueError("Feature JSON must contain non-empty 'specs'.")

    missing_specs = [spec for spec in specs if _spec_kg_missing(spec, repo_root)]
    if not missing_specs:
        return {"created": False, "missing_doc_ids": [], "status": "already_available"}

    for spec in missing_specs:
        specKnowledgeCreatorForEachSpec(
            DOC_ID=spec.get("doc_id"),
            SPEC_PATH=spec.get("downloaded_pdf_path"),
            SPEC_NUM=spec.get("spec_number"),
            RUN_KNOWLEDGE_CREATE=True,
        )

    still_missing = [spec for spec in specs if _spec_kg_missing(spec, repo_root)]
    if still_missing:
        missing_doc_ids = [_extract_doc_id(s) for s in still_missing]
        raise RuntimeError(
            f"Spec KG creation completed but missing graphs remain: {missing_doc_ids}"
        )

    return {
        "created": True,
        "missing_doc_ids": [_extract_doc_id(s) for s in missing_specs],
        "status": "created_missing_sources",
    }


def _ensure_code_knowledge_source(repo_root: Path) -> Dict[str, Any]:
    code_paths = {
        "kg_file": str(
            _code_gen_root()
            / "resources"
            / "Code_knowledge"
            / "OAI"
            / "KnowledgeGraph"
            / "knowledge_graph.pkl"
        ),
        "faiss_index_file": str(
            _code_gen_root()
            / "resources"
            / "Code_knowledge"
            / "OAI"
            / "vector_db"
            / "faiss_index.index"
        ),
        "faiss_metadata_file": str(
            _code_gen_root()
            / "resources"
            / "Code_knowledge"
            / "OAI"
            / "vector_db"
            / "faiss_metadata.json"
        ),
    }
    required_paths = [
        code_paths.get("kg_file", ""),
        code_paths.get("faiss_index_file", ""),
        code_paths.get("faiss_metadata_file", ""),
    ]
    missing = [p for p in required_paths if p and not Path(p).exists()]

    if not missing:
        return {"created": False, "status": "already_available", "paths": code_paths}

    state_with_create = codeKnowledgeCreator({}, RUN_KNOWLEDGE_CREATE=True)
    code_paths = state_with_create.get("code_retrieval_sources", code_paths)
    required_paths = [
        code_paths.get("kg_file", ""),
        code_paths.get("faiss_index_file", ""),
        code_paths.get("faiss_metadata_file", ""),
    ]
    missing_after = [p for p in required_paths if p and not Path(p).exists()]
    if missing_after:
        raise RuntimeError(f"Code KG creation failed for paths: {missing_after}")

    return {"created": True, "status": "created_missing_source", "paths": code_paths}


def _run_spec_retrieval(
    feature_json_input: Any, template_path: Path, repo_root: Path
) -> Tuple[Dict[str, Any], Path]:
    kg_base_dir = _code_gen_root() / "resources" / "Spec_knowledge"
    retrieval_payload = run_agentic_ie_retrieval_phase(
        feature_json_path=feature_json_input,
        template_path=template_path,
        kg_base_dir=kg_base_dir,
        max_depth_kg_expand=2,
        llm_iexpand_max_depth=2,
        llm_iexpand_max_nodes=60,
    )
    retrieval_dir = _code_gen_root() / "outputs" / "spec_chunks"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    run_ts = _now_ist().strftime("%Y%m%d_%H%M%S")
    retrieval_path = retrieval_dir / f"agentic_ie_retrieval_context_{run_ts}.json"
    latest_path = retrieval_dir / "agentic_ie_retrieval_context.json"
    _save_json(retrieval_path, retrieval_payload)
    _save_json(latest_path, retrieval_payload)
    return retrieval_payload, retrieval_path


def _extract_message_names(feature_payload: Dict[str, Any]) -> List[str]:
    messages = (feature_payload.get("message_details", {}) or {}).get("messages", []) or []
    names: List[str] = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("name"):
            names.append(str(msg["name"]))
    if names:
        return names
    intent = str(feature_payload.get("intent", "")).strip()
    return [intent[:120] if intent else "feature"]


def _run_code_retrieval(
    intent: str, feature_payload: Dict[str, Any], code_paths: Dict[str, Any]
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "message_names": _extract_message_names(feature_payload),
    }
    return codeChunkRetrieverAgent(state, intent, code_paths)


def _build_initial_state(
    *,
    intent: str,
    template_path: Path,
    feature_payload: Dict[str, Any],
    feature_json_input: Path | None = None,
) -> CodeGenState:
    state: CodeGenState = {
        "intent": intent,
        "template_path_used": str(template_path),
        "message_names": _extract_message_names(feature_payload),
    }
    if feature_json_input is not None:
        state["input_feature_json"] = str(feature_json_input)
    state["input_intent"] = intent

    if isinstance(feature_payload.get("protocol_classification"), dict):
        state["protocol_classification"] = feature_payload["protocol_classification"]
    if isinstance(feature_payload.get("specs"), list):
        state["specifications"] = feature_payload["specs"]
    if isinstance(feature_payload.get("intent_obj"), dict):
        state["feature_intent"] = feature_payload["intent_obj"]

    template_info = feature_payload.get("template", {}) or {}
    if template_info.get("template_name"):
        state["selected_template_name"] = str(template_info["template_name"])
    if template_info.get("template_path"):
        state["selected_template_path"] = str(template_info["template_path"])

    return state


def _run_template_orchestrator_two_pass(
    *,
    intent: str,
    template_path: Path,
    spec_retrieval_payload: Dict[str, Any],
    code_retrieval_state: Dict[str, Any],
) -> Dict[str, Any]:
    spec_inputs = agentic_ie_retrieval_to_template_filler_inputs(spec_retrieval_payload)
    spec_chunks = spec_inputs["chunks"]

    spec_filler = SpecTemplateFiller(template_file=str(template_path))
    extracted_info = spec_filler.extract_information(query=intent, chunks=spec_chunks)
    partially_filled_template = spec_filler.fill_template(
        extracted_info=extracted_info,
        chunks=spec_chunks,
    )
    spec_template_path = spec_filler.save_output(
        filled_template=partially_filled_template,
        query=intent,
        output_dir=str(_code_gen_root() / "outputs" / "spec_filled_templates"),
    )

    code_filler = CodeTemplateFiller(llm=template_orchestrator_llm)
    code_state = {
        "messages": [Message(content=intent)],
        "code_artifacts_context": code_retrieval_state.get("code_artifacts_context", {}),
    }
    final_filled_template_path = code_filler.template_filler(code_state, spec_template_path)

    return {
        "spec_filled_template_path": spec_template_path,
        "final_filled_template_path": final_filled_template_path,
    }


def _run_prompt_generation(intent: str, final_filled_template_path: str) -> Dict[str, Any]:
    prompt_state = {
        "messages": [Message(content=intent)],
        "final_filled_template_path": final_filled_template_path,
    }
    prompt_state = promptGenerationAgent(prompt_state)
    return {
        "generated_prompt_path": prompt_state.get("code_generation_prompt_path"),
        "generated_prompt": prompt_state.get("code_generation_prompt"),
    }


def _run_self_learning_validation(
    *,
    intent: str,
    final_filled_template_path: str,
    user_resolutions: Any = None,
    use_llm_review: Any = None,
) -> Dict[str, Any]:
    rules_path = _code_gen_root() / "Self_Learning" / "ltm_mapping_rules.json"
    return validate_template_with_mapping_rules(
        intent=intent,
        final_template_path=final_filled_template_path,
        mapping_rules_path=str(rules_path),
        user_resolutions=user_resolutions,
        output_dir=str(_code_gen_root() / "outputs" / "self_learning_templates"),
        use_llm_review=use_llm_review,
    )


def prepare_ambiguity_resolutions_input(
    repo_root: Path,
    *,
    inline: Any = None,
    resolutions_json_path: str = "",
) -> Tuple[Any, Optional[str]]:
    """
    Build a single resolutions payload from inline MCP dict and/or a JSON file.
    File may be a dict (id -> value) or a list of {id, value}.
    """
    out: Any = None
    rjp = (resolutions_json_path or "").strip()
    if rjp:
        p = Path(rjp)
        if not p.is_absolute():
            p = repo_root / p
        if not p.is_file():
            return None, f"resolutions_json_path not found: {p}"
        out = json.loads(p.read_text(encoding="utf-8"))

    if inline is not None and isinstance(inline, dict) and inline:
        if out is None:
            out = dict(inline)
        elif isinstance(out, dict):
            out = {**out, **inline}
        else:
            out = dict(inline)
    elif inline is not None and inline not in ({}, None):
        if out is None:
            out = inline

    if out is None:
        return None, (
            "No resolutions provided. Pass a non-empty `resolutions` object and/or a valid "
            "`resolutions_json_path` (repo-relative or absolute)."
        )
    if isinstance(out, dict) and not out:
        return None, "Resolutions dict is empty."
    if isinstance(out, list) and not out:
        return None, "Resolutions list is empty."
    return out, None


def run_resolve_self_learning_session(
    session_id: str,
    user_resolutions: Any,
    use_llm_review: Any = None,
) -> Dict[str, Any]:
    """
    Apply ambiguity resolutions against the template path stored for `session_id` (no full
    pipeline re-run). Updates SQLite session rows and regenerates the prompt when clear.

    Use this after `generate_enriched_prompt` returns `session_id` and ambiguities, so ambiguity
    ids stay aligned with the same filled template.

    The session row must have status `awaiting_user` (set when ambiguities are reported). Sessions
    already at `final_prompt_ready` must use a new `generate_enriched_prompt` run for a new intent.
    """
    repo_root = _repo_root()
    store = _get_state_store(repo_root)
    sid = str(session_id or "").strip()
    if not sid:
        raise ValueError("session_id must be non-empty")

    sess = store.get_session(sid)
    if not sess:
        raise ValueError(f"Unknown session_id: {sid}")

    st = str(sess.get("status") or "").strip()
    if st != "awaiting_user":
        raise ValueError(
            f"Session status is {st!r}; ambiguity resolution applies only to sessions in "
            "'awaiting_user' (after generate_enriched_prompt reported ambiguities). "
            "For a new intent, call generate_enriched_prompt to start a new session."
        )

    intent = str(sess.get("intent") or "").strip()
    template_path = str(sess.get("template_path") or "").strip()
    if not intent:
        raise ValueError("Session has no intent recorded")
    if not template_path or not Path(template_path).is_file():
        raise ValueError(f"Template file missing for session (template_path={template_path!r})")

    sl_out = _run_self_learning_validation(
        intent=intent,
        final_filled_template_path=template_path,
        user_resolutions=user_resolutions,
        use_llm_review=use_llm_review,
    )

    resolved_template_path = sl_out.get("resolved_template_path") or template_path

    state: Dict[str, Any] = {
        "session_id": sid,
        "intent": intent,
        "template_path_used": template_path,
        "final_filled_template_path": resolved_template_path,
        "self_learning_matched_rule": sl_out.get("matched_rule"),
        "self_learning_has_ambiguities": sl_out.get("has_ambiguities", False),
        "self_learning_ambiguities": sl_out.get("ambiguities", []),
        "self_learning_resolution_applied": sl_out.get("resolution_applied", False),
        "self_learning_validation_policy": sl_out.get("validation_policy") or {},
        "self_learning_llm_ambiguity_review": sl_out.get("llm_ambiguity_review") or {},
        "self_learning_deterministic_ambiguity_count": int(sl_out.get("deterministic_ambiguity_count", 0)),
        "self_learning_human_resolution_required": bool(sl_out.get("human_resolution_required", False)),
        "self_learning_next_step_if_ambiguous": str(sl_out.get("next_step_if_ambiguous") or ""),
        "self_learning_resolved_template_path": resolved_template_path,
    }

    if state["self_learning_has_ambiguities"]:
        state["code_generation_prompt_path"] = ""
        state["code_generation_prompt"] = ""
        store.ensure_session(
            session_id=sid,
            intent=intent,
            status="awaiting_user",
            template_path=str(resolved_template_path),
            ambiguity_questions=state.get("self_learning_ambiguities", []),
        )
    else:
        prompt_out = _run_prompt_generation(
            intent=intent, final_filled_template_path=resolved_template_path
        )
        state["code_generation_prompt_path"] = prompt_out.get("generated_prompt_path", "")
        state["code_generation_prompt"] = prompt_out.get("generated_prompt", "")
        store.ensure_session(
            session_id=sid,
            intent=intent,
            status="final_prompt_ready",
            template_path=str(resolved_template_path),
            code_generation_prompt=state.get("code_generation_prompt", ""),
            code_generation_prompt_path=state.get("code_generation_prompt_path", ""),
            ambiguity_questions=[],
        )

    resolve_manifest = {
        "resolution_only": True,
        "session_id": sid,
        "intent": intent,
        "self_learning": {
            "matched_rule": state.get("self_learning_matched_rule"),
            "has_ambiguities": state.get("self_learning_has_ambiguities", False),
            "ambiguity_count": len(state.get("self_learning_ambiguities", []) or []),
            "resolution_applied": state.get("self_learning_resolution_applied", False),
            "resolved_template_path": state.get("self_learning_resolved_template_path", ""),
        },
        "template_outputs": {
            "final_filled_template_path": state.get("final_filled_template_path"),
            "generated_prompt_path": state.get("code_generation_prompt_path"),
            "generated_prompt": state.get("code_generation_prompt"),
        },
    }
    manifest_path = (
        _code_gen_root()
        / "outputs"
        / "code_gen_runs"
        / f"code_gen_resolve_session_{sid[:8]}_{_now_ist().strftime('%Y%m%d_%H%M%S')}.json"
    )
    _save_json(manifest_path, resolve_manifest)
    state["run_manifest_path"] = str(manifest_path)
    return state


def run_end_to_end(feature_json_path: str) -> Dict[str, Any]:
    repo_root = _repo_root()
    store = _get_state_store(repo_root)
    session_id = str(uuid.uuid4())
    feature_path = Path(feature_json_path)
    if not feature_path.is_absolute():
        feature_path = repo_root / feature_path
    feature_payload = _load_json(feature_path)

    intent = str(feature_payload.get("intent", "")).strip()
    if not intent:
        raise ValueError("Feature JSON must contain non-empty 'intent'.")
    template_path = _resolve_template_path(feature_payload, repo_root)
    store.ensure_session(
        session_id=session_id,
        intent=intent,
        status="started",
        template_path=str(template_path),
    )
    state = _build_initial_state(
        intent=intent,
        template_path=template_path,
        feature_payload=feature_payload,
        feature_json_input=feature_path,
    )
    state["session_id"] = session_id

    kg_stage_run_id = store.upsert_stage_run(
        session_id=session_id,
        stage_name="knowledge_creator_done",
        stage_status="started",
        input_summary={"specs_count": len(feature_payload.get("specs", []) or [])},
    )
    with store.agent_run(
        session_id=session_id,
        stage_run_id=kg_stage_run_id,
        agent_name="specKnowledgeCreatorForEachSpec",
        input_payload={"specs": feature_payload.get("specs", []) or []},
        output_capture_keys=["created", "missing_doc_ids", "status"],
    ):
        spec_kg_status = _ensure_spec_knowledge_sources(
            feature_payload=feature_payload,
            feature_json_path=feature_path,
            repo_root=repo_root,
        )
    with store.agent_run(
        session_id=session_id,
        stage_run_id=kg_stage_run_id,
        agent_name="codeKnowledgeCreator",
        input_payload={},
        output_capture_keys=["created", "status", "paths"],
    ):
        code_kg_status = _ensure_code_knowledge_source(repo_root)
    store.upsert_stage_run(
        session_id=session_id,
        stage_name="knowledge_creator_done",
        stage_status="completed",
        output={"spec_kg": spec_kg_status, "code_kg": code_kg_status},
    )
    state["spec_kg_status"] = spec_kg_status
    state["code_retrieval_sources"] = code_kg_status["paths"]

    retrieval_stage_run_id = store.upsert_stage_run(
        session_id=session_id,
        stage_name="retrieval_done",
        stage_status="started",
        input_summary={"template_path": str(template_path)},
    )
    with store.agent_run(
        session_id=session_id,
        stage_run_id=retrieval_stage_run_id,
        agent_name="run_agentic_ie_retrieval_phase",
        input_payload={"feature_json_path": str(feature_path), "template_path": str(template_path)},
    ):
        spec_retrieval_payload, spec_retrieval_path = _run_spec_retrieval(
            feature_json_input=feature_path,
            template_path=template_path,
            repo_root=repo_root,
        )
    state["spec_retrieval_context"] = spec_retrieval_payload
    state["spec_retrieval_context_path"] = str(spec_retrieval_path)
    with store.agent_run(
        session_id=session_id,
        stage_run_id=retrieval_stage_run_id,
        agent_name="codeChunkRetrieverAgent",
        input_payload={"intent": intent, "message_names": _extract_message_names(feature_payload)},
        output_capture_keys=["code_artifacts_chunks_path"],
    ):
        code_retrieval_state = _run_code_retrieval(
            intent=intent,
            feature_payload=feature_payload,
            code_paths=code_kg_status["paths"],
        )
    state["code_artifacts_context"] = code_retrieval_state.get("code_artifacts_context", {})
    state["code_artifacts_chunks_path"] = code_retrieval_state.get("code_artifacts_chunks_path", "")
    store.upsert_stage_run(
        session_id=session_id,
        stage_name="retrieval_done",
        stage_status="completed",
        output={
            "spec_retrieval_context_path": str(spec_retrieval_path),
            "code_chunks_path": state.get("code_artifacts_chunks_path", ""),
        },
    )
    template_stage_run_id = store.upsert_stage_run(
        session_id=session_id,
        stage_name="template_filled",
        stage_status="started",
        input_summary={"template_path": str(template_path)},
    )
    with store.agent_run(
        session_id=session_id,
        stage_run_id=template_stage_run_id,
        agent_name="SpecTemplateFiller+CodeTemplateFiller",
        input_payload={"intent": intent},
        output_capture_keys=["spec_filled_template_path", "final_filled_template_path"],
    ):
        template_out = _run_template_orchestrator_two_pass(
            intent=intent,
            template_path=template_path,
            spec_retrieval_payload=spec_retrieval_payload,
            code_retrieval_state=code_retrieval_state,
        )
    state["spec_filled_template_path"] = template_out.get("spec_filled_template_path", "")
    state["final_filled_template_path"] = template_out.get("final_filled_template_path", "")
    store.upsert_stage_run(
        session_id=session_id,
        stage_name="template_filled",
        stage_status="completed",
        output={
            "spec_filled_template_path": state.get("spec_filled_template_path"),
            "final_filled_template_path": state.get("final_filled_template_path"),
        },
    )
    sl_stage_run_id = store.upsert_stage_run(
        session_id=session_id,
        stage_name="self_learning_done",
        stage_status="started",
        input_summary={"final_filled_template_path": state.get("final_filled_template_path", "")},
    )
    with store.agent_run(
        session_id=session_id,
        stage_run_id=sl_stage_run_id,
        agent_name="validate_template_with_mapping_rules",
        input_payload={"intent": intent},
        output_capture_keys=[
            "matched_rule",
            "has_ambiguities",
            "resolution_applied",
            "resolved_template_path",
            "validation_policy",
            "llm_ambiguity_review",
            "deterministic_ambiguity_count",
            "human_resolution_required",
            "next_step_if_ambiguous",
        ],
    ):
        sl_out = _run_self_learning_validation(
            intent=intent,
            final_filled_template_path=state["final_filled_template_path"],
            user_resolutions=None,
            use_llm_review=None,
        )
    state["self_learning_matched_rule"] = sl_out.get("matched_rule")
    state["self_learning_has_ambiguities"] = sl_out.get("has_ambiguities", False)
    state["self_learning_ambiguities"] = sl_out.get("ambiguities", [])
    state["self_learning_resolution_applied"] = sl_out.get("resolution_applied", False)
    state["self_learning_validation_policy"] = sl_out.get("validation_policy") or {}
    state["self_learning_llm_ambiguity_review"] = sl_out.get("llm_ambiguity_review") or {}
    state["self_learning_deterministic_ambiguity_count"] = int(sl_out.get("deterministic_ambiguity_count", 0))
    state["self_learning_human_resolution_required"] = bool(sl_out.get("human_resolution_required", False))
    state["self_learning_next_step_if_ambiguous"] = str(sl_out.get("next_step_if_ambiguous") or "")
    resolved_template_path = sl_out.get("resolved_template_path") or state["final_filled_template_path"]
    state["self_learning_resolved_template_path"] = resolved_template_path
    state["final_filled_template_path"] = resolved_template_path
    store.upsert_stage_run(
        session_id=session_id,
        stage_name="self_learning_done",
        stage_status="completed",
        output={
            "has_ambiguities": state.get("self_learning_has_ambiguities", False),
            "ambiguity_count": len(state.get("self_learning_ambiguities", []) or []),
            "resolved_template_path": resolved_template_path,
            "human_resolution_required": state.get("self_learning_human_resolution_required", False),
            "next_step_if_ambiguous": state.get("self_learning_next_step_if_ambiguous", ""),
        },
    )
    if state["self_learning_has_ambiguities"]:
        state["code_generation_prompt_path"] = ""
        state["code_generation_prompt"] = ""
        store.ensure_session(
            session_id=session_id,
            intent=intent,
            status="awaiting_user",
            template_path=str(resolved_template_path),
            ambiguity_questions=state.get("self_learning_ambiguities", []),
        )
    else:
        prompt_stage_run_id = store.upsert_stage_run(
            session_id=session_id,
            stage_name="prompt_generated",
            stage_status="started",
            input_summary={"resolved_template_path": resolved_template_path},
        )
        with store.agent_run(
            session_id=session_id,
            stage_run_id=prompt_stage_run_id,
            agent_name="promptGenerationAgent",
            input_payload={"intent": intent, "final_filled_template_path": resolved_template_path},
            output_capture_keys=["generated_prompt_path"],
        ):
            prompt_out = _run_prompt_generation(intent=intent, final_filled_template_path=resolved_template_path)
        state["code_generation_prompt_path"] = prompt_out.get("generated_prompt_path", "")
        state["code_generation_prompt"] = prompt_out.get("generated_prompt", "")
        store.upsert_stage_run(
            session_id=session_id,
            stage_name="prompt_generated",
            stage_status="completed",
            output={"code_generation_prompt_path": state.get("code_generation_prompt_path", "")},
        )
        store.ensure_session(
            session_id=session_id,
            intent=intent,
            status="final_prompt_ready",
            template_path=str(resolved_template_path),
            code_generation_prompt=state.get("code_generation_prompt", ""),
            code_generation_prompt_path=state.get("code_generation_prompt_path", ""),
        )

    manifest = {
        "input_feature_json": state.get("input_feature_json", str(feature_path)),
        "intent": state.get("intent", intent),
        "template_path_used": state.get("template_path_used", str(template_path)),
        "knowledge_creation": {
            "spec_kg": state.get("spec_kg_status", spec_kg_status),
            "code_kg": code_kg_status,
        },
        "retrieval_outputs": {
            "spec_retrieval_context_path": state.get("spec_retrieval_context_path", str(spec_retrieval_path)),
            "code_chunks_path": state.get("code_artifacts_chunks_path", code_retrieval_state.get("code_artifacts_chunks_path")),
        },
        "template_outputs": {
            "spec_filled_template_path": state.get("spec_filled_template_path", template_out.get("spec_filled_template_path")),
            "final_filled_template_path": state.get("final_filled_template_path", template_out.get("final_filled_template_path")),
            "generated_prompt_path": state.get("code_generation_prompt_path"),
            "generated_prompt": state.get("code_generation_prompt"),
        },
        "self_learning": {
            "matched_rule": state.get("self_learning_matched_rule"),
            "has_ambiguities": state.get("self_learning_has_ambiguities", False),
            "ambiguity_count": len(state.get("self_learning_ambiguities", []) or []),
            "deterministic_ambiguity_count": state.get("self_learning_deterministic_ambiguity_count", 0),
            "resolution_applied": state.get("self_learning_resolution_applied", False),
            "resolved_template_path": state.get("self_learning_resolved_template_path", ""),
            "validation_policy": state.get("self_learning_validation_policy", {}),
            "llm_ambiguity_review": state.get("self_learning_llm_ambiguity_review", {}),
            "human_resolution_required": state.get("self_learning_human_resolution_required", False),
            "next_step_if_ambiguous": state.get("self_learning_next_step_if_ambiguous", ""),
        },
    }
    manifest_path = (
        _code_gen_root()
        / "outputs"
        / "code_gen_runs"
        / f"code_gen_run_{_now_ist().strftime('%Y%m%d_%H%M%S')}.json"
    )
    _save_json(manifest_path, manifest)
    state["run_manifest_path"] = str(manifest_path)
    manifest["run_manifest_path"] = str(manifest_path)
    manifest["session_id"] = session_id
    store.upsert_stage_run(
        session_id=session_id,
        stage_name="manifest_saved",
        stage_status="completed",
        output={"run_manifest_path": str(manifest_path)},
    )
    return manifest


def run_end_to_end_from_intent(
    user_intent: str,
    use_llm_review: Any = None,
) -> Dict[str, Any]:
    """
    End-to-end pipeline:
      Feature_Validation output (dict) -> Knowledge Retrieval -> Template Orchestrator.

    This version does NOT require manual JSON path input for Feature Validation.

    Each call allocates a new session_id. Applying ambiguity resolutions is done only via
    run_resolve_self_learning_session (MCP: resolve_self_learning_ambiguities), not by re-calling
    this function.
    """
    repo_root = _repo_root()
    store = _get_state_store(repo_root)
    session_id = str(uuid.uuid4())
    if not (user_intent or "").strip():
        raise ValueError("user_intent must be a non-empty string")

    # ------------------------------------------------------------
    # STEP 1: Feature Validation (produces feature_payload dict)
    # ------------------------------------------------------------
    # Lazy/dynamic import to avoid package/import issues when
    # `Code_Gen/Feature_Validation` is not a traditional Python package.
    import importlib.util

    fv_file = _code_gen_root() / "Feature_Validation" / "two_stage_spec_agents.py"
    if not fv_file.exists():
        raise FileNotFoundError(f"Feature Validation file not found: {fv_file}")

    spec = importlib.util.spec_from_file_location("two_stage_spec_agents", fv_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Feature Validation module from: {fv_file}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    run_feature_validation_with_intent = getattr(mod, "run_with_intent", None)
    if not callable(run_feature_validation_with_intent):
        raise AttributeError(f"'run_with_intent' not found/callable in: {fv_file}")

    feature_payload: Dict[str, Any] = run_feature_validation_with_intent(user_intent)
    intent = str(feature_payload.get("intent", "")).strip() or user_intent.strip()

    template_path = _resolve_template_path(feature_payload, repo_root)
    store.ensure_session(
        session_id=session_id,
        intent=intent,
        status="started",
        template_path=str(template_path),
    )
    state = _build_initial_state(
        intent=intent,
        template_path=template_path,
        feature_payload=feature_payload,
    )
    state["session_id"] = session_id

    # Persist Feature Validation output for reproducibility.
    run_ts = _now_ist().strftime("%Y%m%d_%H%M%S")
    feature_validation_dir = _code_gen_root() / "outputs" / "feature_validation"
    feature_validation_dir.mkdir(parents=True, exist_ok=True)
    feature_validation_path = feature_validation_dir / f"feature_validation_output_{run_ts}.json"
    _save_json(feature_validation_path, feature_payload)
    state["input_feature_json"] = str(feature_validation_path)
    store.upsert_stage_run(
        session_id=session_id,
        stage_name="feature_validation_done",
        stage_status="completed",
        output={"feature_validation_output_path": str(feature_validation_path)},
    )

    # ------------------------------------------------------------
    # STEP 2: Ensure Knowledge sources are present
    # ------------------------------------------------------------
    kg_stage_run_id = store.upsert_stage_run(
        session_id=session_id,
        stage_name="knowledge_creator_done",
        stage_status="started",
        input_summary={"specs_count": len(feature_payload.get("specs", []) or [])},
    )
    with store.agent_run(
        session_id=session_id,
        stage_run_id=kg_stage_run_id,
        agent_name="specKnowledgeCreatorForEachSpec",
        input_payload={"specs": feature_payload.get("specs", []) or []},
        output_capture_keys=["created", "missing_doc_ids", "status"],
    ):
        spec_kg_status = _ensure_spec_knowledge_sources(
            feature_payload=feature_payload,
            feature_json_path=feature_validation_path,
            repo_root=repo_root,
        )
    with store.agent_run(
        session_id=session_id,
        stage_run_id=kg_stage_run_id,
        agent_name="codeKnowledgeCreator",
        input_payload={},
        output_capture_keys=["created", "status", "paths"],
    ):
        code_kg_status = _ensure_code_knowledge_source(repo_root)
    store.upsert_stage_run(
        session_id=session_id,
        stage_name="knowledge_creator_done",
        stage_status="completed",
        output={"spec_kg": spec_kg_status, "code_kg": code_kg_status},
    )
    state["spec_kg_status"] = spec_kg_status
    state["code_retrieval_sources"] = code_kg_status["paths"]

    # ------------------------------------------------------------
    # STEP 3: Spec / IE Knowledge Retrieval (in-memory feature payload)
    # ------------------------------------------------------------
    retrieval_stage_run_id = store.upsert_stage_run(
        session_id=session_id,
        stage_name="retrieval_done",
        stage_status="started",
        input_summary={"template_path": str(template_path)},
    )
    with store.agent_run(
        session_id=session_id,
        stage_run_id=retrieval_stage_run_id,
        agent_name="run_agentic_ie_retrieval_phase",
        input_payload={"feature_json": True, "template_path": str(template_path)},
    ):
        spec_retrieval_payload, spec_retrieval_path = _run_spec_retrieval(
            feature_json_input=feature_payload,
            template_path=template_path,
            repo_root=repo_root,
        )
    state["spec_retrieval_context"] = spec_retrieval_payload
    state["spec_retrieval_context_path"] = str(spec_retrieval_path)

    # ------------------------------------------------------------
    # STEP 4: Code Chunk Retrieval
    # ------------------------------------------------------------
    with store.agent_run(
        session_id=session_id,
        stage_run_id=retrieval_stage_run_id,
        agent_name="codeChunkRetrieverAgent",
        input_payload={"intent": intent, "message_names": _extract_message_names(feature_payload)},
        output_capture_keys=["code_artifacts_chunks_path"],
    ):
        code_retrieval_state = _run_code_retrieval(
            intent=intent,
            feature_payload=feature_payload,
            code_paths=code_kg_status["paths"],
        )
    state["code_artifacts_context"] = code_retrieval_state.get("code_artifacts_context", {})
    state["code_artifacts_chunks_path"] = code_retrieval_state.get("code_artifacts_chunks_path", "")
    store.upsert_stage_run(
        session_id=session_id,
        stage_name="retrieval_done",
        stage_status="completed",
        output={
            "spec_retrieval_context_path": str(spec_retrieval_path),
            "code_chunks_path": state.get("code_artifacts_chunks_path", ""),
        },
    )

    # ------------------------------------------------------------
    # STEP 5: Template Orchestration
    # ------------------------------------------------------------
    template_stage_run_id = store.upsert_stage_run(
        session_id=session_id,
        stage_name="template_filled",
        stage_status="started",
        input_summary={"template_path": str(template_path)},
    )
    with store.agent_run(
        session_id=session_id,
        stage_run_id=template_stage_run_id,
        agent_name="SpecTemplateFiller+CodeTemplateFiller",
        input_payload={"intent": intent},
        output_capture_keys=["spec_filled_template_path", "final_filled_template_path"],
    ):
        template_out = _run_template_orchestrator_two_pass(
            intent=intent,
            template_path=template_path,
            spec_retrieval_payload=spec_retrieval_payload,
            code_retrieval_state=code_retrieval_state,
        )
    state["spec_filled_template_path"] = template_out.get("spec_filled_template_path", "")
    state["final_filled_template_path"] = template_out.get("final_filled_template_path", "")
    store.upsert_stage_run(
        session_id=session_id,
        stage_name="template_filled",
        stage_status="completed",
        output={
            "spec_filled_template_path": state.get("spec_filled_template_path"),
            "final_filled_template_path": state.get("final_filled_template_path"),
        },
    )

    # ------------------------------------------------------------
    # STEP 6: Self Learning Validation (mapping-rules check)
    # ------------------------------------------------------------
    sl_stage_run_id = store.upsert_stage_run(
        session_id=session_id,
        stage_name="self_learning_done",
        stage_status="started",
        input_summary={"final_filled_template_path": state.get("final_filled_template_path", "")},
    )
    with store.agent_run(
        session_id=session_id,
        stage_run_id=sl_stage_run_id,
        agent_name="validate_template_with_mapping_rules",
        input_payload={"intent": intent},
        output_capture_keys=[
            "matched_rule",
            "has_ambiguities",
            "resolution_applied",
            "resolved_template_path",
            "validation_policy",
            "llm_ambiguity_review",
            "deterministic_ambiguity_count",
            "human_resolution_required",
            "next_step_if_ambiguous",
        ],
    ):
        sl_out = _run_self_learning_validation(
            intent=intent,
            final_filled_template_path=state["final_filled_template_path"],
            user_resolutions=None,
            use_llm_review=use_llm_review,
        )
    state["self_learning_matched_rule"] = sl_out.get("matched_rule")
    state["self_learning_has_ambiguities"] = sl_out.get("has_ambiguities", False)
    state["self_learning_ambiguities"] = sl_out.get("ambiguities", [])
    state["self_learning_resolution_applied"] = sl_out.get("resolution_applied", False)
    state["self_learning_validation_policy"] = sl_out.get("validation_policy") or {}
    state["self_learning_llm_ambiguity_review"] = sl_out.get("llm_ambiguity_review") or {}
    state["self_learning_deterministic_ambiguity_count"] = int(sl_out.get("deterministic_ambiguity_count", 0))
    state["self_learning_human_resolution_required"] = bool(sl_out.get("human_resolution_required", False))
    state["self_learning_next_step_if_ambiguous"] = str(sl_out.get("next_step_if_ambiguous") or "")
    resolved_template_path = sl_out.get("resolved_template_path") or state["final_filled_template_path"]
    state["self_learning_resolved_template_path"] = resolved_template_path
    store.upsert_stage_run(
        session_id=session_id,
        stage_name="self_learning_done",
        stage_status="completed",
        output={
            "has_ambiguities": state.get("self_learning_has_ambiguities", False),
            "ambiguity_count": len(state.get("self_learning_ambiguities", []) or []),
            "resolved_template_path": resolved_template_path,
            "human_resolution_required": state.get("self_learning_human_resolution_required", False),
            "next_step_if_ambiguous": state.get("self_learning_next_step_if_ambiguous", ""),
        },
    )

    # If ambiguities remain unresolved, return state for user input.
    if state["self_learning_has_ambiguities"]:
        state["code_generation_prompt_path"] = ""
        state["code_generation_prompt"] = ""
        store.ensure_session(
            session_id=session_id,
            intent=intent,
            status="awaiting_user",
            template_path=str(resolved_template_path),
            ambiguity_questions=state.get("self_learning_ambiguities", []),
        )
    else:
        prompt_stage_run_id = store.upsert_stage_run(
            session_id=session_id,
            stage_name="prompt_generated",
            stage_status="started",
            input_summary={"resolved_template_path": resolved_template_path},
        )
        with store.agent_run(
            session_id=session_id,
            stage_run_id=prompt_stage_run_id,
            agent_name="promptGenerationAgent",
            input_payload={"intent": intent, "final_filled_template_path": resolved_template_path},
            output_capture_keys=["generated_prompt_path"],
        ):
            prompt_out = _run_prompt_generation(intent=intent, final_filled_template_path=resolved_template_path)
        state["code_generation_prompt_path"] = prompt_out.get("generated_prompt_path", "")
        state["code_generation_prompt"] = prompt_out.get("generated_prompt", "")
        store.upsert_stage_run(
            session_id=session_id,
            stage_name="prompt_generated",
            stage_status="completed",
            output={"code_generation_prompt_path": state.get("code_generation_prompt_path", "")},
        )
        store.ensure_session(
            session_id=session_id,
            intent=intent,
            status="final_prompt_ready",
            template_path=str(resolved_template_path),
            code_generation_prompt=state.get("code_generation_prompt", ""),
            code_generation_prompt_path=state.get("code_generation_prompt_path", ""),
        )

    # ------------------------------------------------------------
    # Manifest for the full run
    # ------------------------------------------------------------
    manifest = {
        "input_feature_json": state.get("input_feature_json", str(feature_validation_path)),
        "input_intent": state.get("input_intent", intent),
        "template_path_used": state.get("template_path_used", str(template_path)),
        "knowledge_creation": {
            "spec_kg": state.get("spec_kg_status", spec_kg_status),
            "code_kg": code_kg_status,
        },
        "retrieval_outputs": {
            "spec_retrieval_context_path": state.get("spec_retrieval_context_path", str(spec_retrieval_path)),
            "code_chunks_path": state.get("code_artifacts_chunks_path", code_retrieval_state.get("code_artifacts_chunks_path")),
        },
        "template_outputs": {
            "spec_filled_template_path": state.get("spec_filled_template_path", template_out.get("spec_filled_template_path")),
            "final_filled_template_path": state.get("final_filled_template_path", template_out.get("final_filled_template_path")),
            "generated_prompt_path": state.get("code_generation_prompt_path"),
            "generated_prompt": state.get("code_generation_prompt"),
        },
        "self_learning": {
            "matched_rule": state.get("self_learning_matched_rule"),
            "has_ambiguities": state.get("self_learning_has_ambiguities", False),
            "ambiguity_count": len(state.get("self_learning_ambiguities", []) or []),
            "deterministic_ambiguity_count": state.get("self_learning_deterministic_ambiguity_count", 0),
            "resolution_applied": state.get("self_learning_resolution_applied", False),
            "resolved_template_path": state.get("self_learning_resolved_template_path", ""),
            "validation_policy": state.get("self_learning_validation_policy", {}),
            "llm_ambiguity_review": state.get("self_learning_llm_ambiguity_review", {}),
            "human_resolution_required": state.get("self_learning_human_resolution_required", False),
            "next_step_if_ambiguous": state.get("self_learning_next_step_if_ambiguous", ""),
        },
    }

    manifest_path = (
        _code_gen_root()
        / "outputs"
        / "code_gen_runs"
        / f"code_gen_run_{_now_ist().strftime('%Y%m%d_%H%M%S')}.json"
    )
    _save_json(manifest_path, manifest)
    state["run_manifest_path"] = str(manifest_path)
    state["session_id"] = session_id
    manifest["session_id"] = session_id
    store.upsert_stage_run(
        session_id=session_id,
        stage_name="manifest_saved",
        stage_status="completed",
        output={"run_manifest_path": str(manifest_path)},
    )
    return state
