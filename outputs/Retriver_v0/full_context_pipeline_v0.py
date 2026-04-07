from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _load_module_from_path(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    # Important for decorators like @dataclass: it expects the module to be present
    # in sys.modules during class processing.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_section_id(section_id: Any) -> str:
    return str(section_id or "").strip()


def _safe_upper(s: Any) -> str:
    return str(s or "").upper().strip()


def _merge_final_context(
    *,
    base_chunks: List[Dict[str, Any]],
    extra_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Dedup by (knowledge_source/doc_id, section_id). Prefer `extra_chunks`
    chunk metadata/role when duplicates exist.
    """
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _key(c: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        doc = str(c.get("knowledge_source") or c.get("source_id") or "").strip()
        sid = _normalize_section_id(c.get("section_id"))
        if not doc or not sid:
            return None
        return (doc, sid)

    for c in base_chunks:
        k = _key(c)
        if k is None:
            continue
        merged[k] = c

    for c in extra_chunks:
        k = _key(c)
        if k is None:
            continue
        merged[k] = c  # override with extra chunk role if present

    out = list(merged.values())
    out.sort(key=lambda x: (_safe_upper(x.get("knowledge_source")), _normalize_section_id(x.get("section_id"))))
    return out


def run_full_pipeline_and_write_final_context() -> Dict[str, Any]:
    """
    Runs:
      - Step-1..3 in-memory (seed-first KG neighborhood expansion; attach full section text)
      - Step-4 in-memory (agentic recursive ASN.1 IE expansion from message_format subtree; LLM-first, deterministic fallback)
    Produces ONE final JSON file containing `final_context` (for template filler).

    IMPORTANT: This function DOES write output to disk only when called.
    """
    repo_root = Path(__file__).resolve().parent.parent
    feature_json_path = repo_root / "Inter-gNB-DU_LTM_handover_procedure_20260323_093447.json"
    if not feature_json_path.exists():
        raise FileNotFoundError(f"Feature JSON not found: {feature_json_path}")

    feature_payload = json.loads(feature_json_path.read_text(encoding="utf-8"))
    intent = str(feature_payload.get("intent", "") or feature_payload.get("query", "") or "").strip()

    template_path = None
    if isinstance(feature_payload.get("template"), dict):
        template_path = feature_payload["template"].get("template_path")

    # Load step modules by file path (Retriver_v0 isn't a Python package).
    section_retriever_path = repo_root / "Retriver_v0" / "section_retriever.py"
    asn_ie_retriever_path = repo_root / "Retriver_v0" / "asn_ie_agentic_retriever.py"
    if not section_retriever_path.exists():
        raise FileNotFoundError(f"Missing: {section_retriever_path}")
    if not asn_ie_retriever_path.exists():
        raise FileNotFoundError(f"Missing: {asn_ie_retriever_path}")

    section_mod = _load_module_from_path("section_retriever_v0", section_retriever_path)
    asn_mod = _load_module_from_path("asn_ie_agentic_retriever_v0", asn_ie_retriever_path)

    # Step-1..3 (in memory).
    step3_output = section_mod.run_step1_to_step3_section_retrieval()
    base_final_context = step3_output.get("final_context") or step3_output.get("specs_context") or []
    if not isinstance(base_final_context, list):
        base_final_context = []

    # Step-4 (in memory).
    step4_output = asn_mod.run_step4_agentic_ie_retrieval()
    asn_final_context = step4_output.get("final_context") or step4_output.get("specs_context") or []
    if not isinstance(asn_final_context, list):
        asn_final_context = []

    # Merge.
    merged_final_context = _merge_final_context(
        base_chunks=base_final_context,
        extra_chunks=asn_final_context,
    )

    final_package = {
        "feature_json_path": str(feature_json_path),
        "template_path": template_path,
        "intent": intent,
        "timestamp": datetime.utcnow().isoformat(),
        "retrieval_config": {
            "phase": "full_pipeline_step1_to_step4",
            "step3_unique_chunks": len(base_final_context),
            "step4_unique_chunks": len(asn_final_context),
            "merged_unique_chunks": len(merged_final_context),
        },
        "asn_trees_by_message_name": step4_output.get("asn_trees_by_message_name", {}),
        "specs_context_count": len(merged_final_context),
        "specs_context": merged_final_context,
        # Adapter compatibility.
        "final_context": merged_final_context,
    }

    # Write final output JSON (single file).
    out_dir = repo_root / "KG_Retriver" / "spec_chunks" / "retrieval_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"final_context_package_v0_{ts}.json"
    out_path.write_text(json.dumps(final_package, indent=2, ensure_ascii=False), encoding="utf-8")
    final_package["final_output_path"] = str(out_path)
    print(f"[Retriver_v0] Wrote final context package: {out_path}")
    return final_package


def main() -> None:
    run_full_pipeline_and_write_final_context()


if __name__ == "__main__":
    main()

