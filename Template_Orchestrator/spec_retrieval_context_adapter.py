"""
Adapt KG-only agentic IE retrieval JSON (e.g. agentic_ie_retrieval_context_*.json)
into the chunk list shape expected by SpecTemplateFiller.

Agentic chunks store doc identity under metadata; the filler reads top-level
knowledge_source / source_id and uses semantic_score (etc.) for ordering before
the top-70%% trim in extract_information.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PayloadInput = Union[str, Path, Dict[str, Any]]


def _load_payload(payload: PayloadInput) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    path = Path(payload)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def normalize_chunks_for_spec_template_filler(
    chunks: List[Dict[str, Any]],
    *,
    assign_order_scores: bool = True,
) -> List[Dict[str, Any]]:
    """
    Ensure each chunk has top-level fields SpecTemplateFiller relies on:
    knowledge_source, source_id (for grouping, dedup, Knowledge_Hints), and
    optional semantic_score so extract_information's sort + top-70%% step
    follows the retrieval merge order when no retrieval scores exist.

    Idempotent: leaves existing top-level knowledge_source / source_id / scores
    if already set.
    """
    n = len(chunks)
    out: List[Dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        c = copy.deepcopy(chunk)
        meta = c.get("metadata") if isinstance(c.get("metadata"), dict) else {}
        doc_id = meta.get("doc_id")
        ks = meta.get("knowledge_source") or doc_id

        if not c.get("knowledge_source") and ks:
            c["knowledge_source"] = str(ks)
        if not c.get("source_id") and doc_id:
            c["source_id"] = str(doc_id)
        elif not c.get("source_id") and c.get("knowledge_source"):
            c["source_id"] = str(c["knowledge_source"])

        if assign_order_scores and n > 0:
            has_score = any(
                k in c and c.get(k) not in (None, 0, 0.0)
                for k in ("semantic_score", "llm_rerank_score", "rank")
            )
            if not has_score:
                c["semantic_score"] = float(n - i) / float(n)

        out.append(c)
    return out


def agentic_ie_retrieval_to_template_filler_inputs(
    payload: PayloadInput,
    *,
    assign_order_scores: bool = True,
) -> Dict[str, Any]:
    """
    Load an agentic IE retrieval output JSON (or accept an already-loaded dict)
    and return values ready for SpecTemplateFiller.

    Returns:
        {
            "chunks": List[dict],       # pass to extract_information / fill_template
            "query": str,               # payload["intent"]
            "template_path": str | None,
            "feature_json_path": str | None,
            "raw_final_context_count": int,
        }

    Uses ``final_context`` from the payload (merged IE + KG-expanded chunks).
    """
    data = _load_payload(payload)
    raw = data.get("final_context")
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ValueError("Payload must contain a list 'final_context'")

    chunks = normalize_chunks_for_spec_template_filler(
        raw, assign_order_scores=assign_order_scores
    )

    intent = data.get("intent") or ""
    template_path = data.get("template_path")
    if template_path is not None:
        template_path = str(template_path)
    feature_path = data.get("feature_json_path")
    if feature_path is not None:
        feature_path = str(feature_path)

    return {
        "chunks": chunks,
        "query": intent if isinstance(intent, str) else str(intent),
        "template_path": template_path,
        "feature_json_path": feature_path,
        "raw_final_context_count": len(raw),
    }
