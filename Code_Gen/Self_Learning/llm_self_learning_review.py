from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ModuleNotFoundError:
        pass


def build_self_learning_llm() -> Any | None:
    """Azure Chat OpenAI; None if deps or credentials missing."""
    try:
        from langchain_openai import AzureChatOpenAI
    except ModuleNotFoundError:
        return None

    _load_dotenv_if_available()
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not api_key or not endpoint:
        return None
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    deployment = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o-mini")
    return AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_deployment=deployment,
        temperature=0.0,
        timeout=120,
        max_retries=2,
    )


def default_use_llm_review_flag() -> bool:
    """
    SELF_LEARNING_LLM_REVIEW=1|true forces on; =0|false forces off.
    If unset: on when Azure env vars are present (opt-in by configuration).
    """
    _load_dotenv_if_available()
    v = os.getenv("SELF_LEARNING_LLM_REVIEW", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return bool(os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"))


def _truncate(s: str, max_chars: int) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 80] + "\n...[truncated]...\n" + s[-60:]


def _norm_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _collect_ie_snippets(template: Dict[str, Any], ie_names: List[str], per_ie: int = 4500) -> str:
    wanted = {_norm_token(n) for n in ie_names if n}
    parts: List[str] = []
    seen: set = set()
    for ie in template.get("Information_Elements") or []:
        if not isinstance(ie, dict):
            continue
        name = str(ie.get("IE_Name") or "").strip()
        if _norm_token(name) not in wanted or _norm_token(name) in seen:
            continue
        seen.add(_norm_token(name))
        defin = str(ie.get("IE_Definition") or "").strip()
        parts.append(f"### IE_Name: {name}\n{_truncate(defin, per_ie)}\n")
    return "\n".join(parts) if parts else "(no matching IE_Definition entries found)"


def _rule_summary_for_llm(rule_key: str, rule: Dict[str, Any]) -> Dict[str, Any]:
    flow = rule.get("message_flow") or rule.get("messages") or []
    slim: List[Dict[str, Any]] = []
    for mf in flow:
        if not isinstance(mf, dict):
            continue
        ies = mf.get("ies") or {}
        ie_keys = list(ies.keys()) if isinstance(ies, dict) else []
        slim.append(
            {
                "msg_name": mf.get("msg_name"),
                "sender": mf.get("sender"),
                "receiver": mf.get("receiver"),
                "ie_names_expected": ie_keys[:80],
            }
        )
    return {
        "rule_key": rule_key,
        "description": rule.get("intent_description") or rule.get("description") or "",
        "feature": rule.get("feature"),
        "messages": slim,
    }


def _parse_llm_json_array(content: str) -> List[Dict[str, Any]]:
    text = str(content or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    if fence:
        text = fence.group(1).strip()
    data = json.loads(text)
    if isinstance(data, dict) and "reviews" in data:
        inner = data["reviews"]
        return inner if isinstance(inner, list) else []
    if isinstance(data, list):
        return data
    return []


def llm_review_self_learning_ambiguities(
    *,
    llm: Any,
    intent: str,
    rule_key: str,
    rule: Dict[str, Any],
    template: Dict[str, Any],
    ambiguities: List[Dict[str, Any]],
    validation_policy: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Enrich ambiguities with optional LLM advisory verdicts.

    Human-in-the-loop: **no ambiguity is removed** here. The user must still respond via
    resolutions; the template is only updated from explicit user input in _apply_resolutions.
    """
    meta: Dict[str, Any] = {
        "ran": True,
        "model": "azure_chat_openai",
        "advisory_only": True,
        "suppressed_count": 0,
        "suppressed": [],
    }

    ie_names: List[str] = []
    for a in ambiguities:
        n = str(a.get("ie_name") or "").strip()
        if n:
            ie_names.append(n)
    ie_context = _collect_ie_snippets(template, list(dict.fromkeys(ie_names)))

    amb_slim = []
    for a in ambiguities:
        amb_slim.append(
            {
                "id": a.get("id"),
                "type": a.get("type"),
                "message_name": a.get("message_name"),
                "ie_name": a.get("ie_name"),
                "field": a.get("field"),
                "question": _truncate(str(a.get("question") or ""), 1200),
                "expected_constraint": a.get("expected_constraint"),
                "detected_ie_definition_preview": _truncate(
                    str(a.get("detected_ie_definition_preview") or a.get("default_value") or ""), 2500
                ),
            }
        )

    prompt = f"""You review 3GPP ASN.1 self-learning ambiguities for a code-generation template.

IMPORTANT
- Your output is **advisory only**. Every ambiguity remains assigned to a **human**; you must NOT assume
  any item is closed or auto-fixed.
- A human will provide resolutions; the system updates the template **only** from that user input.

CONTEXT
- User intent: {intent}
- Validation policy:
{json.dumps(validation_policy, indent=2)}
- Matched mapping rule summary:
{json.dumps(_rule_summary_for_llm(rule_key, rule), indent=2)}

RELEVANT template IE_Definition excerpts (Information_Elements):
{ie_context}

DETERMINISTIC AMBIGUITIES (each may be a real issue or a false positive from shallow heuristics):
{json.dumps(amb_slim, indent=2, ensure_ascii=False)}

TASK
For EACH ambiguity id, suggest:
- "confirm" = likely a real issue; human should fix or answer.
- "false_positive" = likely OK structurally (e.g. typedef elsewhere, SEQUENCE OF item); human may still confirm.
- "needs_human" = unclear without more spec context.

RULES
- Prefer "false_positive" only when ASN.1 structure is plausibly satisfied by referenced types.
- Do not invent missing IEs; if structure is missing, use "confirm".
- Output ONLY valid JSON (no markdown) with this exact shape:
{{"reviews":[{{"id":"<same as input>","verdict":"confirm|false_positive|needs_human","reason":"<one short sentence>"}}]}}
- Include every ambiguity id exactly once.
"""

    try:
        from langchain_core.messages import HumanMessage

        resp = llm.invoke([HumanMessage(content=prompt)])
        raw = getattr(resp, "content", None) or str(resp)
        meta["raw_response_preview"] = _truncate(raw, 4000)
        reviews = _parse_llm_json_array(raw)
    except Exception as e:
        meta["error"] = str(e)
        return [dict(a) for a in ambiguities], meta

    by_id: Dict[str, Dict[str, Any]] = {}
    for r in reviews:
        if not isinstance(r, dict):
            continue
        rid = str(r.get("id") or "").strip()
        if not rid:
            continue
        by_id[rid] = r

    if not by_id:
        meta["parse_warning"] = "no reviews parsed; ambiguities unchanged except copy"
        return [dict(a) for a in ambiguities], meta

    meta["parsed_review_count"] = len(by_id)
    out: List[Dict[str, Any]] = []

    for a in ambiguities:
        enriched = dict(a)
        enriched.setdefault("requires_human_resolution", True)
        aid = str(a.get("id") or "")
        rev = by_id.get(aid)
        if rev:
            verdict = str(rev.get("verdict") or "").strip().lower() or "needs_human"
            reason = str(rev.get("reason") or "").strip()
            enriched["llm_review"] = {
                "verdict": verdict,
                "reason": reason,
                "advisory": True,
                "note": "Human must still submit a resolution for this id when required by the pipeline.",
            }
        out.append(enriched)

    return out, meta
