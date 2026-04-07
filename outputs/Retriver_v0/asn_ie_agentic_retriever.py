from __future__ import annotations

import json
import os
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _normalize_section_id(section_id: Any) -> str:
    return str(section_id or "").strip()


def _safe_upper(s: Any) -> str:
    return str(s or "").upper().strip()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_first_json_object(raw: str) -> Dict[str, Any]:
    """
    Best-effort JSON object extraction from an LLM response.
    """
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*", "", s).strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return {}
    try:
        obj2 = json.loads(m.group(0))
        return obj2 if isinstance(obj2, dict) else {}
    except Exception:
        return {}


def _truncate_text(text: str, max_chars: int) -> str:
    t = text or ""
    if len(t) <= max_chars:
        return t
    head = max(0, max_chars - 500)
    tail = 250
    return t[:head] + "\n\n[TRUNCATED]\n...\n" + t[-tail:]


def _build_llm_azure() -> Any | None:
    """
    LLM-first (Azure OpenAI). Returns None if deps/keys are missing.
    """
    try:
        from dotenv import load_dotenv
        from langchain_openai import AzureChatOpenAI
    except ModuleNotFoundError:
        return None

    load_dotenv()
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not api_key or not endpoint:
        return None

    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    deployment = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o-mini")
    deployment = str(deployment).strip()
    if not deployment:
        deployment = "gpt-4o-mini"

    try:
        return AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            azure_deployment=deployment,
            temperature=0.1,
            timeout=120,
            max_retries=2,
        )
    except Exception:
        return None


def _extract_message_names_from_intent(intent: str) -> List[str]:
    """
    Extract quoted message names from intent:
      ".... 'UE CONTEXT SETUP REQUEST' .... 'UE CONTEXT SETUP RESPONSE' ..."
    """
    if not intent:
        return []
    names = re.findall(r"'([^']+)'", intent)
    cleaned = []
    seen: Set[str] = set()
    for n in names:
        nn = str(n or "").strip()
        if not nn:
            continue
        key = nn.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(nn)
    return cleaned


def _resolve_doc_id_from_spec_number(payload: Dict[str, Any], spec_number: str) -> Optional[str]:
    spec_number_norm = _safe_upper(spec_number)
    if not spec_number_norm:
        return None
    for spec in payload.get("protocol_specs", []) or []:
        sn = _safe_upper(spec.get("spec_number", ""))
        doc_id = str(spec.get("doc_id", "")).strip()
        if sn and doc_id and sn == spec_number_norm:
            return doc_id
    # Some payloads might store mapping under top-level "specs" instead; keep a best-effort.
    for spec in payload.get("specs", []) or []:
        sn = _safe_upper(spec.get("spec_number", ""))
        doc_id = str(spec.get("doc_id", "")).strip()
        if sn and doc_id and sn == spec_number_norm:
            return doc_id
    return None


def _collect_message_format_subtrees(payload: Dict[str, Any], intent: str) -> List[Dict[str, str]]:
    """
    Find message_format sections that correspond to message names mentioned in intent.

    Returns list of:
      {doc_id, message_name, message_format_section_id}
    """
    intent_message_names = set(n.lower() for n in _extract_message_names_from_intent(intent))

    # If intent doesn't have quoted names, we will include all message_format entries.
    include_all = len(intent_message_names) == 0

    out: List[Dict[str, str]] = []
    # Note: multiple messages can share the same message_format_section_id (e.g., both
    # UE CONTEXT SETUP REQUEST/RESPONSE might reference the same abstract syntax section).
    # We must keep per-message_name trees/chunk roles.
    seen: Set[Tuple[str, str, str]] = set()

    for block in payload.get("protocol_message_sections", []) or []:
        protocol = _safe_upper(block.get("protocol", ""))
        spec_number = str(block.get("spec_number", "")).strip()
        doc_id = _resolve_doc_id_from_spec_number(payload, spec_number) if spec_number else ""
        if not doc_id:
            # If spec_number missing, block parsing can't map doc_id; skip.
            continue

        for msg in block.get("messages", []) or []:
            message_name = str(msg.get("message_name", "")).strip()
            if not message_name:
                continue
            if (not include_all) and (message_name.lower() not in intent_message_names):
                continue

            for sec in msg.get("sections", []) or []:
                role = str(sec.get("role", "")).strip()
                if role != "message_format":
                    continue
                section_id = _normalize_section_id(sec.get("section_id"))
                if not section_id:
                    continue

                key = (doc_id, message_name, section_id)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "doc_id": doc_id,
                        "message_name": message_name,
                        "message_format_section_id": section_id,
                        # Keep protocol for debugging.
                        "protocol": protocol,
                    }
                )

    # Deterministic order.
    out.sort(key=lambda d: (d.get("doc_id", ""), d.get("message_format_section_id", ""), d.get("message_name", "")))
    return out


def _build_asn_tree_node(
    *,
    section_id: str,
    section_title: str,
    asn_definition_full: str,
    children: List[Dict[str, Any]],
    asn_definition_max_chars: int = 20000,
) -> Dict[str, Any]:
    return {
        "section_id": section_id,
        "section_title": section_title,
        # Keep the tree payload compact; use full content inside specs_context chunks.
        "asn_definition": _truncate_text(asn_definition_full, asn_definition_max_chars),
        "children": children,
    }


def _derive_ie_definition_name_from_message(message_name: str) -> str:
    """
    Example:
      "UE CONTEXT SETUP REQUEST" -> "UEContextSetupRequestIEs"
    """
    words = re.findall(r"[A-Za-z0-9]+", message_name or "")
    parts: List[str] = []
    for w in words:
        if not w:
            continue
        # Preserve acronyms like UE, RRC, CSI as-is (important for ASN.1 matching).
        if str(w).isupper() and len(str(w)) > 1:
            parts.append(str(w))
        else:
            w = str(w)
            parts.append(w[0].upper() + w[1:].lower())
    camel = "".join(parts)
    if not camel:
        return ""
    if not camel.endswith("IEs"):
        return camel + "IEs"
    return camel


def _is_probably_asn1_definition_text(text: str) -> bool:
    t = (text or "").upper()
    return "::=" in t or "PROTOCOL-IES" in t


def _compile_ie_regex_patterns(identifier: str) -> List[re.Pattern[str]]:
    name = (identifier or "").strip()
    if not name:
        return []
    esc = re.escape(name)
    flags = re.IGNORECASE | re.DOTALL
    return [
        re.compile(rf"{esc}\s+.*?PROTOCOL-IES\s*::=", flags),
        re.compile(rf"{esc}\s*::=", flags),
    ]


def _score_candidate_definition_content(content: str, ie_token: str) -> int:
    """
    Deterministic heuristic score for best-match selection.
    """
    if not ie_token:
        return 0
    c = content or ""
    token_esc = re.escape(ie_token.strip())
    flags = re.IGNORECASE | re.DOTALL

    score = 0
    if re.search(token_esc, c, flags=re.IGNORECASE):
        score += 2
    if re.search(rf"{token_esc}\s*::=", c, flags=flags):
        score += 6
    if re.search(rf"{token_esc}\s+.*?PROTOCOL-IES\s*::=", c, flags=flags):
        score += 10
    if "::=" in c:
        score += 1
    if "PROTOCOL-IES" in c.upper():
        score += 1
    return score


def _find_child_type_candidates_regex(parent_asn_definition_text: str) -> List[str]:
    """
    Deterministic extraction of nested ASN.1 type identifiers.
    """
    text = parent_asn_definition_text or ""
    if not text.strip():
        return []

    # 1) From protocol IE entries: ... TYPE <TypeName>
    type_targets = set(re.findall(r"\bTYPE\s+([A-Za-z0-9][A-Za-z0-9-]*)\b", text, flags=re.IGNORECASE))

    # 2) From SEQUENCE/CHOICE bodies:
    # Heuristic: capture the second identifier after "<something> <TypeName>"
    pair_targets = set(
        re.findall(
            r"\b[A-Za-z0-9][A-Za-z0-9-]*\s+([A-Za-z][A-Za-z0-9-]*)\b(?=\s*(?:,|OPTIONAL|MANDATORY|}|$))",
            text,
            flags=re.IGNORECASE,
        )
    )

    candidates = type_targets.union(pair_targets)
    if not candidates:
        return []

    asn_keywords = {
        "SEQUENCE",
        "SET",
        "CHOICE",
        "OF",
        "OPTIONAL",
        "MANDATORY",
        "ENUMERATED",
        "INTEGER",
        "BOOLEAN",
        "OCTET",
        "STRING",
        "BIT",
        "TRUE",
        "FALSE",
        "SIZE",
        "PRESENT",
        "CRITICALITY",
        "REJECT",
        "IGNORE",
        "PROTOCOL-IES",
        "PROTOCOL",
        "ID",
        "TYPE",
    }

    cleaned: List[str] = []
    seen: Set[str] = set()
    for c in sorted(candidates):
        cc = str(c).strip()
        if not cc:
            continue
        if _safe_upper(cc) in asn_keywords:
            continue
        if cc in seen:
            continue
        seen.add(cc)
        cleaned.append(cc)
    return cleaned[:25]


def _extract_allowed_section_ids_for_message_format(
    *,
    sections_by_sid: Dict[str, Any],
    message_format_section_id: str,
) -> Set[str]:
    sid = _normalize_section_id(message_format_section_id)
    if not sid:
        return set()
    node = sections_by_sid.get(sid)
    if not isinstance(node, dict):
        return {sid}

    allowed: Set[str] = {sid}
    children_recursive = node.get("children_recursive", []) or []
    if isinstance(children_recursive, list):
        for c in children_recursive:
            csid = _normalize_section_id(c)
            if csid:
                allowed.add(csid)
    return allowed


def _load_allowed_chunks_by_sid(
    *,
    chunks_json_path: Path,
    allowed_section_ids: Set[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Returns map: section_id -> chunk dict (with at least content + section_title).
    """
    chunks = _load_json(chunks_json_path)
    if not isinstance(chunks, list):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for c in chunks:
        if not isinstance(c, dict):
            continue
        sid = _normalize_section_id(c.get("section_id"))
        if not sid:
            continue
        if sid not in allowed_section_ids:
            continue
        out[sid] = c
    return out


def _load_allowed_graph_nodes_by_sid(
    *,
    graph: Any,
    allowed_section_ids: Set[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Load node content/title directly from knowledge_graph.pkl.
    This is required for ASN.1 definition chunks because `chunks.json` may
    contain only TOC/headings for some ASN-related sections.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if graph is None or not hasattr(graph, "nodes"):
        return out

    for sid in allowed_section_ids:
        sid_norm = _normalize_section_id(sid)
        if not sid_norm:
            continue
        if sid_norm not in graph.nodes:
            continue
        node = graph.nodes[sid_norm]
        if not isinstance(node, dict):
            continue

        content = str(node.get("content", "") or "")
        section_title = str(node.get("section_title", "") or node.get("title", "") or "").strip()

        metadata: Dict[str, Any] = {}
        node_metadata = node.get("metadata")
        if isinstance(node_metadata, dict):
            metadata = dict(node_metadata)
        else:
            for k in ("level", "parent_section_id", "child_section_ids", "page_numbers"):
                if k in node:
                    metadata[k] = node.get(k)

        out[sid_norm] = {
            "section_id": sid_norm,
            "section_title": section_title,
            "content": content,
            "metadata": metadata,
        }

    return out


def _llm_choose_main_definition(
    *,
    llm: Any | None,
    query_text: str,
    message_name: str,
    message_format_anchor: str,
    candidate_definitions: List[Dict[str, Any]],
    max_keep: int = 3,
) -> List[str]:
    """
    LLM-first selection of best MAIN definition section ids.
    """
    if llm is None:
        # Deterministic: keep top candidates by token presence.
        if not candidate_definitions:
            return []
        # Assume derived token is present in candidates; score via content preview if present.
        scored = []
        for c in candidate_definitions:
            sid = str(c.get("section_id", "")).strip()
            preview = str(c.get("content_preview", "")).strip()
            derived = str(c.get("ie_token", "")).strip()
            score = 0
            if derived:
                score = _score_candidate_definition_content(preview, derived)
            scored.append((score, sid))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [sid for _, sid in scored[:max_keep] if sid]

    items = []
    for c in candidate_definitions:
        items.append(
            {
                "section_id": str(c.get("section_id", "")).strip(),
                "section_title": str(c.get("section_title", "")).strip(),
                "content_preview": _truncate_text(str(c.get("content_preview", "") or ""), 4000),
            }
        )

    prompt = f"""
You are selecting the MAIN/parent ASN.1 IE definition section(s) for a specific F1AP message.

INPUTS:
query_text (feature intent):
{_truncate_text(query_text, 4000)}

message_name:
{message_name}

message_format_anchor (the content of the message_format section):
{_truncate_text(message_format_anchor, 12000)}

CANDIDATE MAIN DEFINITIONS (JSON array):
{json.dumps(items, ensure_ascii=False)}

TASK:
Choose only the section_id(s) that correspond to the MAIN/parent IE definition of the given message.
Return only 1 to {max_keep} section_ids.

Rules:
- Do not summarize.
- Use the ASN.1 definition markers (e.g., '::=', 'PROTOCOL-IES') in the candidates.
- Output JSON only.
"""
    prompt = prompt.strip()
    try:
        resp = llm.invoke(prompt)
        raw = getattr(resp, "content", None) or str(resp)
        obj = _extract_first_json_object(raw)
        ids = obj.get("main_definition_section_ids", [])
        if not isinstance(ids, list):
            # tolerate older key
            ids = obj.get("relevant_section_ids", [])
        if not isinstance(ids, list):
            return []
        ids_clean = [str(x).strip() for x in ids if str(x).strip()]
        return ids_clean[:max_keep]
    except Exception:
        return []


def _llm_extract_child_type_candidates(
    *,
    llm: Any | None,
    query_text: str,
    section_text_anchor: str,
    parent_asn_definition_text: str,
) -> List[str]:
    """
    LLM-first extraction of nested child ASN type identifiers.
    """
    if llm is None:
        return _find_child_type_candidates_regex(parent_asn_definition_text)

    prompt = f"""
Extract nested ASN.1 type identifiers referenced inside the provided parent ASN.1 definition
so we can expand them recursively.

Rules:
- Return only identifiers that look like ASN.1 type names (not keywords like SEQUENCE, CHOICE, OPTIONAL).
- Exclude base types (e.g., INTEGER, BOOLEAN, OCTET STRING, BIT STRING).
- If none, return [].

query_text (intent):
{_truncate_text(query_text, 4000)}

section_text_anchor:
{_truncate_text(section_text_anchor, 8000)}

parent ASN.1 definition text:
{_truncate_text(parent_asn_definition_text, 12000)}

Return JSON ONLY:
{{
  "child_type_candidates": ["<TypeName>", ...]
}}
""".strip()
    try:
        resp = llm.invoke(prompt)
        raw = getattr(resp, "content", None) or str(resp)
        obj = _extract_first_json_object(raw)
        ids = obj.get("child_type_candidates", [])
        if not isinstance(ids, list):
            return []
        cleaned = []
        seen: Set[str] = set()
        for x in ids:
            s = str(x).strip()
            if not s:
                continue
            if s in seen:
                continue
            seen.add(s)
            cleaned.append(s)
        return cleaned[:25]
    except Exception:
        return _find_child_type_candidates_regex(parent_asn_definition_text)


def _llm_choose_best_child_definition_section(
    *,
    llm: Any | None,
    query_text: str,
    section_text_anchor: str,
    child_type_identifier: str,
    parent_asn_definition_text: str,
    candidate_definitions: List[Dict[str, Any]],
) -> Optional[str]:
    """
    LLM-first mapping from child type identifier -> best definition section_id.
    """
    if not candidate_definitions:
        return None
    if llm is None:
        # Deterministic: best token+ASN markers match.
        best_sid = None
        best_score = None
        for c in candidate_definitions:
            sid = str(c.get("section_id", "")).strip()
            content_preview = str(c.get("content_preview", "") or "")
            score = _score_candidate_definition_content(content_preview, child_type_identifier)
            if best_score is None or score > best_score:
                best_score = score
                best_sid = sid
        return best_sid

    items = []
    for c in candidate_definitions:
        items.append(
            {
                "section_id": str(c.get("section_id", "")).strip(),
                "section_title": str(c.get("section_title", "")).strip(),
                "content_preview": _truncate_text(str(c.get("content_preview", "") or ""), 4000),
            }
        )

    prompt = f"""
Select the best matching ASN.1 definition section_id for the given child ASN.1 type identifier.

query_text:
{_truncate_text(query_text, 3000)}

section_text_anchor:
{_truncate_text(section_text_anchor, 5000)}

child_type_identifier:
{child_type_identifier}

parent ASN.1 definition text (context):
{_truncate_text(parent_asn_definition_text, 6000)}

Candidate definition sections (JSON array):
{json.dumps(items, ensure_ascii=False)}

TASK:
Return JSON ONLY with:
{{"best_definition_section_id":"<section_id_or_empty_string>"}}
Choose the most correct section_id that defines the ASN.1 type for the child_type_identifier.
""".strip()

    try:
        resp = llm.invoke(prompt)
        raw = getattr(resp, "content", None) or str(resp)
        obj = _extract_first_json_object(raw)
        sid = str(obj.get("best_definition_section_id", "") or "").strip()
        return sid or None
    except Exception:
        return None


@dataclass
class AsnNodeResult:
    node: Dict[str, Any]
    discovered_section_ids: Set[str]


def _build_asn_tree_recursive(
    *,
    llm: Any | None,
    llm_call_budget: Dict[str, int],
    query_text: str,
    message_format_anchor_text: str,
    allowed_section_ids: Set[str],
    chunks_by_sid: Dict[str, Dict[str, Any]],
    parent_section_id: str,
    parent_section_title: str,
    parent_asn_definition_text: str,
    depth: int,
    max_depth: int,
    visited_section_ids: Set[str],
    max_total_nodes: int,
) -> AsnNodeResult:
    """
    Recursively expands MAIN -> child types using strict allowed_section_ids.
    """
    if depth > max_depth:
        return AsnNodeResult(node=_build_asn_tree_node(
            section_id=parent_section_id,
            section_title=parent_section_title,
            asn_definition_full=parent_asn_definition_text,
            children=[],
        ), discovered_section_ids={parent_section_id})

    if parent_section_id in visited_section_ids:
        # Stop cycles.
        return AsnNodeResult(node=_build_asn_tree_node(
            section_id=parent_section_id,
            section_title=parent_section_title,
            asn_definition_full=parent_asn_definition_text,
            children=[],
        ), discovered_section_ids={parent_section_id})

    discovered: Set[str] = {parent_section_id}
    visited_section_ids.add(parent_section_id)

    # If this doesn't look like ASN definition material, stop.
    if not _is_probably_asn1_definition_text(parent_asn_definition_text):
        return AsnNodeResult(
            node=_build_asn_tree_node(
                section_id=parent_section_id,
                section_title=parent_section_title,
                asn_definition_full=parent_asn_definition_text,
                children=[],
            ),
            discovered_section_ids=discovered,
        )

    if llm_call_budget.get("calls", 0) <= 0:
        llm_for_extraction = None
    else:
        llm_for_extraction = llm

    # 1) Extract child type candidates (LLM-first, deterministic fallback).
    child_type_candidates: List[str] = []
    if llm_for_extraction is not None:
        # budgeted
        llm_call_budget["calls"] -= 1
        child_type_candidates = _llm_extract_child_type_candidates(
            llm=llm_for_extraction,
            query_text=query_text,
            section_text_anchor=message_format_anchor_text,
            parent_asn_definition_text=parent_asn_definition_text,
        )
    if not child_type_candidates:
        child_type_candidates = _find_child_type_candidates_regex(parent_asn_definition_text)

    # Dedup while preserving stable order.
    seen_child_types: Set[str] = set()
    cleaned_child_types: List[str] = []
    for t in child_type_candidates:
        tt = str(t).strip()
        if not tt:
            continue
        if tt in seen_child_types:
            continue
        seen_child_types.add(tt)
        cleaned_child_types.append(tt)
    child_type_candidates = cleaned_child_types[:10]

    children_nodes: List[Dict[str, Any]] = []
    total_nodes_counter = 1  # include current parent

    for child_type in child_type_candidates:
        if total_nodes_counter >= max_total_nodes:
            break

        # Candidate definition nodes restricted to allowed_section_ids.
        candidate_definitions: List[Dict[str, Any]] = []
        for sid in sorted(allowed_section_ids):
            if sid not in chunks_by_sid:
                continue
            c = chunks_by_sid[sid]
            content = str(c.get("content", "") or "")
            if not content:
                continue
            if not _is_probably_asn1_definition_text(content):
                continue
            preview = _truncate_text(content, 5000)
            # Cheap filter: child identifier token must appear at least once.
            if re.search(re.escape(child_type), content, flags=re.IGNORECASE) is None:
                continue
            candidate_definitions.append(
                {
                    "section_id": sid,
                    "section_title": str(c.get("section_title", "") or ""),
                    "content_preview": preview,
                }
            )
            if len(candidate_definitions) >= 8:
                break

        # If nothing matched token-wise, allow a broader candidate set using scoring only (still restricted to allowed).
        if not candidate_definitions:
            scored: List[Tuple[int, str, str]] = []
            for sid in sorted(allowed_section_ids):
                if sid not in chunks_by_sid:
                    continue
                c = chunks_by_sid[sid]
                content = str(c.get("content", "") or "")
                if not content:
                    continue
                if not _is_probably_asn1_definition_text(content):
                    continue
                score = _score_candidate_definition_content(_truncate_text(content, 6000), child_type)
                if score <= 0:
                    continue
                scored.append((score, sid, str(c.get("section_title", "") or "")))
            scored.sort(reverse=True, key=lambda x: x[0])
            for score, sid, stitle in scored[:6]:
                candidate_definitions.append(
                    {"section_id": sid, "section_title": stitle, "content_preview": _truncate_text(str(chunks_by_sid[sid].get("content", "") or ""), 5000)}
                )

        # LLM-first choose best mapping.
        best_child_sid: Optional[str] = None
        llm_for_mapping = llm if llm_call_budget.get("calls", 0) > 0 else None
        if llm_for_mapping is not None and candidate_definitions:
            llm_call_budget["calls"] -= 1
            best_child_sid = _llm_choose_best_child_definition_section(
                llm=llm_for_mapping,
                query_text=query_text,
                section_text_anchor=message_format_anchor_text,
                child_type_identifier=child_type,
                parent_asn_definition_text=parent_asn_definition_text,
                candidate_definitions=candidate_definitions,
            )

        if not best_child_sid:
            # Deterministic best-match if LLM didn't produce a valid sid.
            best_child_sid = _llm_choose_best_child_definition_section(
                llm=None,
                query_text=query_text,
                section_text_anchor=message_format_anchor_text,
                child_type_identifier=child_type,
                parent_asn_definition_text=parent_asn_definition_text,
                candidate_definitions=candidate_definitions,
            )

        best_child_sid = _normalize_section_id(best_child_sid)
        if not best_child_sid:
            continue
        if best_child_sid not in allowed_section_ids:
            continue
        if best_child_sid not in chunks_by_sid:
            continue
        if best_child_sid in visited_section_ids:
            continue

        child_chunk = chunks_by_sid[best_child_sid]
        child_title = str(child_chunk.get("section_title", "") or "")
        child_asn_def = str(child_chunk.get("content", "") or "")

        child_res = _build_asn_tree_recursive(
            llm=llm,
            llm_call_budget=llm_call_budget,
            query_text=query_text,
            message_format_anchor_text=message_format_anchor_text,
            allowed_section_ids=allowed_section_ids,
            chunks_by_sid=chunks_by_sid,
            parent_section_id=best_child_sid,
            parent_section_title=child_title,
            parent_asn_definition_text=child_asn_def,
            depth=depth + 1,
            max_depth=max_depth,
            visited_section_ids=visited_section_ids,
            max_total_nodes=max_total_nodes,
        )
        children_nodes.append(child_res.node)
        discovered.update(child_res.discovered_section_ids)
        total_nodes_counter = len(discovered)

    return AsnNodeResult(
        node=_build_asn_tree_node(
            section_id=parent_section_id,
            section_title=parent_section_title,
            asn_definition_full=parent_asn_definition_text,
            children=children_nodes,
        ),
        discovered_section_ids=discovered,
    )


def _select_main_definition_candidates(
    *,
    allowed_section_ids: Set[str],
    chunks_by_sid: Dict[str, Dict[str, Any]],
    derived_main_ie_token: str,
    max_candidates: int = 12,
) -> List[Dict[str, Any]]:
    """
    Deterministic candidate gathering for MAIN definition nodes.
    """
    candidates: List[Tuple[int, str, str]] = []  # score, sid, preview
    for sid in sorted(allowed_section_ids):
        if sid not in chunks_by_sid:
            continue
        c = chunks_by_sid[sid]
        content = str(c.get("content", "") or "")
        if not content:
            continue
        if not _is_probably_asn1_definition_text(content):
            continue
        if derived_main_ie_token:
            score = _score_candidate_definition_content(
                content=_truncate_text(content, 6000), ie_token=derived_main_ie_token
            )
        else:
            score = 0
            if "::=" in content:
                score += 1
        candidates.append((score, sid, content))

    # If derived token doesn't hit anything, fallback to any ASN-looking nodes.
    candidates.sort(reverse=True, key=lambda x: x[0])
    picked: List[Dict[str, Any]] = []
    for score, sid, content in candidates:
        if not derived_main_ie_token and score <= 0:
            continue
        picked.append(
            {
                "section_id": sid,
                "section_title": str(chunks_by_sid[sid].get("section_title", "") or ""),
                "content_preview": _truncate_text(content, 7000),
                "ie_token": derived_main_ie_token,
                "score": score,
            }
        )
        if len(picked) >= max_candidates:
            break

    if not picked and derived_main_ie_token:
        # last resort: top ASN-like nodes by intrinsic ASN markers
        asn_like = []
        for sid in sorted(allowed_section_ids):
            if sid not in chunks_by_sid:
                continue
            c = chunks_by_sid[sid]
            content = str(c.get("content", "") or "")
            if not content or not _is_probably_asn1_definition_text(content):
                continue
            intrinsic = (6 if "::=" in content else 0) + (2 if "PROTOCOL-IES" in content.upper() else 0)
            asn_like.append((intrinsic, sid, content))
        asn_like.sort(reverse=True, key=lambda x: x[0])
        for _, sid, content in asn_like[:max_candidates]:
            picked.append(
                {
                    "section_id": sid,
                    "section_title": str(chunks_by_sid[sid].get("section_title", "") or ""),
                    "content_preview": _truncate_text(content, 7000),
                    "ie_token": derived_main_ie_token,
                    "score": 0,
                }
            )
    return picked[:max_candidates]


def _pick_message_asn_ie_definitions_for_message(
    *,
    llm: Any | None,
    llm_call_budget: Dict[str, int],
    query_text: str,
    message_name: str,
    message_format_section_id: str,
    allowed_section_ids: Set[str],
    chunks_by_sid: Dict[str, Dict[str, Any]],
    max_tree_depth: int = 4,
    max_total_nodes: int = 80,
) -> Dict[str, Any]:
    """
    Build MAIN -> nested children ASN.1 IE definition tree(s) and return results.
    """
    message_format_sid = _normalize_section_id(message_format_section_id)
    if message_format_sid not in chunks_by_sid:
        raise ValueError(f"message_format_section_id={message_format_sid} missing from chunks_by_sid")

    message_format_chunk = chunks_by_sid[message_format_sid]
    anchor_text = str(message_format_chunk.get("content", "") or "")

    derived_main_ie_token = _derive_ie_definition_name_from_message(message_name)

    candidate_main_defs = _select_main_definition_candidates(
        allowed_section_ids=allowed_section_ids,
        chunks_by_sid=chunks_by_sid,
        derived_main_ie_token=derived_main_ie_token,
        max_candidates=12,
    )

    main_definition_section_ids = _llm_choose_main_definition(
        llm=llm,
        query_text=query_text,
        message_name=message_name,
        message_format_anchor=anchor_text,
        candidate_definitions=candidate_main_defs,
        max_keep=3,
    )

    # If LLM returns empty, deterministic fallback: first best candidate by heuristic.
    if not main_definition_section_ids:
        if candidate_main_defs:
            # choose the highest scored candidate among gathered ones
            scored = sorted(
                candidate_main_defs,
                key=lambda d: int(d.get("score", 0) or 0),
                reverse=True,
            )
            main_definition_section_ids = [str(scored[0].get("section_id", "")).strip()] if scored else []

    discovered_ie_section_ids: Set[str] = set()
    roots: List[Dict[str, Any]] = []

    visited: Set[str] = set()

    for root_sid in main_definition_section_ids[:2]:
        root_sid = _normalize_section_id(root_sid)
        if not root_sid:
            continue
        if root_sid not in allowed_section_ids:
            continue
        if root_sid not in chunks_by_sid:
            continue
        if root_sid in visited:
            continue

        root_chunk = chunks_by_sid[root_sid]
        root_title = str(root_chunk.get("section_title", "") or "")
        root_def = str(root_chunk.get("content", "") or "")

        res = _build_asn_tree_recursive(
            llm=llm,
            llm_call_budget=llm_call_budget,
            query_text=query_text,
            message_format_anchor_text=anchor_text,
            allowed_section_ids=allowed_section_ids,
            chunks_by_sid=chunks_by_sid,
            parent_section_id=root_sid,
            parent_section_title=root_title,
            parent_asn_definition_text=root_def,
            depth=0,
            max_depth=max_tree_depth,
            visited_section_ids=visited,
            max_total_nodes=max_total_nodes,
        )

        roots.append(res.node)
        discovered_ie_section_ids.update(res.discovered_section_ids)

    return {
        "message_name": message_name,
        "message_format_section_id": message_format_sid,
        "derived_main_ie_token": derived_main_ie_token,
        "main_ie_chain_forest": roots,
        "discovered_ie_section_ids": sorted(list(discovered_ie_section_ids)),
    }


def run_step4_agentic_ie_retrieval() -> Dict[str, Any]:
    """
    Step-4:
      Agentic recursive ASN.1 IE discovery starting from message_format section_ids
      (strictly restricted to the message_format subtree).
    """
    repo_root = Path(__file__).resolve().parent.parent

    feature_json_path = repo_root / "Inter-gNB-DU_LTM_handover_procedure_20260323_093447.json"
    if not feature_json_path.exists():
        raise FileNotFoundError(f"Feature JSON not found: {feature_json_path}")
    feature_payload = _load_json(feature_json_path)
    intent = str(feature_payload.get("intent", "") or feature_payload.get("query", "") or "").strip()

    kg_base_dir = repo_root / "KG_Only_Pipeline" / "spec_chunks"
    if not kg_base_dir.exists():
        # Keep same base as step-3.
        kg_base_dir = repo_root / "KG_Retriver" / "spec_chunks"

    seed_out_dir = repo_root / "KG_Retriver" / "spec_chunks" / "retrieval_outputs"
    if not seed_out_dir.exists():
        seed_out_dir = repo_root / "KG_Only_Pipeline" / "spec_chunks" / "retrieval_outputs"

    # Use latest seed_expanded_kg_context output if present.
    seed_files = sorted(
        seed_out_dir.glob("seed_expanded_kg_context_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    seed_expanded_path = seed_files[0] if seed_files else None

    llm = _build_llm_azure()

    # Strictly use the feature JSON mapping to find message_format section ids.
    message_formats = _collect_message_format_subtrees(feature_payload, intent=intent)
    if not message_formats:
        # If we couldn't map message names from intent, fallback: include all message_format entries.
        message_formats = _collect_message_format_subtrees(feature_payload, intent="")

    # Budget: keep number of LLM calls bounded; after that, recursion uses deterministic parsing.
    llm_call_budget = {"calls": 12}

    all_specs_context: List[Dict[str, Any]] = []
    asn_trees_by_message: Dict[str, Any] = {}

    # Dedup across messages and roots.
    # Key includes message_name to avoid dropping distinct trees/roles.
    seen_chunks: Set[Tuple[str, str, str]] = set()

    for mf in message_formats:
        doc_id = str(mf.get("doc_id", "")).strip()
        message_name = str(mf.get("message_name", "")).strip()
        message_format_section_id = _normalize_section_id(mf.get("message_format_section_id"))

        if not doc_id or not message_format_section_id or not message_name:
            continue

        sections_json_path = kg_base_dir / doc_id / "sections.json"
        graph_path = kg_base_dir / doc_id / "KnowledgeGraph" / "knowledge_graph.pkl"
        if not sections_json_path.exists() or not graph_path.exists():
            continue

        sections_by_sid = _load_json(sections_json_path)
        if not isinstance(sections_by_sid, dict):
            continue

        allowed_section_ids = _extract_allowed_section_ids_for_message_format(
            sections_by_sid=sections_by_sid,
            message_format_section_id=message_format_section_id,
        )

        with graph_path.open("rb") as f:
            graph = pickle.load(f)

        chunks_by_sid = _load_allowed_graph_nodes_by_sid(
            graph=graph,
            allowed_section_ids=allowed_section_ids,
        )

        # Ensure we have the anchor message_format chunk.
        if message_format_section_id not in chunks_by_sid:
            # No anchor => cannot build ASN tree for this message.
            continue

        asn_res = _pick_message_asn_ie_definitions_for_message(
            llm=llm,
            llm_call_budget=llm_call_budget,
            query_text=intent,
            message_name=message_name,
            message_format_section_id=message_format_section_id,
            allowed_section_ids=allowed_section_ids,
            chunks_by_sid=chunks_by_sid,
            max_tree_depth=4,
            max_total_nodes=90,
        )

        asn_trees_by_message[message_name] = asn_res

        discovered_ie_sids = set(asn_res.get("discovered_ie_section_ids", []) or [])

        # 1) Add message_format chunk
        mf_chunk = chunks_by_sid.get(message_format_section_id)
        if mf_chunk:
            key = (doc_id, message_name, message_format_section_id)
            if key not in seen_chunks:
                seen_chunks.add(key)
                out_chunk = dict(mf_chunk)
                out_chunk.setdefault("section_title", str(mf_chunk.get("section_title", "") or ""))
                out_chunk["knowledge_source"] = doc_id
                out_chunk["source_id"] = doc_id
                out_chunk["chunk_role"] = "message_format"
                out_chunk["source_message_format_section_id"] = message_format_section_id
                out_chunk["source_message_name"] = message_name
                # Helpful for template filler ordering/selection.
                out_chunk["metadata"] = out_chunk.get("metadata") if isinstance(out_chunk.get("metadata"), dict) else {}
                if isinstance(out_chunk["metadata"], dict):
                    out_chunk["metadata"]["doc_id"] = doc_id
                    out_chunk["metadata"]["knowledge_source"] = doc_id
                all_specs_context.append(out_chunk)

        # 2) Add all discovered ASN IE definition chunks
        for sid in sorted(discovered_ie_sids):
            sid_norm = _normalize_section_id(sid)
            if not sid_norm:
                continue
            if sid_norm == message_format_section_id:
                continue
            if sid_norm not in chunks_by_sid:
                continue
            key = (doc_id, message_name, sid_norm)
            if key in seen_chunks:
                continue
            seen_chunks.add(key)
            ch = chunks_by_sid[sid_norm]
            out_chunk = dict(ch)
            out_chunk["knowledge_source"] = doc_id
            out_chunk["source_id"] = doc_id
            out_chunk["chunk_role"] = "ie_definition_chain"
            out_chunk["source_message_format_section_id"] = message_format_section_id
            out_chunk["source_message_name"] = message_name
            out_chunk["metadata"] = out_chunk.get("metadata") if isinstance(out_chunk.get("metadata"), dict) else {}
            if isinstance(out_chunk["metadata"], dict):
                out_chunk["metadata"]["doc_id"] = doc_id
                out_chunk["metadata"]["knowledge_source"] = doc_id
            all_specs_context.append(out_chunk)

    # Dedup again just in case.
    final_context: List[Dict[str, Any]] = []
    seen2: Set[Tuple[str, str]] = set()
    for c in all_specs_context:
        doc = str(c.get("knowledge_source") or c.get("source_id") or "").strip()
        sid = _normalize_section_id(c.get("section_id"))
        if not doc or not sid:
            continue
        k = (doc, sid)
        if k in seen2:
            continue
        seen2.add(k)
        final_context.append(c)

    final_context.sort(key=lambda x: (_safe_upper(x.get("knowledge_source")), _normalize_section_id(x.get("section_id"))))

    output = {
        "feature_json_path": str(feature_json_path),
        "seed_expanded_kg_context_path": str(seed_expanded_path) if seed_expanded_path is not None else None,
        "intent": intent,
        "timestamp": datetime.utcnow().isoformat(),
        "retrieval_config": {
            "phase": "step4_agentic_asn_ie_expansion",
            "llm_available": llm is not None,
            "llm_call_budget_remaining": llm_call_budget.get("calls", 0),
            "max_tree_depth": 4,
            "max_total_nodes_per_tree": 90,
            "strict_subtree_only": True,
        },
        "asn_trees_by_message_name": asn_trees_by_message,
        "specs_context_count": len(final_context),
        "specs_context": final_context,
        # Adapter compatibility.
        "final_context": final_context,
    }
    return output


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "KG_Retriver" / "spec_chunks" / "retrieval_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_step4_agentic_ie_retrieval()

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"agentic_ie_retrieval_context_v0_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Retriver_v0] Saved Step-4 ASN IE retrieval to: {out_path}")


if __name__ == "__main__":
    main()

