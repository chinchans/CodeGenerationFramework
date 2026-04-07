"""
KG_Retriver: Seed-first KG retrieval with per-seed expansion.

Flow:
1) Parse feature JSON and collect seed section_ids per doc_id (in memory).
2) For each doc_id, load its knowledge graph:
      KG_Only_Pipeline/spec_chunks/<doc_id>/KnowledgeGraph/knowledge_graph.pkl
3) Expand EACH seed independently with:
      max_depth=2, direction="both" (incoming + outgoing)
4) Deduplicate section content across all seeds (no duplicate section chunks),
   but keep a hierarchical view: doc -> root seed -> expanded sections.

Output:
- Returns ONLY the retrieved sections in a hierarchical form (no separate "seed JSON").
- By default prints JSON to stdout; can be captured/piped by caller.
"""

from __future__ import annotations

import json
import os
import pickle
import re
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Feature JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_section_id(section_id: Any) -> str:
    return str(section_id or "").strip()


def _safe_upper(s: str) -> str:
    return (s or "").upper()


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
        # Preserve acronyms like UE, RRC, CSI as-is (important for matching ASN.1 identifiers).
        if w.isupper() and len(w) > 1:
            parts.append(w)
        else:
            parts.append(w[0].upper() + w[1:].lower())
    camel = "".join(parts)
    if not camel:
        return ""
    if not camel.endswith("IEs"):
        return camel + "IEs"
    return camel


def _compile_ie_regex_patterns(identifier: str) -> List[re.Pattern[str]]:
    """
    Compile definition-matching regexes for an ASN.1 identifier.

    We intentionally keep these patterns strict:
    - MAIN/F1AP-style protocol IE definitions often look like:
        <MainIEName> <Something-PROTOCOL-IES> ::= { ... }
    - Regular ASN.1 type definitions look like:
        <TypeName> ::= SEQUENCE { ... }  (or CHOICE/ENUMERATED/etc.)
    """
    name = (identifier or "").strip()
    if not name:
        return []

    esc = re.escape(name)
    flags = re.IGNORECASE | re.DOTALL

    return [
        # MAIN/F1AP-protocol IE definition form:
        #   <MainIEName> <X-PROTOCOL-IES> ::= { ... }
        re.compile(rf"{esc}\s+.*?PROTOCOL-IES\s*::=", flags),
        # Generic ASN.1 type definition form:
        #   <TypeName> ::= ...
        re.compile(rf"{esc}\s*::=", flags),
    ]


def _find_definition_nodes_for_ie(graph: nx.DiGraph, *, ie_name: str) -> List[str]:
    """
    Deterministic lookup: scan node.content for ASN.1 definitions
    of the provided identifier.
    Returns matching node_ids (section_ids).
    """
    patterns = _compile_ie_regex_patterns(ie_name)
    if not patterns:
        return []

    hits: List[str] = []
    for node_id in graph.nodes():
        content = graph.nodes[node_id].get("content", "") or ""
        if not content:
            continue
        if "::=" not in content and "PROTOCOL-IES" not in _safe_upper(content):
            continue
        for pat in patterns:
            if pat.search(content):
                hits.append(str(node_id))
                break

    # Dedup while preserving order.
    seen: Set[str] = set()
    out: List[str] = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def _extract_child_ie_names_regex(parent_ie_definition_text: str) -> List[str]:
    """
    Deterministic fallback for child identifier extraction.

    IMPORTANT: In the MAIN protocol IE definition, nested types are typically
    referenced via "TYPE <TypeName>" (the nested types usually *do not* end with "IEs").
    In SEQUENCE/CHOICE type definitions, nested fields reference other types as:
      <fieldName> <TypeName> [OPTIONAL|MANDATORY] ...
    """
    text = parent_ie_definition_text or ""
    if not text.strip():
        return []

    # 1) From protocol IE entries: ... TYPE <TypeName>
    type_targets = set(re.findall(r"\bTYPE\s+([A-Za-z0-9][A-Za-z0-9-]*)\b", text, flags=re.IGNORECASE))

    # 2) From SEQUENCE/CHOICE bodies: ... <fieldName> <TypeName> [, OPTIONAL ...]
    # Heuristic: capture the second identifier after "<something> <something>".
    pair_targets: Set[str] = set(
        re.findall(
            r"\b[A-Za-z0-9][A-Za-z0-9-]*\s+([A-Za-z][A-Za-z0-9-]*)\b(?=\s*(?:,|OPTIONAL|MANDATORY|}|$))",
            text,
            flags=re.IGNORECASE,
        )
    )

    # Merge and filter ASN.1 keywords / primitives that we never want to expand.
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


def _recursive_ie_definition_discovery(
    *,
    graph: nx.DiGraph,
    main_node_ids: List[str],
    max_depth: int = 2,
    max_nodes: int = 80,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Recursively discover ASN.1 IE definition nodes starting from MAIN node ids.

    Returns:
      - discovered_node_ids (in discovery order, unique)
      - trace edges: [{from_node_id, via_child_ie_name, to_node_id, depth_level}]
    """
    visited_nodes: Set[str] = set()
    discovered: List[str] = []
    edges: List[Dict[str, Any]] = []

    q = deque([(nid, 0) for nid in main_node_ids if nid in graph.nodes])
    while q:
        node_id, depth = q.popleft()
        if node_id in visited_nodes:
            continue
        if depth > max_depth:
            continue
        if len(discovered) >= max_nodes:
            break

        visited_nodes.add(node_id)
        discovered.append(node_id)

        parent_text = graph.nodes[node_id].get("content", "") or ""
        child_ie_names = _extract_child_ie_names_regex(parent_text)[:25]

        for child_ie in child_ie_names:
            found_nodes = _find_definition_nodes_for_ie(graph, ie_name=child_ie)
            for fn in found_nodes:
                if fn not in visited_nodes:
                    q.append((fn, depth + 1))
                edges.append(
                    {
                        "from_node_id": node_id,
                        "via_child_ie_name": child_ie,
                        "to_node_id": fn,
                        "depth_level": depth + 1,
                    }
                )
            if len(discovered) >= max_nodes:
                break

    return discovered, edges


def _truncate_text(text: str, max_chars: int) -> str:
    """
    Keep head+tail so ASN.1 definitions remain more complete while
    reducing prompt size.
    """
    t = text or ""
    if len(t) <= max_chars:
        return t
    head = max(0, max_chars - 120)
    return t[:head] + "\n\n[TRUNCATED]\n...\n" + t[-120:]


def _build_llm() -> Any | None:
    """
    Optional Azure LLM for filtering and IE recursion.
    Falls back to None if credentials/deps are missing.
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
    return AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_deployment=deployment,
        temperature=0.1,
        timeout=120,
        max_retries=2,
    )


def _extract_first_json_object(raw: str) -> Dict[str, Any]:
    """
    Best-effort JSON object extraction from an LLM response.
    """
    s = (raw or "").strip()
    # Strip common markdown fences.
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
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _llm_filter_relevant_sections(
    *,
    llm: Any | None,
    query_text: str,
    section_text_anchor: str,
    candidate_sections: List[Dict[str, Any]],
    max_keep: int = 80,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Returns:
      - relevant_section_ids
      - debug info
    """
    if llm is None:
        return [str(c.get("section_id", "")) for c in candidate_sections if c.get("section_id")], {
            "llm_used": False,
        }

    items = []
    for c in candidate_sections:
        items.append(
            {
                "section_id": str(c.get("section_id", "")),
                "section_title": str(c.get("section_title", "")),
                "content_preview": _truncate_text(str(c.get("content", "")), 700),
            }
        )

    prompt = f"""
You are filtering retrieved 3GPP spec sections for a feature/procedure implementation.

INPUTS:
query_text:
{query_text}

section_text_anchor:
{_truncate_text(section_text_anchor, 6000)}

TASK:
Choose only the section_ids that are relevant to the feature/procedure described by query_text,
and are supported by section_text_anchor.
Return only a small set of relevant section_ids (procedure steps, constraints, and ASN.1 definition material).
Exclude unrelated sections.

CANDIDATES (JSON array):
{json.dumps(items, ensure_ascii=False)[:12000]}

RETURN JSON ONLY:
{{
  "relevant_section_ids": ["<section_id>", ...],
  "reason_short": "<1-2 sentences>"
}}
""".strip()

    resp = llm.invoke(prompt)
    raw = resp.content if hasattr(resp, "content") else str(resp)
    obj = _extract_first_json_object(raw)
    ids = obj.get("relevant_section_ids", []) if isinstance(obj, dict) else []
    if not isinstance(ids, list):
        ids = []
    ids = [str(x) for x in ids if str(x).strip()]
    ids = ids[:max_keep]
    return ids, {"llm_used": True, "llm_keys": list(obj.keys()) if isinstance(obj, dict) else []}


def _is_probably_asn1_definition_text(text: str) -> bool:
    t = (text or "").upper()
    return "::=" in t or "PROTOCOL-IES" in t or "PROTOCOL-IES" in t


def _llm_keep_only_ie_definition_from_section(
    *,
    llm: Any | None,
    query_text: str,
    section_text_anchor: str,
    ie_type_identifier: str,
    section_title: str,
    section_content: str,
    keep_original_if_empty: bool = True,
) -> Dict[str, Any]:
    """
    For a single KG chunk/section, keep ONLY the ASN.1 definition block for
    `ie_type_identifier` (e.g., an ASN.1 type name referenced by the MAIN IE).

    This prevents writing the whole KG section content into the JSON chunks.
    """
    if llm is None:
        return {
            "content": section_content,
            "llm_used": False,
            "extracted_ie_definition": "",
        }

    ie_type_identifier = str(ie_type_identifier or "").strip()
    if not ie_type_identifier:
        return {
            "content": section_content,
            "llm_used": False,
            "extracted_ie_definition": "",
        }

    prompt = f"""
Extract ONLY the COMPLETE ASN.1 definition block for a single identifier from the provided 3GPP spec section.

Context:
USER INTENT (query_text):
{query_text}

SECTION_TEXT_ANCHOR (precision anchor):
{_truncate_text(section_text_anchor, 6000)}

REQUIREMENT:
- Keep ONLY the ASN.1 definition of the identifier: {ie_type_identifier}
- Do NOT include any other ASN.1 definitions or descriptive text.
- Do NOT summarize.
- Return the ASN.1 syntax exactly as it appears in the input (preserve braces, operators, formatting as much as possible).

SECTION TITLE:
{section_title}

SECTION CONTENT:
{_truncate_text(section_content, 20000)}

Return JSON ONLY:
{{
  "extracted_ie_definition": "<asn_definition_string_or_empty>",
  "kept": true/false
}}
""".strip()

    resp = llm.invoke(prompt)
    raw = resp.content if hasattr(resp, "content") else str(resp)
    obj = _extract_first_json_object(raw)
    extracted = ""
    if isinstance(obj, dict):
        extracted = str(obj.get("extracted_ie_definition", "") or "")

    extracted = extracted.strip()
    if extracted:
        return {
            "content": extracted,
            "llm_used": True,
            "extracted_ie_definition": extracted,
        }

    if keep_original_if_empty:
        return {
            "content": section_content,
            "llm_used": True,
            "extracted_ie_definition": "",
        }
    return {
        "content": "",
        "llm_used": True,
        "extracted_ie_definition": "",
    }


def _select_best_definition_section_id(
    *,
    graph: nx.DiGraph,
    candidate_section_ids: List[str],
    ie_name: str,
) -> str | None:
    """
    Deterministic best-effort selection for an IE definition node.
    """
    if not candidate_section_ids:
        return None
    ie_token = (ie_name or "").strip()
    best_id = None
    best_score = None
    esc = re.escape(ie_token)
    flags = re.IGNORECASE | re.DOTALL
    for sid in candidate_section_ids:
        if sid not in graph.nodes:
            continue
        content = graph.nodes[sid].get("content", "") or ""
        score = 0
        if ie_token and re.search(esc, content, flags=re.IGNORECASE):
            score += 4
        # Strong signals: exact identifier definition operator.
        if ie_token and re.search(rf"{esc}\s*::=", content, flags=flags):
            score += 5
        # Strong signals: protocol IE definition operator.
        if ie_token and re.search(rf"{esc}\s+.*?PROTOCOL-IES\s*::=", content, flags=flags):
            score += 8
        if "::=" in content:
            score += 1
        if "PROTOCOL-IES" in _safe_upper(content):
            score += 1
        if best_score is None or score > best_score:
            best_score = score
            best_id = sid
    return best_id


def _llm_choose_main_ie_definition(
    *,
    llm: Any | None,
    query_text: str,
    section_text_anchor: str,
    message_name: str,
    message_format_content: str,
    candidate_definitions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Choose the best MAIN/parent IE definition node for this message.
    """
    if not candidate_definitions:
        return {"main_ie_name": "", "main_definition_section_id": ""}
    if llm is None:
        # Deterministic best-effort choice based on candidate content previews.
        best = None
        best_score = None
        for c in candidate_definitions:
            cand_ie_name = str(c.get("ie_name", "") or "").strip()
            preview = str(c.get("content_preview", "") or "")
            if not cand_ie_name or not preview:
                score = 0
            else:
                esc = re.escape(cand_ie_name)
                flags = re.IGNORECASE | re.DOTALL
                score = 0
                if re.search(esc, preview, flags=re.IGNORECASE):
                    score += 2
                if re.search(rf"{esc}\s+.*?PROTOCOL-IES\s*::=", preview, flags=flags):
                    score += 8
                if re.search(rf"{esc}\s*::=", preview, flags=flags):
                    score += 5
                if "PROTOCOL-IES" in _safe_upper(preview):
                    score += 1
                if "::=" in preview:
                    score += 1

            if best_score is None or score > best_score:
                best_score = score
                best = c

        c0 = best or candidate_definitions[0]
        return {
            "main_ie_name": str(c0.get("ie_name", "")),
            "main_definition_section_id": str(c0.get("section_id", "")),
        }

    prompt = f"""
You are selecting the MAIN/parent ASN.1 IE definition for a specific message.

query_text:
{query_text}

section_text_anchor:
{_truncate_text(section_text_anchor, 6000)}

message_name:
{message_name}

message_format_content (for context):
{_truncate_text(message_format_content, 8000)}

Candidate MAIN IE definitions (JSON array):
{json.dumps(candidate_definitions, ensure_ascii=False)[:12000]}

RETURN JSON ONLY:
{{
  "main_ie_name": "<ie_name ending with IEs>",
  "main_definition_section_id": "<section_id from candidates>"
}}
""".strip()

    resp = llm.invoke(prompt)
    raw = resp.content if hasattr(resp, "content") else str(resp)
    obj = _extract_first_json_object(raw)
    main_ie_name = str(obj.get("main_ie_name", "") or "")
    main_sid = str(obj.get("main_definition_section_id", "") or "")
    if not main_sid:
        c0 = candidate_definitions[0]
        main_ie_name = str(c0.get("ie_name", ""))
        main_sid = str(c0.get("section_id", ""))
    return {"main_ie_name": main_ie_name, "main_definition_section_id": main_sid}


def _llm_extract_child_ie_candidates(
    *,
    llm: Any | None,
    query_text: str,
    section_text_anchor: str,
    parent_ie_name: str,
    parent_asn_definition_text: str,
) -> List[str]:
    """
    Extract child/nested ASN.1 type identifiers for recursive expansion.

    NOTE:
    - These nested identifiers are NOT guaranteed to end with "IEs".
    - For protocol IE definitions, they often appear as targets of "TYPE <TypeName>".
    """
    if not parent_asn_definition_text:
        return []
    if llm is None:
        return _extract_child_ie_names_regex(parent_asn_definition_text)[:15]

    prompt = f"""
Extract nested ASN.1 type identifiers referenced inside the parent ASN.1 definition
so we can expand them recursively.

query_text:
{query_text}

section_text_anchor:
{_truncate_text(section_text_anchor, 6000)}

parent_ie_name:
{parent_ie_name}

parent ASN.1 definition text:
{_truncate_text(parent_asn_definition_text, 12000)}

Rules:
- Return ONLY identifiers (type names) that are explicitly referenced inside the parent ASN.1 text.
- These identifiers may or may not end with "IEs".
- Exclude ASN.1 keywords and primitive base types (e.g., INTEGER, OCTET STRING, BIT STRING, ENUMERATED, BOOLEAN).
- If no confident nested type identifiers exist, return [].

RETURN JSON ONLY:
{{
  "child_type_candidates": ["<TypeNameIdentifier>", ...]
}}
""".strip()

    resp = llm.invoke(prompt)
    raw = resp.content if hasattr(resp, "content") else str(resp)
    obj = _extract_first_json_object(raw)
    ids = []
    if isinstance(obj, dict):
        # Backward/forward compatible parsing.
        ids = obj.get("child_type_candidates", obj.get("child_ie_candidates", []))
    if not isinstance(ids, list):
        return []
    cleaned: List[str] = []
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


def _build_ie_definition_tree(
    *,
    llm: Any | None,
    graph: nx.DiGraph,
    doc_id: str,
    query_text: str,
    section_text_anchor: str,
    message_name: str,
    message_format_section_id: str,
    root_ie_name: str,
    root_section_id: str,
    max_depth: int = 6,
    max_nodes: int = 80,
) -> Dict[str, Any]:
    """
    Build an IE definition tree (MAIN -> child -> ... until leaf).
    Each node contains the full ASN.1 definition from KG.
    """
    visited: Set[str] = set()
    node_counter = 0

    def _node_dict(sid: str, ie_name: str, children: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "ie_name": ie_name,
            "section_id": sid,
            "asn_definition": graph.nodes[sid].get("content", "") or "",
            "children": children,
        }

    def _expand(sid: str, ie_name: str, depth: int) -> Dict[str, Any]:
        nonlocal node_counter
        if node_counter >= max_nodes:
            return _node_dict(sid, ie_name, [])
        if sid in visited or depth > max_depth:
            return _node_dict(sid, ie_name, [])

        visited.add(sid)
        node_counter += 1

        parent_text = graph.nodes[sid].get("content", "") or ""
        child_ie_names = _llm_extract_child_ie_candidates(
            llm=llm,
            query_text=query_text,
            section_text_anchor=section_text_anchor,
            parent_ie_name=ie_name,
            parent_asn_definition_text=parent_text,
        )

        children: List[Dict[str, Any]] = []
        for child_ie in child_ie_names:
            candidate_nodes = _find_definition_nodes_for_ie(graph, ie_name=child_ie)
            best_sid = _select_best_definition_section_id(
                graph=graph, candidate_section_ids=candidate_nodes, ie_name=child_ie
            )
            if not best_sid:
                continue
            children.append(_expand(best_sid, child_ie, depth + 1))
            if node_counter >= max_nodes:
                break

        return _node_dict(sid, ie_name, children)

    root_section_id = _normalize_section_id(root_section_id)
    if not root_section_id or root_section_id not in graph.nodes:
        return {
            "ie_name": root_ie_name,
            "section_id": root_section_id,
            "asn_definition": "",
            "children": [],
        }

    tree = _expand(root_section_id, root_ie_name, depth=0)
    return {
        "message_name": message_name,
        "message_format_section_id": message_format_section_id,
        "doc_id": doc_id,
        "main_ie_chain": tree,
    }


def _discover_main_ie_definition_nodes(
    *,
    graph: nx.DiGraph,
    query_text: str,
    message_name: str,
    message_format_content: str,
    max_nodes: int = 8,
) -> List[Dict[str, str]]:
    """
    Phase-1 (Knowledge_Retrieval style): discover MAIN IE definition nodes first.
    Returns list of {"ie_name": str, "section_id": str}.
    """
    derived_main_ie = _derive_ie_definition_name_from_message(message_name)
    ie_name_candidates: List[str] = []
    seen_ie_names: Set[str] = set()

    for x in [derived_main_ie] + sorted(list(set(re.findall(r"\b[A-Za-z0-9]+IEs\b", message_format_content or "")))):
        xx = str(x or "").strip()
        if not xx or xx in seen_ie_names:
            continue
        seen_ie_names.add(xx)
        ie_name_candidates.append(xx)

    # Query-derived fallback if message format did not expose enough.
    if not ie_name_candidates:
        q_ies = sorted(list(set(re.findall(r"\b[A-Za-z0-9]+IEs\b", query_text or ""))))
        ie_name_candidates.extend(q_ies[:4])
        if derived_main_ie and derived_main_ie not in ie_name_candidates:
            ie_name_candidates.insert(0, derived_main_ie)

    found: List[Dict[str, str]] = []
    seen_pairs: Set[Tuple[str, str]] = set()
    for ie_name in ie_name_candidates:
        node_ids = _find_definition_nodes_for_ie(graph, ie_name=ie_name)
        best_sid = _select_best_definition_section_id(
            graph=graph,
            candidate_section_ids=node_ids,
            ie_name=ie_name,
        )
        if not best_sid:
            continue
        key = (ie_name, best_sid)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        found.append({"ie_name": ie_name, "section_id": best_sid})
        if len(found) >= max_nodes:
            break

    return found


def _expand_ies_agentically_nodes(
    *,
    llm: Any | None,
    graph: nx.DiGraph,
    query_text: str,
    section_text_anchor: str,
    seed_ie_nodes: List[Dict[str, str]],
    max_iterations: int = 2,
    max_total_nodes: int = 120,
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """
    Phase-2 (Knowledge_Retrieval style): iterative agentic IE expansion.
    Starts from main IE nodes and recursively discovers nested ASN.1 type definition nodes.

    Returns:
      - discovered_ie_nodes: [{"ie_name", "section_id"}]
      - expansion_trace: list of expansion edges/steps
    """
    discovered: List[Dict[str, str]] = []
    discovered_sids: Set[str] = set()
    frontier: List[Dict[str, str]] = []
    trace: List[Dict[str, Any]] = []

    for n in seed_ie_nodes:
        sid = _normalize_section_id(n.get("section_id"))
        ie_name = str(n.get("ie_name", "")).strip() or "UnknownIE"
        if not sid or sid not in graph.nodes or sid in discovered_sids:
            continue
        discovered_sids.add(sid)
        node = {"ie_name": ie_name, "section_id": sid}
        discovered.append(node)
        frontier.append(node)

    for iteration in range(1, max_iterations + 1):
        if not frontier or len(discovered) >= max_total_nodes:
            break
        new_frontier: List[Dict[str, str]] = []

        for parent in frontier:
            parent_sid = _normalize_section_id(parent.get("section_id"))
            parent_ie_name = str(parent.get("ie_name", "")).strip()
            if not parent_sid or parent_sid not in graph.nodes:
                continue
            parent_text = graph.nodes[parent_sid].get("content", "") or ""
            child_types = _llm_extract_child_ie_candidates(
                llm=llm,
                query_text=query_text,
                section_text_anchor=section_text_anchor,
                parent_ie_name=parent_ie_name,
                parent_asn_definition_text=parent_text,
            )

            for child_type in child_types:
                node_ids = _find_definition_nodes_for_ie(graph, ie_name=child_type)
                best_sid = _select_best_definition_section_id(
                    graph=graph,
                    candidate_section_ids=node_ids,
                    ie_name=child_type,
                )
                if not best_sid or best_sid not in graph.nodes:
                    continue
                if best_sid in discovered_sids:
                    continue

                child_node = {"ie_name": child_type, "section_id": best_sid}
                discovered_sids.add(best_sid)
                discovered.append(child_node)
                new_frontier.append(child_node)
                trace.append(
                    {
                        "iteration": iteration,
                        "from_section_id": parent_sid,
                        "from_ie_name": parent_ie_name,
                        "via_child_type": child_type,
                        "to_section_id": best_sid,
                    }
                )

                if len(discovered) >= max_total_nodes:
                    break
            if len(discovered) >= max_total_nodes:
                break

        frontier = new_frontier

    return discovered, trace


def _node_chunk(graph: nx.DiGraph, node_id: str, doc_id: str) -> Dict[str, Any]:
    node = graph.nodes[node_id]
    return {
        "section_id": node_id,
        "section_title": node.get("section_title", ""),
        "content": node.get("content", ""),
        # Template orchestrator expects top-level knowledge_source / source_id sometimes.
        "knowledge_source": doc_id,
        "source_id": doc_id,
        "metadata": {
            "doc_id": doc_id,
            "level": node.get("level"),
            "parent_section_id": node.get("parent_section_id"),
            "child_section_ids": node.get("child_section_ids", []),
            "page_numbers": node.get("page_numbers", []),
            "knowledge_source": doc_id,
            # Populated later:
            # - expanded_from_seed_section_ids: List[str]
            # - is_seed: bool
        },
    }


def _edge_rel_types(graph: nx.DiGraph, u: str, v: str) -> List[str]:
    attrs = graph.get_edge_data(u, v) or {}
    rels = attrs.get("relationship_types", []) or []
    if isinstance(rels, str):
        return [rels]
    return [str(r) for r in rels]


def _resolve_doc_id_from_spec_number(
    payload: Dict[str, Any], spec_number: str
) -> Optional[str]:
    """
    Map TS spec_number -> doc_id using payload["specs"] (preferred),
    otherwise None.
    """
    spec_number_norm = _safe_upper(spec_number).strip()
    if not spec_number_norm:
        return None

    for spec in payload.get("specs", []) or []:
        sn = _safe_upper(str(spec.get("spec_number", "")).strip())
        doc_id = str(spec.get("doc_id", "")).strip()
        if sn and doc_id and sn == spec_number_norm:
            return doc_id
    return None


def _collect_seed_section_ids_by_doc(payload: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Collect seed section_ids per doc_id (deduped, deterministic order).

    Note:
    - Seeds are only used as roots for hierarchy + KG traversal.
    - We do NOT emit any "seed artifacts" in the returned output.
    """
    seeds_by_doc: Dict[str, Set[str]] = {}

    def _add_seed(
        *,
        doc_id: str,
        section_id: str,
    ) -> None:
        dsid = _normalize_section_id(section_id)
        ddid = str(doc_id).strip()
        if not ddid or not dsid:
            return
        seeds_by_doc.setdefault(ddid, set()).add(dsid)

    # 1) procedure_spec_info seed(s)
    proc = payload.get("procedure_spec_info", {}) or {}
    proc_section_id = _normalize_section_id(proc.get("section_id"))
    proc_spec_number = str(proc.get("spec_number", "")).strip()
    if proc_section_id and proc_spec_number:
        doc_id = _resolve_doc_id_from_spec_number(payload, proc_spec_number)
        if doc_id:
            _add_seed(
                doc_id=doc_id,
                section_id=proc_section_id,
            )

    # 2) protocol_specs[] seeds (explicit section_id per spec entry, if provided)
    for spec in payload.get("protocol_specs", []) or []:
        spec_number = str(spec.get("spec_number", "")).strip()
        section_id = _normalize_section_id(spec.get("section_id"))
        doc_id = str(spec.get("doc_id", "")).strip()
        if not doc_id and spec_number:
            doc_id = _resolve_doc_id_from_spec_number(payload, spec_number) or ""
        if doc_id and section_id:
            _add_seed(
                doc_id=doc_id,
                section_id=section_id,
            )

    # 3) protocol_message_sections[] seeds (message-specific sections)
    protocol_doc_map: Dict[Tuple[str, str], str] = {}
    for spec in payload.get("protocol_specs", []) or []:
        protocol = str(spec.get("protocol", "")).strip().upper()
        spec_number = str(spec.get("spec_number", "")).strip().upper()
        doc_id = str(spec.get("doc_id", "")).strip()
        if protocol and spec_number and doc_id:
            protocol_doc_map[(protocol, spec_number)] = doc_id

    for block in payload.get("protocol_message_sections", []) or []:
        protocol = str(block.get("protocol", "")).strip().upper()
        spec_number = str(block.get("spec_number", "")).strip().upper()
        doc_id = protocol_doc_map.get((protocol, spec_number), "") if (protocol and spec_number) else ""
        if not doc_id:
            continue

        for m in block.get("messages", []) or []:
            for sec in m.get("sections", []) or []:
                sid = _normalize_section_id(sec.get("section_id"))
                if not sid:
                    continue
                _add_seed(
                    doc_id=doc_id,
                    section_id=sid,
                )

    # Deterministic ordering: doc_id then section_id.
    return {doc_id: sorted(list(sids)) for doc_id, sids in sorted(seeds_by_doc.items(), key=lambda x: x[0])}


def _collect_message_format_seeds(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Collect message_format seeds for ASN.1 IE discovery.
    Returns a list of:
      {doc_id, message_name, section_id}
    """
    spec_number_to_doc: Dict[str, str] = {}
    for spec in payload.get("specs", []) or []:
        sn = str(spec.get("spec_number", "")).strip().upper()
        did = str(spec.get("doc_id", "")).strip()
        if sn and did:
            spec_number_to_doc[sn] = did

    out: List[Dict[str, Any]] = []
    for block in payload.get("protocol_message_sections", []) or []:
        spec_number = str(block.get("spec_number", "")).strip().upper()
        doc_id = spec_number_to_doc.get(spec_number, "")
        if not doc_id:
            continue
        for msg in block.get("messages", []) or []:
            message_name = str(msg.get("message_name", "")).strip()
            for sec in msg.get("sections", []) or []:
                if str(sec.get("role", "")).strip() != "message_format":
                    continue
                sid = _normalize_section_id(sec.get("section_id"))
                if not sid:
                    continue
                out.append(
                    {
                        "doc_id": doc_id,
                        "message_name": message_name,
                        "section_id": sid,
                    }
                )

    seen: Set[Tuple[str, str, str]] = set()
    dedup: List[Dict[str, Any]] = []
    for x in sorted(out, key=lambda d: (d["doc_id"], d["message_name"], d["section_id"])):
        key = (x["doc_id"], x["message_name"], x["section_id"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(x)
    return dedup


def _resolve_graph_path(kg_base_dir: Path, doc_id: str) -> Path:
    """
    Resolve knowledge graph for exact doc_id.
    """
    p = kg_base_dir / doc_id / "KnowledgeGraph" / "knowledge_graph.pkl"
    if not p.exists():
        raise FileNotFoundError(f"KG graph not found for doc_id={doc_id}: {p}")
    return p


def _expand_from_single_seed(
    graph: nx.DiGraph,
    *,
    seed_id: str,
    max_depth: int,
    direction: str,
    allowed_relations: Optional[Set[str]],
) -> Tuple[List[str], Set[str]]:
    """
    Expand KG from a single seed section_id.

    Returns:
      expanded_neighbor_ids (excluding seed itself),
      visited_ids (including seed).
    """
    if seed_id not in graph.nodes:
        return [], set()

    visited: Set[str] = {seed_id}
    expanded_neighbor_ids: List[str] = []

    q = deque([(seed_id, 0)])
    while q:
        current, depth = q.popleft()
        if depth >= max_depth:
            continue

        # Deterministic expansion order:
        neighbors_out: List[str] = []
        neighbors_in: List[str] = []
        if direction in {"out", "both"}:
            neighbors_out = sorted(list(graph.successors(current)))
        if direction in {"in", "both"}:
            neighbors_in = sorted(list(graph.predecessors(current)))

        for nbr in neighbors_out:
            if nbr in visited:
                continue
            if allowed_relations:
                rels = _edge_rel_types(graph, current, nbr)
                if not any(r in allowed_relations for r in rels):
                    continue
            visited.add(nbr)
            expanded_neighbor_ids.append(nbr)
            q.append((nbr, depth + 1))

        for nbr in neighbors_in:
            if nbr in visited:
                continue
            if allowed_relations:
                rels = _edge_rel_types(graph, nbr, current)
                if not any(r in allowed_relations for r in rels):
                    continue
            visited.add(nbr)
            expanded_neighbor_ids.append(nbr)
            q.append((nbr, depth + 1))

    return expanded_neighbor_ids, visited


def run_section_retrieval_hierarchy(
    *,
    feature_json_path: Path,
    kg_base_dir: Path,
    max_depth: int = 2,
    direction: str = "both",
    allowed_relations: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    payload = _load_json(feature_json_path)
    query_text = str(payload.get("intent", "") or payload.get("query", "") or "").strip()
    section_text_anchor = str(payload.get("section_text", "") or "").strip()
    llm = _build_llm()
    seeds_by_doc = _collect_seed_section_ids_by_doc(payload)
    message_format_seeds = _collect_message_format_seeds(payload)
    message_format_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for s in message_format_seeds:
        message_format_by_doc.setdefault(s["doc_id"], []).append(s)

    docs_out: List[Dict[str, Any]] = []
    all_final_context: List[Dict[str, Any]] = []

    for doc_id, seed_section_ids in sorted(seeds_by_doc.items(), key=lambda x: x[0]):
        # Load KG
        graph_path = _resolve_graph_path(kg_base_dir, doc_id)
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        if not isinstance(graph, nx.DiGraph):
            raise ValueError(f"Expected nx.DiGraph in {graph_path}")

        # Unique sections stored once per doc.
        sections_by_sid: Dict[str, Dict[str, Any]] = {}
        missing_seed_section_ids: List[str] = []

        roots: List[Dict[str, Any]] = []
        message_ie_chains: List[Dict[str, Any]] = []
        asn1_ie_section_ids_used: Set[str] = set()

        for seed_sid in seed_section_ids:
            seed_sid = _normalize_section_id(seed_sid)
            if seed_sid not in graph.nodes:
                missing_seed_section_ids.append(seed_sid)
                continue

            # Ensure root chunk exists (stored once).
            if seed_sid not in sections_by_sid:
                root_chunk = _node_chunk(graph, seed_sid, doc_id)
                root_chunk["metadata"]["is_root_seed"] = True
                root_chunk["metadata"]["expanded_from_seed_section_ids"] = []
                sections_by_sid[seed_sid] = root_chunk
            else:
                sections_by_sid[seed_sid]["metadata"]["is_root_seed"] = True

            expanded_neighbor_ids, _visited = _expand_from_single_seed(
                graph,
                seed_id=seed_sid,
                max_depth=max_depth,
                direction=direction,
                allowed_relations=allowed_relations,
            )

            # Update unique section map for expanded nodes.
            expanded_section_ids_dedup: List[str] = []
            seen_in_this_seed: Set[str] = set()
            for nbr_sid in expanded_neighbor_ids:
                nbr_sid_norm = _normalize_section_id(nbr_sid)
                if nbr_sid_norm not in graph.nodes:
                    continue
                if nbr_sid_norm in seen_in_this_seed:
                    continue
                seen_in_this_seed.add(nbr_sid_norm)
                expanded_section_ids_dedup.append(nbr_sid_norm)
                if nbr_sid_norm not in sections_by_sid:
                    chunk = _node_chunk(graph, nbr_sid_norm, doc_id)
                    chunk["metadata"]["is_root_seed"] = False
                    chunk["metadata"]["expanded_from_seed_section_ids"] = []
                    sections_by_sid[nbr_sid_norm] = chunk

                seed_list = sections_by_sid[nbr_sid_norm]["metadata"].setdefault(
                    "expanded_from_seed_section_ids", []
                )
                if seed_sid not in seed_list:
                    seed_list.append(seed_sid)

            roots.append(
                {
                    "root_section_id": seed_sid,
                    "expanded_section_ids": expanded_section_ids_dedup,
                }
            )

        # 1) LLM-filter relevant expanded sections for the feature.
        candidate_sections = list(sections_by_sid.values())
        relevant_ids, filter_debug = _llm_filter_relevant_sections(
            llm=llm,
            query_text=query_text,
            section_text_anchor=section_text_anchor,
            candidate_sections=candidate_sections,
            max_keep=90,
        )
        if not relevant_ids:
            relevant_ids = [str(c.get("section_id", "")) for c in candidate_sections if c.get("section_id")]

        relevant_sections_by_sid: Dict[str, Dict[str, Any]] = {}
        for sid in relevant_ids:
            sid_norm = _normalize_section_id(sid)
            if sid_norm in sections_by_sid:
                relevant_sections_by_sid[sid_norm] = sections_by_sid[sid_norm]

        # 2) For each message_format seed, run two-phase agentic IE retrieval:
        #    Phase-1: discover MAIN IE definition nodes
        #    Phase-2: expand recursively/iteratively to nested sub-IE definitions
        for mf in message_format_by_doc.get(doc_id, []):
            mf_section_id = _normalize_section_id(mf.get("section_id"))
            message_name = str(mf.get("message_name", "")).strip()
            if not mf_section_id or mf_section_id not in graph.nodes:
                continue

            message_format_content = graph.nodes[mf_section_id].get("content", "") or ""
            main_seed_nodes = _discover_main_ie_definition_nodes(
                graph=graph,
                query_text=query_text,
                message_name=message_name,
                message_format_content=message_format_content,
                max_nodes=6,
            )
            if not main_seed_nodes:
                # Fallback keeps pipeline alive even if main IE couldn't be found.
                main_seed_nodes = [{"ie_name": _derive_ie_definition_name_from_message(message_name) or "UnknownIEs", "section_id": mf_section_id}]

            discovered_nodes, expansion_trace = _expand_ies_agentically_nodes(
                llm=llm,
                graph=graph,
                query_text=query_text,
                section_text_anchor=section_text_anchor,
                seed_ie_nodes=main_seed_nodes,
                max_iterations=2,
                max_total_nodes=120,
            )

            # Map IE type identifier -> section_id so we can filter chunk content precisely.
            sid_to_ie_type: Dict[str, str] = {}
            for n in discovered_nodes:
                s = _normalize_section_id(n.get("section_id"))
                it = str(n.get("ie_name", "")).strip()
                if s and it:
                    sid_to_ie_type[s] = it

            used_ie_section_ids: Set[str] = set()
            for d in discovered_nodes:
                sid = _normalize_section_id(d.get("section_id"))
                if sid:
                    used_ie_section_ids.add(sid)
            asn1_ie_section_ids_used.update(used_ie_section_ids)

            # Union IE definitions into relevant section set.
            for sid in used_ie_section_ids:
                if sid in graph.nodes:
                    if sid not in relevant_sections_by_sid:
                        chunk = _node_chunk(graph, sid, doc_id)
                        # IMPORTANT: do not dump entire ASN.1 KG section.
                        # For ASN-like chunks, keep ONLY the IE definition that we need.
                        if _is_probably_asn1_definition_text(chunk.get("content", "")) and llm is not None:
                            ie_type_identifier = sid_to_ie_type.get(sid, "")
                            if ie_type_identifier:
                                filtered = _llm_keep_only_ie_definition_from_section(
                                    llm=llm,
                                    query_text=query_text,
                                    section_text_anchor=section_text_anchor,
                                    ie_type_identifier=ie_type_identifier,
                                    section_title=str(chunk.get("section_title", "")),
                                    section_content=str(chunk.get("content", "")),
                                    keep_original_if_empty=True,
                                )
                                chunk["content"] = filtered.get("content", chunk.get("content", ""))
                                chunk.setdefault("metadata", {})
                                chunk["metadata"]["asn_ie_filtered_by_llm"] = True
                                chunk["metadata"]["asn_ie_type_identifier"] = ie_type_identifier
                        relevant_sections_by_sid[sid] = chunk
                else:
                    # Should not happen; skip.
                    pass

            # Store per-message IE chain output.
            main_root = main_seed_nodes[0] if main_seed_nodes else {"ie_name": "", "section_id": ""}
            tree = _build_ie_definition_tree(
                llm=llm,
                graph=graph,
                doc_id=doc_id,
                query_text=query_text,
                section_text_anchor=section_text_anchor,
                message_name=message_name,
                message_format_section_id=mf_section_id,
                root_ie_name=str(main_root.get("ie_name", "")),
                root_section_id=str(main_root.get("section_id", "")),
                max_depth=6,
                max_nodes=80,
            )
            message_ie_chains.append(
                {
                    "message_name": message_name,
                    "message_format_section_id": mf_section_id,
                    "main_ie_chain": tree.get("main_ie_chain", {}),
                    "agentic_ie_discovery": {
                        "main_ie_seed_nodes": main_seed_nodes,
                        "discovered_ie_nodes": discovered_nodes,
                        "expansion_trace": expansion_trace,
                    },
                }
            )

        # Final relevant sections (deterministic order by section_id).
        unique_sections = list(relevant_sections_by_sid.values())
        unique_sections.sort(key=lambda c: str(c.get("section_id", "")))
        all_final_context.extend(unique_sections)

        asn1_ie_section_count = len(asn1_ie_section_ids_used.intersection(set(relevant_sections_by_sid.keys())))

        docs_out.append(
            {
                "doc_id": doc_id,
                "roots": roots,
                "relevant_sections_by_id": {c["section_id"]: c for c in unique_sections},
                "messages": message_ie_chains,
                "missing_root_section_ids": sorted(set(missing_seed_section_ids)),
                "counts": {
                    "root_seed_count": len(roots),
                    "relevant_section_count": len(unique_sections),
                    "asn1_ie_section_count": asn1_ie_section_count,
                },
            }
        )

    # Global dedup across docs (should already be unique per doc_id).
    # Key by (doc_id, section_id)
    seen_global: Set[Tuple[str, str]] = set()
    final_context_dedup: List[Dict[str, Any]] = []
    for c in all_final_context:
        doc = str(c.get("knowledge_source") or c.get("metadata", {}).get("doc_id") or "").strip()
        sid = _normalize_section_id(c.get("section_id"))
        if not doc or not sid:
            continue
        key = (doc, sid)
        if key in seen_global:
            continue
        seen_global.add(key)
        final_context_dedup.append(c)

    # Sort final flat context.
    final_context_dedup.sort(key=lambda c: (str(c.get("knowledge_source", "")), str(c.get("section_id", ""))))

    doc_ids_in_run = [d.get("doc_id") for d in docs_out if d.get("doc_id")]

    total_unique_sections = sum(
        int((d.get("counts", {}) or {}).get("relevant_section_count", 0)) for d in docs_out
    )
    total_asn1_ie_sections = sum(int((d.get("counts", {}) or {}).get("asn1_ie_section_count", 0)) for d in docs_out)
    total_root_seeds = sum(int((d.get("counts", {}) or {}).get("root_seed_count", 0)) for d in docs_out)

    output = {
        # Header fields (top-of-JSON)
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "query": query_text,
        "doc_ids": doc_ids_in_run,
        "counts": {
            "doc_count": len(docs_out),
            "root_seed_count": total_root_seeds,
            "unique_section_count": total_unique_sections,
            "asn1_ie_section_count": total_asn1_ie_sections,
            "final_context_count": len(final_context_dedup),
        },
        "feature_json_path": str(feature_json_path),
        "retrieval_config": {
            "max_depth": max_depth,
            "direction": direction,
            "allowed_relations": sorted(list(allowed_relations)) if allowed_relations else None,
        },
        "docs": docs_out,
        # Optional flat list, useful for downstream template filling.
        "final_context": final_context_dedup,
        # Compatibility alias with retriever-agent style consumers.
        "specs_context": final_context_dedup,
        "specs_context_count": len(final_context_dedup),
    }
    return output


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    # Static / hardcoded inputs (per your instruction: no args).
    FEATURE_JSON_PATH = repo_root / "Inter-gNB-DU_LTM_handover_procedure_20260323_093447.json"
    KG_BASE_DIR = repo_root / "KG_Only_Pipeline" / "spec_chunks"
    OUT_DIR = repo_root / "KG_Retriver" / "spec_chunks" / "retrieval_outputs"

    # Static retrieval config.
    MAX_DEPTH = 2
    DIRECTION = "both"  # incoming + outgoing
    ALLOWED_RELATIONS: Optional[Set[str]] = None  # None => allow all

    output = run_section_retrieval_hierarchy(
        feature_json_path=FEATURE_JSON_PATH,
        kg_base_dir=KG_BASE_DIR,
        max_depth=MAX_DEPTH,
        direction=DIRECTION,
        allowed_relations=ALLOWED_RELATIONS,
    )

    # Save a single final output JSON per run (timestamped).
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Cleanup old outputs so the folder contains only one output JSON.
    for old in OUT_DIR.glob("seed_expanded_kg_context*.json"):
        try:
            old.unlink()
        except Exception:
            # Best-effort cleanup; continue even if a file is locked.
            pass

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"seed_expanded_kg_context_{ts}.json"

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    # Also print (optional) so other scripts can pipe/consume stdout.
    # print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Saved seed-expanded KG context to: {out_path}")


if __name__ == "__main__":
    main()

