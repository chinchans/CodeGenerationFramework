"""
Agentic IE retrieval (KG-only, no embeddings).

End-to-end:
1) Load feature JSON (your Inter-gNB-DU_LTM_handover_procedure_*.json).
2) For each `doc_id` spec, load its KG: `spec_chunks/<doc_id>/KnowledgeGraph/knowledge_graph.pkl`.
3) Seed KG traversal from:
   - `procedure_spec_info.section_id`
   - all `protocol_message_sections[*].messages[*].sections[*].section_id` (all roles)
4) For each `role == "message_format"` section:
   - take the full KG `content` for that section
   - identify the relevant MAIN/parent IE definition
   - recursively discover relevant sub-IE definitions
   - deduplicate per spec (doc_id, section_id)

Output:
- Writes agentic IE definitions context to:
  KG_Only_Pipeline/spec_chunks/retrieval_outputs/agentic_ie_retrieval_context.json
"""

from __future__ import annotations

import json
import pickle
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
IST = timezone(timedelta(hours=5, minutes=30))
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_feature_payload(feature_json_path: Path | Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Accept Feature Validation output either as:
    - an on-disk JSON path (legacy)
    - an in-memory dict (end-to-end pipeline)
    """
    if isinstance(feature_json_path, dict):
        return feature_json_path, "in_memory"
    return _load_json(feature_json_path), str(feature_json_path)


def _normalize_section_id(section_id: Any) -> str:
    return str(section_id or "").strip()


def _safe_upper(s: str) -> str:
    return (s or "").upper()


def _extract_intent_keywords(intent: str) -> List[str]:
    """
    Small heuristic keyword extractor for LTM-related filtering.
    Keep it conservative to avoid dropping valid candidates.
    """
    text = _safe_upper(intent)
    keywords = set()
    if "LTM" in text:
        keywords.add("LTM")
    # Common strings seen in LTM handover / UE context setup.
    for k in [
        "UE CONTEXT",
        "UE CONTEXT SETUP",
        "UE CONTEXT SETUP REQUEST",
        "CSI",
        "PRACH",
        "LOWER LAYER",
        "CONFIGURATION",
        "CONFIG",
        "MAPPING",
        "RESOURCE",
        "CONTEXT",
        "HANDOVER",
    ]:
        if k.replace(" ", "") in text.replace(" ", "") or k in text:
            # Store without collapsing spaces (for substring checks later).
            keywords.add(k)

    # Add 2-4 character tokens and "words" longer than 4.
    words = re.findall(r"[A-Z][A-Z0-9\-]{3,}", text)
    for w in words:
        keywords.add(w)
    return sorted(list(keywords))


def _derive_ie_definition_name_from_message(message_name: str) -> str:
    """
    Example:
      "UE CONTEXT SETUP REQUEST" -> "UEContextSetupRequestIEs"
    """
    words = re.findall(r"[A-Za-z0-9]+", message_name or "")
    camel = "".join(w.capitalize() for w in words if w)
    if not camel:
        return ""
    if not camel.endswith("IEs"):
        return camel + "IEs"
    return camel


def _build_llm() -> Any | None:
    """
    Optional Azure LLM. If credentials aren't present, the script falls back
    to regex-based IE child extraction.
    """
    import os

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        load_dotenv = None

    try:
        from langchain_openai import AzureChatOpenAI
    except ModuleNotFoundError:
        return None

    if load_dotenv is not None:
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


def _truncate_text(text: str, max_chars: int) -> str:
    t = text or ""
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 200] + "\n\n[TRUNCATED]" + t[-200:]


def _llm_extract_main_ie_candidates(
    llm: Any,
    *,
    intent: str,
    template: Dict[str, Any],
    message_name: str,
    message_format_section_content: str,
) -> List[str]:
    """
    Ask LLM to propose MAIN/parent IE names (identifiers ending with IEs).
    """
    prompt = f"""
You are helping retrieve ASN.1 Information Elements (IEs) from a 3GPP spec.

TASK:
Given:
1) intent: {intent}
2) template schema: {json.dumps(template)[:1500]} ...
3) message_name: {message_name}
4) message_format_section_content: (ASN.1-related text)

Identify the relevant MAIN/parent IE definition name(s) that should be used to fill the template for this intent.

Rules:
- Return ONLY a JSON object with key "main_ie_candidates": a list of strings.
- Each string MUST be an IE definition identifier that ends with "IEs" (example: "UEContextSetupRequestIEs").
- Prefer identifiers that are clearly implied by the message_name and intent.
- Do not include generic words; only IE definition identifiers.

message_format_section_content:
{_truncate_text(message_format_section_content, 8000)}
"""
    resp = llm.invoke(prompt)
    raw = resp.content if hasattr(resp, "content") else str(resp)
    raw = raw.strip()
    try:
        data = json.loads(raw)
    except Exception:
        # Try to recover a JSON substring.
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return []
        data = json.loads(m.group(0))
    cands = data.get("main_ie_candidates", []) if isinstance(data, dict) else []
    if not isinstance(cands, list):
        return []
    cleaned = []
    for c in cands:
        cs = str(c).strip()
        if cs and cs.endswith("IEs"):
            cleaned.append(cs)
    # Deduplicate while preserving order.
    seen: Set[str] = set()
    out: List[str] = []
    for x in cleaned:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _llm_extract_child_ie_candidates(
    llm: Any,
    *,
    intent: str,
    parent_ie_name: str,
    parent_ie_definition_text: str,
) -> List[str]:
    """
    Ask LLM to extract likely child/sub-IE definition identifiers from the parent IE definition.
    """
    prompt = f"""
You are parsing ASN.1 definitions for 3GPP IEs.

TASK:
From the ASN.1 definition text below for a parent IE definition "{parent_ie_name}",
extract the child/sub-IE definition identifiers that appear inside it and are relevant for the following intent:
intent: {intent}

Rules:
- Return ONLY JSON: {{"child_ie_candidates": ["..."]}}
- Each candidate MUST be an identifier string that ends with "IEs"
- Candidates must be identifiers that are present (or very clearly referenced) in the definition text.
- If no confident child IEs exist, return an empty list.

ASN.1 parent IE definition text:
{_truncate_text(parent_ie_definition_text, 9000)}
"""
    resp = llm.invoke(prompt)
    raw = resp.content if hasattr(resp, "content") else str(resp)
    raw = raw.strip()
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return []
        data = json.loads(m.group(0))
    cands = data.get("child_ie_candidates", []) if isinstance(data, dict) else []
    if not isinstance(cands, list):
        return []
    cleaned = []
    for c in cands:
        cs = str(c).strip()
        if cs and cs.endswith("IEs"):
            cleaned.append(cs)
    # Deduplicate.
    seen: Set[str] = set()
    out: List[str] = []
    for x in cleaned:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _node_chunk(graph: nx.DiGraph, node_id: str, doc_id: str) -> Dict[str, Any]:
    node = graph.nodes[node_id]
    return {
        "section_id": node_id,
        "section_title": node.get("section_title", ""),
        "content": node.get("content", ""),
        "metadata": {
            "doc_id": doc_id,
            "level": node.get("level"),
            "parent_section_id": node.get("parent_section_id"),
            "child_section_ids": node.get("child_section_ids", []),
            "page_numbers": node.get("page_numbers", []),
            "knowledge_source": doc_id,
        },
    }


def _edge_rel_types(graph: nx.DiGraph, u: str, v: str) -> List[str]:
    attrs = graph.get_edge_data(u, v) or {}
    rels = attrs.get("relationship_types", []) or []
    if isinstance(rels, str):
        return [rels]
    return [str(r) for r in rels]


def _expand_from_seeds(
    graph: nx.DiGraph,
    seed_ids: Set[str],
    *,
    max_depth: int,
    direction: str = "both",
    allowed_relations: Optional[Set[str]] = None,
    doc_id: str,
) -> List[Dict[str, Any]]:
    """
    KG-only expansion: BFS from seed section_ids using graph edges.
    Returns a list of chunk dicts (seed nodes included).
    """
    existing = {sid for sid in seed_ids if sid in graph.nodes}
    initial_chunks = [_node_chunk(graph, sid, doc_id) for sid in sorted(existing)]

    visited: Set[str] = set(existing)
    q = deque([(sid, 0) for sid in sorted(existing)])
    expanded: List[Dict[str, Any]] = []

    while q:
        current, depth = q.popleft()
        if depth >= max_depth:
            continue

        neighbors: List[str] = []
        if direction in {"out", "both"}:
            neighbors.extend(list(graph.successors(current)))
        if direction in {"in", "both"}:
            neighbors.extend(list(graph.predecessors(current)))

        for nbr in neighbors:
            rel = None
            if direction in {"out", "both"} and graph.has_edge(current, nbr):
                rels = _edge_rel_types(graph, current, nbr)
                rel = rels[0] if rels else None
            if direction in {"in", "both"} and graph.has_edge(nbr, current):
                rels = _edge_rel_types(graph, nbr, current)
                rel = rels[0] if rels else rel

            if allowed_relations and rel and rel not in allowed_relations:
                continue
            if nbr in visited:
                continue
            visited.add(nbr)
            expanded.append(_node_chunk(graph, nbr, doc_id))
            q.append((nbr, depth + 1))

    # Dedup returned chunks by node_id, preserve order: seeds first.
    seen: Set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for c in initial_chunks + expanded:
        sid = _normalize_section_id(c.get("section_id"))
        if sid and sid not in seen:
            seen.add(sid)
            deduped.append(c)
    return deduped


def _expand_from_seeds_with_trace(
    graph: nx.DiGraph,
    seed_ids: Set[str],
    *,
    max_depth: int,
    direction: str = "both",
    allowed_relations: Optional[Set[str]] = None,
    doc_id: str,
    max_trace_edges: int = 20000,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Same KG expansion as `_expand_from_seeds`, but also returns a BFS trace graph:
    edges with depth + root_seed attribution.
    """
    existing = {sid for sid in seed_ids if sid in graph.nodes}
    initial_chunks = [_node_chunk(graph, sid, doc_id) for sid in sorted(existing)]

    visited: Set[str] = set(existing)
    q = deque([(sid, 0, sid) for sid in sorted(existing)])  # (current, depth, root_seed)
    expanded: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    while q:
        current, depth, root_seed = q.popleft()
        if depth >= max_depth:
            continue

        neighbors: List[str] = []
        if direction in {"out", "both"}:
            neighbors.extend(list(graph.successors(current)))
        if direction in {"in", "both"}:
            neighbors.extend(list(graph.predecessors(current)))

        for nbr in neighbors:
            rels: List[str] = []
            if direction in {"out", "both"} and graph.has_edge(current, nbr):
                rels = _edge_rel_types(graph, current, nbr)
            elif direction in {"in", "both"} and graph.has_edge(nbr, current):
                rels = _edge_rel_types(graph, nbr, current)

            rel_primary = rels[0] if rels else None
            if allowed_relations and rel_primary and rel_primary not in allowed_relations:
                continue
            if nbr in visited:
                continue

            visited.add(nbr)
            expanded.append(_node_chunk(graph, nbr, doc_id))

            if len(edges) < max_trace_edges:
                edges.append(
                    {
                        "from_section_id": current,
                        "to_section_id": nbr,
                        "root_seed_section_id": root_seed,
                        "depth_level": depth + 1,
                        "relationship_types": rels,
                    }
                )

            q.append((nbr, depth + 1, root_seed))

    # Dedup returned chunks by node_id, preserve order: seeds first.
    seen: Set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for c in initial_chunks + expanded:
        sid = _normalize_section_id(c.get("section_id"))
        if sid and sid not in seen:
            seen.add(sid)
            deduped.append(c)

    trace = {
        "doc_id": doc_id,
        "seeds": sorted(list(existing)),
        "max_depth": max_depth,
        "direction": direction,
        "visited_count": len(visited),
        "edges_count": len(edges),
        "truncated": len(visited) > 0 and len(edges) >= max_trace_edges,
        "edges": edges,
    }
    return deduped, trace


def _compile_ie_regex_patterns(ie_name: str) -> List[re.Pattern[str]]:
    """
    Create regex patterns that match likely ASN.1 definition lines referencing ie_name.
    """
    name = (ie_name or "").strip()
    if not name:
        return []

    # Ensure we search for both exact token and token-with-IEs variations.
    candidates = {name}
    if not name.endswith("IEs"):
        candidates.add(name + "IEs")
    base = name[:-3] if name.endswith("IEs") else name
    if base:
        candidates.add(base + "IEs")

    patterns: List[re.Pattern[str]] = []
    for c in sorted(candidates):
        esc = re.escape(c)
        # Match definition signature snippets.
        patterns.append(re.compile(rf"\b{esc}\b\s+.*PROTOCOL-IES", re.IGNORECASE))
        patterns.append(re.compile(rf"\b{esc}\b\s*::=", re.IGNORECASE))
        patterns.append(re.compile(rf"\b{esc}\b", re.IGNORECASE))
    return patterns


def _find_definition_nodes_for_ie(graph: nx.DiGraph, *, doc_id: str, ie_name: str) -> List[str]:
    """
    Deterministic lookup: scan node.content for ie_name and ASN.1 definition markers.
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
        for pat in patterns:
            if pat.search(content):
                # Also ensure it looks like ASN.1 definition-ish.
                # (Keeps precision higher without needing embeddings.)
                if "::=" in content or "PROTOCOL-IES" in content or "PROTOCOL-IES" in _safe_upper(content):
                    hits.append(str(node_id))
                    break
    # Dedup.
    seen: Set[str] = set()
    out: List[str] = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


@dataclass(frozen=True)
class MessageFormatSeed:
    doc_id: str
    message_name: str
    section_id: str
    role: str
    reason: str


def _collect_doc_id_map(payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Map spec_number -> doc_id for stable graph loading.
    """
    out: Dict[str, str] = {}
    for spec in payload.get("specs", []) or []:
        spec_number = str(spec.get("spec_number", "")).strip()
        doc_id = str(spec.get("doc_id", "")).strip()
        if spec_number and doc_id:
            out[spec_number.upper()] = doc_id
    return out


def _collect_seeds(payload: Dict[str, Any]) -> Tuple[Dict[str, Set[str]], List[MessageFormatSeed]]:
    """
    Returns:
      - all_seed_section_ids_by_doc: doc_id -> {section_id...} across all roles
      - message_format_seeds: list of MessageFormatSeed entries
    """
    spec_number_to_doc = _collect_doc_id_map(payload)

    all_seed_section_ids_by_doc: Dict[str, Set[str]] = {}
    message_format_seeds: List[MessageFormatSeed] = []

    # 1) procedure_spec_info seed(s)
    proc = payload.get("procedure_spec_info", {}) or {}
    proc_section_id = _normalize_section_id(proc.get("section_id"))
    proc_spec_number = str(proc.get("spec_number", "")).strip().upper()
    if proc_section_id and proc_spec_number in spec_number_to_doc:
        doc_id = spec_number_to_doc[proc_spec_number]
        all_seed_section_ids_by_doc.setdefault(doc_id, set()).add(proc_section_id)

    # 2) protocol_message_sections seeds for all roles + message_format seeds
    for block in payload.get("protocol_message_sections", []) or []:
        block_spec_number = str(block.get("spec_number", "")).strip().upper()
        doc_id = spec_number_to_doc.get(block_spec_number, "")
        if not doc_id:
            continue

        for msg in block.get("messages", []) or []:
            message_name = str(msg.get("message_name", "")).strip()
            for sec in msg.get("sections", []) or []:
                sid = _normalize_section_id(sec.get("section_id"))
                if not sid:
                    continue
                all_seed_section_ids_by_doc.setdefault(doc_id, set()).add(sid)

                role = str(sec.get("role", "")).strip()
                reason = str(sec.get("reason", "")).strip()
                if role == "message_format":
                    message_format_seeds.append(
                        MessageFormatSeed(
                            doc_id=doc_id,
                            message_name=message_name,
                            section_id=sid,
                            role=role,
                            reason=reason,
                        )
                    )

    return all_seed_section_ids_by_doc, message_format_seeds


def _dedup_chunks_by_doc_and_section(chunks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[str, str]] = set()
    out: List[Dict[str, Any]] = []
    for c in chunks:
        sid = _normalize_section_id(c.get("section_id"))
        doc_id = str(c.get("metadata", {}).get("doc_id", "")).strip()
        if not sid or not doc_id:
            continue
        key = (doc_id, sid)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _fallback_extract_child_ie_names_regex(parent_ie_definition_text: str) -> List[str]:
    """
    Extract likely child IE names deterministically from definition text.
    """
    text = parent_ie_definition_text or ""
    # Common pattern: tokens like SomethingIEs
    cands = set(re.findall(r"\b[A-Za-z0-9]+IEs\b", text))
    return sorted(list(cands))


def _recursive_ie_expansion_from_main_node(
    *,
    llm: Any | None,
    graph: nx.DiGraph,
    doc_id: str,
    intent: str,
    template: Dict[str, Any],
    message_format_context: str,
    main_node_ids: Sequence[str],
    max_depth: int,
    max_nodes: int,
) -> List[Dict[str, Any]]:
    """
    Recursively expand definitions by extracting child IE identifiers from each
    parent IE definition text, then locating matching definition nodes.
    """
    visited_nodes: Set[str] = set()
    out_chunks: List[Dict[str, Any]] = []

    # Cache: parent_node_id -> child_ie_candidates
    children_cache: Dict[str, List[str]] = {}

    queue: deque[Tuple[str, int]] = deque()
    for nid in main_node_ids:
        if nid in graph.nodes:
            queue.append((nid, 0))

    while queue:
        node_id, depth = queue.popleft()
        if node_id in visited_nodes:
            continue
        if len(out_chunks) >= max_nodes:
            break
        if depth > max_depth:
            continue

        visited_nodes.add(node_id)

        chunk = _node_chunk(graph, node_id, doc_id)
        out_chunks.append(chunk)

        parent_ie_definition_text = chunk.get("content", "") or ""
        # Extract a best-effort parent IE name token.
        # (For prompt context only; recursion is node-driven.)
        parent_ie_name_guess = ""
        m = re.search(r"\b([A-Za-z0-9]+IEs)\b", parent_ie_definition_text)
        if m:
            parent_ie_name_guess = m.group(1)

        if node_id not in children_cache:
            if llm is not None:
                child_ie_candidates = _llm_extract_child_ie_candidates(
                    llm,
                    intent=intent,
                    parent_ie_name=parent_ie_name_guess or "UnknownIEs",
                    parent_ie_definition_text=parent_ie_definition_text,
                )
            else:
                child_ie_candidates = _fallback_extract_child_ie_names_regex(
                    parent_ie_definition_text
                )
            children_cache[node_id] = child_ie_candidates

        child_ie_candidates = children_cache[node_id]

        # Locate definition nodes for each child IE name.
        for child_ie_name in child_ie_candidates:
            found_nodes = _find_definition_nodes_for_ie(graph, doc_id=doc_id, ie_name=child_ie_name)
            for fn in found_nodes:
                if fn not in visited_nodes:
                    queue.append((fn, depth + 1))

            # Stop early if we've grown large.
            if len(out_chunks) >= max_nodes:
                break

    return out_chunks


def _recursive_ie_expansion_from_main_node_with_trace(
    *,
    llm: Any | None,
    graph: nx.DiGraph,
    doc_id: str,
    intent: str,
    template: Dict[str, Any],
    message_format_context: str,
    main_node_ids: Sequence[str],
    max_depth: int,
    max_nodes: int,
    max_trace_edges: int = 5000,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Same recursive IE expansion as `_recursive_ie_expansion_from_main_node`, but also
    returns a trace of how child/sub-IE candidates lead to discovered definition nodes.
    """
    visited_nodes: Set[str] = set()
    out_chunks: List[Dict[str, Any]] = []

    # Cache: parent_node_id -> child_ie_candidates
    children_cache: Dict[str, List[str]] = {}

    queue: deque[Tuple[str, int]] = deque()
    for nid in main_node_ids:
        if nid in graph.nodes:
            queue.append((nid, 0))

    edges: List[Dict[str, Any]] = []

    while queue:
        node_id, depth = queue.popleft()
        if node_id in visited_nodes:
            continue
        if len(out_chunks) >= max_nodes:
            break
        if depth > max_depth:
            continue

        visited_nodes.add(node_id)

        chunk = _node_chunk(graph, node_id, doc_id)
        out_chunks.append(chunk)

        parent_ie_definition_text = chunk.get("content", "") or ""
        parent_ie_name_guess = ""
        m = re.search(r"\b([A-Za-z0-9]+IEs)\b", parent_ie_definition_text)
        if m:
            parent_ie_name_guess = m.group(1)

        if node_id not in children_cache:
            if llm is not None:
                child_ie_candidates = _llm_extract_child_ie_candidates(
                    llm,
                    intent=intent,
                    parent_ie_name=parent_ie_name_guess or "UnknownIEs",
                    parent_ie_definition_text=parent_ie_definition_text,
                )
            else:
                child_ie_candidates = _fallback_extract_child_ie_names_regex(
                    parent_ie_definition_text
                )
            children_cache[node_id] = child_ie_candidates

        child_ie_candidates = children_cache[node_id]

        for child_ie_name in child_ie_candidates:
            found_nodes = _find_definition_nodes_for_ie(
                graph, doc_id=doc_id, ie_name=child_ie_name
            )
            for fn in found_nodes:
                if fn not in visited_nodes:
                    queue.append((fn, depth + 1))

                if len(edges) < max_trace_edges:
                    edges.append(
                        {
                            "from_ie_definition_node_id": node_id,
                            "via_child_ie_name": child_ie_name,
                            "to_ie_definition_node_id": fn,
                            "depth_level": depth + 1,
                        }
                    )

            if len(out_chunks) >= max_nodes:
                break

    trace = {
        "doc_id": doc_id,
        "message_format_context_truncated_chars": len(_truncate_text(message_format_context, 4000)),
        "main_node_ids": list(main_node_ids),
        "visited_node_count": len(visited_nodes),
        "edges_count": len(edges),
        "truncated": len(edges) >= max_trace_edges,
        "edges": edges,
        "visited_node_ids": sorted(list(visited_nodes)) if len(visited_nodes) <= 500 else [],
    }
    return out_chunks, trace


def run_agentic_ie_retrieval_phase(
    *,
    feature_json_path: Path | Dict[str, Any],
    template_path: Path,
    kg_base_dir: Path,
    max_depth_kg_expand: int = 2,
    llm_iexpand_max_depth: int = 2,
    llm_iexpand_max_nodes: int = 60,
) -> Dict[str, Any]:
    payload, feature_source = _load_feature_payload(feature_json_path)
    intent = str(payload.get("intent", "")).strip()
    template = _load_json(template_path)

    all_seed_section_ids_by_doc, message_format_seeds = _collect_seeds(payload)

    llm = _build_llm()

    retrieval_config = {
        "max_depth_kg_expand": max_depth_kg_expand,
        "llm_iexpand_max_depth": llm_iexpand_max_depth,
        "llm_iexpand_max_nodes": llm_iexpand_max_nodes,
        "llm_available": llm is not None,
    }

    # Debug traces that can be visualized as JSON graphs/trees.
    kg_expansion_trace_by_doc_id: Dict[str, Any] = {}
    ie_recursive_trace_by_message: List[Dict[str, Any]] = []

    # Expand context per doc_id from all seeds (all roles).
    per_doc_expanded_context: Dict[str, List[Dict[str, Any]]] = {}
    for doc_id, seed_ids in all_seed_section_ids_by_doc.items():
        graph_path = kg_base_dir / doc_id / "KnowledgeGraph" / "knowledge_graph.pkl"
        if not graph_path.exists():
            continue
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        if not isinstance(graph, nx.DiGraph):
            continue
        expanded_chunks, kg_trace = _expand_from_seeds_with_trace(
            graph,
            seed_ids,
            max_depth=max_depth_kg_expand,
            direction="both",
            allowed_relations=None,
            doc_id=doc_id,
            max_trace_edges=25000,
        )
        per_doc_expanded_context[doc_id] = expanded_chunks
        kg_expansion_trace_by_doc_id[doc_id] = kg_trace

    # Agentic IE extraction per message_format seed.
    per_doc_ie_chunks: Dict[str, List[Dict[str, Any]]] = {}
    intent_keywords = _extract_intent_keywords(intent)

    for seed in message_format_seeds:
        graph_path = kg_base_dir / seed.doc_id / "KnowledgeGraph" / "knowledge_graph.pkl"
        if not graph_path.exists():
            continue

        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        if not isinstance(graph, nx.DiGraph):
            continue

        if seed.section_id not in graph.nodes:
            continue

        message_format_chunk = _node_chunk(graph, seed.section_id, seed.doc_id)
        message_format_content = message_format_chunk.get("content", "") or ""

        # Derive candidate main IE names from message name (deterministic heuristic).
        derived_main_ie_name = _derive_ie_definition_name_from_message(seed.message_name)
        main_ie_candidates: List[str] = []
        if derived_main_ie_name:
            main_ie_candidates.append(derived_main_ie_name)

        # If that fails, use LLM to propose candidates.
        found_main_nodes: List[str] = []
        for cand in main_ie_candidates:
            hits = _find_definition_nodes_for_ie(graph, doc_id=seed.doc_id, ie_name=cand)
            found_main_nodes.extend(hits)

        found_main_nodes = sorted(list(set(found_main_nodes)))

        if not found_main_nodes and llm is not None:
            llm_cands = _llm_extract_main_ie_candidates(
                llm,
                intent=intent,
                template=template,
                message_name=seed.message_name,
                message_format_section_content=message_format_content,
            )
            for cand in llm_cands:
                hits = _find_definition_nodes_for_ie(graph, doc_id=seed.doc_id, ie_name=cand)
                found_main_nodes.extend(hits)
            found_main_nodes = sorted(list(set(found_main_nodes)))

        if not found_main_nodes:
            # As a last resort, pick the seed node itself (agentic recursion may still
            # reveal embedded ASN.1 identifiers).
            found_main_nodes = [seed.section_id]

        main_chunks, ie_trace = _recursive_ie_expansion_from_main_node_with_trace(
            llm=llm,
            graph=graph,
            doc_id=seed.doc_id,
            intent=intent,
            template=template,
            message_format_context=message_format_content,
            main_node_ids=found_main_nodes[:3],  # limit branching
            max_depth=llm_iexpand_max_depth,
            max_nodes=llm_iexpand_max_nodes,
            max_trace_edges=6000,
        )
        ie_trace.update(
            {
                "message_name": seed.message_name,
                "message_format_section_id": seed.section_id,
                "message_format_reason": seed.reason,
            }
        )

        # Optional keyword filter: keep nodes that look LTM-related.
        # But do not over-filter: if none match, keep all.
        if intent_keywords:
            filtered = []
            for c in main_chunks:
                cc = _safe_upper(c.get("content", ""))
                if any(k.replace(" ", "") in cc.replace(" ", "") for k in intent_keywords):
                    filtered.append(c)
            if filtered:
                main_chunks = filtered

        ie_trace["filtered_ie_definition_section_ids"] = [
            _normalize_section_id(c.get("section_id")) for c in main_chunks if _normalize_section_id(c.get("section_id"))
        ]
        ie_recursive_trace_by_message.append(ie_trace)

        per_doc_ie_chunks.setdefault(seed.doc_id, []).extend(main_chunks)

    # Final dedup and merge context.
    all_ie_chunks: List[Dict[str, Any]] = []
    for chunks in per_doc_ie_chunks.values():
        all_ie_chunks.extend(chunks)

    dedup_ie_chunks = _dedup_chunks_by_doc_and_section(all_ie_chunks)

    # Also include KG-expanded context snippets from all seeds (helps template fill).
    expanded_all: List[Dict[str, Any]] = []
    for chunks in per_doc_expanded_context.values():
        expanded_all.extend(chunks)
    dedup_expanded = _dedup_chunks_by_doc_and_section(expanded_all)

    final_context = _dedup_chunks_by_doc_and_section(dedup_ie_chunks + dedup_expanded)

    return {
        "feature_json_path": feature_source,
        "template_path": str(template_path),
        "intent": intent,
        "timestamp": datetime.now(IST).isoformat(),
        "retrieval_config": retrieval_config,
        "kg_expansion_trace_by_doc_id": kg_expansion_trace_by_doc_id,
        "ie_recursive_trace_by_message": ie_recursive_trace_by_message,
        "per_doc_ie_definition_chunks": {
            doc_id: [c for c in _dedup_chunks_by_doc_and_section(chunks)]
            for doc_id, chunks in per_doc_ie_chunks.items()
        },
        "final_context_count": len(final_context),
        "final_context": final_context,
    }


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent  # KG_Only_Pipeline
    REPO_ROOT = SCRIPT_DIR.parent

    FEATURE_JSON_PATH = REPO_ROOT / "Inter-gNB-DU_LTM_handover_procedure_20260323_093447.json"
    TEMPLATE_PATH = REPO_ROOT / "inputs" / "Template.json"

    KG_BASE_DIR = SCRIPT_DIR / "spec_chunks"

    RUN_TS = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
    OUT_DIR = SCRIPT_DIR / "spec_chunks" / "retrieval_outputs"
    out_path = OUT_DIR / f"agentic_ie_retrieval_context_{RUN_TS}.json"
    latest_path = OUT_DIR / "agentic_ie_retrieval_context.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = run_agentic_ie_retrieval_phase(
        feature_json_path=FEATURE_JSON_PATH,
        template_path=TEMPLATE_PATH,
        kg_base_dir=KG_BASE_DIR,
        max_depth_kg_expand=2,
        llm_iexpand_max_depth=2,
        llm_iexpand_max_nodes=60,
    )

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    # Update "latest" pointer (useful for downstream modules that expect a fixed filename).
    latest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("KG-only agentic IE retrieval completed.")
    print(f"Output: {out_path}")
    print(f"Latest pointer: {latest_path}")

