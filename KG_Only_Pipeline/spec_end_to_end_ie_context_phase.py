"""
End-to-end IE context retrieval (KG expansion + cosine filtering + agentic IE chain extraction).

What this phase does (no template filling):
1) Load feature validation JSON.
2) Collect all seed section_ids across all roles (incl. every message_format section).
3) Expand KG from seeds up to KG depth=2 (recall).
4) For each message_format section, score expanded KG candidates with cosine similarity
   against the feature-validation intent/section_text (precision gate).
5) Ground each message_format section to an IE chain:
   - pick MAIN/parent IE definition nodes
   - recursively discover child/sub-IE definitions until leaf nodes
   - cycle protection + max depth/nodes
6) Output a JSON "context package" that downstream Template Orchestrator can use as `specs_context`.

Output contract (recommended):
- `specs_context`: List[chunk_dict] deduped by (knowledge_source, section_id)
  Chunk dict fields:
    - section_id, section_title, content
    - knowledge_source (top-level) = doc_id
    - metadata (optional) with doc_id, level, etc.

Additionally includes:
- `message_format_groundings`: message-level grounding trace
"""

from __future__ import annotations

import json
import pickle
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_section_id(section_id: Any) -> str:
    return str(section_id or "").strip()


def _safe_upper(s: str) -> str:
    return (s or "").upper()


def _truncate_text(text: str, max_chars: int) -> str:
    t = text or ""
    if len(t) <= max_chars:
        return t
    # Keep both head and tail; spec definitions often contain both signature and nested structure.
    head = max(0, max_chars - 200)
    return t[:head] + "\n\n[TRUNCATED]\n...\n" + t[-200:]


def _build_llm() -> Any | None:
    """
    Optional Azure OpenAI LLM.
    If credentials aren't present, returns None.
    """
    import os

    from dotenv import load_dotenv
    from langchain_openai import AzureChatOpenAI

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
""".strip()

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

    cands = data.get("main_ie_candidates", []) if isinstance(data, dict) else []
    if not isinstance(cands, list):
        return []

    cleaned: List[str] = []
    for c in cands:
        cs = str(c).strip()
        if cs and cs.endswith("IEs"):
            cleaned.append(cs)

    # Dedup while preserving order.
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
""".strip()

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

    cleaned: List[str] = []
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


def _extract_intent_keywords(intent: str) -> List[str]:
    """
    Conservative heuristic keyword extractor for filtering IE candidates.
    Keep it permissive: we only use it to avoid obviously irrelevant IEs.
    """
    text = _safe_upper(intent)
    keywords: Set[str] = set()

    # Common LTM-related tokens.
    if "LTM" in text:
        keywords.add("LTM")
    for k in [
        "UE CONTEXT",
        "UE CONTEXT SETUP",
        "UE CONTEXT SETUP REQUEST",
        "CSI",
        "PRACH",
        "LOWER LAYER",
        "CONFIGURATION",
        "MAPPING",
        "RESOURCE",
        "CONTEXT",
        "HANDOVER",
    ]:
        compact_k = k.replace(" ", "")
        if compact_k and compact_k in text.replace(" ", ""):
            keywords.add(k)

    # Add 2-4 character tokens and "words" longer than 4.
    words = re.findall(r"[A-Z][A-Z0-9\\-]{3,}", text)
    for w in words:
        keywords.add(w)

    # Stabilize ordering.
    return sorted(keywords)


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


def _compile_ie_regex_patterns(ie_name: str) -> List[re.Pattern[str]]:
    name = (ie_name or "").strip()
    if not name:
        return []

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
        patterns.append(re.compile(rf"\\b{esc}\\b\\s+.*PROTOCOL-IES", re.IGNORECASE))
        patterns.append(re.compile(rf"\\b{esc}\\b\\s*::=", re.IGNORECASE))
        patterns.append(re.compile(rf"\\b{esc}\\b", re.IGNORECASE))
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
                if "::=" in content or "PROTOCOL-IES" in content or "PROTOCOL-IES" in _safe_upper(content):
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


def _node_chunk(graph: nx.DiGraph, node_id: str, doc_id: str) -> Dict[str, Any]:
    node = graph.nodes[node_id]
    chunk: Dict[str, Any] = {
        "section_id": node_id,
        "section_title": node.get("section_title", ""),
        "content": node.get("content", ""),
        # Template Orchestrator prefers top-level knowledge_source.
        "knowledge_source": doc_id,
        "metadata": {
            "doc_id": doc_id,
            "level": node.get("level"),
            "parent_section_id": node.get("parent_section_id"),
            "child_section_ids": node.get("child_section_ids", []),
            "page_numbers": node.get("page_numbers", []),
            "knowledge_source": doc_id,
        },
    }
    return chunk


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
    KG recall expansion (BFS) from seed section_ids.
    Returns list of chunk dicts (seed nodes included).
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


def _dedup_chunks(chunks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[str, str]] = set()
    out: List[Dict[str, Any]] = []
    for c in chunks:
        sid = _normalize_section_id(c.get("section_id"))
        doc_id = str(c.get("knowledge_source") or c.get("metadata", {}).get("doc_id") or "").strip()
        if not sid or not doc_id:
            continue
        key = (doc_id, sid)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _safe_hf_embeddings(model_name: str = "BAAI/bge-base-en-v1.5") -> Any:
    """
    Build embeddings instance.
    If embedding dependencies are missing, raise a helpful error.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model_name)


def _cosine_sim_matrix(query_vec: np.ndarray, cand_vecs: np.ndarray) -> np.ndarray:
    """
    query_vec: (d,)
    cand_vecs: (n,d)
    returns: (n,)
    """
    q = query_vec.astype("float32")
    c = cand_vecs.astype("float32")
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return np.zeros((c.shape[0],), dtype="float32")
    q = q / q_norm
    c_norm = np.linalg.norm(c, axis=1, keepdims=True)
    c_norm[c_norm == 0] = 1.0
    c = c / c_norm
    return np.dot(c, q)


def _score_candidates_cosine(
    *,
    embeddings: Any,
    query_text: str,
    candidates: List[Dict[str, Any]],
    candidate_content_key: str = "content",
    max_candidate_text_chars: int = 2000,
    top_k: int = 60,
    min_score: Optional[float] = None,
    truncate_query_chars: int = 4000,
) -> List[Dict[str, Any]]:
    """
    Compute cosine similarity between query_text and candidate chunk content.
    Adds:
      - semantic_score
      - rank
    """
    q_text = _truncate_text(query_text or "", truncate_query_chars)
    if not candidates:
        return []

    # Prepare candidate texts (truncate to reduce compute).
    cand_texts: List[str] = []
    for c in candidates:
        t = c.get(candidate_content_key, "") or ""
        cand_texts.append(_truncate_text(t, max_candidate_text_chars))

    # Embed query and candidates in batches via HF embeddings.
    query_vec = np.array(embeddings.embed_query(q_text), dtype="float32")
    cand_vecs = np.array(embeddings.embed_documents(cand_texts), dtype="float32")

    scores = _cosine_sim_matrix(query_vec, cand_vecs)
    scored: List[Dict[str, Any]] = []
    for i, (c, s) in enumerate(zip(candidates, scores.tolist())):
        if min_score is not None and s < float(min_score):
            continue
        cc = c.copy()
        cc["semantic_score"] = round(float(s), 6)
        scored.append(cc)

    scored.sort(key=lambda x: x.get("semantic_score", 0.0), reverse=True)
    # Rank after sorting.
    for i, c in enumerate(scored[:top_k], 1):
        c["rank"] = i
    return scored[:top_k]


@dataclass(frozen=True)
class MessageFormatSeed:
    doc_id: str
    message_name: str
    section_id: str
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
                            reason=reason,
                        )
                    )

    return all_seed_section_ids_by_doc, message_format_seeds


def _fallback_extract_child_ie_names_regex(parent_ie_definition_text: str) -> List[str]:
    """
    Extract likely child/sub-IE identifiers deterministically from parent IE definition text.
    """
    text = parent_ie_definition_text or ""
    # Common pattern: tokens like SomethingIEs.
    cands = set(re.findall(r"\\b[A-Za-z0-9]+IEs\\b", text))
    return sorted(list(cands))


def _filter_ie_candidates_by_keywords(ie_names: List[str], *, keywords: List[str]) -> List[str]:
    if not keywords:
        return ie_names
    if not ie_names:
        return []
    kw_compact = [k.replace(" ", "") for k in keywords if k and k.strip()]
    out: List[str] = []
    seen: Set[str] = set()
    for name in ie_names:
        n = _safe_upper(name).replace(" ", "")
        ok = any(kw.replace(" ", "") in n for kw in kw_compact)
        if ok and name not in seen:
            seen.add(name)
            out.append(name)
    # If filter is too aggressive, keep original list.
    return out if out else ie_names


def _recursive_ie_chain_extraction(
    *,
    graph: nx.DiGraph,
    doc_id: str,
    intent_text: str,
    message_format_context: str,
    main_node_ids: Sequence[str],
    max_depth: int,
    max_nodes: int,
    llm: Any | None = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Recursively discover IE definition nodes:
    - Start from MAIN/parent node(s)
    - Extract child IE identifiers from definition text (regex)
    - Locate corresponding definition nodes by scanning KG contents
    - Repeat until max depth/nodes

    Returns:
      (extracted_chunks, visited_node_ids)
    """
    visited_nodes: Set[str] = set()
    out_chunks: List[Dict[str, Any]] = []
    # Track an IE name hint per node so the LLM can anchor the parent IE extraction.
    queue: deque[Tuple[str, int, str]] = deque()
    for nid in main_node_ids:
        if nid in graph.nodes:
            queue.append((nid, 0, ""))

    children_cache: Dict[str, List[str]] = {}
    ie_name_hint_cache: Dict[str, str] = {}

    while queue:
        node_id, depth, ie_name_hint = queue.popleft()
        if node_id in visited_nodes:
            continue
        if len(out_chunks) >= max_nodes:
            break
        if depth > max_depth:
            continue

        visited_nodes.add(node_id)

        chunk = _node_chunk(graph, node_id, doc_id)
        out_chunks.append(chunk)

        parent_text = chunk.get("content", "") or ""

        if node_id not in children_cache:
            # Prefer LLM extraction for IE-to-child IE identifiers.
            if llm is not None:
                # Derive (or reuse) parent IE name hint for better LLM anchoring.
                if node_id in ie_name_hint_cache and ie_name_hint_cache[node_id]:
                    parent_ie_name_guess = ie_name_hint_cache[node_id]
                elif ie_name_hint:
                    parent_ie_name_guess = ie_name_hint
                else:
                    m = re.search(r"\b([A-Za-z0-9]+IEs)\b", parent_text)
                    parent_ie_name_guess = m.group(1) if m else "UnknownIEs"

                ie_name_hint_cache[node_id] = parent_ie_name_guess
                child_ie_candidates = _llm_extract_child_ie_candidates(
                    llm,
                    intent=intent_text,
                    parent_ie_name=parent_ie_name_guess,
                    parent_ie_definition_text=parent_text,
                )
                # If LLM yields nothing, fall back to deterministic regex.
                if not child_ie_candidates:
                    child_ie_candidates = _fallback_extract_child_ie_names_regex(parent_text)
            else:
                child_ie_candidates = _fallback_extract_child_ie_names_regex(parent_text)

            # Limit branching to avoid context blow-up, but allow enough breadth.
            children_cache[node_id] = child_ie_candidates[:25]

        for child_ie_name in children_cache[node_id]:
            found_nodes = _find_definition_nodes_for_ie(graph, doc_id=doc_id, ie_name=child_ie_name)
            for fn in found_nodes:
                if fn not in visited_nodes:
                    queue.append((fn, depth + 1, child_ie_name))

            if len(out_chunks) >= max_nodes:
                break

    return out_chunks, sorted(list(visited_nodes))


def _find_main_ie_candidates(
    graph: nx.DiGraph,
    *,
    doc_id: str,
    message_name: str,
    message_format_content: str,
    intent_text: str,
    keywords: List[str],
    max_candidates: int = 5,
    llm: Any | None = None,
    template: Dict[str, Any] | None = None,
) -> List[str]:
    """
    Determine MAIN/parent IE definition candidate nodes.
    Primary strategy:
      - derived camel+IEs token from message_name
    Fallback:
      - any IEs identifiers found inside message_format_content (filtered by keywords if possible)
    """
    derived_main_ie_name = _derive_ie_definition_name_from_message(message_name)

    ie_name_candidates: List[str] = []
    if derived_main_ie_name:
        ie_name_candidates.append(derived_main_ie_name)

    # Pull any IEs identifiers from message_format content (heuristic fallback).
    extracted = set(re.findall(r"\\b[A-Za-z0-9]+IEs\\b", message_format_content or ""))
    if extracted:
        extracted_list = sorted(list(extracted))
        # Keep heuristic extracted candidates; do not over-filter here because
        # the main IE selection is sensitive and should be assisted by LLM.
        ie_name_candidates.extend(extracted_list)

    # Dedup while preserving order.
    seen: Set[str] = set()
    ie_name_candidates = [x for x in ie_name_candidates if not (x in seen or seen.add(x))]

    # Locate matching definition nodes for each IE name.
    found_nodes: List[str] = []
    for ie_name in ie_name_candidates:
        nodes = _find_definition_nodes_for_ie(graph, doc_id=doc_id, ie_name=ie_name)
        found_nodes.extend(nodes)
        if len(found_nodes) >= max_candidates:
            break

    # If nothing was found, ask LLM to propose MAIN/parent IE identifiers.
    if not found_nodes and llm is not None:
        llm_template = template or {}
        llm_candidates = _llm_extract_main_ie_candidates(
            llm,
            intent=intent_text,
            template=llm_template,
            message_name=message_name,
            message_format_section_content=message_format_content,
        )
        for cand in llm_candidates:
            hits = _find_definition_nodes_for_ie(graph, doc_id=doc_id, ie_name=cand)
            found_nodes.extend(hits)
            if len(found_nodes) >= max_candidates:
                break

    # Dedup while preserving order.
    seen_nodes: Set[str] = set()
    out: List[str] = []
    for n in found_nodes:
        if n not in seen_nodes:
            seen_nodes.add(n)
            out.append(n)
    return out


def run_end_to_end_ie_context_phase(
    *,
    feature_json_path: Path,
    kg_base_dir: Path,
    output_path: Path,
    template_path: Optional[Path] = None,
    max_depth_kg_expand: int = 2,
    cosine_top_k: int = 80,
    cosine_min_score: Optional[float] = None,
    cosine_max_candidate_text_chars: int = 2000,
    kg_expansion_direction: str = "both",
    recursive_ie_max_depth: int = 2,
    recursive_ie_max_nodes: int = 80,
    max_candidates_for_scoring: int = 2500,
    cosine_embedding_model: str = "BAAI/bge-base-en-v1.5",
) -> Dict[str, Any]:
    """
    Standalone function to generate enriched KG context package as JSON.
    """
    def _now_utc() -> str:
        # ISO8601 with seconds for compact logs.
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def _log(msg: str) -> None:
        print(f"[{_now_utc()}] {msg}")

    payload = _load_json(feature_json_path)
    _log(f"Loaded feature JSON: {feature_json_path}")
    intent_text = str(payload.get("intent", "")).strip() or str(payload.get("message_details", {}).get("message", "")).strip()

    # Use provided feature section_text as the precision-gating query anchor.
    feature_section_text = str(payload.get("section_text", "") or "")
    global_query_text = (payload.get("intent", "") or "").strip()
    if feature_section_text:
        global_query_text = f"{global_query_text}\n\n{feature_section_text}".strip()

    all_seed_section_ids_by_doc, message_format_seeds = _collect_seeds(payload)
    _log(
        "Collected seeds: "
        f"docs={len(all_seed_section_ids_by_doc)}, "
        f"total_seed_sections={sum(len(v) for v in all_seed_section_ids_by_doc.values())}, "
        f"message_format_seeds={len(message_format_seeds)}"
    )
    embeddings = None
    try:
        embeddings = _safe_hf_embeddings(model_name=cosine_embedding_model)
        _log(f"Initialized embeddings: {cosine_embedding_model}")
    except Exception as e:
        # If embeddings are unavailable, we still produce output with no cosine filtering.
        embeddings = None
        _log(f"WARNING: embeddings not available, cosine filtering disabled. Error: {str(e)[:200]}")

    # Optional LLM for agentic IE discovery (MAIN + child/sub-IE).
    llm = _build_llm()
    if llm is None:
        _log("WARNING: LLM not available, recursive IE discovery will use deterministic regex only.")
    else:
        _log("Initialized LLM for recursive IE discovery.")

    template: Dict[str, Any] = {}
    if template_path and template_path.exists():
        try:
            template = json.loads(template_path.read_text(encoding="utf-8"))
            _log(f"Loaded template schema for LLM prompts: {template_path}")
        except Exception as e:
            _log(f"WARNING: failed to load template schema ({template_path}). Error: {str(e)[:200]}")

    # Expand KG per doc_id from all seeds.
    per_doc_expanded_candidates: Dict[str, List[Dict[str, Any]]] = {}
    for doc_id, seed_ids in all_seed_section_ids_by_doc.items():
        graph_path = kg_base_dir / doc_id / "KnowledgeGraph" / "knowledge_graph.pkl"
        if not graph_path.exists():
            _log(f"KG graph not found (skip): {graph_path}")
            continue
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        if not isinstance(graph, nx.DiGraph):
            _log(f"KG graph invalid type (skip): {graph_path}")
            continue

        _log(
            f"Expanding KG: doc_id={doc_id}, seeds={len(seed_ids)}, "
            f"depth={max_depth_kg_expand}, nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}"
        )
        expanded = _expand_from_seeds(
            graph,
            seed_ids,
            max_depth=max_depth_kg_expand,
            direction=kg_expansion_direction,
            allowed_relations=None,
            doc_id=doc_id,
        )
        _log(f"KG expanded: doc_id={doc_id}, expanded_sections={len(expanded)}")
        per_doc_expanded_candidates[doc_id] = expanded

    # Index KG graphs per doc for fast lookup during IE extraction.
    per_doc_graph: Dict[str, nx.DiGraph] = {}
    for doc_id in per_doc_expanded_candidates.keys():
        graph_path = kg_base_dir / doc_id / "KnowledgeGraph" / "knowledge_graph.pkl"
        if not graph_path.exists():
            continue
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        if isinstance(graph, nx.DiGraph):
            per_doc_graph[doc_id] = graph

    # Ground per message_format seed.
    message_format_groundings: List[Dict[str, Any]] = []
    specs_context_chunks: List[Dict[str, Any]] = []

    for seed in message_format_seeds:
        graph = per_doc_graph.get(seed.doc_id)
        candidates = per_doc_expanded_candidates.get(seed.doc_id, [])
        if graph is None or not candidates:
            _log(f"Skipping message_format (no graph/candidates): doc_id={seed.doc_id}, section_id={seed.section_id}")
            continue

        # Pull message_format section content from KG for grounding.
        if seed.section_id not in graph.nodes:
            continue
        message_format_chunk = _node_chunk(graph, seed.section_id, seed.doc_id)
        message_format_content = message_format_chunk.get("content", "") or ""

        # Precision gating: cosine similarity from feature-validation query -> KG candidates.
        # Keep it per-message: query includes message_name to disambiguate request/response.
        scored_candidates = candidates
        _log(
            f"Scoring candidates: doc_id={seed.doc_id}, message={seed.message_name}, "
            f"message_format_section_id={seed.section_id}, candidates={len(candidates)}"
        )
        if embeddings is not None:
            query_text = global_query_text
            if seed.message_name:
                query_text = f"{seed.message_name}\n\n{query_text}"

            # Pre-cap candidate pool for scoring to control memory/time.
            if len(candidates) > max_candidates_for_scoring:
                # Simple heuristic pre-trim using keywords.
                kw = _extract_intent_keywords(intent_text + " " + seed.message_name)
                kw_compact = [k.replace(" ", "") for k in kw if k and k.strip()]
                trimmed: List[Dict[str, Any]] = []
                for c in candidates:
                    t = _safe_upper(c.get("content", "") or "")
                    t_compact = t.replace(" ", "")
                    if any(k in t_compact for k in kw_compact):
                        trimmed.append(c)
                    if len(trimmed) >= max_candidates_for_scoring:
                        break
                if trimmed:
                    candidates_for_scoring = trimmed
                else:
                    candidates_for_scoring = candidates[:max_candidates_for_scoring]
            else:
                candidates_for_scoring = candidates
            _log(f"Cosine scoring: candidates_for_scoring={len(candidates_for_scoring)}")

            scored_candidates = _score_candidates_cosine(
                embeddings=embeddings,
                query_text=query_text,
                candidates=candidates_for_scoring,
                top_k=cosine_top_k,
                min_score=cosine_min_score,
                max_candidate_text_chars=cosine_max_candidate_text_chars,
            )
            _log(
                f"Cosine done: doc_id={seed.doc_id}, top_section_ids="
                f"{[_normalize_section_id(c.get('section_id')) for c in scored_candidates[:5]]}"
            )

        # Also ensure message_format section itself is present.
        # (Downstream IE extraction relies on KG definition contexts.)
        message_format_section_id = _normalize_section_id(seed.section_id)
        if message_format_section_id and all(_normalize_section_id(c.get("section_id")) != message_format_section_id for c in scored_candidates):
            scored_candidates = [message_format_chunk] + list(scored_candidates)

        # IE chain extraction: MAIN/parent candidates + recursive child extraction.
        main_node_ids = _find_main_ie_candidates(
            graph,
            doc_id=seed.doc_id,
            message_name=seed.message_name,
            message_format_content=message_format_content,
            intent_text=intent_text,
            keywords=_extract_intent_keywords(intent_text + " " + seed.message_name),
            llm=llm,
            template=template,
        )

        # If main candidates are empty, fallback to using the message_format section itself.
        if not main_node_ids:
            main_node_ids = [seed.section_id]

        _log(
            f"IE chain extraction start: doc_id={seed.doc_id}, message={seed.message_name}, "
            f"main_node_ids={main_node_ids[:5]}"
        )
        ie_definition_chunks, visited_node_ids = _recursive_ie_chain_extraction(
            graph=graph,
            doc_id=seed.doc_id,
            intent_text=intent_text,
            message_format_context=message_format_content,
            main_node_ids=main_node_ids,
            max_depth=recursive_ie_max_depth,
            max_nodes=recursive_ie_max_nodes,
            llm=llm,
        )
        _log(
            f"IE chain extraction done: doc_id={seed.doc_id}, extracted_chunks={len(ie_definition_chunks)}, "
            f"visited_nodes={len(visited_node_ids)}"
        )

        # Add grounding evidence + selected KG chunks for downstream template filling.
        # `specs_context` should include:
        # - message_format section
        # - IE definition chain nodes
        # - top cosine-scored related chunks (helpful for filling descriptive fields)
        related_chunks = scored_candidates[: min(20, max(5, cosine_top_k // 4))]
        chunk_role_index: Dict[str, str] = {}
        for c in related_chunks:
            chunk_role_index[_normalize_section_id(c.get("section_id"))] = "related_candidate"
        for c in ie_definition_chunks:
            chunk_role_index[_normalize_section_id(c.get("section_id"))] = "ie_definition_chain"

        # Mark roles lightly (does not affect template filler; it may include extra keys).
        for c in related_chunks:
            c["chunk_role"] = chunk_role_index.get(_normalize_section_id(c.get("section_id")), "related_candidate")
            c["source_message_format_section_id"] = seed.section_id
            c["source_message_name"] = seed.message_name

        for c in ie_definition_chunks:
            c["chunk_role"] = "ie_definition_chain"
            c["source_message_format_section_id"] = seed.section_id
            c["source_message_name"] = seed.message_name

        message_format_chunk["chunk_role"] = "message_format"
        message_format_chunk["source_message_format_section_id"] = seed.section_id
        message_format_chunk["source_message_name"] = seed.message_name

        specs_context_chunks.extend([message_format_chunk] + related_chunks + ie_definition_chunks)

        message_format_groundings.append(
            {
                "doc_id": seed.doc_id,
                "message_name": seed.message_name,
                "message_format_section_id": seed.section_id,
                "message_format_reason": seed.reason,
                "message_format_content_truncated_chars": len(_truncate_text(message_format_content, 4000)),
                "cosine_used": embeddings is not None,
                "cosine_top_k": cosine_top_k,
                "cosine_min_score": cosine_min_score,
                "selected_cosine_candidate_section_ids": [
                    _normalize_section_id(c.get("section_id")) for c in scored_candidates[:cosine_top_k]
                ],
                "main_ie_candidate_node_ids": list(main_node_ids),
                "ie_chain_visited_node_ids": visited_node_ids,
                "ie_chain_section_ids": [_normalize_section_id(c.get("section_id")) for c in ie_definition_chunks],
            }
        )
        _log(f"Grounding complete for message_format: section_id={seed.section_id}")

    # Final dedup for specs_context.
    specs_context_chunks = _dedup_chunks(specs_context_chunks)
    _log(f"Dedup complete: specs_context_count={len(specs_context_chunks)}")

    output = {
        "feature_json_path": str(feature_json_path),
        "intent": intent_text,
        "timestamp": datetime.utcnow().isoformat(),
        "retrieval_config": {
            "max_depth_kg_expand": max_depth_kg_expand,
            "cosine_top_k": cosine_top_k,
            "cosine_min_score": cosine_min_score,
            "kg_expansion_direction": kg_expansion_direction,
            "recursive_ie_max_depth": recursive_ie_max_depth,
            "recursive_ie_max_nodes": recursive_ie_max_nodes,
            "max_candidates_for_scoring": max_candidates_for_scoring,
            "cosine_embedding_model": cosine_embedding_model,
        },
        "specs_context_count": len(specs_context_chunks),
        "specs_context": specs_context_chunks,
        "message_format_groundings": message_format_groundings,
    }

    _log(f"Writing output JSON: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"Saved output JSON: {output_path} (size_bytes={output_path.stat().st_size})")
    return output


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent  # KG_Only_Pipeline
    REPO_ROOT = SCRIPT_DIR.parent

    FEATURE_JSON_PATH = REPO_ROOT / "Inter-gNB-DU_LTM_handover_procedure_20260323_093447.json"
    KG_BASE_DIR = SCRIPT_DIR / "spec_chunks"
    RUN_TS = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    OUT_DIR = SCRIPT_DIR / "spec_chunks" / "retrieval_outputs"
    OUT_PATH = OUT_DIR / f"end_to_end_ie_context_package_{RUN_TS}.json"
    LATEST_PATH = OUT_DIR / "end_to_end_ie_context_package.json"
    TEMPLATE_PATH = REPO_ROOT / "inputs" / "Template.json"

    result = run_end_to_end_ie_context_phase(
        feature_json_path=FEATURE_JSON_PATH,
        kg_base_dir=KG_BASE_DIR,
        output_path=OUT_PATH,
        template_path=TEMPLATE_PATH,
        max_depth_kg_expand=2,
        cosine_top_k=80,
        cosine_min_score=None,
        recursive_ie_max_depth=2,
        recursive_ie_max_nodes=80,
    )

    # Also write/overwrite the "latest" pointer file for convenience.
    LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved end-to-end IE context package to: {OUT_PATH}")
    print(f"Saved latest pointer JSON to: {LATEST_PATH}")

