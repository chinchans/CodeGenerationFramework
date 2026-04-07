from __future__ import annotations

import json
import pickle
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import networkx as nx


def _normalize_section_id(section_id: Any) -> str:
    return str(section_id or "").strip()


def _safe_upper(s: Any) -> str:
    return str(s or "").upper().strip()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_doc_id_from_spec_number(payload: Dict[str, Any], spec_number: str) -> Optional[str]:
    spec_number_norm = _safe_upper(spec_number)
    if not spec_number_norm:
        return None
    for s in payload.get("specs", []) or []:
        sn = _safe_upper(s.get("spec_number", ""))
        doc_id = str(s.get("doc_id", "")).strip()
        if sn and doc_id and sn == spec_number_norm:
            return doc_id
    return None


def collect_seed_sections_by_doc(payload: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Step-1: collect all seed section_ids per spec/doc_id.
    Seeds are derived from:
      - procedure_spec_info.section_id
      - protocol_specs[].section_id (if present)
      - protocol_message_sections[].messages[].sections[].section_id
    """

    def _add_seed(seeds_by_doc: Dict[str, Set[str]], doc_id: str, section_id: Any) -> None:
        sid = _normalize_section_id(section_id)
        ddid = str(doc_id or "").strip()
        if not sid or not ddid:
            return
        seeds_by_doc.setdefault(ddid, set()).add(sid)

    seeds_by_doc: Dict[str, Set[str]] = {}

    # 1) procedure_spec_info seed (explicit section_id + spec_number)
    proc = payload.get("procedure_spec_info", {}) or {}
    proc_section_id = _normalize_section_id(proc.get("section_id"))
    proc_spec_number = str(proc.get("spec_number", "")).strip()
    if proc_section_id and proc_spec_number:
        doc_id = _resolve_doc_id_from_spec_number(payload, proc_spec_number)
        if doc_id:
            _add_seed(seeds_by_doc, doc_id=doc_id, section_id=proc_section_id)

    # 2) protocol_specs[] seed (may include section_id per entry)
    for spec in payload.get("protocol_specs", []) or []:
        spec_number = str(spec.get("spec_number", "")).strip()
        section_id = _normalize_section_id(spec.get("section_id"))
        if not section_id or not spec_number:
            continue
        doc_id = _resolve_doc_id_from_spec_number(payload, spec_number)
        if doc_id:
            _add_seed(seeds_by_doc, doc_id=doc_id, section_id=section_id)

    # 3) protocol_message_sections[] seeds (message-specific sections)
    protocol_doc_map: Dict[Tuple[str, str], str] = {}
    for spec in payload.get("protocol_specs", []) or []:
        protocol = _safe_upper(spec.get("protocol", ""))
        spec_number = _safe_upper(spec.get("spec_number", ""))
        doc_id = str(spec.get("doc_id", "")).strip()  # may not exist in incoming seeds
        if not doc_id and spec_number:
            # resolve from payload["specs"]
            doc_id = _resolve_doc_id_from_spec_number(payload, spec_number) or ""
        if protocol and spec_number and doc_id:
            protocol_doc_map[(protocol, spec_number)] = doc_id

    for block in payload.get("protocol_message_sections", []) or []:
        protocol = _safe_upper(block.get("protocol", ""))
        spec_number = _safe_upper(block.get("spec_number", ""))
        doc_id = protocol_doc_map.get((protocol, spec_number), "") if (protocol and spec_number) else ""
        if not doc_id:
            # Fall back: use spec_number only
            doc_id = _resolve_doc_id_from_spec_number(payload, str(block.get("spec_number", "")).strip()) or ""

        if not doc_id:
            continue

        for msg in block.get("messages", []) or []:
            for sec in msg.get("sections", []) or []:
                sid = _normalize_section_id(sec.get("section_id"))
                if sid:
                    _add_seed(seeds_by_doc, doc_id=doc_id, section_id=sid)

    # Deterministic ordering: doc_id then section_id
    return {doc_id: sorted(list(sids)) for doc_id, sids in sorted(seeds_by_doc.items(), key=lambda x: x[0])}


def expand_kg_from_seed(
    graph: nx.DiGraph,
    *,
    seed_id: str,
    max_depth: int = 2,
    direction: str = "both",
) -> Dict[str, int]:
    """
    Step-2: BFS expansion in the KG for connected nodes up to `max_depth`.
    Returns depth_map including the seed itself (depth 0).
    """
    seed_id = _normalize_section_id(seed_id)
    if not seed_id or seed_id not in graph.nodes:
        return {}

    visited_depth: Dict[str, int] = {seed_id: 0}
    q: Deque[Tuple[str, int]] = deque([(seed_id, 0)])

    while q:
        current, depth = q.popleft()
        if depth >= max_depth:
            continue

        neighbors: Set[str] = set()
        if direction in {"out", "both"}:
            neighbors.update(str(n) for n in graph.successors(current))
        if direction in {"in", "both"}:
            neighbors.update(str(n) for n in graph.predecessors(current))

        for nbr in sorted(neighbors):
            if nbr not in graph.nodes:
                # If graph uses non-string ids, this conversion could break.
                # Current pipeline KG node ids are expected to be section_id strings.
                continue
            if nbr in visited_depth:
                continue
            visited_depth[nbr] = depth + 1
            q.append((nbr, depth + 1))

    return visited_depth


def load_chunks_content_index(chunks_json_path: Path, wanted_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
    """
    Step-3: build a lookup of chunks.json entries by section_id, but only keep wanted_ids.
    """
    if not chunks_json_path.exists():
        raise FileNotFoundError(f"chunks.json not found: {chunks_json_path}")

    with chunks_json_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(chunks, list):
        return out

    for c in chunks:
        if not isinstance(c, dict):
            continue
        sid = _normalize_section_id(c.get("section_id"))
        if sid and sid in wanted_ids:
            out[sid] = c
    return out


def run_step1_to_step3_section_retrieval() -> Dict[str, Any]:
    """
    Executes Step-1..Step-3:
      1) Collect seed section_ids from the provided feature JSON (per spec/doc_id)
      2) For each doc_id, expand KG neighborhood from each seed to depth=2 (both directions)
      3) Attach full section_text from chunks.json for every expanded node (including seeds at depth 0)

    No ranking/filtering or IE extraction is performed in this v0 implementation.
    """
    # Repo-relative hardcoded inputs (no CLI args as requested).
    repo_root = Path(__file__).resolve().parent.parent

    feature_json_path = repo_root / "Inter-gNB-DU_LTM_handover_procedure_20260323_093447.json"
    if not feature_json_path.exists():
        raise FileNotFoundError(f"Feature JSON not found: {feature_json_path}")

    query_text = str(_load_json(feature_json_path).get("intent", "") or "").strip()
    if not query_text:
        # best-effort fallback to intent wording
        query_text = str(_load_json(feature_json_path).get("query", "") or "").strip()

    kg_base_dir = repo_root / "KG_Only_Pipeline" / "spec_chunks"
    if not kg_base_dir.exists():
        raise FileNotFoundError(f"KG base dir not found: {kg_base_dir}")

    seeds_by_doc = collect_seed_sections_by_doc(_load_json(feature_json_path))

    max_depth = 2
    direction = "both"

    docs_out: List[Dict[str, Any]] = []
    all_final_context: List[Dict[str, Any]] = []

    total_root_seed_count = 0
    total_unique_section_count = 0

    for doc_id, seed_section_ids in seeds_by_doc.items():
        graph_path = kg_base_dir / doc_id / "KnowledgeGraph" / "knowledge_graph.pkl"
        with graph_path.open("rb") as f:
            graph = pickle.load(f)
        if not isinstance(graph, nx.DiGraph):
            raise ValueError(f"Expected nx.DiGraph in: {graph_path}")

        seed_set: Set[str] = set(_normalize_section_id(s) for s in seed_section_ids)
        total_root_seed_count += len(seed_set)

        missing_root_section_ids: List[str] = []

        # Step-2 expansion bookkeeping.
        # - global_min_depth_by_sid: sid -> min depth reached across all seeds
        # - expanded_from_seed_section_ids: sid -> set of seeds that reached it with depth>0
        global_min_depth_by_sid: Dict[str, int] = {}
        expanded_from_seed_section_ids: Dict[str, Set[str]] = defaultdict(set)
        per_root_depth_maps: Dict[str, Dict[str, int]] = {}

        for seed_sid in sorted(seed_set):
            if seed_sid not in graph.nodes:
                missing_root_section_ids.append(seed_sid)
                continue
            depth_map = expand_kg_from_seed(graph, seed_id=seed_sid, max_depth=max_depth, direction=direction)
            per_root_depth_maps[seed_sid] = depth_map

            for sid, depth in depth_map.items():
                sid_norm = _normalize_section_id(sid)
                if not sid_norm:
                    continue
                prev = global_min_depth_by_sid.get(sid_norm)
                if prev is None or depth < prev:
                    global_min_depth_by_sid[sid_norm] = depth
                if depth > 0:
                    expanded_from_seed_section_ids[sid_norm].add(seed_sid)

        expanded_ids = sorted(global_min_depth_by_sid.keys())
        total_unique_section_count += len(expanded_ids)

        # Step-3: resolve full text for each expanded node from chunks.json.
        chunks_path = kg_base_dir / doc_id / "chunks.json"
        chunks_by_sid = load_chunks_content_index(chunks_path, set(expanded_ids))

        relevant_sections_by_id: Dict[str, Dict[str, Any]] = {}
        for sid in expanded_ids:
            # Prefer KG node content for ASN.1 definitions.
            node = graph.nodes.get(sid, {}) if hasattr(graph, "nodes") else {}
            node_content = str(node.get("content", "") or "")
            node_title = str(
                node.get("section_title", "") or node.get("title", "") or ""
            ).strip()
            chunk = chunks_by_sid.get(sid)
            if chunk:
                section_title = str(chunk.get("section_title", "") or "")
                # If KG has richer ASN.1 content, prefer it over chunks.json.
                content = node_content if node_content else str(chunk.get("content", "") or "")
                if node_title:
                    section_title = node_title
                metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
            else:
                section_title = node_title
                content = node_content
                metadata = {}

            # Attach retrieval trace metadata for Step-3.
            depth_from_seed = int(global_min_depth_by_sid.get(sid, 0))
            is_seed = sid in seed_set and depth_from_seed == 0

            if not isinstance(metadata, dict):
                metadata = {}
            metadata = dict(metadata)
            metadata.setdefault("doc_id", doc_id)
            metadata.setdefault("knowledge_source", doc_id)
            metadata["depth_from_seed"] = depth_from_seed
            metadata["is_root_seed"] = bool(is_seed)

            # Match existing v1 output intent:
            # - for true seeds (depth 0), leave list empty
            # - for expanded nodes, list the seed(s) that reached them (depth>0)
            if is_seed:
                metadata["expanded_from_seed_section_ids"] = []
            else:
                metadata["expanded_from_seed_section_ids"] = sorted(
                    expanded_from_seed_section_ids.get(sid, set())
                )

            chunk_obj = {
                "section_id": sid,
                "section_title": section_title,
                "content": content,
                "knowledge_source": doc_id,
                "source_id": doc_id,
                "metadata": metadata,
            }
            relevant_sections_by_id[sid] = chunk_obj
            all_final_context.append(chunk_obj)

        roots_out: List[Dict[str, Any]] = []
        for seed_sid in sorted(seed_set):
            depth_map = per_root_depth_maps.get(seed_sid, {})
            expanded_section_ids = sorted([s for s in depth_map.keys() if s != seed_sid])
            roots_out.append(
                {
                    "root_section_id": seed_sid,
                    "expanded_section_ids": expanded_section_ids,
                }
            )

        docs_out.append(
            {
                "doc_id": doc_id,
                "roots": roots_out,
                "relevant_sections_by_id": relevant_sections_by_id,
                "messages": [],
                "missing_root_section_ids": sorted(set(missing_root_section_ids)),
                "counts": {
                    "root_seed_count": len(seed_set) - len(set(missing_root_section_ids)),
                    "relevant_section_count": len(expanded_ids),
                    "asn1_ie_section_count": 0,
                },
            }
        )

    # Global dedup by (doc_id, section_id). Our construction is already unique per doc_id,
    # but this keeps the output shape consistent with other retrieval outputs.
    seen: Set[Tuple[str, str]] = set()
    final_context_dedup: List[Dict[str, Any]] = []
    for c in all_final_context:
        doc = str(c.get("knowledge_source") or c.get("metadata", {}).get("doc_id") or "").strip()
        sid = _normalize_section_id(c.get("section_id"))
        if not doc or not sid:
            continue
        key = (doc, sid)
        if key in seen:
            continue
        seen.add(key)
        final_context_dedup.append(c)

    final_context_dedup.sort(key=lambda c: (str(c.get("knowledge_source", "")), str(c.get("section_id", ""))))

    output = {
        # Header fields (align to your existing KG_Retriver outputs).
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "query": query_text,
        # Some adapters look for "intent" even if older outputs only had "query".
        "intent": query_text,
        "doc_ids": [str(d) for d in seeds_by_doc.keys()],
        "counts": {
            "doc_count": len(seeds_by_doc),
            "root_seed_count": total_root_seed_count,
            "unique_section_count": len(final_context_dedup),
            "asn1_ie_section_count": 0,
            "final_context_count": len(final_context_dedup),
        },
        "feature_json_path": str(feature_json_path),
        "retrieval_config": {
            "max_depth": max_depth,
            "direction": direction,
            "allowed_relations": None,
            # In v0 we do not apply any LLM filtering or IE extraction.
            "phase": "step1_step3_only",
        },
        "docs": docs_out,
        # Compatibility aliases with downstream template-filling code.
        "final_context": final_context_dedup,
        "specs_context": final_context_dedup,
        "specs_context_count": len(final_context_dedup),
    }

    return output


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "KG_Retriver" / "spec_chunks" / "retrieval_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_result = run_step1_to_step3_section_retrieval()

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"seed_expanded_kg_context_{ts}.json"
    out_path.write_text(json.dumps(run_result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Retriver_v0] Saved Step-1..3 expanded context to: {out_path}")


if __name__ == "__main__":
    main()

