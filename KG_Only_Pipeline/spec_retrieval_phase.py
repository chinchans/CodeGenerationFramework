"""
KG-only retrieval phase for specification context expansion.

Inputs:
- Feature validation output JSON (seed section IDs + doc IDs/specs)
- Built NetworkX knowledge graphs under KG_Only_Pipeline/spec_chunks/<doc_id>/KnowledgeGraph/

Output:
- Expanded retrieval context JSON under KG_Only_Pipeline/spec_chunks/retrieval_outputs/
"""

from __future__ import annotations

import json
import pickle
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import networkx as nx


def _load_feature_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Feature JSON not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def _safe_doc_id_from_spec_number(spec_number: str) -> str:
    # Example: "TS 38.401" -> "ts_138401v180600p" is unknown version.
    # We only derive base prefix "ts_138401" for fallback matching.
    digits = "".join(ch for ch in (spec_number or "") if ch.isdigit())
    if not digits:
        return ""
    if len(digits) >= 5:
        return f"ts_{digits[:3]}{digits[3:]}"
    return f"ts_{digits}"


def _normalize_section_id(section_id: Any) -> str:
    return str(section_id or "").strip()


def _collect_seed_sections_by_doc(feature_data: Dict[str, Any]) -> Dict[str, Set[str]]:
    """
    Build seed map:
      doc_id -> {section_id...}
    """
    seeds: Dict[str, Set[str]] = {}

    # 1) Procedure-level seed (often architecture spec section).
    proc = feature_data.get("procedure_spec_info", {}) or {}
    proc_section = _normalize_section_id(proc.get("section_id"))
    proc_spec_number = proc.get("spec_number", "")
    proc_doc_base = _safe_doc_id_from_spec_number(proc_spec_number)
    if proc_section and proc_doc_base:
        seeds.setdefault(proc_doc_base, set()).add(str(proc_section))

    # 2) protocol_specs[] seeds (with explicit doc_id).
    for spec in feature_data.get("protocol_specs", []) or []:
        doc_id = str(spec.get("doc_id", "")).strip()
        section_id = _normalize_section_id(spec.get("section_id", ""))
        if doc_id and section_id:
            seeds.setdefault(doc_id, set()).add(section_id)

    # 3) protocol_message_sections[] seeds (message-specific sections).
    # Map these to doc_id via protocol_specs by spec_number/protocol.
    protocol_doc_map: Dict[Tuple[str, str], str] = {}
    for spec in feature_data.get("protocol_specs", []) or []:
        protocol = str(spec.get("protocol", "")).strip().upper()
        spec_number = str(spec.get("spec_number", "")).strip().upper()
        doc_id = str(spec.get("doc_id", "")).strip()
        if protocol and spec_number and doc_id:
            protocol_doc_map[(protocol, spec_number)] = doc_id

    for block in feature_data.get("protocol_message_sections", []) or []:
        protocol = str(block.get("protocol", "")).strip().upper()
        spec_number = str(block.get("spec_number", "")).strip().upper()
        doc_id = protocol_doc_map.get((protocol, spec_number), "")
        if not doc_id:
            continue
        for m in block.get("messages", []) or []:
            for sec in m.get("sections", []) or []:
                sid = _normalize_section_id(sec.get("section_id", ""))
                if sid:
                    seeds.setdefault(doc_id, set()).add(sid)

    return seeds


def _resolve_graph_path(base_dir: Path, doc_id_or_prefix: str) -> Path | None:
    """
    Resolve graph path for exact doc_id, with prefix fallback.
    """
    exact = base_dir / "spec_chunks" / doc_id_or_prefix / "KnowledgeGraph" / "knowledge_graph.pkl"
    if exact.exists():
        return exact

    # Prefix fallback: e.g. "ts_138401" -> match "ts_138401v180600p"
    parent = base_dir / "spec_chunks"
    if not parent.exists():
        return None
    candidates = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith(doc_id_or_prefix)]
    for c in sorted(candidates):
        kp = c / "KnowledgeGraph" / "knowledge_graph.pkl"
        if kp.exists():
            return kp
    return None


def _load_graph(path: Path) -> nx.DiGraph:
    with open(path, "rb") as f:
        g = pickle.load(f)
    if not isinstance(g, nx.DiGraph):
        raise ValueError(f"Expected nx.DiGraph at {path}")
    return g


def _extract_node_chunk(graph: nx.DiGraph, node_id: str, doc_id: str) -> Dict[str, Any]:
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


def _seed_neighbor_snapshot(
    graph: nx.DiGraph,
    seed_id: str,
    allowed_relations: Set[str] | None = None,
) -> Dict[str, Any]:
    if seed_id not in graph.nodes:
        return {"seed_id": seed_id, "exists": False, "outgoing": [], "incoming": []}

    outgoing_map: Dict[str, Set[str]] = {}
    for n in graph.successors(seed_id):
        rels = [
            r
            for r in _edge_rel_types(graph, seed_id, n)
            if (not allowed_relations or r in allowed_relations)
        ]
        if rels:
            key = str(n)
            outgoing_map.setdefault(key, set()).update(rels)

    incoming_map: Dict[str, Set[str]] = {}
    for n in graph.predecessors(seed_id):
        rels = [
            r
            for r in _edge_rel_types(graph, n, seed_id)
            if (not allowed_relations or r in allowed_relations)
        ]
        if rels:
            key = str(n)
            incoming_map.setdefault(key, set()).update(rels)

    outgoing = [
        {"to": node_id, "relationship_types": sorted(list(rel_set))}
        for node_id, rel_set in sorted(outgoing_map.items(), key=lambda x: x[0])
    ]
    incoming = [
        {"from": node_id, "relationship_types": sorted(list(rel_set))}
        for node_id, rel_set in sorted(incoming_map.items(), key=lambda x: x[0])
    ]

    return {
        "seed_id": seed_id,
        "exists": True,
        "outgoing_count": len(outgoing),
        "incoming_count": len(incoming),
        "outgoing": outgoing,
        "incoming": incoming,
    }


def _expand_from_seeds(
    graph: nx.DiGraph,
    doc_id: str,
    seed_ids: Set[str],
    max_depth: int = 2,
    allowed_relations: Set[str] | None = None,
    direction: str = "both",
) -> Dict[str, Any]:
    existing = {sid for sid in seed_ids if sid in graph.nodes}
    missing = sorted(list(seed_ids - existing))

    initial_chunks = [_extract_node_chunk(graph, sid, doc_id) for sid in sorted(existing)]

    visited: Set[str] = set(existing)
    q = deque([(sid, 0, None, None) for sid in sorted(existing)])
    expanded_chunks: List[Dict[str, Any]] = []

    while q:
        current, depth, from_node, via_rel = q.popleft()
        if depth >= max_depth:
            continue

        neighbors: List[Tuple[str, str]] = []  # (neighbor, rel)
        if direction in {"out", "both"}:
            for n in graph.successors(current):
                for rel in _edge_rel_types(graph, current, n):
                    neighbors.append((n, rel))
        if direction in {"in", "both"}:
            for n in graph.predecessors(current):
                for rel in _edge_rel_types(graph, n, current):
                    neighbors.append((n, rel))

        for nbr, rel in neighbors:
            if allowed_relations and rel not in allowed_relations:
                continue
            if nbr in visited:
                continue
            visited.add(nbr)

            chunk = _extract_node_chunk(graph, nbr, doc_id)
            chunk["expanded_chunk_from"] = {
                "section_id": current,
                "section_title": graph.nodes[current].get("section_title", ""),
            }
            chunk["depth_level"] = depth + 1
            chunk["expansion_relation"] = [rel]
            expanded_chunks.append(chunk)

            q.append((nbr, depth + 1, current, rel))

    return {
        "seed_ids_found": sorted(list(existing)),
        "seed_ids_missing": missing,
        "initial_chunks": initial_chunks,
        "expanded_chunks": expanded_chunks,
    }


def run_retrieval_phase(
    feature_json_path: str,
    base_dir: str,
    max_depth: int = 2,
    allowed_relations: Set[str] | None = None,
    direction: str = "both",
    manual_seeds_by_doc: Dict[str, Set[str]] | None = None,
) -> Dict[str, Any]:
    base = Path(base_dir)
    feature_data = _load_feature_json(feature_json_path)
    seeds_by_doc = _collect_seed_sections_by_doc(feature_data)
    if manual_seeds_by_doc:
        seeds_by_doc = {
            str(doc): {_normalize_section_id(sid) for sid in sids if _normalize_section_id(sid)}
            for doc, sids in manual_seeds_by_doc.items()
        }

    per_doc_results = {}
    final_context = []

    for doc_id_hint, seed_ids in seeds_by_doc.items():
        graph_path = _resolve_graph_path(base, doc_id_hint)
        if not graph_path:
            per_doc_results[doc_id_hint] = {
                "status": "graph_not_found",
                "seed_ids": sorted(list(seed_ids)),
            }
            continue

        actual_doc_id = graph_path.parent.parent.name
        graph = _load_graph(graph_path)
        expanded = _expand_from_seeds(
            graph=graph,
            doc_id=actual_doc_id,
            seed_ids=seed_ids,
            max_depth=max_depth,
            allowed_relations=allowed_relations,
            direction=direction,
        )
        seed_neighbors = [
            _seed_neighbor_snapshot(graph, sid, allowed_relations=allowed_relations)
            for sid in sorted(expanded["seed_ids_found"])
        ]
        per_doc_results[actual_doc_id] = {
            "status": "ok",
            "graph_path": str(graph_path),
            "seed_ids": sorted(list(seed_ids)),
            "seed_ids_found": expanded["seed_ids_found"],
            "seed_ids_missing": expanded["seed_ids_missing"],
            "initial_count": len(expanded["initial_chunks"]),
            "expanded_count": len(expanded["expanded_chunks"]),
            "seed_neighbors": seed_neighbors,
        }
        final_context.extend(expanded["initial_chunks"])
        final_context.extend(expanded["expanded_chunks"])

    # Deduplicate by normalized (doc_id, section_id)
    seen = set()
    deduped = []
    for c in final_context:
        normalized_sid = _normalize_section_id(c.get("section_id"))
        key = (str(c.get("metadata", {}).get("doc_id", "")).strip(), normalized_sid)
        if not normalized_sid:
            continue
        if key in seen:
            continue
        seen.add(key)
        c["section_id"] = normalized_sid
        deduped.append(c)

    return {
        "feature_json_path": feature_json_path,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "retrieval_config": {
            "max_depth": max_depth,
            "direction": direction,
            "allowed_relations": sorted(list(allowed_relations)) if allowed_relations else "ALL",
            "manual_seed_mode": bool(manual_seeds_by_doc),
        },
        "seeds_by_doc": {k: sorted(list(v)) for k, v in seeds_by_doc.items()},
        "per_doc_results": per_doc_results,
        "final_context_count": len(deduped),
        "final_context": deduped,
    }


def save_retrieval_output(payload: Dict[str, Any], output_path: str) -> str:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(p)


if __name__ == "__main__":
    # Static config (no args)
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent
    BASE_DIR = str(SCRIPT_DIR)
    FEATURE_JSON_PATH = str(REPO_ROOT / "./Inter-gNB-DU_LTM_handover_procedure_20260323_093447.json")
    MAX_DEPTH = 2
    DIRECTION = "both"  # out | in | both
    # Manual override for debugging: set to True to bypass feature-json seed mapping.
    USE_MANUAL_SEEDS = True
    MANUAL_SEEDS_BY_DOC = {
        "ts_138401v180600p": {"8.2.1.5"},
    }
    ALLOWED_RELATIONS = {
        "DEPENDS_ON",
        "PREREQUISITE_FOR",
        "USES",
        "DEFINES",
        "RELATED_TO",
        "CHILD_OF",
        "PARENT_OF",
        "SIBLING_OF",
        "REFERENCES",
        "REFERENCED_BY",
    }

    OUT_FILE = str(SCRIPT_DIR / "spec_chunks" / "retrieval_outputs" / "spec_retrieval_context.json")

    print("=" * 80)
    print("KG-Only Spec Retrieval Phase")
    print("=" * 80)
    print(f"feature_json : {FEATURE_JSON_PATH}")
    print(f"base_dir     : {BASE_DIR}")
    print(f"max_depth    : {MAX_DEPTH}")
    print(f"direction    : {DIRECTION}")
    print(f"manual_seeds : {USE_MANUAL_SEEDS}")
    print("-" * 80)

    payload = run_retrieval_phase(
        feature_json_path=FEATURE_JSON_PATH,
        base_dir=BASE_DIR,
        max_depth=MAX_DEPTH,
        allowed_relations=ALLOWED_RELATIONS,
        direction=DIRECTION,
        manual_seeds_by_doc=MANUAL_SEEDS_BY_DOC if USE_MANUAL_SEEDS else None,
    )
    out = save_retrieval_output(payload, OUT_FILE)

    print("seeds_by_doc       :", payload.get("seeds_by_doc"))
    print("per_doc_results    :", json.dumps(payload.get("per_doc_results", {}), indent=2))
    print(f"final_context_count: {payload.get('final_context_count')}")
    print(f"output            : {out}")
    print("=" * 80)

