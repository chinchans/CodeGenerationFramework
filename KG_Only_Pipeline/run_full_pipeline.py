"""
Single entrypoint to run full KG-only spec pipeline for ALL specs listed in an input JSON.

Stages (per spec):
1) TOC parsing
2) Spec ingestion + chunking
3) Knowledge graph creation
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from spec_ingestion_chunking import SpecIngestionChunker, save_ingestion_outputs
from spec_knowledge_graph_builder import SpecKnowledgeGraphBuilder
from toc_parser import parse_toc_sections, save_toc_sections


def _extract_doc_id(spec: Dict[str, Any]) -> str:
    doc_id = str(spec.get("doc_id", "")).strip()
    if doc_id:
        return doc_id
    pdf_path = str(spec.get("downloaded_pdf_path", "")).strip()
    if not pdf_path:
        raise ValueError("Spec entry must include either `doc_id` or `downloaded_pdf_path`.")
    # In this repo the doc_id matches the PDF stem (e.g. ts_138473v180600p).
    return Path(pdf_path).stem


def _spec_pdf_path(spec: Dict[str, Any]) -> Path:
    pdf_path = str(spec.get("downloaded_pdf_path", "")).strip()
    if not pdf_path:
        raise ValueError("Spec entry missing `downloaded_pdf_path`.")
    return Path(pdf_path)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent  # .../KG_Only_Pipeline
    repo_root = base_dir.parent  # .../CodeGenerationFramework

    # Hardcoded input (per your request).
    input_json_path = repo_root / "Inter-gNB-DU_LTM_handover_procedure_20260323_093447.json"

    # Where the pipeline writes toc_sections.json, chunks.json, and KnowledgeGraph outputs.
    output_root = base_dir / "spec_chunks"
    output_root.mkdir(parents=True, exist_ok=True)

    # Stage defaults (mirrors the static configs inside the stage scripts).
    # Use auto-TOC detection: toc_parser will search for the TOC pages in the PDF.
    TOC_START_PAGE: Optional[int] = None
    TOC_END_PAGE: Optional[int] = None
    # If strict TOC parsing fails, toc_parser can fall back to LLM parsing and regex.
    STRICT_TOC_ONLY = False
    SKIP_START = 9
    SKIP_FROM: Optional[int] = None

    # Per-spec TOC overrides.
    # Format: { "<doc_id>": (<toc_start_page>, <toc_end_page>) }
    # If a doc_id is missing here, the pipeline falls back to auto-TOC detection.
    TOC_PAGE_OVERRIDES: Dict[str, tuple[int, int]] = {
        # Example (fill these for each spec you want stable results for):
        "ts_138401v180600p": (4, 8),
        "ts_138473v180600p": (4, 21),
    }

    # Safety: since you reported auto-TOC sometimes picks wrong TOC pages,
    # fail fast if a doc_id isn't in the override dictionary.
    ALLOW_AUTO_TOC_FALLBACK = False

    INCLUDE_SEMANTIC_RELATIONS = True
    SEMANTIC_TOP_K_CANDIDATES = 10
    SEMANTIC_FINAL_K = 5
    SEMANTIC_MAX_CHUNKS = 350

    print("#" * 80)
    print("KG-Only Full Pipeline Entry (All Specs)")
    print("#" * 80)
    print(f"Base Dir      : {base_dir}")
    print(f"Input JSON    : {input_json_path}")
    print(f"Output Root   : {output_root}")
    print(
        f"TOC Range     : {'auto' if TOC_START_PAGE is None and TOC_END_PAGE is None else str(TOC_START_PAGE)+'-'+str(TOC_END_PAGE)} "
        f"(strict={STRICT_TOC_ONLY})"
    )
    print("#" * 80)

    payload = json.loads(input_json_path.read_text(encoding="utf-8"))
    specs = payload.get("specs", [])
    if not isinstance(specs, list) or not specs:
        raise ValueError("Input JSON must contain a non-empty `specs` array.")

    total_start = time.time()
    for idx, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ValueError(f"Each item in `specs` must be an object; got: {type(spec)}")

        doc_id = _extract_doc_id(spec)
        pdf_path = _spec_pdf_path(spec)
        spec_number = str(spec.get("spec_number", "")).strip()

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found for doc_id={doc_id}: {pdf_path}")

        print()
        print("=" * 80)
        print(
            f"[{idx+1}/{len(specs)}] Building KG for doc_id={doc_id}"
            + (f" ({spec_number})" if spec_number else "")
        )
        print("=" * 80)
        spec_start = time.time()

        # Per-spec TOC pages override (doc_id-specific) to avoid incorrect auto-detection.
        override = TOC_PAGE_OVERRIDES.get(doc_id)
        if not override and not ALLOW_AUTO_TOC_FALLBACK:
            raise ValueError(
                f"Missing TOC override for doc_id={doc_id}. "
                f"Add it to TOC_PAGE_OVERRIDES in run_full_pipeline.py."
            )

        toc_start_page: Optional[int] = override[0] if override else TOC_START_PAGE
        toc_end_page: Optional[int] = override[1] if override else TOC_END_PAGE
        toc_mode = "override" if override else "auto"
        print(f"TOC mode      : {toc_mode} (start={toc_start_page}, end={toc_end_page})")

        # Stage 1) TOC parsing
        print("[Stage 1/3] TOC parsing ...")
        toc_entries = parse_toc_sections(
            str(pdf_path),
            toc_start_page=toc_start_page,
            toc_end_page=toc_end_page,
            strict_toc_only=STRICT_TOC_ONLY,
        )
        toc_sections_path = save_toc_sections(toc_entries, str(output_root), doc_id)
        print(f"TOC sections     : {toc_sections_path}")

        # Stage 2) Ingestion + chunking
        print("[Stage 2/3] Ingestion + chunking ...")
        chunker = SpecIngestionChunker(doc_id=doc_id)
        result = chunker.run(
            pdf_path=str(pdf_path),
            skip_start=SKIP_START,
            skip_from=SKIP_FROM,
            toc_sections_path=toc_sections_path,
        )
        paths = save_ingestion_outputs(
            doc_id=doc_id,
            sections=result["sections"],
            chunks=result["chunks"],
            output_root=str(output_root),
        )
        print(f"Sections parsed  : {result['sections_count']}")
        print(f"Chunks generated : {result['chunks_count']}")
        print(f"chunks.json      : {paths['chunks_path']}")

        # Stage 3) Knowledge graph build
        print("[Stage 3/3] Knowledge graph build ...")
        builder = SpecKnowledgeGraphBuilder(doc_id=doc_id)
        chunks = builder._load_chunks(paths["chunks_path"])
        graph = builder.build_graph(
            chunks,
            include_semantic_relations=INCLUDE_SEMANTIC_RELATIONS,
            semantic_top_k_candidates=SEMANTIC_TOP_K_CANDIDATES,
            semantic_final_k=SEMANTIC_FINAL_K,
            semantic_max_chunks=SEMANTIC_MAX_CHUNKS,
        )

        graph_dir = output_root / doc_id / "KnowledgeGraph"
        graph_dir.mkdir(parents=True, exist_ok=True)
        output_graph_path = str(graph_dir / "knowledge_graph.pkl")
        output_summary_path = str(graph_dir / "graph_summary.json")
        output_adjacency_json_path = str(graph_dir / "knowledge_graph_adjacency.json")

        builder.save_graph(graph, output_graph_path)
        builder.save_graph_summary(graph, output_summary_path)
        builder.save_graph_adjacency_json(graph, output_adjacency_json_path, doc_id=doc_id)

        print(f"Nodes            : {graph.number_of_nodes()}")
        print(f"Edges            : {graph.number_of_edges()}")
        print(f"Graph saved      : {output_graph_path}")
        print(f"Summary saved    : {output_summary_path}")
        print("=" * 80)
        print(f"[DONE] doc_id={doc_id} in {time.time() - spec_start:.1f}s")

    total_elapsed = time.time() - total_start
    print()
    print("#" * 80)
    print(f"Full pipeline completed successfully in {total_elapsed:.1f}s for {len(specs)} specs")
    print("#" * 80)

