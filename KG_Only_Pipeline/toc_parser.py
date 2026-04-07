"""
TOC-first section mapper for 3GPP specs.

This module:
1) Extracts likely table-of-contents pages from a PDF
2) Uses LLM to parse section_id, section_title, toc_page
3) Builds recursive parent/child hierarchy
4) Saves normalized TOC section mapping JSON

No CLI args are required; static config is provided in __main__.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pdfplumber
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()


@dataclass
class TocEntry:
    section_id: str
    title: str
    toc_page: int
    level: int
    parent: Optional[str]
    children: List[str]
    children_recursive: List[str]


def _parse_section_numbers(section_id: str) -> Optional[List[int]]:
    match = re.fullmatch(r"\d+(?:\.\d+){0,4}", section_id.strip())
    if not match:
        return None
    return [int(x) for x in section_id.split(".")]


def _is_parent_id(parent_id: str, child_id: str) -> bool:
    p = _parse_section_numbers(parent_id)
    c = _parse_section_numbers(child_id)
    if not p or not c:
        return False
    return len(p) + 1 == len(c) and c[: len(p)] == p


def _extract_toc_pages_text(pdf_path: str, max_pages_scan: int = 40, max_toc_pages: int = 10) -> str:
    pages_text: List[str] = []
    toc_started = False
    toc_pages_collected = 0

    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages[:max_pages_scan]):
            page_num = idx + 1
            text = page.extract_text() or ""
            compact = " ".join(text.lower().split())

            has_toc_marker = "table of contents" in compact or compact.startswith("contents")
            has_toc_lines = bool(re.search(r"^\s*\d+(?:\.\d+){0,4}\s+.+?\s+\.{2,}\s*\d+\s*$", text, flags=re.MULTILINE))

            if has_toc_marker:
                toc_started = True

            if toc_started or has_toc_lines:
                pages_text.append(f"--- TOC_PDF_PAGE_{page_num} ---\n{text}")
                toc_pages_collected += 1
                if toc_pages_collected >= max_toc_pages:
                    break

            # Stop once TOC likely ended and sections body started.
            if toc_started and toc_pages_collected >= 2:
                if re.search(r"^\s*1(?:\.0)?\s+[A-Za-z].*$", text, flags=re.MULTILINE) and not has_toc_lines:
                    break

    return "\n\n".join(pages_text)


def _extract_fixed_toc_pages_text(pdf_path: str, toc_start_page: int, toc_end_page: int) -> str:
    if toc_start_page <= 0 or toc_end_page <= 0:
        raise ValueError("TOC page numbers must be >= 1.")
    if toc_end_page < toc_start_page:
        raise ValueError("TOC_END_PAGE must be >= TOC_START_PAGE.")

    pages_text: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        start_idx = toc_start_page - 1
        end_idx = min(toc_end_page, total_pages) - 1
        for idx in range(start_idx, end_idx + 1):
            text = pdf.pages[idx].extract_text() or ""
            pages_text.append(f"--- TOC_PDF_PAGE_{idx + 1} ---\n{text}")
    return "\n\n".join(pages_text)


def _build_llm() -> AzureChatOpenAI:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", os.getenv("OPENAI_API_VERSION", "2024-02-15-preview"))
    deployment = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o-mini")

    if not api_key or not endpoint:
        raise ValueError("Azure OpenAI credentials are required for TOC parser.")

    return AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_deployment=deployment
    )


def _extract_json_array(text: str) -> List[Dict]:
    text = (text or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            return []
        data = json.loads(match.group(0))
        return data if isinstance(data, list) else []


def _parse_toc_with_llm(toc_text: str, llm: AzureChatOpenAI) -> List[Tuple[str, str, int]]:
    prompt = f"""
You are parsing 3GPP PDF table-of-contents text.
Return ONLY a JSON array. No markdown.

Each element must be:
{{
  "section_id": "9.4",
  "section_title": "UE CONTEXT SETUP RESPONSE",
  "page_num": 147
}}

Rules:
- Keep only numeric hierarchical sections: 1, 1.1, 1.1.1, 1.1.1.1, 1.1.1.1.1
- Ignore annexes, bibliography, references, and non-section lines.
- section_id must be exact and unique.
- toc_page must be integer page number shown in TOC (logical page).
- Preserve official section title text as-is.

TOC TEXT:
{toc_text}
"""
    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    items = _extract_json_array(raw)

    parsed: List[Tuple[str, str, int]] = []
    seen = set()
    for item in items:
        section_id = str(item.get("section_id", "")).strip()
        title = str(item.get("section_title", "")).strip()
        toc_page = item.get("toc_page")
        if not section_id or not title:
            continue
        nums = _parse_section_numbers(section_id)
        if not nums:
            continue
        if not isinstance(toc_page, int):
            try:
                toc_page = int(str(toc_page).strip())
            except Exception:
                continue
        if section_id in seen:
            continue
        seen.add(section_id)
        parsed.append((section_id, title, toc_page))
    return parsed


def _fallback_parse_toc_regex(toc_text: str) -> List[Tuple[str, str, int]]:
    pattern = re.compile(r"^\s*(\d+(?:\.\d+){0,4})\s+(.+?)\s+\.{2,}\s*(\d+)\s*$", re.MULTILINE)
    parsed: List[Tuple[str, str, int]] = []
    seen = set()
    for section_id, title, toc_page in pattern.findall(toc_text):
        if section_id in seen:
            continue
        nums = _parse_section_numbers(section_id)
        if not nums:
            continue
        seen.add(section_id)
        parsed.append((section_id, title.strip(), int(toc_page)))
    return parsed


def _extract_toc_lines_strict(toc_text: str) -> List[Tuple[str, str, int]]:
    """
    Strict TOC extraction from dotted TOC lines only.
    This is deterministic and avoids narrative/body leakage.
    """
    parsed = _fallback_parse_toc_regex(toc_text)
    if not parsed:
        return []

    # Extra guardrails: remove noisy entries that are very unlikely to be real TOC titles.
    bad_title_patterns = [
        r"\bthe ue\b",
        r"\bthe gnb\b",
        r"\bsends\b",
        r"\bincludes\b",
        r"\bif the ue is\b",
        r"\bshall\b",
    ]
    filtered: List[Tuple[str, str, int]] = []
    for sid, title, page in parsed:
        t = " ".join(title.lower().split())
        if any(re.search(p, t) for p in bad_title_patterns):
            continue
        filtered.append((sid, title, page))
    return filtered


def _validate_llm_entries_against_toc_text(
    llm_entries: List[Tuple[str, str, int]],
    toc_text: str,
) -> List[Tuple[str, str, int]]:
    """
    Keep only LLM entries that can be matched back to TOC-like text.
    """
    valid: List[Tuple[str, str, int]] = []
    for sid, title, page in llm_entries:
        sid_re = re.escape(sid)
        page_re = re.escape(str(page))
        # Require same section id and page in a TOC-like dotted line.
        line_re = re.compile(
            rf"^\s*{sid_re}\s+.+?\.{{2,}}\s*{page_re}\s*$",
            re.MULTILINE,
        )
        if line_re.search(toc_text):
            valid.append((sid, title, page))
    return valid


def _build_recursive_hierarchy(sections: List[Tuple[str, str, int]]) -> Dict[str, TocEntry]:
    # Sort by section numeric depth and value
    def sort_key(item: Tuple[str, str, int]) -> Tuple[int, List[int]]:
        sid = item[0]
        nums = _parse_section_numbers(sid) or [999999]
        return (len(nums), nums)

    entries: Dict[str, TocEntry] = {}
    ordered = sorted(sections, key=sort_key)

    for sid, title, toc_page in ordered:
        level = len(_parse_section_numbers(sid) or [])
        entries[sid] = TocEntry(
            section_id=sid,
            title=title,
            toc_page=toc_page,
            level=level,
            parent=None,
            children=[],
            children_recursive=[],
        )

    # Assign parent/children
    all_ids = list(entries.keys())
    for child_id in all_ids:
        child_nums = _parse_section_numbers(child_id) or []
        if len(child_nums) <= 1:
            continue
        # Nearest parent candidate: strip one segment each step.
        for k in range(len(child_nums) - 1, 0, -1):
            parent_id = ".".join(str(x) for x in child_nums[:k])
            if parent_id in entries and _is_parent_id(parent_id, child_id):
                entries[child_id].parent = parent_id
                entries[parent_id].children.append(child_id)
                break

    # Build recursive children
    def dfs_descendants(sid: str) -> List[str]:
        out: List[str] = []
        for c in entries[sid].children:
            out.append(c)
            out.extend(dfs_descendants(c))
        return out

    for sid in entries:
        entries[sid].children_recursive = dfs_descendants(sid)

    return entries


def parse_toc_sections(
    pdf_path: str,
    toc_start_page: Optional[int] = None,
    toc_end_page: Optional[int] = None,
    strict_toc_only: bool = True,
) -> Dict[str, TocEntry]:
    if toc_start_page and toc_end_page:
        toc_text = _extract_fixed_toc_pages_text(pdf_path, toc_start_page, toc_end_page)
    else:
        toc_text = _extract_toc_pages_text(pdf_path)

    if not toc_text.strip():
        raise ValueError("Could not extract TOC pages text from PDF.")

    # Stage-1 should be strict: section_id/title/page from TOC lines only.
    parsed = _extract_toc_lines_strict(toc_text)
    if not parsed and not strict_toc_only:
        llm = _build_llm()
        llm_parsed = _parse_toc_with_llm(toc_text, llm)
        parsed = _validate_llm_entries_against_toc_text(llm_parsed, toc_text)
        if not parsed:
            parsed = _fallback_parse_toc_regex(toc_text)
    if not parsed:
        raise ValueError("Failed to parse TOC sections via both LLM and regex fallback.")

    return _build_recursive_hierarchy(parsed)


def save_toc_sections(entries: Dict[str, TocEntry], output_dir: str, doc_id: str) -> str:
    out_dir = Path(output_dir) / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "toc_sections.json"
    flat_path = out_dir / "toc_sections_flat.json"

    serializable_flat = {
        sid: {
            "section_id": e.section_id,
            "title": e.title,
            "toc_page": e.toc_page,
            "level": e.level,
            "parent": e.parent,
            "children": e.children,
            "children_recursive": e.children_recursive,
        }
        for sid, e in entries.items()
    }

    def sort_sid(sid: str) -> Tuple[int, List[int]]:
        nums = _parse_section_numbers(sid) or [999999]
        return (len(nums), nums)

    def build_node(sid: str) -> Dict:
        e = entries[sid]
        return {
            "section_id": e.section_id,
            "title": e.title,
            "toc_page": e.toc_page,
            "level": e.level,
            "parent": e.parent,
            "children": [build_node(c) for c in sorted(e.children, key=sort_sid)],
            "children_recursive": e.children_recursive,
        }

    root_ids = [sid for sid, e in entries.items() if e.parent is None]
    root_ids = sorted(root_ids, key=sort_sid)
    serializable_tree = {
        "doc_id": doc_id,
        "sections_tree": [build_node(sid) for sid in root_ids],
    }

    # Primary output now uses nested tree format.
    path.write_text(json.dumps(serializable_tree, indent=2), encoding="utf-8")
    # Keep a flat compatibility file for debugging/older tools.
    flat_path.write_text(json.dumps(serializable_flat, indent=2), encoding="utf-8")
    return str(path)


if __name__ == "__main__":
    # Static run config (no args)
    DOC_ID = "ts_138401v180600p"
    PDF_PATH = "../data/specs/ts_138401v180600p.pdf"
    OUTPUT_DIR = "./spec_chunks"
    # Set explicit TOC page range (1-based PDF pages). If either is None, auto-detect TOC pages.
    TOC_START_PAGE: Optional[int] = 4
    TOC_END_PAGE: Optional[int] = 9
    STRICT_TOC_ONLY = True

    print("=" * 72)
    print("TOC-First Section Mapper")
    print("=" * 72)
    print(f"Entry point: {Path(__file__).name}")
    print(f"doc_id      : {DOC_ID}")
    print(f"pdf_path    : {PDF_PATH}")
    print(f"output_dir  : {OUTPUT_DIR}")
    print(f"toc_range   : {TOC_START_PAGE} - {TOC_END_PAGE}" if TOC_START_PAGE and TOC_END_PAGE else "toc_range   : auto-detect")
    print("-" * 72)

    pdf = Path(PDF_PATH)
    if not pdf.exists():
        raise FileNotFoundError(
            f"Configured PDF not found: {PDF_PATH}\n"
            "Update PDF_PATH in this file before running."
        )

    toc_entries = parse_toc_sections(
        str(pdf),
        toc_start_page=TOC_START_PAGE,
        toc_end_page=TOC_END_PAGE,
        strict_toc_only=STRICT_TOC_ONLY,
    )
    saved_path = save_toc_sections(toc_entries, OUTPUT_DIR, DOC_ID)

    print(f"Sections mapped : {len(toc_entries)}")
    print(f"toc_sections.json: {saved_path}")
    print("=" * 72)

