"""
Spec ingestion and section-based chunking for KG-only pipeline.

This module intentionally excludes embeddings/FAISS/LLM logic.
It only:
1) Parses a spec PDF into hierarchical section nodes
2) Extracts chunks from leaf sections + parent direct text
3) Optionally saves parsed/chunked artifacts to JSON
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from toc_parser import parse_toc_sections, save_toc_sections


@dataclass
class SectionNode:
    section_id: str
    title: str
    level: int
    toc_page: Optional[int] = None
    page_num: Optional[int] = None
    page_numbers: set = field(default_factory=set)
    content_lines: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    has_direct_text: bool = False
    is_leaf: bool = False

    def __post_init__(self) -> None:
        if self.page_num is not None:
            self.page_numbers.add(self.page_num)

    def add_content_line(self, line: str, page_num: Optional[int] = None) -> None:
        self.content_lines.append(line)
        if page_num is not None:
            self.page_numbers.add(page_num)
        if line.strip() and not re.match(r"^\d+(\.\d+)+", line.strip()):
            self.has_direct_text = True

    def get_full_content(self) -> str:
        content = f"{self.section_id} {self.title}\n"
        content += "\n".join(self.content_lines)
        return content.strip()


class SpecIngestionChunker:
    def __init__(self, doc_id: str) -> None:
        self.doc_id = doc_id

    @staticmethod
    def parse_section_number(section_str: str) -> Tuple[Optional[List[int]], str]:
        match = re.match(r"^(\d+(?:\.\d+)*)", section_str.strip())
        if match:
            section_part = match.group(1)
            numbers = [int(x) for x in section_part.split(".")]
            remaining = section_str[match.end() :].strip()
            return numbers, remaining
        return None, section_str

    @staticmethod
    def is_parent_of(parent: List[int], child: List[int]) -> bool:
        if len(parent) >= len(child):
            return False
        return child[: len(parent)] == parent and len(child) == len(parent) + 1

    @staticmethod
    def _is_valid_heading_title(title: str) -> bool:
        clean = title.strip()
        if not clean:
            return False
        # Filter out obvious non-heading lines from PDF noise.
        bad_prefixes = (
            "3gpp ts",
            "etsi",
            "version",
            "page ",
            "table of contents",
            "foreword",
        )
        lower = clean.lower()
        if any(lower.startswith(p) for p in bad_prefixes):
            return False
        # Require at least one alphabetic character in title.
        return any(ch.isalpha() for ch in clean)

    def _extract_heading(self, line: str) -> Optional[Tuple[str, str, List[int]]]:
        """
        Extract (section_id, title, section_nums) from a heading line.
        Accepts section IDs up to 5 levels: 7.7.2.1.3.
        """
        line = line.strip()
        if not line:
            return None

        # Allow optional trailing dot after section id (e.g., "3. Title").
        match = re.match(r"^(\d+(?:\.\d+){0,4})\.?\s+(.+)$", line)
        if not match:
            return None

        section_id = match.group(1)
        title = match.group(2).strip()
        section_nums, _ = self.parse_section_number(section_id)
        if section_nums is None:
            return None
        if len(section_nums) > 5:
            return None
        if not self._is_valid_heading_title(title):
            return None
        return section_id, title, section_nums

    def _load_toc_sections(self, toc_sections_path: str) -> Dict[str, SectionNode]:
        raw = json.loads(Path(toc_sections_path).read_text(encoding="utf-8"))
        sections: Dict[str, SectionNode] = {}

        # Accept both:
        # 1) Flat format: { "3": {...}, "3.1": {...} }
        # 2) Tree format: { "doc_id": "...", "sections_tree": [ {..children..} ] }
        flat_items: Dict[str, Dict[str, Any]] = {}

        if isinstance(raw, dict) and "sections_tree" in raw and isinstance(raw["sections_tree"], list):
            def flatten_tree(nodes: List[Dict[str, Any]]) -> None:
                for n in nodes:
                    sid = str(n.get("section_id", "")).strip()
                    if not sid:
                        continue
                    flat_items[sid] = {
                        "section_id": sid,
                        "title": n.get("title", ""),
                        "level": n.get("level"),
                        "parent": n.get("parent"),
                        "children": [str(c.get("section_id")) for c in n.get("children", []) if isinstance(c, dict)],
                    }
                    flatten_tree(n.get("children", []))

            flatten_tree(raw["sections_tree"])
        elif isinstance(raw, dict):
            # Flat map mode
            for sid, item in raw.items():
                if isinstance(item, dict):
                    flat_items[sid] = item

        for sid, item in flat_items.items():
            level = int(item.get("level") or len(sid.split(".")))
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            sections[sid] = SectionNode(
                section_id=sid,
                title=title,
                level=level,
                toc_page=item.get("toc_page"),
                page_num=None,
            )

        # Wire parent/children from TOC mapping.
        for sid, item in flat_items.items():
            if sid not in sections:
                continue
            parent = item.get("parent")
            children = item.get("children", []) or []
            if isinstance(parent, str) and parent in sections:
                sections[sid].parent = parent
            sections[sid].children = [c for c in children if c in sections]

        return sections

    @staticmethod
    def _match_section_id_prefix(line: str, section_ids_desc: List[str]) -> Optional[str]:
        stripped = line.strip()
        if not stripped:
            return None
        for sid in section_ids_desc:
            # Match heading-like start: "9.4 title..." or "9.4. title..."
            if re.match(rf"^{re.escape(sid)}\.?(\s+|$)", stripped):
                return sid
        return None

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()

    def _section_title_tokens(self, title: str, max_tokens: int = 5) -> List[str]:
        norm = self._normalize_for_match(title)
        tokens = [t for t in norm.split() if len(t) > 1]
        return tokens[:max_tokens]

    def _match_toc_heading_line(
        self,
        line: str,
        section_ids_desc: List[str],
        sections: Dict[str, SectionNode],
        title_tokens_by_sid: Dict[str, List[str]],
    ) -> Optional[str]:
        stripped = line.strip()
        if not stripped:
            return None

        for sid in section_ids_desc:
            # Strict heading start: section ID must be at line-start.
            m = re.match(rf"^{re.escape(sid)}\.?(\s+|$)(.*)$", stripped)
            if not m:
                continue

            # Validate heading using TOC title tokens to avoid random list items (e.g. "1. The UE ...").
            remainder = m.group(2).strip()
            remainder_norm = self._normalize_for_match(remainder)
            title_tokens = title_tokens_by_sid.get(sid, [])

            if not title_tokens:
                return sid

            # Accept when at least two title tokens appear in the remainder.
            hit_count = sum(1 for t in title_tokens if t in remainder_norm)
            if hit_count >= min(2, len(title_tokens)):
                return sid

        return None

    @staticmethod
    def _build_filtered_text_with_page_markers(
        pages: List[Any],
        skip_start: int,
        skip_from: Optional[int],
    ) -> Tuple[str, Dict[int, int]]:
        """
        Build a single text blob with [PAGE n] markers and return
        {page_num -> start_index_of_marker} for quick page-anchored searches.
        """
        chunks: List[str] = []
        page_start_index: Dict[int, int] = {}

        total_pages = len(pages)
        start_idx = max(0, skip_start)
        end_idx_exclusive = total_pages if skip_from is None else min(total_pages, max(start_idx, skip_from - 1))

        cursor = 0
        for idx in range(start_idx, end_idx_exclusive):
            page_num = idx + 1
            marker = f"\n[PAGE {page_num}]\n"
            text = pages[idx].page_content if hasattr(pages[idx], "page_content") else ""
            text = text or ""
            block = marker + text
            page_start_index[page_num] = cursor
            chunks.append(block)
            cursor += len(block)

        return "".join(chunks), page_start_index

    def _find_section_heading_match(
        self,
        text: str,
        section_id: str,
        title: str,
        start_pos: int = 0,
    ) -> Optional[re.Match]:
        # Prefer section-id + title-token validation, then fallback to id-only.
        title_tokens = self._section_title_tokens(title)
        heading_re = re.compile(rf"(^|\n)\s*{re.escape(section_id)}\.?(\s+|$)(.*)$", flags=re.MULTILINE)
        for m in heading_re.finditer(text, start_pos):
            remainder = self._normalize_for_match(m.group(3))
            if not title_tokens:
                return m
            hit_count = sum(1 for t in title_tokens if t in remainder)
            if hit_count >= min(2, len(title_tokens)):
                return m

        # Fallback: strict section-id heading only.
        sid_only_re = re.compile(rf"(^|\n)\s*{re.escape(section_id)}\.?(\s+|$)", flags=re.MULTILINE)
        return sid_only_re.search(text, start_pos)

    @staticmethod
    def _is_header_footer_line(line: str) -> bool:
        s = (line or "").strip()
        if not s:
            return False
        s_norm = " ".join(s.lower().split())

        # Common ETSI/3GPP running headers/footers and page-only noise.
        if s_norm in {"etsi", "3gpp", "release 18"}:
            return True
        if re.fullmatch(r"\d+", s_norm):
            return True
        if re.search(r"\betsi\s+ts\s+\d{3}\s+\d{3}\b", s_norm):
            return True
        if re.search(r"\b3gpp\s+ts\s+\d+\.\d+\b", s_norm):
            return True
        if re.search(r"\bversion\s+\d+\.\d+\.\d+\b", s_norm) and "3gpp ts" in s_norm:
            return True
        if re.search(r"\brelease\s+\d+\b", s_norm) and "3gpp ts" in s_norm:
            return True
        if re.search(r"\(20\d{2}-\d{2}\)", s_norm) and "etsi ts" in s_norm:
            return True
        return False

    def _clean_section_lines(self, text: str) -> List[str]:
        lines = text.split("\n")
        cleaned: List[str] = []
        for line in lines:
            # Drop page markers and known recurring headers/footers.
            if re.search(r"\[PAGE\s+\d+\]", line):
                continue
            if self._is_header_footer_line(line):
                continue
            cleaned.append(line.rstrip())

        # Collapse excessive blank lines.
        normalized: List[str] = []
        blank_run = 0
        for line in cleaned:
            if line.strip():
                blank_run = 0
                normalized.append(line)
            else:
                blank_run += 1
                if blank_run <= 1:
                    normalized.append("")
        # Trim leading/trailing blanks.
        while normalized and not normalized[0].strip():
            normalized.pop(0)
        while normalized and not normalized[-1].strip():
            normalized.pop()
        return normalized

    def load_and_parse_pdf(
        self,
        pdf_path: str,
        skip_start: int = 10,
        skip_from: Optional[int] = None,
        toc_sections_path: Optional[str] = None,
    ) -> Dict[str, SectionNode]:
        if toc_sections_path:
            return self.load_and_parse_pdf_with_toc(
                pdf_path=pdf_path,
                toc_sections_path=toc_sections_path,
                skip_start=skip_start,
                skip_from=skip_from,
            )

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        end_index = skip_from - 1
        filtered_pages = pages[skip_start:end_index] if end_index > skip_start else pages[skip_start:]

        sections: Dict[str, SectionNode] = {}
        current_section_stack: List[Tuple[List[int], str]] = []
        _sections_by_level = defaultdict(int)

        for page_idx, page in enumerate(filtered_pages):
            page_num = skip_start + page_idx + 1
            lines = page.page_content.split("\n")

            for line in lines:
                line_stripped = line.strip()
                heading = self._extract_heading(line_stripped)

                if heading:
                    section_id, title, section_nums = heading
                    level = len(section_nums)
                    _sections_by_level[level] += 1

                    # Do not overwrite existing section IDs; duplicate captures are usually noise.
                    if section_id in sections:
                        if current_section_stack:
                            current_section_id = current_section_stack[-1][1]
                            sections[current_section_id].add_content_line(line, page_num)
                        continue

                    section_node = SectionNode(section_id, title, level, page_num)
                    sections[section_id] = section_node

                    while current_section_stack:
                        parent_nums, parent_id = current_section_stack[-1]
                        if self.is_parent_of(parent_nums, section_nums):
                            parent_node = sections[parent_id]
                            parent_node.children.append(section_id)
                            section_node.parent = parent_id
                            break
                        current_section_stack.pop()

                    current_section_stack.append((section_nums, section_id))
                else:
                    if current_section_stack:
                        current_section_id = current_section_stack[-1][1]
                        sections[current_section_id].add_content_line(line, page_num)

        for section_id, node in sections.items():
            if not node.children:
                node.is_leaf = True

        return sections

    def load_and_parse_pdf_with_toc(
        self,
        pdf_path: str,
        toc_sections_path: str,
        skip_start: int = 10,
        skip_from: Optional[int] = None,
    ) -> Dict[str, SectionNode]:
        """
        Parse section content using TOC-derived structure.
        Section IDs/titles/hierarchy come from toc_sections.json.
        Body scanning only assigns content lines to the currently active section.
        """
        sections = self._load_toc_sections(toc_sections_path)
        if not sections:
            raise ValueError(f"No valid TOC sections found in: {toc_sections_path}")

        # Use page-anchored section slicing: current section heading -> next section heading
        # in TOC order (any next section: sibling/child/parent level).
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        body_text, page_start_index = self._build_filtered_text_with_page_markers(
            pages=pages,
            skip_start=skip_start,
            skip_from=skip_from,
        )

        if not body_text.strip():
            raise ValueError("No body text available for TOC-based extraction.")

        def sid_sort_key(sid: str) -> Tuple[int, List[int]]:
            toc_page = sections[sid].toc_page if isinstance(sections[sid].toc_page, int) else 10**9
            return (toc_page, [int(x) for x in sid.split(".")])

        ordered_ids = sorted(sections.keys(), key=sid_sort_key)

        for i, sid in enumerate(ordered_ids):
            node = sections[sid]
            anchor_page = node.toc_page if isinstance(node.toc_page, int) else (skip_start + 1)
            start_anchor_idx = page_start_index.get(anchor_page, 0)

            start_match = self._find_section_heading_match(
                text=body_text,
                section_id=sid,
                title=node.title,
                start_pos=start_anchor_idx,
            )
            if not start_match:
                # Global fallback if anchor page mapping is imperfect.
                start_match = self._find_section_heading_match(
                    text=body_text,
                    section_id=sid,
                    title=node.title,
                    start_pos=0,
                )
            if not start_match:
                continue

            start_idx = start_match.start()
            end_idx = len(body_text)
            if i + 1 < len(ordered_ids):
                next_sid = ordered_ids[i + 1]
                next_node = sections[next_sid]
                next_anchor_page = next_node.toc_page if isinstance(next_node.toc_page, int) else anchor_page
                next_anchor_idx = page_start_index.get(next_anchor_page, start_idx + 1)
                next_match = self._find_section_heading_match(
                    text=body_text,
                    section_id=next_sid,
                    title=next_node.title,
                    start_pos=max(start_idx + 1, next_anchor_idx),
                )
                if not next_match:
                    # fallback global search after current start
                    next_match = self._find_section_heading_match(
                        text=body_text,
                        section_id=next_sid,
                        title=next_node.title,
                        start_pos=start_idx + 1,
                    )
                if next_match:
                    end_idx = next_match.start()

            section_slice = body_text[start_idx:end_idx]
            page_hits = [int(x) for x in re.findall(r"\[PAGE\s+(\d+)\]", section_slice)]
            clean_text = re.sub(r"\n?\[PAGE\s+\d+\]\n?", "\n", section_slice).strip()
            if not clean_text:
                continue

            heading_line_re = re.compile(
                rf"^\s*{re.escape(sid)}\.?\s+.*?$",
                flags=re.IGNORECASE | re.MULTILINE,
            )
            clean_body_only = heading_line_re.sub("", clean_text, count=1).strip()
            cleaned_lines = self._clean_section_lines(clean_body_only)

            if page_hits:
                node.page_numbers.update(page_hits)
                if node.page_num is None:
                    node.page_num = min(page_hits)
            elif isinstance(node.toc_page, int):
                node.page_num = node.toc_page
                node.page_numbers.add(node.toc_page)

            if cleaned_lines:
                node.content_lines = cleaned_lines
                node.has_direct_text = any(line.strip() for line in node.content_lines)

        for section_id, node in sections.items():
            if not node.children:
                node.is_leaf = True

        return sections

    @staticmethod
    def _get_parent_path(section_id: str, sections: Dict[str, SectionNode]) -> List[str]:
        path: List[str] = []
        current_id: Optional[str] = section_id
        while current_id:
            path.insert(0, current_id)
            if current_id in sections and sections[current_id].parent:
                current_id = sections[current_id].parent
            else:
                break
        return path

    @staticmethod
    def _get_descendant_ids(section_id: str, sections: Dict[str, SectionNode]) -> List[str]:
        descendants: List[str] = []

        def dfs(node_id: str) -> None:
            for child_id in sections[node_id].children:
                descendants.append(child_id)
                dfs(child_id)

        if section_id in sections:
            dfs(section_id)
        return descendants

    def extract_deepest_chunks(self, sections: Dict[str, SectionNode]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        leaf_sections = [sec_id for sec_id, node in sections.items() if node.is_leaf]

        for section_id in leaf_sections:
            node = sections[section_id]
            chunks.append(
                {
                    "section_id": section_id,
                    "section_title": node.title,
                    "content": node.get_full_content(),
                    "metadata": {
                        "doc_id": self.doc_id,
                        "section_number": section_id,
                        "section_title": node.title,
                        "parent_section_id": node.parent,
                        "parent_path": self._get_parent_path(section_id, sections),
                        "level": node.level,
                        "child_section_ids": [],
                        "child_section_ids_recursive": [],
                        "page_numbers": sorted(list(node.page_numbers)) if node.page_numbers else [],
                        "has_children": False,
                        "is_leaf": True,
                        "direct_text_only": False,
                    },
                }
            )

        processed_parents: set = set()

        def process_parent(section_id: str) -> None:
            if section_id in processed_parents or section_id not in sections:
                return

            node = sections[section_id]
            if node.children:
                direct_text_lines: List[str] = []

                for line in node.content_lines:
                    is_child_section = False
                    for child_id in node.children:
                        child_node = sections[child_id]
                        if line.strip().startswith(child_node.section_id):
                            is_child_section = True
                            break
                    if is_child_section:
                        break
                    direct_text_lines.append(line)

                direct_content = f"{node.section_id} {node.title}"
                if direct_text_lines:
                    direct_content += "\n" + "\n".join(direct_text_lines)

                chunks.append(
                    {
                        "section_id": section_id,
                        "section_title": node.title,
                        "content": direct_content.strip(),
                        "metadata": {
                            "doc_id": self.doc_id,
                            "section_number": section_id,
                            "section_title": node.title,
                            "parent_section_id": node.parent,
                            "parent_path": self._get_parent_path(section_id, sections),
                            "level": node.level,
                            "child_section_ids": node.children,
                            "child_section_ids_recursive": self._get_descendant_ids(section_id, sections),
                            "page_numbers": sorted(list(node.page_numbers)) if node.page_numbers else [],
                            "has_children": True,
                            "is_leaf": False,
                            "direct_text_only": True,
                        },
                    }
                )
                processed_parents.add(section_id)

            if node.parent:
                process_parent(node.parent)

        for section_id in leaf_sections:
            parent = sections[section_id].parent
            if parent:
                process_parent(parent)

        return chunks

    def run(
        self,
        pdf_path: str,
        skip_start: int = 10,
        skip_from: Optional[int] = None,
        toc_sections_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        sections = self.load_and_parse_pdf(
            pdf_path=pdf_path,
            skip_start=skip_start,
            skip_from=skip_from,
            toc_sections_path=toc_sections_path,
        )
        chunks = self.extract_deepest_chunks(sections)
        return {
            "doc_id": self.doc_id,
            "pdf_path": pdf_path,
            "sections_count": len(sections),
            "chunks_count": len(chunks),
            "sections": sections,
            "chunks": chunks,
        }


def save_ingestion_outputs(
    doc_id: str,
    sections: Dict[str, SectionNode],
    chunks: List[Dict[str, Any]],
    output_root: str,
) -> Dict[str, str]:
    output_dir = Path(output_root) / doc_id
    output_dir.mkdir(parents=True, exist_ok=True)

    sections_path = output_dir / "sections.json"
    chunks_path = output_dir / "chunks.json"

    serializable_sections = {
        sid: {
            "section_id": node.section_id,
            "title": node.title,
            "level": node.level,
            "toc_page": node.toc_page,
            "page_num": node.page_num,
            "page_numbers": sorted(list(node.page_numbers)),
            "children": node.children,
            "children_recursive": SpecIngestionChunker._get_descendant_ids(sid, sections),
            "parent": node.parent,
            "has_direct_text": node.has_direct_text,
            "is_leaf": node.is_leaf,
        }
        for sid, node in sections.items()
    }

    sections_path.write_text(json.dumps(serializable_sections, indent=2), encoding="utf-8")
    chunks_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")

    return {
        "sections_path": str(sections_path),
        "chunks_path": str(chunks_path),
    }


if __name__ == "__main__":
    # Static input configuration (no CLI args).
    # Update these values as needed for your local run.
    DOC_ID = "ts_138401v180600p"
    PDF_PATH = "../data/specs/ts_138401v180600p.pdf"
    OUTPUT_ROOT = "./spec_chunks"
    TOC_SECTIONS_PATH = "./spec_chunks/ts_138401v180600p/toc_sections.json"
    RUN_TOC_PARSER_FIRST = True
    TOC_START_PAGE: Optional[int] = 4
    TOC_END_PAGE: Optional[int] = 9
    STRICT_TOC_ONLY = True
    SKIP_START = 9
    # Keep as None to avoid truncating section content in later pages.
    SKIP_FROM: Optional[int] = None

    print("=" * 72)
    print("KG-Only End-to-End: TOC -> Ingestion -> Chunking")
    print("=" * 72)
    print(f"Entry point: {Path(__file__).name}")
    print(f"doc_id      : {DOC_ID}")
    print(f"pdf_path    : {PDF_PATH}")
    print(f"output_root : {OUTPUT_ROOT}")
    print(f"toc_sections : {TOC_SECTIONS_PATH}")
    print(f"skip_start   : {SKIP_START}")
    print(f"skip_from    : {SKIP_FROM if SKIP_FROM is not None else 'EOF'}")
    print(f"run_toc_first: {RUN_TOC_PARSER_FIRST}")
    if RUN_TOC_PARSER_FIRST:
        print(f"toc_range   : {TOC_START_PAGE} - {TOC_END_PAGE}")
    print("-" * 72)

    pdf_file = Path(PDF_PATH)
    if not pdf_file.exists():
        raise FileNotFoundError(
            f"Configured PDF not found: {PDF_PATH}\n"
            "Update PDF_PATH in this file before running."
        )

    # Stage 1: Build TOC mapping first (optional but recommended).
    if RUN_TOC_PARSER_FIRST:
        print("[Stage 1/2] Parsing TOC and saving toc_sections.json ...")
        toc_entries = parse_toc_sections(
            str(pdf_file),
            toc_start_page=TOC_START_PAGE,
            toc_end_page=TOC_END_PAGE,
            strict_toc_only=STRICT_TOC_ONLY,
        )
        saved_toc_path = save_toc_sections(toc_entries, OUTPUT_ROOT, DOC_ID)
        TOC_SECTIONS_PATH = saved_toc_path
        print(f"TOC sections mapped: {len(toc_entries)}")
        print(f"toc_sections.json : {TOC_SECTIONS_PATH}")
        print("-" * 72)

    print("[Stage 2/2] Extracting section content and building chunks ...")
    chunker = SpecIngestionChunker(doc_id=DOC_ID)
    result = chunker.run(
        pdf_path=str(pdf_file),
        skip_start=SKIP_START,
        skip_from=SKIP_FROM,
        toc_sections_path=TOC_SECTIONS_PATH if Path(TOC_SECTIONS_PATH).exists() else None,
    )

    paths = save_ingestion_outputs(
        doc_id=DOC_ID,
        sections=result["sections"],
        chunks=result["chunks"],
        output_root=OUTPUT_ROOT,
    )

    print(f"Sections parsed : {result['sections_count']}")
    print(f"Chunks generated: {result['chunks_count']}")
    print(f"sections.json   : {paths['sections_path']}")
    print(f"chunks.json     : {paths['chunks_path']}")
    print("=" * 72)

