"""
Extract chunks (functions, structs/unions/enums, variables) from parsed C code.
Each chunk = full source of one entity + metadata. No KG edges yet.
"""

import json
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

from tree_sitter import Node, Parser, Tree

from .parse_codebase import (
    CODEBASE_ROOT,
    TARGET_DIRS,
    C_LANGUAGE,
    collect_c_h_files,
    parse_file,
)


def get_node_text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _first_identifier(node: Node, source_bytes: bytes) -> str | None:
    """First identifier text found in subtree (depth-first)."""
    if node.type == "identifier":
        return get_node_text(node, source_bytes).strip()
    for child in node.children:
        out = _first_identifier(child, source_bytes)
        if out:
            return out
    return None


def get_function_name(func_def_node: Node, source_bytes: bytes) -> str:
    """Function name from function_definition node."""
    decl = func_def_node.child_by_field_name("declarator")
    if decl is None:
        for c in func_def_node.children:
            if c.type == "declarator":
                decl = c
                break
    if decl is not None:
        name = _first_identifier(decl, source_bytes)
        if name:
            return name
    return f"anonymous_func_L{func_def_node.start_point[0] + 1}"


def get_type_specifier_name(spec_node: Node, source_bytes: bytes) -> str:
    """Name from struct_specifier / union_specifier / enum_specifier."""
    name_node = spec_node.child_by_field_name("name")
    if name_node is not None and name_node.type in ("type_identifier", "identifier"):
        return get_node_text(name_node, source_bytes).strip()
    # fallback: first type_identifier or identifier in children
    for c in spec_node.children:
        if c.type in ("type_identifier", "identifier"):
            return get_node_text(c, source_bytes).strip()
    line = spec_node.start_point[0] + 1
    kind = spec_node.type.replace("_specifier", "")
    return f"anonymous_{kind}_L{line}"


def get_variable_name_from_decl(decl_node: Node, source_bytes: bytes) -> str | None:
    """First variable name from a declaration. Handles tree-sitter-c: declarator field(s) or init_declarator."""
    # Single declarator from field
    single = decl_node.child_by_field_name("declarator")
    if single is not None:
        name = _first_identifier(single, source_bytes)
        if name:
            return name
    # Walk all children: init_declarator, identifier, pointer_declarator, array_declarator
    for c in decl_node.children:
        if c.type == "init_declarator":
            decl = c.child_by_field_name("declarator")
            if decl is None:
                for cc in c.children:
                    if cc.type in ("declarator", "identifier", "pointer_declarator", "array_declarator"):
                        decl = cc
                        break
            if decl is not None:
                name = _first_identifier(decl, source_bytes)
                if name:
                    return name
        elif c.type in ("identifier", "pointer_declarator", "array_declarator", "declarator"):
            name = _first_identifier(c, source_bytes)
            if name:
                return name
    return None


def _has_body(spec_node: Node) -> bool:
    """True if struct/union has field_declaration_list; enum has enumerator_list."""
    for c in spec_node.children:
        if c.type in ("field_declaration_list", "enumerator_list"):
            return True
    return False


def _is_top_level_declaration(decl_node: Node, root: Node) -> bool:
    """True if this declaration is a direct child of translation_unit."""
    return decl_node.parent is not None and decl_node.parent.id == root.id


def _declaration_has_variables(decl_node: Node) -> bool:
    """True if declaration declares variable(s). Skip function declarations (only function_declarator)."""
    var_like = ("init_declarator", "identifier", "pointer_declarator", "array_declarator", "declarator")
    has_var = any(c.type in var_like for c in decl_node.children)
    has_only_func = any(c.type == "function_declarator" for c in decl_node.children) and not has_var
    if has_only_func:
        return False
    return has_var


def extract_chunks_from_tree(
    file_path: Path,
    tree: Tree,
    source_bytes: bytes,
    root_path: Path,
) -> list[dict]:
    """
    Extract one chunk per function, struct/union/enum (with body), and top-level variable.
    Returns list of dicts: node_id, name, entity_type, file_path, line_start, line_end, text.
    """
    chunks = []
    root = tree.root_node
    try:
        rel_path = file_path.relative_to(root_path)
    except ValueError:
        rel_path = Path(file_path.name)

    def visit(node: Node, at_file_scope: bool = True) -> None:
        if node.type == "function_definition":
            name = get_function_name(node, source_bytes)
            text = get_node_text(node, source_bytes)
            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1
            node_id = f"function:{rel_path}::{name}"
            chunks.append({
                "node_id": node_id,
                "name": name,
                "entity_type": "function",
                "file_path": str(rel_path),
                "line_start": line_start,
                "line_end": line_end,
                "text": text,
            })
            for child in node.children:
                if child.type == "compound_statement":
                    visit(child, False)
                else:
                    visit(child, at_file_scope)
            return
        if node.type == "struct_specifier" and _has_body(node):
            name = get_type_specifier_name(node, source_bytes)
            text = get_node_text(node, source_bytes)
            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1
            node_id = f"struct:{rel_path}::{name}"
            chunks.append({
                "node_id": node_id,
                "name": name,
                "entity_type": "struct",
                "file_path": str(rel_path),
                "line_start": line_start,
                "line_end": line_end,
                "text": text,
            })
            return
        if node.type == "union_specifier" and _has_body(node):
            name = get_type_specifier_name(node, source_bytes)
            text = get_node_text(node, source_bytes)
            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1
            node_id = f"union:{rel_path}::{name}"
            chunks.append({
                "node_id": node_id,
                "name": name,
                "entity_type": "union",
                "file_path": str(rel_path),
                "line_start": line_start,
                "line_end": line_end,
                "text": text,
            })
            return
        if node.type == "enum_specifier" and _has_body(node):
            name = get_type_specifier_name(node, source_bytes)
            text = get_node_text(node, source_bytes)
            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1
            node_id = f"enum:{rel_path}::{name}"
            chunks.append({
                "node_id": node_id,
                "name": name,
                "entity_type": "enum",
                "file_path": str(rel_path),
                "line_start": line_start,
                "line_end": line_end,
                "text": text,
            })
            return
        if node.type == "declaration" and at_file_scope and _declaration_has_variables(node):
            name = get_variable_name_from_decl(node, source_bytes)
            if name:
                text = get_node_text(node, source_bytes)
                line_start = node.start_point[0] + 1
                line_end = node.end_point[0] + 1
                node_id = f"variable:{rel_path}::{name}"
                chunks.append({
                    "node_id": node_id,
                    "name": name,
                    "entity_type": "variable",
                    "file_path": str(rel_path),
                    "line_start": line_start,
                    "line_end": line_end,
                    "text": text,
                    "is_global": True,  # file-scope only; used by KG to create function_uses_variable edges
                })
            return
        for child in node.children:
            visit(child, at_file_scope)

    for child in root.children:
        visit(child, True)

    # Deduplicate by node_id (same entity can appear in tree in multiple ways)
    seen = set()
    unique = []
    for c in chunks:
        if c["node_id"] not in seen:
            seen.add(c["node_id"])
            unique.append(c)
    return unique


def run_extraction(
    root: Path | None = None,
    target_dirs: list | None = None,
    max_files: int | None = None,
    output_json: Path | None = None,
) -> list[dict]:
    """
    Parse entire codebase and extract chunks. Returns list of all chunk dicts.
    If output_json is set, writes chunks to that file (texts may be large).
    """
    root = root or CODEBASE_ROOT
    target_dirs = target_dirs or TARGET_DIRS
    if not root.exists():
        raise FileNotFoundError(f"Codebase root not found: {root}")

    paths = collect_c_h_files(root, target_dirs=target_dirs, max_files=max_files)
    parser = Parser(C_LANGUAGE)
    all_chunks = []
    errors = 0
    for i, file_path in enumerate(paths):
        result = parse_file(file_path, parser)
        if result is None:
            errors += 1
            continue
        tree, source_bytes = result
        try:
            chunks = extract_chunks_from_tree(file_path, tree, source_bytes, root)
            all_chunks.extend(chunks)
        except Exception as e:
            errors += 1
            if errors <= 3:
                
                pass  # print(f"Extract error {file_path}: {e}")
        # if (i + 1) % 500 == 0:
            pass  # print(f"  Processed {i + 1}/{len(paths)} files, {len(all_chunks)} chunks so far...")

    # print(f"Extracted {len(all_chunks)} chunks from {len(paths) - errors} files ({errors} errors).")
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        # print(f"Wrote {output_json}")
    return all_chunks


if __name__ == "__main__":
    import sys
    out = Path(__file__).parent / "outputs" / "chunks.json"
    run_extraction(output_json=out)
    sys.exit(0)
