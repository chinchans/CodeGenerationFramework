"""
Codebase parser using Tree-sitter (C grammar).
Reads .c/.h files and shows how the syntax tree is built for each file.
Use this to verify parsing before building the KG in the next step.
"""

import sys
from pathlib import Path

# Tree-sitter: core parser + C language
from tree_sitter import Language, Parser, Node, Tree

import logging

logger = logging.getLogger(__name__)

# C grammar (pre-built wheel exposes language())
try:
    import tree_sitter_c as tsc
except ImportError:
    # print("Install C grammar: pip install tree-sitter-c")
    sys.exit(1)




# Build C language once
C_LANGUAGE = Language(tsc.language())

# Set your codebase root (full path)
CODEBASE_ROOT = Path(r"./openairinterface5g-develop")

# Directories under CODEBASE_ROOT to scan for .c/.h (None = all)
TARGET_DIRS = ["openair1", "openair2", "openair3", "common"]


# -----------------------------------------------------------------------------
# Tree display helpers
# -----------------------------------------------------------------------------

def get_node_text(node: Node, source_bytes: bytes) -> str:
    """Return the source text span for a node (for display)."""
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def get_node_preview(node: Node, source_bytes: bytes, max_len: int = 60) -> str:
    """Short one-line preview of node text."""
    raw = get_node_text(node, source_bytes).replace("\n", " ").strip()
    if len(raw) > max_len:
        return raw[: max_len - 3] + "..."
    return raw or "(empty)"


def print_tree_recursive(
    node: Node,
    source_bytes: bytes,
    indent: int = 0,
    max_depth: int = 20,
    show_preview: bool = True,
    max_text_len: int = 50,
) -> None:
    """
    Print the syntax tree with indentation.
    - node_type (line_start, line_end) [preview]
    """
    if indent > max_depth:
        return
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    prefix = "  " * indent
    preview = ""
    if show_preview and node.child_count == 0:
        preview = "  # " + get_node_preview(node, source_bytes, max_text_len)
    # print(f"{prefix}{node.type} (L{line_start}-{line_end}){preview}")
    for child in node.children:
        print_tree_recursive(
            child, source_bytes, indent + 1, max_depth, show_preview, max_text_len
        )


def print_tree_sexp(tree: Tree) -> None:
    """Print the full tree as an s-expression (compact)."""
    # print(tree.root_node.sexp())


# -----------------------------------------------------------------------------
# Codebase discovery
# -----------------------------------------------------------------------------

def collect_c_h_files(
    root: Path,
    target_dirs: list | None = None,
    max_files: int | None = None,
) -> list:
    """
    Collect all .c and .h files under root.
    If target_dirs is set (e.g. ['openair1','openair2','openair3','common']),
    only those direct children of root are scanned.
    """
    if target_dirs:
        paths = []
        for d in target_dirs:
            dir_path = root / d
            if dir_path.is_dir():
                paths.extend(dir_path.rglob("*.c"))
                paths.extend(dir_path.rglob("*.h"))
        paths = sorted(set(paths))
    else:
        paths = sorted(root.rglob("*.c")) + sorted(root.rglob("*.h"))
        paths = sorted(set(paths))

    if max_files is not None:
        paths = paths[: max_files]
    return paths


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------

def parse_file(file_path: Path, parser: Parser) -> tuple[Tree, bytes] | None:
    """Parse a single file. Returns (tree, source_bytes) or None on read error."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        # print(f"Read error {file_path}: {e}")
        return None
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    return tree, source_bytes


def run_single_file(
    file_path: Path,
    format: str = "indent",
    max_depth: int = 20,
    show_preview: bool = True,
) -> None:
    """Parse one file and print its tree."""
    parser = Parser(C_LANGUAGE)
    result = parse_file(file_path, parser)
    if result is None:
        return
    tree, source_bytes = result
    root = tree.root_node

    # print("=" * 70)
    # print(f"FILE: {file_path}")
    # print(f"ROOT: {root.type}  (L{root.start_point[0]+1}-{root.end_point[0]+1})")
    # print("=" * 70)

    if format == "sexp":
        print_tree_sexp(tree)
    else:
        print_tree_recursive(
            root, source_bytes, indent=0, max_depth=max_depth,
            show_preview=show_preview,
        )
    # print()


def run_multiple_files(
    file_paths: list[Path],
    format: str = "indent",
    max_depth: int = 8,
    show_preview: bool = False,
) -> None:
    """Parse multiple files and print a short tree summary for each."""
    parser = Parser(C_LANGUAGE)
    for i, file_path in enumerate(file_paths):
        result = parse_file(file_path, parser)
        if result is None:
            continue
        tree, source_bytes = result
        root = tree.root_node
        # print(f"[{i+1}/{len(file_paths)}] {file_path}")
        # print(f"    root: {root.type}  children: {root.child_count}  L{root.start_point[0]+1}-{root.end_point[0]+1}")
        if format == "indent" and max_depth > 0:
            print_tree_recursive(
                root, source_bytes, indent=1, max_depth=max_depth,
                show_preview=show_preview,
            )
        # print()


# -----------------------------------------------------------------------------
# Main: parse entire codebase (no args)
# -----------------------------------------------------------------------------

def main() -> None:
    """Parse entire codebase with no arguments. One-line summary per file."""
    root = CODEBASE_ROOT
    if not root.exists():
        # print(f"Codebase root not found: {root}")
        sys.exit(1)

    paths = collect_c_h_files(root, target_dirs=TARGET_DIRS, max_files=None)
    # print(f"Codebase root: {root}")
    # print(f"Parsing {len(paths)} .c/.h files...")
    # print()

    parser = Parser(C_LANGUAGE)
    ok = 0
    err = 0
    for i, file_path in enumerate(paths):
        result = parse_file(file_path, parser)
        if result is None:
            err += 1
            continue
        tree, source_bytes = result
        root_node = tree.root_node
        line_start = root_node.start_point[0] + 1
        line_end = root_node.end_point[0] + 1
        # print(f"[{i+1}/{len(paths)}] {file_path}  root={root_node.type}  L{line_start}-{line_end}  children={root_node.child_count}")
        ok += 1

    # print()
    # print(f"Done. Parsed {ok} files, {err} errors.")


if __name__ == "__main__":
    main()
