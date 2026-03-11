#!/usr/bin/env python3
"""
After pulling a repo (e.g. you created a branch on client, pushed, then pulled on server),
export changed files into two flat folders:

1. Git_Diff         - Full content of each changed file (same filename, flat).
2. Difference_Folder - Only the added/changed code per file (no diff format). Same filename as source (.c/.h).

Compares current branch ONLY (e.g. new_feature) to its base (main/origin/main):
only changes in that branch are exported. Run from the repo root of that branch.
Override base with 2nd arg: python git_diff_export.py . main
"""

import os
import re
import subprocess
import sys
from datetime import datetime


def get_current_branch(repo_path):
    """Current branch name, or None if detached."""
    r = subprocess.run(
        ["git", "-C", repo_path, "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        return None
    name = r.stdout.strip()
    return name if name != "HEAD" else None


def get_default_base_ref(repo_path):
    """Base ref for 'branch vs where it came from': main, origin/main, master, or HEAD~1."""
    for ref in ("main", "origin/main", "master", "origin/master"):
        r = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "--verify", ref],
            capture_output=True,
        )
        if r.returncode == 0:
            return ref
    return "HEAD~1"


def get_changed_files(repo_path, base_ref):
    """Return list of relative paths of files changed in current branch only (base_ref...HEAD)."""
    cmd = ["git", "-C", repo_path, "diff", "--name-only", base_ref + "...HEAD"]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError("git diff failed: %s" % (result.stderr or result.stdout))
    paths = [p.strip() for p in (result.stdout or "").splitlines() if p.strip()]
    # Skip ASN specification files; we only want C/C headers in Git_Diff/Difference_Folder.
    paths = [p for p in paths if not p.lower().endswith(".asn")]
    return paths


def build_unique_basename_map(rel_paths):
    """Map rel_path -> flat filename. Same basename when unique; add path prefix if duplicate."""
    base_count = {}
    out = {}
    for rel in rel_paths:
        base = os.path.basename(rel)
        if base not in base_count:
            base_count[base] = []
        base_count[base].append(rel)
    for base, paths in base_count.items():
        if len(paths) == 1:
            out[paths[0]] = base
        else:
            for i, rel in enumerate(paths):
                # e.g. openair2/F1AP/f1ap.c -> openair2_F1AP_f1ap.c
                prefix = rel.replace("\\", "/").rsplit("/", 1)[0].replace("/", "_")
                out[rel] = "%s_%s" % (prefix, base) if prefix else base
    return out


def export_full_files(repo_path, rel_paths, out_dir):
    """Copy full content of each changed file into out_dir (flat, same basename style)."""
    os.makedirs(out_dir, exist_ok=True)
    name_map = build_unique_basename_map(rel_paths)
    copied = 0
    for rel in rel_paths:
        src = os.path.join(repo_path, rel)
        if not os.path.isfile(src):
            continue
        dest_name = name_map[rel]
        dest = os.path.join(out_dir, dest_name)
        try:
            with open(src, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            with open(dest, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
            copied += 1
        except Exception as e:
            sys.stderr.write("Skip %s: %s\n" % (rel, e))
    return copied


def diff_to_code_only(patch_text):
    """Extract only added lines from git diff output, as plain code (no +/-, no @@, no ---/+++)."""
    lines = []
    for line in patch_text.splitlines():
        if line.startswith("+++ ") or line.startswith("--- ") or line.startswith("@@"):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            lines.append(line[1:].rstrip("\r"))
        # skip - lines (removed) and diff header (diff --git ...)
    return "\n".join(lines) + ("\n" if lines else "")


# Pattern for start of typedef struct (e.g. "typedef struct f1ap_ue_context_setup_req_s {")
_TYPEDEF_STRUCT_RE = re.compile(
    r"typedef\s+struct\s+(\w+)\s*\{",
    re.MULTILINE,
)


def _brace_match_from(content, open_pos):
    """
    Given content and the index of an opening '{', return the index of the matching '};'
    (the '}' that closes the brace, followed by optional whitespace and ';').
    Returns -1 if not found. Counts only { and } for balance (no string/comment skip).
    """
    depth = 0
    i = open_pos
    n = len(content)
    while i < n:
        if content[i] == "{":
            depth += 1
        elif content[i] == "}":
            depth -= 1
            if depth == 0:
                # we reached the closing brace of this struct; now require a ';'
                # somewhere after the brace to treat it as a complete typedef
                j = i + 1
                while j < n and content[j] != ";":
                    j += 1
                if j < n and content[j] == ";":
                    return j + 1  # position just after ';'
                # no semicolon found => treat as incomplete
                return -1
        i += 1
    return -1


def extract_complete_struct(full_content, struct_name):
    """
    Extract the full typedef struct definition for struct_name from full_content.
    Returns the substring including "typedef struct name { ... };", or None if not found.
    """
    # Match "typedef struct struct_name " or "typedef struct struct_name\n{"
    pattern = re.compile(
        r"typedef\s+struct\s+" + re.escape(struct_name) + r"\s*\{",
        re.MULTILINE,
    )
    m = pattern.search(full_content)
    if not m:
        return None
    start = m.start()
    open_brace = full_content.index("{", m.end() - 1)
    end = _brace_match_from(full_content, open_brace)
    if end == -1:
        return None
    return full_content[start:end]


def find_incomplete_struct_spans(code_only):
    """
    Find typedef struct blocks in code_only that are incomplete (no matching };).
    Returns list of (struct_name, start_index, end_index). end_index is exclusive.
    """
    spans = []
    for m in _TYPEDEF_STRUCT_RE.finditer(code_only):
        name = m.group(1)
        start = m.start()
        open_brace = code_only.index("{", m.end() - 1)
        end = _brace_match_from(code_only, open_brace)
        if end == -1:
            # Incomplete: from this typedef to next "typedef struct" or end of content
            next_typedef = _TYPEDEF_STRUCT_RE.search(code_only, start + 1)
            end = next_typedef.start() if next_typedef else len(code_only)
            spans.append((name, start, end))
    return spans


def fix_incomplete_structs(code_only, full_content):
    """
    If code_only contains incomplete typedef structs (e.g. only the opening line),
    replace each with the complete struct from full_content (original file).
    """
    incomplete = find_incomplete_struct_spans(code_only)
    if not incomplete:
        return code_only
    # Replace from end to start so indices stay valid
    result = code_only
    for name, start, end in reversed(incomplete):
        full_struct = extract_complete_struct(full_content, name)
        if full_struct:
            result = result[:start] + full_struct + result[end:]
    return result


def export_diffs(repo_path, rel_paths, out_dir, base_ref):
    """Write per-file: only the added/changed code (no diff format). Same filename as source (.c/.h, no .patch).
    If the diff-only output contains an incomplete typedef struct (e.g. only the opening line),
    it is replaced with the complete struct from the full file in the repo."""
    os.makedirs(out_dir, exist_ok=True)
    name_map = build_unique_basename_map(rel_paths)
    written = 0
    for rel in rel_paths:
        dest_name = name_map[rel]  # same as Git_Diff: .c or .h, no .patch
        dest = os.path.join(out_dir, dest_name)
        cmd = ["git", "-C", repo_path, "diff", base_ref + "...HEAD", "--", rel]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            continue
        code_only = diff_to_code_only(result.stdout or "")
        if not code_only.strip():
            continue
        try:
            full_path = os.path.join(repo_path, rel)
            if os.path.isfile(full_path):
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    full_content = f.read()
                code_only = fix_incomplete_structs(code_only, full_content)
            with open(dest, "w", encoding="utf-8", newline="\n") as f:
                f.write(code_only)
            written += 1
        except Exception as e:
            sys.stderr.write("Skip diff %s: %s\n" % (rel, e))
    return written


def main():
    repo_path = os.getcwd()
    if len(sys.argv) > 1:
        repo_path = os.path.abspath(sys.argv[1])
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        print("Not a git repo: %s" % repo_path)
        sys.exit(1)
    base_ref = sys.argv[2] if len(sys.argv) > 2 else get_default_base_ref(repo_path)
    branch = get_current_branch(repo_path)
    print("Branch: %s" % (branch or "HEAD (detached)"))
    print("Diff: %s...HEAD (this branch only)" % base_ref)

    changed = get_changed_files(repo_path, base_ref)
    if not changed:
        print("No changed files in this branch.")
        sys.exit(0)

    # Create folders next to this script: Git_Diff/<timestamp>/ and Difference_Folder/<timestamp>/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_diff_base = os.path.join(script_dir, "Git_Diff")
    diff_folder_base = os.path.join(script_dir, "Difference_Folder")
    git_diff_dir = os.path.join(git_diff_base, timestamp)
    diff_folder_dir = os.path.join(diff_folder_base, timestamp)

    n1 = export_full_files(repo_path, changed, git_diff_dir)
    n2 = export_diffs(repo_path, changed, diff_folder_dir, base_ref)

    print("Changed files: %d" % len(changed))
    print("Git_Diff: %d files (full content) -> %s" % (n1, git_diff_dir))
    print("Difference_Folder: %d files (code only) -> %s" % (n2, diff_folder_dir))


if __name__ == "__main__":
    main()
