import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

from fastmcp import FastMCP


# Ensure repo-root imports work when running: python Code_Gen/mcp_server.py
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


mcp = FastMCP(name="CodeGeneratorServer")

# Last state from generate_enriched_prompt; used by validate_repo_code for intent + template_path.
_last_orchestrator_state = None


def _read_prompt_text(state: dict) -> str:
    prompt = state.get("code_generation_prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt

    prompt_path = state.get("code_generation_prompt_path")
    if isinstance(prompt_path, str) and prompt_path.strip() and os.path.isfile(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def _read_template_text(state: dict) -> str:
    template_path = state.get("final_filled_template_path")
    if isinstance(template_path, str) and template_path.strip() and os.path.isfile(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


@mcp.tool()
def generate_enriched_prompt(intent: str) -> dict:
    """
    Runs the Code_Gen pipeline end-to-end for the given intent.

    Always starts a new session (new session_id). If self-learning reports ambiguities, use
    resolve_self_learning_ambiguities with that session_id to continue—do not call this tool again
    for the same ambiguity round expecting resolutions to apply.
    """
    global _last_orchestrator_state

    if not (intent or "").strip():
        return {"success": False, "error": "Intent is required."}

    from Code_Gen.pipeline import run_end_to_end_from_intent

    state = run_end_to_end_from_intent(intent)
    prompt_text = _read_prompt_text(state if isinstance(state, dict) else {})
    template_text = _read_template_text(state if isinstance(state, dict) else {})
    has_ambiguities = bool((state or {}).get("self_learning_has_ambiguities"))
    ambiguities = (state or {}).get("self_learning_ambiguities", [])

    _last_orchestrator_state = {
        "session_id": state.get("session_id") if isinstance(state, dict) else None,
        "messages": [SimpleNamespace(content=intent)],
        "final_filled_template_path": state.get("final_filled_template_path"),
        "code_generation_prompt": prompt_text,
        "code_generation_prompt_path": state.get("code_generation_prompt_path"),
        "run_manifest_path": state.get("run_manifest_path"),
    }

    if has_ambiguities:
        return {
            "success": True,
            "type": "self_learning_ambiguity_review",
            "requires_user_input": True,
            "intent": intent,
            "session_id": (state or {}).get("session_id"),
            "ambiguities": ambiguities,
            "next_action": (
                "Human required: for each ambiguity `id`, collect the user's answer. "
                "Call resolve_self_learning_ambiguities(session_id=<this_session_id>, resolutions=...) "
                "to update the same filled template and regenerate the prompt (same session only). "
                "For large ASN.1 text, use resolutions_json_path (repo-relative or absolute). "
                "Do not use resolve for a new intent—call generate_enriched_prompt again to start from scratch. "
                "Approve the MCP tool in the IDE so the server can apply updates."
            ),
            "run_manifest_path": state.get("run_manifest_path"),
        }

    if not prompt_text:
        return {
            "success": False,
            "error": "Pipeline completed but did not return a final prompt.",
            "state_keys": sorted(list(state.keys())) if isinstance(state, dict) else [],
        }

    response = {
        "success": True,
        "type": "draft_prompt",
        "requires_user_approval": True,
        "prompt": prompt_text,
        "next_action": "IDE should display preview and wait for user approval",
        "intent": intent,
        "session_id": (state or {}).get("session_id"),
        "template": template_text,
        "prompt_path": state.get("code_generation_prompt_path"),
        "run_manifest_path": state.get("run_manifest_path"),
    }
    return response


@mcp.tool()
def resolve_self_learning_ambiguities(
    session_id: str,
    resolutions: Optional[Dict[str, Any]] = None,
    resolutions_json_path: str = "",
) -> dict:
    """
    Continuation-only: apply human answers for self-learning ambiguities on an existing session
    (status awaiting_user) returned by generate_enriched_prompt. Updates the stored template and
    regenerates the prompt when clear. New intents must use generate_enriched_prompt (new session).
    """
    from Code_Gen.pipeline import (
        prepare_ambiguity_resolutions_input,
        run_resolve_self_learning_session,
    )

    merged, prep_err = prepare_ambiguity_resolutions_input(
        REPO_ROOT,
        inline=resolutions,
        resolutions_json_path=resolutions_json_path,
    )
    if prep_err:
        return {"success": False, "error": prep_err}

    sid = (session_id or "").strip()
    if not sid:
        return {
            "success": False,
            "error": "session_id is required (copy it from the generate_enriched_prompt ambiguity response).",
        }
    try:
        state = run_resolve_self_learning_session(sid, user_resolutions=merged)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    resolved_intent = str((state or {}).get("intent") or "")
    has_ambiguities = bool((state or {}).get("self_learning_has_ambiguities"))
    if has_ambiguities:
        return {
            "success": True,
            "type": "self_learning_ambiguity_review",
            "requires_user_input": True,
            "intent": resolved_intent,
            "session_id": (state or {}).get("session_id"),
            "ambiguities": (state or {}).get("self_learning_ambiguities", []),
            "next_action": (
                "Some ambiguities remain or new ones appeared after applying resolutions. "
                "Ask the human for the remaining answers and call resolve_self_learning_ambiguities again "
                "with the same session_id and id -> value entries for every open item."
            ),
            "run_manifest_path": state.get("run_manifest_path"),
        }

    prompt_text = _read_prompt_text(state if isinstance(state, dict) else {})
    template_text = _read_template_text(state if isinstance(state, dict) else {})
    if not prompt_text:
        return {
            "success": False,
            "error": "Prompt regeneration failed after applying resolutions.",
            "state_keys": sorted(list(state.keys())) if isinstance(state, dict) else [],
        }

    global _last_orchestrator_state
    _last_orchestrator_state = {
        "session_id": state.get("session_id") if isinstance(state, dict) else None,
        "messages": [SimpleNamespace(content=resolved_intent)],
        "final_filled_template_path": state.get("final_filled_template_path"),
        "code_generation_prompt": prompt_text,
        "code_generation_prompt_path": state.get("code_generation_prompt_path"),
        "run_manifest_path": state.get("run_manifest_path"),
    }

    return {
        "success": True,
        "type": "draft_prompt",
        "requires_user_approval": True,
        "prompt": prompt_text,
        "intent": resolved_intent,
        "session_id": (state or {}).get("session_id"),
        "template": template_text,
        "prompt_path": state.get("code_generation_prompt_path"),
        "run_manifest_path": state.get("run_manifest_path"),
        "resolution_applied": True,
    }


@mcp.tool()
def validate_repo_code(branch: str = "new_feature") -> dict:
    """
    Validates code after client pushes. Client sends only the branch name (the one pushed).
    Server uses openairinterface5g-develop for fetch, checkout, pull, git_diff. No repo path
    or workspace path is shared between client and server.

    CRITICAL - Branch detection: Before calling, run "git branch --show-current" in the
    repo where the user made changes and pushed (e.g. openairinterface5g-develop). Pass
    that branch. Do NOT assume master. Wrong branch fills Git_Diff/Difference_Folder
    with incorrect files.
    """
    global _last_orchestrator_state

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_path = r"C:\Users\ChanduVangala\Desktop\Code_Generation_v0\openairinterface-5g"

    user_intent = None
    template_path = None
    session_id = None
    if _last_orchestrator_state:
        state = _last_orchestrator_state
        session_id = state.get("session_id")
        messages = state.get("messages") or []
        if messages and hasattr(messages[0], "content"):
            user_intent = messages[0].content
        tp = state.get("final_filled_template_path")
        if tp and os.path.isfile(tp):
            template_path = tp

    if not os.path.isdir(repo_path):
        return {
            "success": False,
            "error": f"Repository path does not exist or is not a directory: {repo_path}",
        }

    git_dir = os.path.join(repo_path, ".git")
    if not os.path.isdir(git_dir):
        return {
            "success": False,
            "error": f"Not a git repository: {repo_path}",
        }

    fetch_proc = subprocess.run(
        ["git", "-C", repo_path, "fetch"],
        capture_output=True,
        text=True,
    )
    if fetch_proc.returncode != 0:
        return {
            "success": False,
            "error": "git fetch failed",
            "details": fetch_proc.stderr or fetch_proc.stdout,
        }

    checkout_proc = subprocess.run(
        ["git", "-C", repo_path, "checkout", branch],
        capture_output=True,
        text=True,
    )
    if checkout_proc.returncode != 0:
        return {
            "success": False,
            "error": f"git checkout {branch} failed",
            "details": checkout_proc.stderr or checkout_proc.stdout,
        }

    pull_proc = subprocess.run(
        ["git", "-C", repo_path, "pull", "origin", branch],
        capture_output=True,
        text=True,
    )
    if pull_proc.returncode != 0:
        return {
            "success": False,
            "error": f"git pull origin {branch} failed",
            "details": pull_proc.stderr or pull_proc.stdout,
        }

    git_diff_script = os.path.join(script_dir, "git_diff.py")
    if not os.path.isfile(git_diff_script):
        return {
            "success": False,
            "error": f"git_diff.py not found at expected location: {git_diff_script}",
        }

    cmd = [sys.executable, git_diff_script, repo_path]
    diff_proc = subprocess.run(cmd, capture_output=True, text=True)
    if diff_proc.returncode != 0:
        return {
            "success": False,
            "error": "git_diff.py execution failed",
            "details": diff_proc.stderr or diff_proc.stdout,
        }

    git_diff_dir = os.path.join(script_dir, "Git_Diff")
    diff_folder_dir = os.path.join(script_dir, "Difference_Folder")

    from Code_Validation.code_validation_agent import CodeValidationAgent

    agent = CodeValidationAgent(
        codebase_path=git_diff_dir,
        code_diff_folder=diff_folder_dir,
        template_path=template_path,
        user_intent=user_intent,
    )

    results = agent.run_noninteractive()

    memory_issues = results.get("memory")
    dependency_issues = results.get("dependency")
    mandatory = results.get("mandatory_ie")
    functional = results.get("functional")

    status_messages = {}
    if not memory_issues:
        status_messages["memory"] = "No memory leaks found."
    if not dependency_issues:
        status_messages["dependency"] = "All dependency structures are valid."
    if isinstance(mandatory, dict):
        any_missing = any(v for v in mandatory.values())
        if not any_missing:
            status_messages["mandatory_ie"] = "All mandatory IEs are present in the codebase."
    if isinstance(functional, list) and not functional:
        status_messages["functional"] = "All functional test cases passed."

    # Persist into outputs/session_state.sqlite if we have a session_id.
    if session_id:
        try:
            from Code_Gen.sqlite_state_store import SqliteStateStore

            db_path = REPO_ROOT / "outputs" / "session_state.sqlite"
            store = SqliteStateStore(db_path)
            store.ensure_session(
                session_id=session_id,
                intent=user_intent,
                status="validation_completed",
                template_path=template_path,
            )
            store.insert_code_validation(
                session_id=session_id,
                branch=branch,
                results=results,
                status_messages=status_messages,
                git_pull_output=pull_proc.stdout,
                git_diff_output=diff_proc.stdout,
            )
        except Exception:
            # Validation should still return results even if persistence fails.
            pass

    return {
        "success": True,
        "type": "code_validation_results",
        "branch": branch,
        "repo_path": repo_path,
        "results": results,
        "status_messages": status_messages,
        "git_pull_output": pull_proc.stdout,
        "git_diff_output": diff_proc.stdout,
    }


def _git_run(repo_path, args, capture_output=True):
    cmd = ["git", "-C", repo_path] + args
    r = subprocess.run(cmd, capture_output=capture_output, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()
    return r.returncode, out, err


@mcp.tool()
def commit_and_push_feature_branch(
    feature_branch: str = "new_feature",
    main_branch: str = "main",
    commit_message: str = "Feature update (Keep)",
    phase: str = "client_commands",
) -> dict:
    """
    Two-phase sync: (1) Get commands for MCP client to run; (2) Server pull.
    """
    repo_path = r"C:\Users\ChanduVangala\Desktop\Code_Generation_v0\openairinterface-5g"

    if not os.path.isdir(repo_path):
        return {"success": False, "error": f"Repository path does not exist: {repo_path}"}
    git_dir = os.path.join(repo_path, ".git")
    if not os.path.isdir(git_dir):
        return {"success": False, "error": f"Not a git repository: {repo_path}"}

    steps = []
    if phase == "client_commands":
        client_commands = [
            f"git checkout {main_branch}",
            f"git checkout {feature_branch} 2>nul || git checkout -b {feature_branch}",
            "git add -A",
            f'git commit -m "{commit_message}" --allow-empty',
            f"git push -u origin {feature_branch}",
        ]
        client_commands_unix = [
            f"git checkout {main_branch}",
            f"git checkout {feature_branch} 2>/dev/null || git checkout -b {feature_branch}",
            "git add -A",
            f'git commit -m "{commit_message}" --allow-empty',
            f"git push -u origin {feature_branch}",
        ]
        return {
            "success": True,
            "type": "client_commands",
            "message": "Run these commands in your client repo (workspace), then call this tool again with phase='server_pull' so the server can pull.",
            "client_commands": client_commands,
            "client_commands_unix": client_commands_unix,
            "next_step": "Call commit_and_push_feature_branch with phase='server_pull' after the above commands succeed.",
        }

    code, _, err = _git_run(repo_path, ["fetch", "origin"])
    if code != 0:
        return {
            "success": False,
            "error": f"git fetch origin failed: {err}",
            "steps": steps,
        }
    steps.append("Fetched latest refs from origin")

    code, current, err = _git_run(repo_path, ["rev-parse", "--abbrev-ref", "HEAD"])
    if code != 0:
        return {"success": False, "error": f"Could not get current branch: {err}", "steps": steps}
    steps.append(f"Current branch: {current}")

    code, branches, _ = _git_run(repo_path, ["branch", "--list", feature_branch])
    feature_exists = any(
        line.strip().lstrip("*").strip() == feature_branch
        for line in (branches.splitlines() if branches else [])
    )

    if feature_exists:
        code, _, err = _git_run(repo_path, ["checkout", feature_branch])
        if code != 0:
            return {
                "success": False,
                "error": f"Checkout existing branch {feature_branch} failed: {err}",
                "steps": steps,
            }
        steps.append(f"Checked out existing branch {feature_branch}")

        code, _, err = _git_run(repo_path, ["pull", "origin", feature_branch])
        if code != 0:
            return {
                "success": False,
                "error": f"Pull origin/{feature_branch} failed: {err}",
                "steps": steps,
            }
        steps.append(f"Pulled latest from origin/{feature_branch}")
    else:
        code, _, err = _git_run(
            repo_path,
            ["checkout", "-B", feature_branch, f"origin/{feature_branch}"],
        )
        if code != 0:
            return {
                "success": False,
                "error": f"Checkout tracking branch origin/{feature_branch} failed: {err}",
                "steps": steps,
            }
        steps.append(f"Created/reset local {feature_branch} from origin/{feature_branch}")

        code, _, err = _git_run(repo_path, ["pull", "origin", feature_branch])
        if code != 0:
            return {
                "success": False,
                "error": f"Pull origin/{feature_branch} failed: {err}",
                "steps": steps,
            }
        steps.append(f"Pulled latest from origin/{feature_branch}")

    return {
        "success": True,
        "type": "git_sync",
        "repo_path": repo_path,
        "branch": feature_branch,
        "steps": steps,
    }


if __name__ == "__main__":
    mcp.run()
