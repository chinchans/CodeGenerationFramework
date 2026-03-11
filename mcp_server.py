# mcp_server.py
# import logging
import warnings
import os
import subprocess
import sys
# Suppress SWIG/FAISS deprecation warnings (from dependencies, not our code)
# warnings.filterwarnings("ignore", message="builtin type .* has no __module__", category=DeprecationWarning)

from fastmcp import FastMCP

# logger = logging.getLogger(__name__)

# Create MCP Server
mcp = FastMCP(name="CodeGeneratorServer")

# Last state from generate_enriched_prompt; used by validate_repo_code for intent + template_path.
_last_orchestrator_state = None

# Gateway is imported lazily inside the tool to keep server startup fast.
# Listing tools (listOfferingsForUI) would otherwise timeout while loading
# FeatureValidation, Knowledge_Retrieval, Template_Orchestrator, LLMs, etc.


@mcp.tool()
def generate_enriched_prompt(intent: str) -> dict:
    """
    Generates enriched prompt from Template Orchestrator.
    This prompt must be reviewed by user before code generation.
    """

    # logger.info("Received intent: %s", intent)

    from gateway import AIServiceGateway
    global _last_orchestrator_state
    gateway = AIServiceGateway()
    state = gateway.orchestrator(intent)
    _last_orchestrator_state = state

    # full_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
    # with open(full_path, "r", encoding="utf-8") as f:
    #     code_generation_prompt = f.read()
   

    # state = {"code_generation_prompt": code_generation_prompt}

    if "code_generation_prompt" not in state:
        return {
            "success": False,
            "error": "Template Orchestrator failed to generate prompt."
        }

    enriched_prompt = state["code_generation_prompt"]

    # Optional extras to help downstream validation:
    session_id = state.get("session_id")
    template_path = state.get("final_filled_template_path")

    response = {
        "success": True,
        "type": "draft_prompt",
        "requires_user_approval": True,
        "prompt": enriched_prompt,
        "next_action": "IDE should display preview and wait for user approval",
    }

    # Include additional metadata if available, without breaking existing clients.
    if session_id:
        response["session_id"] = session_id
    if template_path:
        response["template_path"] = template_path
    # Echo back the intent so the client can reuse it directly.
    response["intent"] = intent

    return response


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
    # Use the actual repo path you have locally (no '-develop' suffix).
    repo_path = r"C:\Users\ChanduVangala\Desktop\Code_Generation_v0\openairinterface-5g"

    # Intent and template: use server state only if present and valid; else None → agent defaults
    user_intent = None
    template_path = None
    if _last_orchestrator_state:
        state = _last_orchestrator_state
        messages = state.get("messages") or []
        if messages and hasattr(messages[0], "content"):
            user_intent = messages[0].content
        tp = state.get("final_filled_template_path")
        if tp and os.path.isfile(tp):
            template_path = tp
    # If user_intent or template_path still None, CodeValidationAgent uses its __init__ defaults

    # Basic validation
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

    # 1) Ensure we are on the desired branch and up to date
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

    # 2) Run git_diff.py to export Git_Diff and Difference_Folder
    git_diff_script = os.path.join(script_dir, "git_diff.py")

    if not os.path.isfile(git_diff_script):
        return {
            "success": False,
            "error": f"git_diff.py not found at expected location: {git_diff_script}",
        }

    # git_diff.py: repo_path and optional base_ref (e.g. main). Branch is already checked out above.
    cmd = [sys.executable, git_diff_script, repo_path]
    diff_proc = subprocess.run(cmd, capture_output=True, text=True)
    if diff_proc.returncode != 0:
        return {
            "success": False,
            "error": "git_diff.py execution failed",
            "details": diff_proc.stderr or diff_proc.stdout,
        }

    # 3) Run CodeValidationAgent using the exported folders
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

    # Optional helper messages for \"all good\" cases only.
    # If any dimension is non-empty/non-None, the client can render its own
    # detailed summary from the raw results.
    memory_issues = results.get("memory")
    dependency_issues = results.get("dependency")
    mandatory = results.get("mandatory_ie")
    functional = results.get("functional")

    status_messages = {}

    if not memory_issues:
        status_messages["memory"] = "No memory leaks found."

    if not dependency_issues:
        status_messages["dependency"] = "All dependency structures are valid."

    # mandatory_ie is a dict { path -> missing_list }. All valid means every list is empty.
    if isinstance(mandatory, dict):
        any_missing = any(v for v in mandatory.values())
        if not any_missing:
            status_messages["mandatory_ie"] = "All mandatory IEs are present in the codebase."

    # functional is a list of failed test cases; empty list => all passed.
    if isinstance(functional, list) and not functional:
        status_messages["functional"] = "All functional test cases passed."

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
    """Run git in repo_path. Returns (returncode, stdout, stderr)."""
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

    phase='client_commands' (default): Returns client_commands for the MCP client
    to run in the client repo: ensure new_feature branch exists, add, commit, push.
    The MCP client (Cursor agent) should run those commands, then call again with
    phase='server_pull'.

    phase='server_pull': Runs on the server repo only: checkout new_feature, pull
    from origin. Call this after the client has run the client_commands and pushed.
    """
    repo_path = r"C:\Users\ChanduVangala\Desktop\Code_Generation_v0\openairinterface-5g"

    if not os.path.isdir(repo_path):
        return {"success": False, "error": f"Repository path does not exist: {repo_path}"}
    git_dir = os.path.join(repo_path, ".git")
    if not os.path.isdir(git_dir):
        return {"success": False, "error": f"Not a git repository: {repo_path}"}

    steps = []

    if phase == "client_commands":
        # Return commands for MCP client to run in the client repo (create branch if needed, add, commit, push).
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

    # phase == "server_pull": run server-side fetch + checkout + pull only
    # 1) Make sure we see the latest remote branches from origin
    code, _, err = _git_run(repo_path, ["fetch", "origin"])
    if code != 0:
        return {
            "success": False,
            "error": f"git fetch origin failed: {err}",
            "steps": steps,
        }
    steps.append("Fetched latest refs from origin")

    # 2) Get current branch (for diagnostics only)
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
