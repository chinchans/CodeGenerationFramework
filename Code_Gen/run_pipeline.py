from __future__ import annotations

import traceback
import sys
from datetime import datetime
from pathlib import Path

# Allow running as: python Code_Gen/run_pipeline.py
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def main() -> None:
    output_dir = REPO_ROOT / "Code_Gen" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_log = output_dir / f"run_pipeline_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    user_intent = 'gNB-CU has to prepare and send F1AP "UE CONTEXT MODIFICATION REQUEST" message to the source gNB-DU and Source gNB-DU responds with a "UE CONTEXT MODIFICATION RESPONSE" message and this message has to be handled on gNB-CU for the Inter-gNB-DU LTM handover'
    

    def _log(msg: str) -> None:
        print(msg, flush=True)
        with open(debug_log, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    _log(f"[Code_Gen] Starting end-to-end pipeline from intent.")
    _log(f"[Code_Gen] Debug log: {debug_log}")

    try:
        from Code_Gen.pipeline import run_end_to_end_from_intent
    except BaseException:
        _log("[Code_Gen] Failed while importing Code_Gen.pipeline (run_end_to_end_from_intent)")
        _log(traceback.format_exc())
        raise

    result = run_end_to_end_from_intent(user_intent)
    _log("[Code_Gen] Pipeline completed.")
    _log(f"[Code_Gen] Run manifest: {result.get('run_manifest_path', '')}")
    _log(f"[Code_Gen] Final template: {result.get('final_filled_template_path', '')}")
    _log(f"[Code_Gen] Prompt file: {result.get('code_generation_prompt_path', '')}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        print("[Code_Gen] Pipeline failed with exception:", flush=True)
        traceback.print_exc()
        raise
