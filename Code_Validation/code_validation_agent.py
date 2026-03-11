import os
import sys
import subprocess
import re
from datetime import datetime
import time
import json
from typing import List, Dict, Set, Tuple, Optional
import threading

from openai import AzureOpenAI
from dotenv import load_dotenv
loaded = load_dotenv(verbose=True)

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME")


def _resolve_latest_timestamp_subdir(base_path: str) -> str:
    """
    If base_path contains subdirs named like YYYYMMDD_HHMMSS, return the path to the latest one.
    Otherwise return base_path unchanged.
    """
    if not base_path or not os.path.isdir(base_path):
        return base_path
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not subdirs:
        return base_path
    # Filter to timestamp-like names (YYYYMMDD_HHMMSS) so we don't pick random folders
    timestamp_like = [d for d in subdirs if re.match(r"^\d{8}_\d{6}$", d)]
    if not timestamp_like:
        return base_path
    timestamp_like.sort()
    return os.path.join(base_path, timestamp_like[-1])


class CodeValidationAgent:
    def __init__(
        self,
        oai_path: Optional[List[str]] = None,
        codebase_path: Optional[str] = None,
        template_path: Optional[str] = None,
        user_intent: Optional[str] = None,
        code_diff_folder: Optional[str] = None,
    ):
        # Allow overriding defaults from callers (e.g., gateway) while
        # preserving existing hardcoded behavior when no arguments are given.
        if oai_path is not None:
            self.OAI_PATH = oai_path if isinstance(oai_path, list) else [oai_path]
        else:
            self.OAI_PATH = [
                r"C:\Users\ChanduVangala\Desktop\Code_Generation_v0\openairinterface-5g"
            ]

        _diff_folder = (
            code_diff_folder
            or r"C:\Users\ChanduVangala\Desktop\GTest\Difference_Folder"
        )
        _codebase = (
            codebase_path
            or r"C:\Users\ChanduVangala\Desktop\GTest\Git_Diff"
        )
        self.CODE_DIFF_FOLDER = _resolve_latest_timestamp_subdir(_diff_folder)
        self.CODEBASE_PATH = _resolve_latest_timestamp_subdir(_codebase)
        self.USER_INTENT = (
            user_intent
            or "Implement F1AP UE CONTEXT SETUP RESPONSE Message Procedure of Inter-gNB-DU LTM Handover"
        )
        self.TEMPLATE_PATH = (
            template_path
            or r"C:\Users\ChanduVangala\Desktop\Code_Generation_v0\outputs\final_filled_templates\F1AP_UE_CONTEXT_SETUP_Response_Message_Procedure_of_Inter-gNB-DU_LTM_Handover_20260226_080721.json"
        )
        self.CPPCHECK_TIMEOUT_SEC = 480
        self.CPPCHECK_JOBS = 8
        self.C_EXTENSIONS = (".c",)

        self.results = {
            # "build": None,
            "memory": None,
            "mandatory_ie": None,
            "dependency": None,
            "functional": None,
        }

    def run_with_state(self, state: Dict) -> Dict:
        """
        Convenience helper for running validation given a pipeline-like state.
        It pulls template path and user intent from the state if available,
        then delegates to the standard run() method.
        """
        template_path = state.get("final_filled_template_path")
        if template_path and os.path.isfile(template_path):
            self.TEMPLATE_PATH = template_path

        messages = state.get("messages") or []
        if messages and hasattr(messages[0], "content"):
            self.USER_INTENT = messages[0].content

        return self.run()
    def build_phase(self):
        compilation_output=self.run_build()
        print("\n===COMPILATION RESULTS===")
        for k, v in compilation_output.items():
            print(f"\nPath: {k}\nResult:{v}\n")      
        self.results["build"]=compilation_output
        return compilation_output
    def memory_phase(self):
        # Use updated/diff files only (CODE_DIFF_FOLDER)
        path = self.CODE_DIFF_FOLDER
        candidates = []
        start = time.time()
        candidates = self.run_pattern_scan(path)
        elapsed = time.time() - start
        for path_, a, f in candidates[:80]:
            print("  %s  alloc=%d free=%d" % (path_, a, f))
        if len(candidates) > 80:
            print("  ... +%d more" % (len(candidates) - 80))

        print("\nRunning Cppcheck...\n")
        start = time.time()
        timed_out, leak_lines ,full_output= self.run_cppcheck_time_limited(
            path,
          
        )
    
        
        
        if leak_lines:
            print("Leak-related (%d):" % len(leak_lines))
            for line in leak_lines:
                print(" ",line)
            self.results["memory"]=leak_lines
        else:
            print("No leak-related lines in cppcheck output.")
            self.results["memory"]=None
        return self.results

    def mandatory_phase(self):
        print("===Mandatory IE Check===")
        if not os.path.isfile(self.TEMPLATE_PATH):
            raise SystemExit("Invalid TEMPLATE_PATH")
        missed_ies={}
        for path in self.OAI_PATH:
            if not os.path.isdir(path):
                print("Skipping this Invalid path",path)
                continue
            
            output=self.validate_mandatory_ies_for_folder(path)
         
            missed_ies[path]=output["templates_processed"][0]["scan_report"]["missing_mandatory_ies"]
        print("Missed IE's",missed_ies)
        self.results["mandatory_ie"]=missed_ies
        return missed_ies
    def dependency_phase(self):
        print("===Dependency check===", flush=True)
        output = self.validate_folder(recursive_json=False)
        invalid_structs = []
        for name, result in output.items():
            if result.startswith("INVALID STRUCTURE"):
                invalid_structs.append(name)
        print()  # newline so summary is visible after possible long LLM run
        if invalid_structs:
            print("Invalid structure(s):", flush=True)
            for s in invalid_structs:
                print(f"  - {s}", flush=True)
            print(f"Dependency check complete: {len(invalid_structs)} invalid structure(s).", flush=True)
        else:
            print("All structures valid.", flush=True)
            print("Dependency check complete: 0 invalid.", flush=True)
        sys.stdout.flush()
        self.results["dependency"] = invalid_structs
        return invalid_structs
    def functional_phase(self, use_llm_only: bool = False):
        print("===Functional Check===")
        out = self.functional_check(use_llm_only=use_llm_only)
        if out is None:
            self.results["functional"] = None
            return
        testcases, passedtc, failedtc = out
        print("Total test cases:", len(testcases))
        if failedtc:
            for f in failedtc:
                print(f"- {f['test_case']}")
                print(f"  Reason: {f['reason']}")
        else:
            print("No Test cases are Failed")
        self.results["functional"] = failedtc
    def collect_c_files(self,root_dirs):
        files = []
        for root in root_dirs:
            if not os.path.isdir(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                for name in filenames:
                    if any(name.endswith(ext) for ext in self.C_EXTENSIONS):
                        files.append(os.path.join(dirpath, name))
        return files
    def count_alloc_free(self,content):
        alloc = len(re.findall(r"\b(malloc|calloc|realloc)\s*\(", content))
        free = len(re.findall(r"\bfree\s*\(", content))
        return alloc, free
    def run_pattern_scan(self,codebase_path):
        root_dirs = [codebase_path] if isinstance(codebase_path, str) else list(codebase_path)
        files = self.collect_c_files(root_dirs)
        candidates = []
        for path in files:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue
            alloc, free = self.count_alloc_free(content)
            if alloc > free and alloc > 0:
                candidates.append((path, alloc, free))
        return candidates
    def run_cppcheck_time_limited(self,codebase_path):
        if isinstance(codebase_path, (list, tuple)):
            paths_to_check = list(codebase_path)
        else:
            paths_to_check = [codebase_path]

        cmd = [
            "cppcheck",
            "--enable=warning,style",
            "--force",
            "-j", str(self.CPPCHECK_JOBS),
            "--inline-suppr",
            "-q",
        ]
        cmd.extend(paths_to_check)

        proc = None
        timed_out = False
        output_lines = []

        def run():
            nonlocal proc
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            for line in proc.stdout:
                output_lines.append(line.rstrip())
            proc.wait()

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout=self.CPPCHECK_TIMEOUT_SEC)

        if thread.is_alive() and proc:
            timed_out = True
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        leak_keywords = ("memleak", "memory leak", "leak of", "Resource leak", "leak:")
        leak_lines = [
            line for line in output_lines
            if any(k in line for k in leak_keywords)
        ]

        return timed_out, leak_lines, output_lines
    def run_build(self):
        SUCCESS_MARKER = "BUILD SHOULD BE SUCCESSFUL"
        LOG_ROOT = os.path.abspath(os.path.join(os.getcwd(), "oai_build_logs"))
        os.makedirs(LOG_ROOT, exist_ok=True)
        results = {} 
        for path in self.OAI_PATH:
            cmake_path = os.path.join(path, "cmake_targets")
            build_dir = os.path.join(cmake_path, "ran_build", "build")

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # no spaces
            project = os.path.basename(os.path.normpath(path)) or "oai_project"
            log_file = os.path.join(LOG_ROOT, f"{project}_{ts}.log")

            
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, "w", encoding="utf-8") as _:
                pass

            if not os.path.exists(cmake_path):
                results[path] = f'Path not found: {cmake_path}'
                continue

            if not os.path.exists(build_dir):
                build_steps = f'cd "{cmake_path}" && ./build_oai -I && ./build_oai --gNB'
            else:
                build_steps = f'cd "{cmake_path}" && ./build_oai --gNB'

        
            full_cmd = (
                f'( set -o pipefail; {build_steps} ) 2>&1 | tee -a "{log_file}"'
            )

            proc = subprocess.Popen([
                "gnome-terminal",
                "--title", f"OAI Build: {project}",
                "--wait",
                "--",
                "bash",
                "-lc",
                full_cmd
            ])
            proc.wait()

        
            try:
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except FileNotFoundError:
                results[path] = f"Log not found (unexpected). Expected at: {log_file}"
                continue

            if SUCCESS_MARKER.lower() in content.lower():
                results[path] = "compilation successful"
            else:
                errors = self.extract_errors_from_run_build(content)
                results[path] = errors if errors else "unknown error (no error lines found in output)"

        return results 
    def remove_comments(self,code: str) -> str:
        code = re.sub(r'//.*', '', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        return code

    def extract_errors_from_run_build(self,text: str) -> str:
        """
        Keep only error-like lines.
        """
        error_lines = []
        pattern = re.compile(r"(fatal error|Error|failed|undefined reference|collect2: error|ld: error)", re.IGNORECASE)
        for line in self.text.splitlines():
            if pattern.search(line):
                error_lines.append(line)
    
        seen = set()
        deduped = []
        for l in error_lines:
            if l not in seen:
                deduped.append(l)
                seen.add(l)
        return "\n".join(deduped).strip()
    
    def load_information_elements(self,json_path: str) -> List[Dict]:
        with open(json_path, "r") as f:
            data = json.load(f)

        if "Information_Elements" not in data:
            raise ValueError(f"Information_Elements not found in JSON: {json_path}")

        return data["Information_Elements"]

    # -----------------------------
    # Step 2: Ask LLM to extract mandatory IEs
    # (PROMPT IS UNCHANGED)
    # -----------------------------
    def extract_mandatory_ies_via_llm(self,info_elements: List[Dict]) -> Dict:
        prompt = """
        You are a senior 3GPP protocol expert.

        Below is the COMPLETE Information_Elements block extracted from a 3GPP specification.

        Feature Context:
        We are specifically integrating or adding a new feature based on the template and the user intent.
        Your primary goal is to determine which Information Elements are mandatory for this particular feature
        to run correctly, using BOTH:
        - The template / ASN.1 definitions, and
        - The explicit user intent describing what behaviour the feature must achieve.
        Pick up those Information Elements which you think, feature-wise and semantically, are required even if
        in the template or ASN.1 they are marked as CRITICALLY REJECT or PRESENCE OPTIONAL.

        Your task is to determine feature-level mandatory IEs in a fully generic way,
        without hardcoding any specific feature name.

        ========================
        TASK INSTRUCTIONS
        ========================

        1. Identify all IE names present in the Information_Elements JSON.

        2. PRESENCE-Based Extraction:
        Extract ALL IEs marked as:
        - "PRESENCE mandatory"

        These MUST always be included in the output.

        3. Recursive Analysis:
        For every IE selected (presence-based OR feature-required):

        a. Locate its full definition inside Information_Elements.
        b. Analyze its internal structure.
        c. Extract nested IEs that are:
            - Marked as PRESENCE mandatory, OR
            - Required for runtime correctness of the feature.
        d. Repeat recursively until no further mandatory sub-IEs exist.

        
        4.Note: This is Highly Important Point
        Determine runtime necessity purely from semantic role,
        not from explicit feature name matching.Whatever IE's feature specifically you believe
        that should be Mandatory list out all the IE's even if in template if in ANS.1 or template mentions it is 
        mentioned as critically Reject or Presence Optional also. Related Example will be LTM Related Features .
        The above is just an example you need to pickup these type of features and their IE's as mentioned above.
        As you can figure out the features which are newly implementing as you already have idea regarding OAI 
        existing codebase what features are placed already.Inside each IE Definition when it is not placed any thing inside like PRESENCE OPTIONAL or OPTIONAL then
        consider it as Mandatory IE 

        4.a Name / representation differences between spec and code:
        In the C codebase, the same Information Element may appear with a structurally different name
        (for example, converted to snake_case, prefixed/suffixed, or wrapped in a struct whose name
        does not exactly match the ASN.1 IE_Name string). When you reason about whether two IEs are
        the “same”, treat such semantically equivalent but differently named representations as the
        same Information Element.

        5. Feature Subtree Rule:
        If multiple IEs are semantically related to the feature,
        treat the entire required subtree for correct runtime operation as mandatory.

        6. Do NOT rely only on "PRESENCE mandatory".
        OPTIONAL IEs required for runtime correctness must also be included.

        7. Do NOT hallucinate.
        Only use IE names that exist inside the provided Information_Elements block.

        ========================
        OUTPUT FORMAT RULES
        ========================

        - Output ONLY valid JSON.
        - Output MUST be hierarchical.
        - Do NOT flatten nested mandatory IEs.
        - Leaf IEs must still appear.
        - Do NOT include explanations.
        - Do NOT include markdown.
        - Do NOT include comments.
        - Output must be strictly parseable JSON.

        Output format must be EXACTLY:

        {
        "feature_level_mandatory_ies": [
            {
            "IE_Name": "IE_NAME",
            "mandatory_sub_ies": [
                {
                "IE_Name": "SUB_IE_NAME"
                }
            ]
            }
        ]
        }

        ========================
        STRICT CONSTRAINTS
        ========================

        - Do NOT assume feature names.
        - Do NOT skip nested IE definitions.
        - Do NOT include optional IEs unless:
            • They are PRESENCE mandatory, OR
            • They are required for runtime correctness.
        - Do NOT generate synthetic IEs.
        - Use only IEs present in Information_Elements.

        =========================
        Important Note
        =========================
        -Inside each IE Definition when it is not placed any thing inside like PRESENCE OPTIONAL or OPTIONAL then
        consider it as Mandatory IE 
        ========================
        Information_Elements:
        ========================
        """
        # Attach the concrete user intent so the LLM can decide feature-level mandatory IEs
        # based on what behaviour must be implemented.
        if self.USER_INTENT:
            prompt += f"\n        User Intent:\n        {self.USER_INTENT}\n\n"

        prompt += json.dumps(info_elements, indent=2)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()


        return self.extract_json_safely(content)

    # -----------------------------
    # Step 3: Flatten hierarchical mandatory IEs (improved)
    # -----------------------------
    def _collect_ie_names(self,node, result: Set[str]):
        """
        Recursively collect IE_Name values from the hierarchical structure returned by the LLM.
        Expected shapes:
        - {"IE_Name": "X", "mandatory_sub_ies": [ {...}, ... ]}
        - {"IE_Name": "Y"} as leaf
        - ["list", "of", "nodes"]
        """
        if isinstance(node, dict):
            # Only add the value of IE_Name, not the key name
            if "IE_Name" in node and isinstance(node["IE_Name"], str):
                result.add(node["IE_Name"])
            # Recurse into sub-ies if present
            if "mandatory_sub_ies" in node and isinstance(node["mandatory_sub_ies"], list):
                for item in node["mandatory_sub_ies"]:
                    self._collect_ie_names(item, result)
            # Also recurse other dict fields in case structure varies
            for k, v in node.items():
                if k not in ("IE_Name", "mandatory_sub_ies"):
                    self._collect_ie_names(v, result)

        elif isinstance(node, list):
            for item in node:
                self._collect_ie_names(item, result)

    def get_flat_mandatory_ie_list(self,hierarchical_output: Dict) -> List[str]:
        flat_set: Set[str] = set()
        if not isinstance(hierarchical_output, dict) or "feature_level_mandatory_ies" not in hierarchical_output:
            return []
        self._collect_ie_names(hierarchical_output["feature_level_mandatory_ies"], flat_set)
        return sorted(flat_set)

    def extract_json_safely(self,text: str) -> dict:
        """
        Extracts the first valid JSON object from LLM output.
        Raises a clear error if not possible.
        """
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in LLM output")

        json_str = text[start:end + 1]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print("❌ Invalid JSON returned by LLM:")
            print(json_str)
            raise e

    # -----------------------------
    # Step 4: Scan codebase for mandatory IEs
    # -----------------------------
    def generate_ie_variants(self,ie: str) -> List[str]:
        variants = set()

        # Original ASN.1 name
        variants.add(ie)

        # Hyphen to underscore
        variants.add(ie.replace("-", "_"))

        # First-letter lowercase (safe guard)
        if ie:
            variants.add(ie[0].lower() + ie[1:])

        # Both combined
        ie_us = ie.replace("-", "_")
        if ie_us:
            variants.add(ie_us[0].lower() + ie_us[1:])

        return list(variants)

    def scan_codebase(self,codebase_path: str, mandatory_ies: List[str]) -> Dict:
        found = set()
        missing = set(mandatory_ies)

        file_cache = []

        # Read files once
        for root, _, files in os.walk(codebase_path):
            for file in files:
                if file.endswith((".c", ".cpp", ".h")):
                    try:
                        with open(os.path.join(root, file), "r", errors="ignore") as f:
                            file_cache.append(f.read())
                    except Exception:
                        pass

        # ---------- PASS 1: Exact ----------
        for ie in list(missing):
            for content in file_cache:
                if ie in content:
                    found.add(ie)
                    missing.discard(ie)
                    break

        # ---------- PASS 2: Variant recovery ----------
        for ie in list(missing):
            variants = self.generate_ie_variants(ie)
            for content in file_cache:
                if any(v in content for v in variants):
                    found.add(ie)
                    missing.discard(ie)
                    break

        return {
            "found_mandatory_ies": sorted(found),
            "missing_mandatory_ies": sorted(missing)
        }

    # -----------------------------
    # Step 5A: Single-template pipeline (returns a result dict)
    # -----------------------------
    def validate_mandatory_ies_for_file(self,json_path: str, codebase_path: str) -> Dict:
    
        info_elements = self.load_information_elements(json_path)

        
        hierarchical_output = self.extract_mandatory_ies_via_llm(info_elements)

        flat_mandatory_ies = self.get_flat_mandatory_ie_list(hierarchical_output)

    
        # for ie in flat_mandatory_ies:
        #     print("   -", ie)

        
        report = self.scan_codebase(codebase_path, flat_mandatory_ies)

        # print("\n✅ Found mandatory IEs:")
        # for ie in report["found_mandatory_ies"]:
        #     print("   ✔", ie)

        # print("\n❌ Missing mandatory IEs:")
        # for ie in report["missing_mandatory_ies"]:
        #     print("   ✘", ie)

        return {
            "template": os.path.basename(json_path),
            "hierarchical_output": hierarchical_output,
            "flat_mandatory_ies": flat_mandatory_ies,
            "scan_report": report,
            
        }

    # -----------------------------
    # Step 5B: Single template pipeline (uses TEMPLATE_PATH)
    # -----------------------------
    def validate_mandatory_ies_for_folder(self,codebase_path: str) -> Dict:
        if not os.path.isfile(self.TEMPLATE_PATH):
            raise ValueError(f"Template file not found: {self.TEMPLATE_PATH}")

        json_files = [self.TEMPLATE_PATH]

        if not json_files:
            return {"templates_processed": [], "errors": []}

        all_results = []
        errors = []

    
        for path in sorted(json_files):
            try:
                result = self.validate_mandatory_ies_for_file(path, codebase_path)
                all_results.append(result)
            except Exception as e:
                print(f"❌ Error processing {os.path.basename(path)}: {e}")
                errors.append({"template": os.path.basename(path), "error": str(e)})

        # ----- Aggregated summary -----
        # print("\n==============================")
        # print("📊 Aggregated Summary (Folder)")
        # print("==============================")
        # for r in all_results:
        #     tname = r["template"]
        #     found_cnt = len(r["scan_report"]["found_mandatory_ies"])
        #     missing_cnt = len(r["scan_report"]["missing_mandatory_ies"])
        #     total = len(r["flat_mandatory_ies"])
        #     print(f"• {tname}: total={total}, found={found_cnt}, missing={missing_cnt}")

        if errors:
            print("\n⚠️ Templates with errors:")
            for err in errors:
                print(f"   - {err['template']}: {err['error']}")

        return {"templates_processed": all_results, "errors": errors}
    def load_information_elements_from_dir(self,recursive: bool = False) -> Tuple[Dict[str, Dict], List[str]]:
        """
        Loads Information_Elements from all JSON files in a directory.
        Returns:
        ie_map: { IE_Name -> full IE dict (last one wins if duplicate) }
        ie_name_list: list of all IE_Name values (may contain duplicates across files)
        """
        json_files = self.list_json_files_in_dir(recursive=recursive)

        ie_map: Dict[str, Dict] = {}
        ie_name_list: List[str] = []

        for file in json_files:
            try:
                with open(file, "r") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"❌ Failed to parse JSON {file}: {e}")
                continue

            if "Information_Elements" not in data or not isinstance(data["Information_Elements"], list):
                print(f"⚠️ 'Information_Elements' missing or not a list in {file}, skipping.")
                continue

            for ie in data["Information_Elements"]:
                name = ie.get("IE_Name")
                if not name or not isinstance(name, str):
                    continue
                ie_map[name] = ie  # last occurrence wins
                ie_name_list.append(name)

        
        return ie_map, ie_name_list
    def list_json_files_in_dir(self, recursive: bool = False) -> List[str]:
        if not os.path.isfile(self.TEMPLATE_PATH):
            raise ValueError(f"Template file not found: {self.TEMPLATE_PATH}")
        if not self.TEMPLATE_PATH.lower().endswith(".json"):
            print(f"⚠️ Template path is not a JSON file: {self.TEMPLATE_PATH}")
            return []
        return [self.TEMPLATE_PATH]
    def extract_structs_from_headers(self) -> Tuple[Dict[str, str], List[str]]:
        struct_map: Dict[str, str] = {}
        struct_name_list: List[str] = []
        typedef_struct_pattern = re.compile(
            r'typedef\s+struct'
            r'(?:\s+\w+)?'
            r'\s*\{',
            re.MULTILINE
        )
        for root, _, files in os.walk(self.CODE_DIFF_FOLDER):
            for file in files:
                if file.endswith(".h"):
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, "r", errors="ignore") as f:
                            content = f.read()

                        content = self.remove_comments(content)

                        pos = 0
                        while True:
                            match = typedef_struct_pattern.search(content, pos)
                            if not match:
                                break

                            start_brace = match.end() - 1
                            brace_count = 1
                            i = start_brace + 1

                            while i < len(content) and brace_count > 0:
                                if content[i] == "{":
                                    brace_count += 1
                                elif content[i] == "}":
                                    brace_count -= 1
                                i += 1

                            struct_body = content[start_brace:i]

                            name_match = re.match(
                                r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*;',
                                content[i:]
                            )

                            if name_match:
                                struct_name = name_match.group(1)

                                full_struct_text = (
                                    "typedef struct " +
                                    struct_body +
                                    " " +
                                    struct_name +
                                    ";"
                                )

                                struct_map[struct_name] = full_struct_text
                                struct_name_list.append(struct_name)

                            pos = i

                    except Exception:
                        print(f"Skipping {file_path} due to error")

    
        return struct_map, struct_name_list
    def semantic_name_mapping(self,struct_names: List[str], ie_names: List[str]) -> Dict[str, str]:
        """
        Ask LLM to map each C struct name to the best matching ASN.1 IE name.
        Returns: { struct_name -> matched_ie_name or "NO_MATCH" }
        """
        prompt = f"""
    You are a 3GPP protocol expert.

    Below is list of C struct names:
    {struct_names}

    Below is list of ASN.1 Information Element names:
    {ie_names}

    Task:
    For EACH C struct, pick the most semantically matching ASN.1 IE name
    (ignoring case, underscores, hyphens, common abbreviations).
    If no good match exists, use "NO_MATCH".

    Return a SINGLE JSON object whose KEYS are the C struct names
    and whose VALUES are the matched IE names or "NO_MATCH".

    Example format:
    {{ "MyStructA": "ServCellIndex", "MyStructB": "NO_MATCH" }}

    Only return JSON. No comments, no text, no markdown.
    """

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}  # ensures strict JSON object
            )
            response_text = response.choices[0].message.content.strip()
            return json.loads(response_text)

        except Exception as e:
            print("❌ Error during semantic name mapping:", str(e))
            return {}
    def validate_struct_recursive(self,struct_name: str,
                              struct_map: Dict[str, str],
                              mapping: Dict[str, str],
                              ie_map: Dict[str, Dict],
                              validated_cache: Dict[str, str]) -> str:

        if struct_name in validated_cache:
            return validated_cache[struct_name]

        struct_body = struct_map.get(struct_name, "")

        # 1️⃣ Extract dependencies
        dependencies = self.extract_dependencies(
            struct_body,
            list(struct_map.keys()),
            struct_name
        )

        # 2️⃣ Validate children first (DFS)
        for dep in dependencies:
            self.validate_struct_recursive(
                dep,
                struct_map,
                mapping,
                ie_map,
                validated_cache
            )

        # 3️⃣ Now validate current struct
        matched_ie = mapping.get(struct_name, "NO_MATCH")

        ie_definition = ""
        if matched_ie != "NO_MATCH":
            ie_definition = ie_map.get(matched_ie, {}).get("IE_Definition", "")

    
        child_asn, child_c = self.collect_child_context(struct_name, struct_map, mapping, ie_map,
                                                visited_structs=set(), visited_asn=set())

        result = self.validate_structure(
            struct_name,
            struct_body,
            matched_ie,
            ie_definition,
            child_asn,
            child_c
        )

    
        validated_cache[struct_name] = result
        return result
    def extract_dependencies(self,struct_body: str, all_struct_names: List[str], current_struct: str) -> List[str]:
        dependencies = []
        for name in all_struct_names:
            if name != current_struct:
                pattern = r'\b' + re.escape(name) + r'\b'
                if re.search(pattern, struct_body):
                    dependencies.append(name)
        return dependencies
    def extract_asn_dependencies(self,asn_definition: str, ie_map: Dict[str, Dict]) -> List[str]:
        """
        Extract ASN.1 referenced types from a definition.
        Returns list of referenced type names that exist in ie_map.
        """
        deps: Set[str] = set()
        if not asn_definition:
            return []

        # Find capitalized type-like tokens; conservative
        pattern = r'\b([A-Z][A-Za-z0-9\-]+)\b'
        tokens = re.findall(pattern, asn_definition)

        for token in tokens:
            if token in ie_map:
                deps.add(token)

        return list(deps)
    def collect_child_context(self,struct_name: str,
                          struct_map: Dict[str, str],
                          mapping: Dict[str, str],
                          ie_map: Dict[str, Dict],
                          visited_structs: Set[str],
                          visited_asn: Set[str]) -> Tuple[str, str]:

        child_asn_context = ""
        child_c_context = ""

        # Prevent loops
        if struct_name in visited_structs:
            return "", ""

        visited_structs.add(struct_name)

        struct_body = struct_map.get(struct_name, "")
        matched_ie = mapping.get(struct_name, "NO_MATCH")

        # Add current C struct
        child_c_context += f"\nC Struct {struct_name}:\n{struct_body}\n"

        if matched_ie != "NO_MATCH" and matched_ie in ie_map:
            asn_definition = ie_map[matched_ie].get('IE_Definition', "")

            # Add current ASN
            if matched_ie not in visited_asn:
                visited_asn.add(matched_ie)
                child_asn_context += f"\nASN.1 Type {matched_ie}:\n{asn_definition}\n"

            # 🔥 NEW: Extract ASN dependencies
            asn_deps = self.extract_asn_dependencies(asn_definition, ie_map)

            for dep in asn_deps:
                if dep not in visited_asn:
                    visited_asn.add(dep)
                    child_asn_context += f"\nASN.1 Type {dep}:\n{ie_map[dep].get('IE_Definition', '')}\n"

        # 🔥 Existing C dependency recursion
        c_deps = self.extract_dependencies(struct_body, list(struct_map.keys()), struct_name)

        for dep in c_deps:
            asn_sub, c_sub = self.collect_child_context(
                dep,
                struct_map,
                mapping,
                ie_map,
                visited_structs,
                visited_asn
            )
            child_asn_context += asn_sub
            child_c_context += c_sub

        return child_asn_context, child_c_context
    def validate_structure(self,struct_name, struct_body, ie_name, ie_definition, child_asn_context, child_c_context):
        if ie_name == "NO_MATCH":
            return "NO CORRESPONDING IE FOUND"

        prompt = f"""
    You are a senior 3GPP ASN.1 ↔ C structure validation expert.

    Your task is strict semantic validation of structural and representational compatibility.

    ========================
    ROOT C STRUCT:
    Name: {struct_name}
    Definition:
    {struct_body}

    ROOT ASN.1 TYPE:
    Name: {ie_name}
    Definition:
    {ie_definition}

    ========================
    EXPANDED ASN.1 CONTEXT:
    {child_asn_context}

    EXPANDED C STRUCT CONTEXT:
    {child_c_context}

    ========================
    VALIDATION RULES (STRICT):

    1. STRUCTURE FLATTENING RULE (MANDATORY PRIORITY):

    If an ASN.1 field is defined as a structured SEQUENCE type,
    it is VALID for the C struct to represent the subfields
    directly as individual members, provided semantic equivalence is preserved.

    The C struct is NOT required to contain a nested struct
    with the same ASN.1 type name.

    Flattened representations MUST be accepted
    if all required semantic components are present.


    2. OPTIONAL FIELD RULE (OVERRIDES MISMATCH):

    If an ASN.1 field is marked OPTIONAL,
    its absence in the C struct MUST NOT cause INVALID STRUCTURE.

    This includes optional SEQUENCE members and extension containers.
    Missing OPTIONAL fields must NOT be treated as:
    - Structural mismatch
    -Incomplete representation
    -Validation failure 
    Strictly make them VALID if the struct do not have OPTIONAL fields no matter what

    Recursive validation MUST ignore missing OPTIONAL fields.


    3. RECURSIVE VALIDATION RULE:

    All referenced ASN.1 types MUST be validated recursively.

    Validation must continue through all nested dependencies
    until only primitive types remain.

    You MUST NOT declare INVALID
    before completing recursive dependency validation.


    4. NUMERIC RANGE VALIDATION RULE:

    If ASN.1 defines numeric bounds,
    the C datatype must be capable of representing
    all values within the ASN.1 allowed range.

    A C datatype with a wider representable range is VALID.

    C types are NOT required to enforce ASN.1 bounds exactly.
    They must only be capable of representing them.

    If numeric bounds cannot be determined from the provided context,
    respond INSUFFICIENT CONTEXT.


    5. REPRESENTATIONAL COMPATIBILITY RULE:

    Validation must be based on representational compatibility,
    not literal name matching or exact type keyword matching.

    Exact ASN.1 field names are NOT required
    to appear identically in the C struct.

    Exact ASN.1 type keywords are NOT required
    to map to identically named C types.

    C representations are required to be capable of representing
    the ASN.1 structure and value ranges.

    Exact constraint enforcement at compile-time is NOT required.


    6. OCTET STRING RULE:

    If ASN.1 defines OCTET STRING,
    the C representation is VALID if it semantically represents
    a contiguous sequence of bytes.

    The internal definition of custom byte container types
    does not need to be provided.

    Validation must be based on semantic intent.


    7. DEPENDENCY REQUIREMENT:

    If any referenced ASN.1 type is missing from the expanded context,
    respond:

    INSUFFICIENT CONTEXT.

    You MUST NOT assume missing definitions.


    8. DETERMINISM RULE:

    Do not speculate.
    Do not assume.
    Use only provided definitions.

    Validation must be performed in this order:

    1. Apply structure flattening acceptance.
    2. Apply optional field rule.
    3. Perform recursive dependency validation.
    4. Perform numeric range validation.
    5. Apply representational compatibility rule.

    You MUST NOT declare INVALID
    before all applicable rules are evaluated.

    ========================
    OUTPUT FORMAT (STRICT):

    For ASN.1 types that are simple SEQUENCE "-Item" or "Item" (e.g. LTMConfigurationIDMapping-Item, LTMgNB-DU-IDs-Item): if the C struct has the same number of semantically matching fields (e.g. plmn+cellid+id, or single id), respond VALID STRUCTURE. Do not require exact name match.

    If valid:
    VALID STRUCTURE.

    If invalid:
    INVALID STRUCTURE.
    Followed by precise technical reason.

    If required definitions are missing:
    INSUFFICIENT CONTEXT.
    State exactly which ASN.1 type definition is missing.

    No extra explanation.
    No conversational text.
    """
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return response.choices[0].message.content.strip()
    def validate_folder(self, recursive_json: bool = False) -> None:
        ie_map, ie_name_list = self.load_information_elements_from_dir(recursive=recursive_json)
        struct_map, struct_name_list = self.extract_structs_from_headers()
        # Deterministic order so dependency results are consistent across runs
        struct_name_list = sorted(struct_name_list)
        mapping = self.semantic_name_mapping(struct_name_list, ie_name_list)
        validated_cache: Dict[str, str] = {}
        for struct_name in struct_name_list:
            self.validate_struct_recursive(
                struct_name,
                struct_map,
                mapping,
                ie_map,
                validated_cache
            )
        return validated_cache
    def load_codebase(self):
        """Load code from updated/diff files only (CODE_DIFF_FOLDER)."""
        file_contents = {}
        for root, _, files in os.walk(self.CODE_DIFF_FOLDER):
            for file in files:
                if file.endswith((".c", ".cpp", ".h", ".hpp")):
                    full_path = os.path.join(root, file)
                    try:
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        relative_path = os.path.relpath(full_path, self.CODE_DIFF_FOLDER)
                        file_contents[relative_path] = content
                    except Exception as e:
                        print(f"Error reading {file}:{e}")
        return file_contents
    def load_template_json(self):
        """
        Load the template JSON and remove any Knowledge_Hints sections.
        Returns cleaned JSON object.
        """
        with open(self.TEMPLATE_PATH, "r", encoding="utf-8") as f:
            template_data = json.load(f)
        return template_data

    def generate_functional_tests(self, template_json):
        template_str = json.dumps(template_json, indent=2)

        prompt = f"""
        You are a telecom protocol stack validation expert.

        Generate **functional unit test cases** for the entire template provided below.

        Template (Knowledge_Hints removed):
        {template_str}
        Also refer to the User Intent what we have provided here:
        {self.USER_INTENT}
        Do NOT do any mandatory IE Check and Dependency IE Check
        Provide test cases from template only and based on user intent.
        Do not provide  your test cases outside content from template.
        Try to focus more on user intent features.
        Provide brief explanations about what is being tested in each test case.
        Do not hallucinate.
        Return strictly in the following JSON format:
        [
        {{
            "name": "Test case 1",
            "check": "Description of functional/unit check",
            "Explanation":"Breif explanation about what is being tested in each test case"
            "success_criteria": "What defines PASS/FAIL"
        }}
        ]

        Return ONLY valid raw JSON.
        Do NOT use markdown or code blocks.

        """
        response = self.call_llm(prompt)
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        test_cases = json.loads(cleaned)
        
        return test_cases
    def call_llm(self,prompt):
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_MODEL_NAME"),
            messages=[
                {"role": "system", "content": "You are a strict Validation assistant"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content.strip()
    def split_code_into_chunks(self,code, chunk_size=1200):
        lines = code.splitlines()
        for i in range(0, len(lines), chunk_size):
            yield "\n".join(lines[i:i+chunk_size])

    def validate_test_case(self,test_case, files_dict):
        last_reason = "Test case not satisfied for new code generation task"
        # Deterministic file order for consistent results across runs
        for filename, code in sorted(files_dict.items()):
            chunks = list(self.split_code_into_chunks(code, chunk_size=1200))

            for idx, chunk in enumerate(chunks):
                

                prompt = f"""
                You are Senior expert in validation where you need to thoroughly check
                the test cases and code logically and structurally.

                Test case:
                {test_case}

                Modified File:
                {filename}

                Code changes:Do NOT do any mandatory IE Check and Dependency IE Check
                {chunk}
                You can remember relevant information from previous chunks and use it
                to check the current chunk.

                Determine whether this particular test case portion is present in code 
                and code must satisfy either structurally or logically the test case.

                Do not hallucinate.
                Properly compare the code and test case.

                You can use  the 'Previously observed code/data' as context to maintain 
                history across chunks of the same file for this test case.
                Each test case must reference specific function names,structure names or field names
                from the template.
                
                Do not generate high level behavioral checks.
                If test case is semantically matching with the code then make that test case passed.
                Reply ONLY in JSON format:
                {{
                    "result":"PASS" or "FAIL",
                    "reason":"one clear reason if FAIL , empty if PASS",
                    "observed_info":"any relevant info from this chunk for the test case"
                }}
                Return ONLY valid raw JSON.
                Do NOT use markdown or code blocks.
                """
                try:
                    response = self.call_llm(prompt)
                
                    cleaned = response.strip()
                    if cleaned.startswith("```"):
                        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
                    parsed = json.loads(cleaned)
                    result = parsed.get("result", "").upper()
                    reason = parsed.get("reason", "").strip()

                    if result == "PASS":
                        return "PASS"
                    else:
                        last_reason = reason if reason else last_reason
                except Exception as e:
                    last_reason = f"Parsing/Error issue: {str(e)}"

        return f"FAIL:{last_reason}"
    def functional_check(self, use_llm_only: bool = False):
        changed_files = self.load_codebase()
        if not changed_files:
            print("No modified files found in filtered codebase folder.")
            return None

        # Load template JSON
        template_json = self.load_template_json()

        if use_llm_only:
            test_cases = self.generate_functional_tests(template_json)
        else:
            # Ask whether to use LLM or Developer test cases
            choice = input(
                """Who will provide test cases? 
    1-Developer
    2-LLM Generate functional simulation tests
    Enter choice (1/2):
    """
            )

            if choice == "1":
                print("Enter test cases. Type DONE when finished")
                test_cases = []
                while True:
                    tc = input("Enter test case (JSON format): ")
                    if tc.strip().upper() == "DONE":
                        break
                    if tc.strip():
                        try:
                            test_cases.append(json.loads(tc))
                        except Exception as e:
                            print(f"Invalid JSON: {e}")
                if not test_cases:
                    print("No test cases provided")
                    return None
            elif choice == "2":
                test_cases = self.generate_functional_tests(template_json)
            else:
                print("Invalid choice")
                return None

        passed = []
        failed = []
        # Deterministic order: sort test cases by name so results are consistent across runs
        test_cases_sorted = sorted(
            test_cases,
            key=lambda t: (t.get("name") if isinstance(t, dict) else str(t))
        )
        for tc in test_cases_sorted:
            result = self.validate_test_case(tc, changed_files)
            if result == "PASS":
                passed.append(tc)
            else:
                failed.append({
                    "test_case": tc,
                    "reason": result.replace("FAIL:", "").strip()
                })

        return test_cases, passed, failed

    def run(self):
        print("=== Validation run started ===\n", flush=True)
        # self.build_phase()
        self.memory_phase()
        self.mandatory_phase()
        self.dependency_phase()
        self.functional_phase(use_llm_only=False)
        return self.results

    def run_noninteractive(self):
        """Run validation without any input(). Use from MCP server / automation."""
        print("=== Validation run (non-interactive) ===\n", flush=True)
        self.memory_phase()
        self.mandatory_phase()
        self.dependency_phase()
        self.functional_phase(use_llm_only=True)
        return self.results


def main():
    agent=CodeValidationAgent()
    final_results=agent.run()
   
    return final_results

if __name__=="__main__":
    main()
            