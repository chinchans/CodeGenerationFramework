# template_filler.py
import json
import re
import os
import logging
from datetime import datetime
from typing import List,Dict,Any
from pathlib import Path

logger = logging.getLogger(__name__)


class CodeTemplateFiller:
    def __init__(self, llm):
        self.llm = llm
        self.codebase_path = os.getenv("CODEBASE_PATH", "./openairinterface5g-develop")

    
    def format_chunks_as_context(self, chunks: List[Dict[str, Any]], max_chunks: int = 20) -> str:
        context_parts = []
        chunks_to_use = chunks[:max_chunks]
        for i, chunk in enumerate(chunks_to_use, 1):
            chunk_type = chunk.get("chunk_type", "UNKNOWN")
            meta = chunk.get("metadata", {})
            name = meta.get("name", "Unknown")
            file_path = meta.get("file_path", "Unknown")
            line_start = meta.get("line_start", 0)
            module = meta.get("module", "")
            protocol = meta.get("protocol", "")
            chunk_text = chunk.get("chunk_text", "")
            score = chunk.get("score", 0.0)
            cosine_score = chunk.get("cosine_score", 0.0)
            keyword_score = chunk.get("keyword_score", 0.0)
            is_expanded = chunk.get("expanded_from_kg", False)
            expanded_tag = " [HELPER/RELATED]" if is_expanded else ""
            context_part = f"""
                --- Code Chunk {i}{expanded_tag} ---
                Type: {chunk_type}
                Name: {name}
                File: {file_path}
                Line: {line_start}
                Module: {module}
                Protocol: {protocol}
                Relevance Score: {score:.4f} (Semantic: {cosine_score:.4f}, Keyword: {keyword_score:.4f})

                Code:
                {chunk_text}
            """
            context_parts.append(context_part)
        return "\n".join(context_parts)

 
    def create_llm_prompt(self, template: Dict[str, Any], code_chunks_context: str, user_query: str) -> str:
        feature_name = template.get("Feature_Name", "Unknown Feature")
        feature_description = "\n".join(template.get("Feature_Description", []))
        steps_raw = template.get("Feature_Implementation_Steps", [])[:10]
        implementation_steps_list = []
        for step in steps_raw:
            if isinstance(step, str):
                implementation_steps_list.append(step)
            elif isinstance(step, dict):
                desc = step.get("Description", "")
                step_no = step.get("Step_No", "")
                if step_no and desc:
                    implementation_steps_list.append(f"{step_no}. {desc}")
                elif desc:
                    implementation_steps_list.append(desc)
        implementation_steps = "\n".join(implementation_steps_list)

        code_artifacts_structure = {
            "Code_Artifacts": [
                {
                    "Codebase_Name": "OpenAirInterface 5G",
                    "Code_Instructions": [
                        {"type": "helper_function", "function_name": "example", "file_path": "path/to/file.c", "line_start": 123, "line_end": 145, "description": "...", "called_functions": [], "calling_functions": [], "uses_structs": []},
                        {"type": "data_structure", "struct_name": "example_struct", "file_path": "path/to/file.h", "line_start": 50, "description": "..."},
                        {"type": "variable", "variable_name": "example_var", "file_path": "path/to/file.c", "line_start": 200, "var_type": "int", "description": "..."},
                    ]
                }
            ]
        }

        prompt = f"""
            You are an expert code analyst specializing in 5G NR protocol implementation guidance.

            TASK: Analyze the provided code chunks and identify ONLY code elements that would be USEFUL BUILDING BLOCKS when IMPLEMENTING or MODIFYING the feature described in the template.

            CRITICAL - FOCUS ON IMPLEMENTATION BUILDING BLOCKS (NOT the implementation itself):
            - DO NOT include functions that directly implement the procedure (e.g., "CU_send_UE_CONTEXT_SETUP_REQUEST", "DU_handle_UE_CONTEXT_SETUP_REQUEST")
            - INSTEAD, include code elements that would be NEEDED/USED when implementing the feature:
            * Data structures (structs) - these define the message/data formats
            * Encoding/decoding helper functions - needed to create/parse messages
            * Initialization functions - needed to set up data structures
            * Supporting utility functions that are called during implementation
            - EXCLUDE: Main handler functions (they ARE the implementation, not building blocks)
            - EXCLUDE: Header file declarations (duplicates of .c files)
            - EXCLUDE: Memory cleanup/free functions (utility, not implementation building blocks)
            - LIMIT output to 10-15 code artifacts MAXIMUM (5-8 functions, 3-5 structs, 2-5 variables)
            - Be highly selective: Include ONLY what's needed to build/implement the feature

            CRITICAL REQUIREMENT - 5G NR ONLY:
            - ONLY use code chunks that are for 5G NR (New Radio) technology
            - EXCLUDE all chunks related to LTE, 4G, or any older technologies
            - Check the file path in each chunk's metadata to determine if it's 5G NR or LTE

            USER QUERY: {user_query}

            FEATURE INFORMATION:
            Feature Name: {feature_name}

            Feature Description:
            {feature_description}

            Key Implementation Steps:
            {implementation_steps}

            RETRIEVED CODE CHUNKS (Related/Helper Code):
            {code_chunks_context}

            INSTRUCTIONS:
            1. Filter chunks to 5G NR only.
            2. Analyze and identify NEEDED code elements: data structures, helper functions, variables.
            3. For each useful chunk, extract: helper_function, data_structure, or variable.
            4. Be SELECTIVE - 5-8 functions, 3-5 structs, 2-5 variables maximum (10-15 total).
            5. EXCLUDE main handler/implementation functions, duplicates, cleanup utilities.

            EXPECTED OUTPUT STRUCTURE:
            {json.dumps(code_artifacts_structure, indent=2)}

            Return ONLY the Code_Artifacts field as valid JSON (no markdown blocks).
        """

        return prompt


    def call_llm(self, prompt: str) -> Dict[str, Any]:
        # print("Calling LLM to identify related/helper code elements...")
        # print("-" * 60)
        response_text = None
        try:
            # print("   Sending prompt to LLM...")
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            response_text = re.sub(r'"{3,}', '"', response_text)
            response_text = re.sub(r',(\s*[}\]])', r'\1', response_text)
            # print("   Parsing LLM response...")
            try:
                code_artifacts = json.loads(response_text)
            except json.JSONDecodeError as first_error:
                # Common failure from LLM output: unescaped backslashes in paths
                # (e.g. C:\foo\bar) causing "Invalid \escape".
                if "Invalid \\escape" in str(first_error):
                    escaped_text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', response_text)
                    code_artifacts = json.loads(escaped_text)
                else:
                    raise
            if not isinstance(code_artifacts, dict):
                raise ValueError(f"Expected dict, got {type(code_artifacts)}")
            if "Code_Artifacts" not in code_artifacts:
                if isinstance(code_artifacts, list):
                    code_artifacts = {"Code_Artifacts": code_artifacts}
                else:
                    raise ValueError("Response missing 'Code_Artifacts' field")
            # print("   LLM response parsed successfully")
            if isinstance(code_artifacts.get("Code_Artifacts"), list) and len(code_artifacts["Code_Artifacts"]) > 0:
                instructions = code_artifacts["Code_Artifacts"][0].get("Code_Instructions", [])
                func_count = len([i for i in instructions if i.get("type") in ("helper_function", "implementing_function")])
                struct_count = len([i for i in instructions if i.get("type") == "data_structure"])
                var_count = len([i for i in instructions if i.get("type") == "variable"])
                pass  # print("   Identified: %d functions, %d structs, %d variables", func_count, struct_count, var_count)
            return code_artifacts
        except json.JSONDecodeError as e:
            # logger.error("   JSON parsing error: %s", e)
            if response_text:
                pass  # logger.debug("   Response preview: %s", response_text[:500])
            raise
        except Exception as e:
            # logger.error("   Error during LLM call: %s", e)
            if response_text:
                pass  # logger.debug("   Response preview: %s", response_text[:500])
            raise


    
    def _resolve_codebase_root(self) -> Path:
        codebase_root = self.codebase_path
        if not codebase_root:
            current_dir = Path(__file__).parent.absolute()
            for path in [
                current_dir / "openairinterface5g-develop",
                current_dir.parent / "openairinterface5g-develop",
                Path("./openairinterface5g-develop"),
                Path("../openairinterface5g-develop"),
            ]:
                if path.exists() and path.is_dir():
                    codebase_root = str(path)
                    break
        if not codebase_root or not os.path.exists(codebase_root):
            codebase_root = os.getcwd()
        return Path(codebase_root).resolve()


    def _infer_repo_name_from_paths(self, file_paths: List[str]) -> str:
        for raw_path in file_paths:
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            normalized = raw_path.replace("\\", "/")
            parts = [p for p in normalized.split("/") if p]
            for i, part in enumerate(parts):
                if part.lower() == "openairinterface5g-develop":
                    return parts[i]
        return ""


    def detect_codebase_metadata(self, file_paths: List[str] = None) -> Dict[str, str]:
        metadata = {"Code_Language": "", "Build_System": "", "Repo_Path": "", "Code_Output_Type": "", "Execution_Mode": "Runtime", "CI_Pipeline": "", "Version_Control": ""}

        # Prefer inference from retrieved code chunk paths so metadata reflects
        # the actual target implementation codebase even if CODEBASE_PATH is unset.
        file_paths = file_paths or []
        normalized_paths = [p.replace("\\", "/") for p in file_paths if isinstance(p, str) and p.strip()]
        if normalized_paths:
            has_c = any(p.endswith(".c") or p.endswith(".h") for p in normalized_paths)
            has_cpp = any(p.endswith(".cpp") or p.endswith(".hpp") for p in normalized_paths)
            has_py = any(p.endswith(".py") for p in normalized_paths)
            if has_c:
                metadata["Code_Language"] = "C"
            elif has_cpp:
                metadata["Code_Language"] = "C++"
            elif has_py:
                metadata["Code_Language"] = "Python"
            else:
                metadata["Code_Language"] = "Unknown"

            repo_name = self._infer_repo_name_from_paths(normalized_paths)
            metadata["Repo_Path"] = repo_name if repo_name else "."
            metadata["Build_System"] = "CMake" if has_c or has_cpp else "Unknown"
            metadata["Version_Control"] = "Git"
            metadata["Code_Output_Type"] = "Library/Executable"
            metadata["Execution_Mode"] = "Runtime"
            metadata["CI_Pipeline"] = ""
            return metadata

        codebase_path = self._resolve_codebase_root()
        try:
            cwd = Path.cwd().resolve()
            metadata["Repo_Path"] = str(codebase_path.relative_to(cwd)) if codebase_path.is_relative_to(cwd) else str(codebase_path)
        except (ValueError, AttributeError):
            metadata["Repo_Path"] = str(codebase_path)

        # Infer language by dominant file extension; this avoids accidental
        # "Python" when running from the framework repo instead of the OAI repo.
        ext_counts = {
            "C": sum(1 for _ in codebase_path.rglob("*.c")),
            "C++": sum(1 for _ in codebase_path.rglob("*.cpp")),
            "Python": sum(1 for _ in codebase_path.rglob("*.py")),
            "Java": sum(1 for _ in codebase_path.rglob("*.java")),
        }
        dominant_lang = max(ext_counts, key=ext_counts.get) if ext_counts else "Unknown"
        metadata["Code_Language"] = dominant_lang if ext_counts.get(dominant_lang, 0) > 0 else "Unknown"
        metadata["Build_System"] = "CMake" if (codebase_path / "CMakeLists.txt").exists() else ("Make" if (codebase_path / "Makefile").exists() else "Unknown")
        metadata["Version_Control"] = "Git" if (codebase_path / ".git").exists() else "Unknown"
        metadata["Code_Output_Type"] = "Library/Executable"
        return metadata


    def fill_template(self, template: Dict[str, Any], code_artifacts: Dict[str, Any], code_chunks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        # print("Filling template with Code_Artifacts...")
        filled_template = json.loads(json.dumps(template))
        filled_template["Code_Artifacts"] = code_artifacts.get("Code_Artifacts", [])
        # print("   Detecting codebase metadata...")
        chunk_paths = []
        for chunk in (code_chunks or []):
            chunk_path = (chunk.get("metadata", {}) or {}).get("file_path", "")
            if isinstance(chunk_path, str) and chunk_path.strip():
                chunk_paths.append(chunk_path)
        filled_template["Codebase_Metadata"] = self.detect_codebase_metadata(chunk_paths)
        # print("   Template filled successfully")
        return filled_template


    def save_template(self, template: Dict[str, Any], OUTPUT_DIR: str) -> str:
        """
        Save the filled template into OUTPUT_DIR with an auto-generated filename.
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        feature_name = template.get("Feature_Name", "Unknown_Feature")

        # Windows filenames cannot include: < > : " / \ | ? *
        # Your Feature_Name can contain strings like "L1/L2 ...", so we must sanitize.
        feature_name = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", str(feature_name))
        feature_name = feature_name.replace(" ", "_")
        feature_name = feature_name.strip("_")
        if not feature_name:
            feature_name = "Unknown_Feature"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"{feature_name}_{timestamp}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        # print("Saved filled template to: %s", output_path)
        return output_path

    def template_filler(self,state, SPEC_TEMPLATE_PATH):
        configured_path = state.get("codebase_path") or os.getenv("CODEBASE_PATH")
        if configured_path:
            self.codebase_path = configured_path

        template_file = SPEC_TEMPLATE_PATH
        # print("Loading template from: %s", SPEC_TEMPLATE_PATH)
        if not os.path.exists(SPEC_TEMPLATE_PATH):
            raise FileNotFoundError(f"Spec filled Template file not found: {SPEC_TEMPLATE_PATH}")
        with open(SPEC_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            template = json.load(f)
        feature_name = template.get("Feature_Name", "Unknown")
        # print("   Loaded template: %s", feature_name)
        
        output_root = Path(__file__).resolve().parent.parent / "outputs"
        OUTPUT_DIR = str(output_root / "final_filled_template")


        # Normalize code_artifacts_context into a flat list of chunk dicts
        raw_context = state.get("code_artifacts_context")
        code_chunks = []

        # If an earlier version already stored a flat list, just use it
        if isinstance(raw_context, list):
            code_chunks = raw_context
        elif isinstance(raw_context, dict) and raw_context is not None:
            # 1) Semantic chunks already have full metadata + chunk_text
            semantic_chunks = raw_context.get("semantic_chunks", [])
            code_chunks.extend(semantic_chunks)

            # 2) Expanded chunks are grouped by depth; flatten and adapt metadata
            expanded_chunks = raw_context.get("expanded_chunks", {})
            for depth_key, depth_chunks in expanded_chunks.items():
                for ch in depth_chunks:
                    meta = ch.get("metadata", {}) or {}
                    line_range = meta.get("line_range") or [0, 0]
                    line_start = line_range[0] if isinstance(line_range, list) and len(line_range) > 0 else 0

                    unified_chunk = {
                        "chunk_type": ch.get("chunk_type", "UNKNOWN"),
                        "metadata": {
                            "name": meta.get("name", "Unknown"),
                            "file_path": meta.get("file_path", "Unknown"),
                            "line_start": line_start,
                            "module": meta.get("module", ""),
                            "protocol": meta.get("protocol", ""),
                        },
                        # Expanded KG chunks may not have inline code text; keep empty string if missing
                        "chunk_text": ch.get("chunk_text", ""),
                        # Scores are unknown for expanded chunks – default to 0
                        "score": 0.0,
                        "cosine_score": 0.0,
                        "keyword_score": 0.0,
                        "expanded_from_kg": True,
                    }
                    code_chunks.append(unified_chunk)
        else:
            code_chunks = []

        user_query = state.get("messages")[0].content 

        # print("Formatting code chunks as context...")
        context = self.format_chunks_as_context(code_chunks, max_chunks=20)
        # print("   Context formatted: %s characters", f"{len(context):,}")

        prompt = self.create_llm_prompt(template, context, user_query)
        code_artifacts = self.call_llm(prompt)
        filled_template = self.fill_template(template, code_artifacts, code_chunks)
        final_filled_template_path = self.save_template(filled_template, OUTPUT_DIR)

    
        return final_filled_template_path
