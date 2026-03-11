# prompt_generator.py

import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Azure OpenAI Configuration (ONLY PROVIDER)
# ----------------------------------------------------------

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o-mini")

if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
    raise ValueError("Azure OpenAI credentials not found for Prompt Generator.")

llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    temperature=0.2
)

# print("Using Azure OpenAI for Prompt Generation")

# ----------------------------------------------------------
# Prompt Generation Agent
# ----------------------------------------------------------

def promptGenerationAgent(state):
    """
    Generates a code-generation prompt using:
    - User intent (from state messages)
    - Filled specification template

    Saves:
    - Prompt inside GlobalState
    - Prompt inside a text file

    Works with any user message and any valid filled template path from state.
    """
    # print("=" * 60)
    # print("Prompt Generation Agent")
    # print("=" * 60)

    messages = state.get("messages") or []
    if not messages:
        raise ValueError("No messages in state for prompt generation.")
    user_query = messages[0].content if hasattr(messages[0], "content") else str(messages[0])
    filled_template_path = state.get("final_filled_template_path")
    # filled_template_path = "..\\outputs\\final_filled_templates\\F1AP_UE_CONTEXT_SETUP_REQUEST_Message_Procedure_of_Inter-gNB-DU_LTM_Handover_20260219_053852.json"
    # user_query = "Implement F1AP UE CONTEXT SETUP REQUEST Message Procedure of Inter-gNB-DU LTM Handover"

    if not filled_template_path or not os.path.exists(filled_template_path):
        raise ValueError("Filled specification template not found for prompt generation.")

    # --------------------------------------------------
    # Load Filled Specification
    # --------------------------------------------------

    with open(filled_template_path, "r", encoding="utf-8") as f:
        filled_spec = json.load(f)

    # --------------------------------------------------
    # Meta Prompt
    # --------------------------------------------------

    # This contract is injected verbatim into the final code-generation prompt
    # to prevent shallow implementations that update only a single IE while
    # bypassing transitive dependencies (connected IEs, container structs, and call-sites).
    COVERAGE_CONTRACT = f"""
---
## MANDATORY COVERAGE CONTRACT (NON-NEGOTIABLE)

You MUST implement the requested feature with **complete transitive coverage** across:
- All connected Information Elements (IEs) and referenced ASN.1 types
- All container / parent chain structures that carry those IEs
- All encoder + decoder paths (producer + consumer)
- All call-sites and dependent code that uses the decoded/encoded structures

Partial / shallow changes are forbidden. If IE-A implies IE-B (via ASN.1 type reference, container membership, SEQUENCE field, CHOICE alternative, SEQUENCE OF item type, ProtocolIE container, or extension container), you MUST update **both** the representation and all affected encode/decode + consumer code.

### Required phases (you MUST follow in order)

#### Phase 0 — Dependency Closure (Depth-First)
Before modifying any file, you MUST compute a **Dependency Closure** starting from the message root(s) implied by the user request and the provided template JSON.

You MUST output a machine-checkable JSON object named `dependency_closure_report` containing:
- `roots`: list of root container(s) (message IE container / top-level PDU / top-level C struct(s) used for this message)
- `nodes`: list of **all** reachable IEs/types (depth-first), each with:
  - `name`
  - `kind`: `IE|ASN_TYPE|C_STRUCT|ENCODER|DECODER|CALLSITE`
  - `depth`: integer (root is 0)
  - `parents`: list of parent node names (for traceability)
- `edges`: list of `{ "from": "...", "to": "...", "reason": "field|choice|list|container|extension|encode|decode|uses|calls" }`
- `max_depth`
- `unresolved_symbols`: list (MUST be empty before coding starts; if not empty you MUST keep searching/expanding until it is empty)

Rules:
- The closure MUST include all connected types/IEs, not only those explicitly listed in the user request.
- For ASN.1: treat `::=` definitions as authoritative; traverse until primitives.
- For code: traverse `struct` fields, encode/decode helper usage, and consumer call-sites.
- You MUST include BOTH directions for code impact: producers (encode/build) AND consumers (decode/use).

#### Phase 1 — Impact Matrix (No Orphans Allowed)
You MUST output an `impact_matrix` that maps every closure node (except primitives) to concrete code changes:
- Struct definition locations (.h/.c)
- Encoder/decoder functions to update
- Helper/init/copy/free functions to update (as required by project patterns)
- All consumer/call-site locations that read/write the fields

Hard rule: **No orphan nodes**.
- If a node is in dependency closure and has no planned code touchpoints, that is a failure: keep searching until you find the right integration points.

#### Phase 2 — Implementation (Propagation Required)
Implement the feature and propagate changes through:
- All structs/types from the closure
- Encode + decode paths
- All call-sites in each actor that handle or consume the message

Hard rule: For every optional IE you add/change, you MUST implement presence tracking and wire it through encode/decode + consumer logic.

#### Phase 3 — Self-Audit (Coverage Gate)
After coding, you MUST run a self-audit and output `coverage_gate_report`:
- `closure_nodes_total`
- `closure_nodes_covered`
- `missing_nodes` (MUST be empty)
- `max_depth_covered` (MUST equal `dependency_closure_report.max_depth`)
- `evidence`: for each node, cite the exact file + symbol/function touched

If anything is missing, you MUST do another patch round until coverage is complete.

### Prohibited behaviors
- Do NOT “skip” connected IEs because they seem generic.
- Do NOT update only encode or only decode.
- Do NOT update only one actor when multiple actors consume/produce the message.
- Do NOT stop after updating a single struct if there are downstream consumers.

---
""".strip()


    meta_prompt = f"""
        You are an expert Prompt Architect specializing in generating large-scale,
        production-grade, highly structured prompts for advanced code generation systems.

        Your task:
        Generate a SINGLE, EXTREMELY DETAILED, STRUCTURED, AND EXHAUSTIVE prompt
        that will instruct another LLM to generate production-ready source code.

        CRITICAL BEHAVIOR REQUIREMENTS:

        - The generated prompt MUST be long-form and comprehensive.
        - The generated prompt MUST include clearly separated sections using headings.
        - The generated prompt MUST enforce implementation rigor across an entire codebase.
        - The generated prompt MUST include implementation phases.
        - The generated prompt MUST include validation checklists.
        - The generated prompt MUST include strict compliance rules.
        - The generated prompt MUST include integration instructions.
        - The generated prompt MUST include compilation and safety requirements.
        - The generated prompt MUST explicitly forbid partial implementation.
        - The generated prompt MUST reflect enterprise-level production standards.
        - The generated prompt MUST be comparable in depth and scope to a formal engineering specification.

        STRICT RULES:

        - Do NOT generate any code.
        - Do NOT summarize the specification.
        - Do NOT add new requirements.
        - Use the specification strictly as the source of truth.
        - Do NOT shorten sections for brevity.
        - Expand constraints thoroughly.
        - Be exhaustive rather than concise.
        - The output prompt should be at least 3–5 pages worth of structured instruction.
        - Favor over-specification rather than minimalism.

        User Request:
        {user_query}

        Filled Technical Specification (JSON):
        {json.dumps(filled_spec, indent=2)}

        The generated prompt MUST include the following major sections:

        1. Role Definition (senior engineer persona with domain expertise)
        2. Task Overview
        3. Authoritative Input Context Description
        4. Actor-Specific Codebase Rules (if applicable)
        5. Type & Struct Alignment Enforcement
        6. Implementation Strategy (Phased Approach)
        7. Codebase Modification Identification Process
        8. Mandatory Data Structure Updates
        9. IE / ASN / Encoding Handling Rules
        10. Consumer / Call-Site Update Requirements
        11. Memory Safety & Resource Management Requirements
        12. Compilation Safety & Type Resolution Rules
        13. Code Quality & Naming Standards
        14. Error Handling & Validation Rules
        15. Output Format Requirements
        16. Integration Summary Requirements
        17. Final Engineering Checklist
        18. Critical Reminders Section
        19. Recursive ASN.1 Information Element Expansion Requirements
        20. Mandatory Coverage Contract (Dependency Closure + Impact Matrix + Self-Audit Coverage Gate)

        The generated prompt MUST require **full recursive expansion of all Information Elements (IEs) that are relevant to the current user request / message type**, starting from the top-level parent IE(s) defined in the filled specification JSON.

        CRITICAL COVERAGE REQUIREMENT (MOST IMPORTANT):
        The generated prompt MUST embed and enforce a **Coverage Contract** that prevents shallow code generation.
        The implementing LLM MUST be forced to:
        - Compute and output a **Dependency Closure Report** (depth-first) BEFORE coding
        - Produce an **Impact Matrix** mapping every closure node to concrete code touchpoints
        - Implement changes for ALL closure nodes (no orphans)
        - Run a **Self-Audit Coverage Gate** and patch again if anything is missing
        - Cover BOTH producer (encode/build) and consumer (decode/use) code paths and all call-sites

        The prompt MUST instruct the implementing LLM to:

        - Identify, from the filled JSON specification, the **main / root IE container(s)** for the concrete message and direction implied by the user request .
        - For each such root IE container, locate its exact ASN.1 definition and treat that ASN.1 definition as the **single source of truth** for included IEs and their types (never rely only on a natural-language IE list in the prompt).
        - Extract **every referenced TYPE and IE name** from the ASN.1 definition of the root container(s), and cross-check that every IE name mentioned anywhere in the filled JSON for this message is also covered; if the JSON lists additional IEs for this intent, they MUST also be resolved in the ASN.1 tree.
        - For each referenced TYPE:
            - If SEQUENCE → expand all fields and recursively resolve each field type.
            - If SEQUENCE OF → resolve the item type and recursively expand that item type.
            - If CHOICE → expand all alternatives and recursively resolve each alternative type.
            - If ENUMERATED → generate enum definitions and ensure every enumerated value is represented.
            - If INTEGER with range → enforce range validation in the generated code.
            - If BIT STRING with size → enforce bit length and mask/flag handling.
            - If OCTET STRING → preserve encoding semantics and size constraints.
            - If ProtocolIE-SingleContainer / ProtocolIE-Container → unwrap the inner IE and continue recursion on its ASN.1 type.
            - If ProtocolExtensionContainer / Extension container → include extension-safe handling and recursive resolution of extension IEs.

        - Perform **strict depth-first recursive traversal** of the ASN.1 dependency tree until only primitive types remain.
        - Ensure **no referenced ASN.1 type, IE name, or nested container mentioned anywhere in the ASN.1 or filled JSON for this message is left unresolved**.
        - For every resolved IE / ASN.1 type, **generate or update the corresponding C struct definition(s)**, including nested structs and lists, aligned exactly with the ASN.1 field names and constraints.
        - Update or generate **encode/decode logic** for every expanded type, including per-field mapping, presence conditions, and list iteration.
        - Update or generate **copy / helper / utility functions** for every expanded type that is used in the local codebase.
        - Maintain strict type and naming alignment with both the ASN.1 definitions and the existing local header files for the relevant actor.
        - Ensure all nested containers (lists, choices, extension containers, protocol IE containers) are correctly represented in memory with correct ownership and lifetime.
        - Respect OPTIONAL / CONDITIONAL presence rules, including conditions expressed in the filled JSON template for this specific message.
        - Enforce CRITICALITY handling behavior as defined in the specification and local implementation style.
        - Implement presence bitmask or flags for optional IEs, and ensure they are wired through encode/decode and helper functions.
        - Ensure list size constraints and cardinality constraints are validated at runtime with robust error handling.
        - Generate nested allocation / deallocation logic where required, consistent with the project’s memory-management model.
        - Produce **explicit, actor- and file-specific instructions** for where each struct/IE/encode/decode/helper change MUST be implemented in the existing codebase (paths, function names, and integration points), so that another LLM cannot skip or partially implement these requirements.

        The generated prompt MUST require **dynamic, data-driven IE traversal based on the ASN.1 tree and the filled JSON**, NOT a hard-coded or shallow list of IEs.
        It MUST explicitly forbid shallow or partial ASN expansion, or coverage limited only to the IEs mentioned in the natural-language user request.
        It MUST enforce whole-tree coverage of ASN.1 dependencies for the concrete message / intent, while remaining **generic enough to work for any future user request and message type** that can be expressed by the same JSON template structure.

        SPECIAL CASE: ASN.1 VERSION / FEATURE GAPS
        - Even if the local ASN.1 version appears older or does not explicitly define some of the feature-related IEs mentioned in the user request or JSON (for example Rel‑18 LTM IEs on a Rel‑15 F1AP tree), the implementing LLM MUST STILL PROCEED and implement the feature as far as safely possible in this codebase.
        - In such cases, the generated prompt MUST instruct the implementing LLM to:
          - Never invent or reference ASN.1-generated symbols (types, enums, IE IDs) that do not actually exist in the current headers / generated code.
          - Instead, introduce or extend internal C structs, helper functions, configuration paths, and call flows to model the requested feature behavior (e.g., LTM-like behavior) inside the existing actors, even if the on-the-wire ASN.1 cannot yet carry the exact standard IEs.
          - Use existing extension hooks, vendor-specific fields, or internal-only data paths where available, but WITHOUT breaking compilation or type-safety.
          - Clearly mark any places where a future ASN.1/codegen upgrade would be required to map the internal representation onto real standard IEs, using TODO-style comments and explicit notes in the integration summary.
        - IMPORTANT: In this special-case mode, the implementing LLM MUST NOT abort or refuse the task solely because the ASN.1 version is older; it MUST produce a best-effort, compilable, internal implementation that gets as close as possible to the requested feature within the current codebase constraints.

        The generated prompt must:
        - Use clear separators like "---"
        - Use bullet points and sub-bullets
        - Use strong imperative language (MUST, CRITICAL, REQUIRED)
        - Include compliance verification checklists
        - Explicitly require whole-codebase consistency

        MANDATORY ENDING STRUCTURE:
        The generated prompt MUST end with a consolidated, actionable section that has the following structure (generate this as the final part of your output, before any placeholder for the template):

        1. **Role** — One line: senior engineer persona with domain expertise.
        2. **TASK** — Clear task statement plus these subsections (adapt to the specification):
           - CRITICAL REQUIREMENT: exact implementation location and scope of codebase changes
           - IE NAMING REQUIREMENT: exact match of IE names from the specification
           - TYPE AND STRUCT DEFINITION ALIGNMENT (PER ACTOR): member names, list/container shape, value vs pointer from local headers
           - CONSUMER / CALL-SITE CODE (PER ACTOR): update all files that use the decoded message/struct per actor
           - ACTOR-SPECIFIC CODEBASE REQUIREMENT: only modify folders matching Call_Flow actors
           - ASN.1 FILES (.asn) - THIRD PARTY: do not modify .asn files; document changes for third-party services only
        3. **INPUT CONTEXT** — JSON template description and existing codebase context (what to analyze).
        4. **IMPLEMENTATION STRATEGY** — Phased approach:
           - Phase 1: Specification Analysis (parse JSON, extract actors, IEs, constraints)
           - Phase 2: Codebase Integration & Modification Points Identification (actor folders, exact locations, comprehensive search for structs, IE definitions, encode/decode, helpers, consumer/call-site code; alignment with local headers)
           - Phase 3: Code Generation & Codebase Modifications (target location, data structure updates, IE updates, initialization, field mapping, encoding/decoding, helpers, validation, memory safety, documentation)
        5. **CODE QUALITY REQUIREMENTS** — Naming conventions, code style, error handling, performance, compilation requirements (type resolution from same actor headers, no cross-actor assumptions).
        6. **OUTPUT FORMAT** — Mandatory list of what must be provided: primary function implementation, data structure modifications, ASN.1/IE updates (not .asn), encoding/decoding updates, header updates, helper updates, dependency notes, integration summary; with actor-specific file paths and clear labels.
        7. **SPECIAL CONSIDERATIONS** — Protocol compliance, encoding/decoding, state management, concurrency, testing hooks as applicable.
        8. **FINAL CHECKLIST** — Checkboxes for: implementation location, codebase modifications, functional requirements (IE names exact, optional fields, piggy-backed IEs, memory safety), code quality, output completeness.
        9. **Closing reminder** — One short paragraph stating that the implementation must be complete at the exact location with all related components updated, and that .asn files are not modified in the codebase.
        10. End with the exact line: "Here is the template :" (the template JSON will be appended automatically after your output).

        Do not reference any external file; generate this ending structure from the specification and user request so it is self-contained and applicable to any similar message.

        And also strictly follow the following rules:
            - For IE definitions, you must strictly follow the ASN.1 definition and not add any additional information
            - For IE defintions , you must go the IEs ASN.1 definition and extract the IE definitions recursively from the main IE to the child and if the child IE also has a substructure then you must extract the substructure recursively and so on.

        Return ONLY the final generated prompt text.
    """
    # --------------------------------------------------
    # Invoke LLM
    # --------------------------------------------------

    response = llm.invoke(meta_prompt)
    generated_prompt = response.content.strip()

    # Inject hard coverage contract to eliminate shallow/bypassed implementations
    generated_prompt = f"{COVERAGE_CONTRACT}\n\n{generated_prompt}"
    if "Here is the template" not in generated_prompt[-100:]:
        generated_prompt = generated_prompt + "\n\nHere is the template :\n"
    generated_prompt = generated_prompt + json.dumps(filled_spec, indent=2)

    # --------------------------------------------------
    # Save Prompt to File
    # --------------------------------------------------

    OUTPUT_DIR = "./outputs/code_generation_prompts"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"code_prompt_{timestamp}.txt"
    output_path = os.path.join(OUTPUT_DIR, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(generated_prompt)

    # --------------------------------------------------
    # Store in GlobalState
    # --------------------------------------------------

    state["code_generation_prompt"] = generated_prompt
    state["code_generation_prompt_path"] = output_path

    print("Prompt saved to: %s", output_path)
    print("Prompt generation completed successfully")

    return state


# if __name__ == "__main__":
#     # For standalone testing only. In production, state is provided by the orchestrator
#     # with "messages" and "final_filled_template_path" from the workflow.
#     from types import SimpleNamespace
#     state = { # Set to a valid path for local testing
#     }
#     promptGenerationAgent(state)