# Code_Gen Framework: End-to-End Flow

This document describes the complete runtime flow of the `Code_Gen` framework from client request to final prompt output, including every major agent/stage, its role, input/output, dependencies, and how data moves through the pipeline.

---

## 1) End-to-End Flow (Client -> Server -> Pipeline -> Output)

```mermaid
flowchart TD
    A[Client sends intent] --> B[MCP tool: generate_enriched_prompt(intent)]
    B --> C[Pipeline: run_end_to_end_from_intent]
    C --> D[Feature Validation agents]
    D --> E[Knowledge Creation agents]
    E --> F[Retrieval agents: Spec + Code]
    F --> G[Template Orchestrator pass 1: SpecTemplateFiller]
    G --> H[Template Orchestrator pass 2: CodeTemplateFiller]
    H --> I[Self-Learning validation]
    I -->|ambiguities| J[Return questions to client]
    J --> K[MCP tool: resolve_self_learning_ambiguities(intent,resolutions)]
    K --> I
    I -->|resolved/no ambiguities| L[Prompt Generation agent]
    L --> M[Return draft prompt + template + paths]
    M --> N[Client executes generated prompt in coding workflow]
    N --> O[Commit_and_push_feature_branch]
    O --> P[Validate_repo_code(branch)]
```

---

## 2) MCP Server Entry and Control

### 2.1 `generate_enriched_prompt(intent)`

- **Role**: Main public MCP entrypoint for orchestration.
- **Input**: User intent string from client.
- **Output**:
  - `draft_prompt` payload (prompt text + template text), or
  - `self_learning_ambiguity_review` payload if ambiguity resolution is required.

### 2.2 `resolve_self_learning_ambiguities(intent, resolutions)`

- **Role**: Re-runs the pipeline with explicit user resolutions for ambiguity IDs.
- **Input**:
  - `intent`
  - `resolutions` map keyed by ambiguity ID.
- **Output**:
  - Another ambiguity review (if unresolved ambiguities remain), or
  - `draft_prompt` when fully resolved.

### 2.3 Post-Code-Generation MCP tools

- **`commit_and_push_feature_branch(...)`**
  - Two-phase branch sync helper (client commands phase + server pull phase).
- **`validate_repo_code(branch)`**
  - Pulls specified branch on server-side target repo and runs code validation agent.
  - Persists validation results in session database when session is available.

---

## 3) Pipeline Core Orchestrator

The full runtime path is controlled by `run_end_to_end_from_intent(user_intent, ambiguity_resolutions=None)`.

High-level stages:

1. **Feature Validation**
2. **Knowledge Source Readiness**
3. **Spec Retrieval**
4. **Code Retrieval**
5. **Template Orchestration (spec pass + code pass)**
6. **Self-Learning Validation**
7. **Prompt Generation (if no pending ambiguities)**

---

## 4) Stage-by-Stage Agent

## Agent 1: Feature Validation Agent Stack

Entry function: `run_with_intent(user_intent)` from two-stage feature validation module.

### Stage A: Intent Classifier

- **Role**: Converts detailed user intent into canonical high-level 3GPP feature/procedure label.
- **Input**: Raw `user_intent`.
- **Output**: Normalized `feature_name` (with confidence/generic handling).

### Stage B: Architecture/Procedure Agent

- **Role**:
  - Identifies primary spec and section for the feature.
  - Retrieves latest ETSI PDF where available.
  - Extracts section text and message details.
- **Input**: `feature_name` (+ contextual text).
- **Output**:
  - `procedure_spec_info`
  - `section_text`
  - `message_details` (messages + feature protocols)

### Stage C: Protocol Spec Agent

- **Role**: Identifies protocol-specific specs relevant to extracted messages/protocols.
- **Input**: feature + procedure spec context + protocols.
- **Output**: `protocol_specs[]`.

### Stage D: Message Section Agent

- **Role**: Locates protocol-specific sections for each filtered message.
- **Input**: `protocol_specs[]` + `message_details`.
- **Output**: `protocol_message_sections[]`.

### Stage E: Template Selector

- **Role**: Selects template based on protocol priority.
- **Input**: feature protocols and protocol specs.
- **Output**:
  - `template_name`
  - `template_path`
  - selected protocol marker.

### Agent 1 Final Output Payload

Feature validation returns canonical payload containing (key fields):

- `intent`
- `specs[]` (spec number/link/doc id/downloaded path)
- `message_details`
- `protocol_specs`
- `protocol_message_sections`
- `template` (selected template info)

This payload is the source for downstream agents.

---

## Agent 2: Knowledge Retrieval Agent

The pipeline ensures required retrieval sources exist before retrieval starts.

### Stage: `specKnowledgeCreatorForEachSpec`

- **Role**: Ensure each spec has KG artifacts.
- **Input**: per-spec `doc_id`, `downloaded_pdf_path`, `spec_number`.
- **Output**: path metadata for spec KG/vector artifacts.

- **Working flow**:
  1. Parse TOC + sections from PDF.
  2. Chunk spec content.
  3. Build KG + summary/adjacency.
  4. Copy artifacts to canonical retrieval location.

### Stage: `codeKnowledgeCreator`

- **Role**: Ensure code KG and vector embeddings exist.
- **Input**: state + `RUN_KNOWLEDGE_CREATE` flag.
- **Output**: `code_retrieval_sources` with KG/FAISS paths.


---

### Stage: `Spec Retrieval Agent`

Entry function: `run_agentic_ie_retrieval_phase(...)`

- **Role**: Retrieve high-signal spec context using section seeds and recursive IE-aware expansion.
- **Input**:
  - Feature payload (from Agent 1),
  - template path,
  - spec KG base directory.
- **Output**:
  - `final_context` (deduplicated spec chunks),
  - retrieval traces (`kg_expansion_trace_by_doc_id`, `ie_recursive_trace_by_message`),
  - retrieval metadata/config.

- **Working flow**:
  1. Collect seed nodes/sections from feature payload.
  2. Expand KG neighborhoods around seeds.
  3. Infer main IE definitions (heuristic + LLM fallback).
  4. Recursively expand child IE structures.
  5. Merge IE chunks + expanded context and deduplicate.

---

### Stage: `Code Retrieval Agent`

Entry function: `codeChunkRetrieverAgent(state, query, CODE_KNOWLEDGE_PATHS)`

- **Role**: Retrieve implementation-relevant code chunks for each target message.
- **Input**:
  - user query/intent,
  - message names from feature payload,
  - code KG + FAISS paths.
- **Output**:
  - `code_artifacts_context`:
    - semantic chunks
    - KG-expanded chunks
    - metadata
  - serialized chunk files per-message and combined output.

- **Working flow**:
  1. Semantic retrieve top chunks for each message.
  2. Expand graph neighbors using selected relations.
  3. Merge all messages into one context package.

---

## Agent 3: Template Orchestration (Two Pass)

### Pass 1: Spec Template Filler (`SpecTemplateFiller`)

- **Role**: Fill spec/protocol/IE-centric template fields from spec retrieval context.
- **Input**:
  - selected template JSON,
  - adapted retrieval chunks (`final_context`),
  - query/intent.
- **Output**: spec-filled template JSON (`spec_filled_template_path`).

### Pass 2: Code Template Filler (`CodeTemplateFiller`)

- **Role**: Enrich template with code artifacts/building blocks from code retrieval.
- **Input**:
  - spec-filled template,
  - code retrieval chunks/context,
  - user intent.
- **Output**: final filled template JSON (`final_filled_template_path`).

---

## Agent 4: Self-Learning Validation Agent

Entry: `validate_template_with_mapping_rules(...)`

- **Role**:
  - Validate final filled template against long-term mapping rules.
  - Identify mismatches/ambiguities.
  - Optionally apply user-provided resolutions and emit resolved template.
- **Input**:
  - intent,
  - final template path,
  - mapping rules path,
  - optional user resolutions.
- **Output**:
  - `matched_rule`
  - `has_ambiguities`
  - `ambiguities[]`
  - `resolution_applied`
  - `resolved_template_path`.
- **Branching behavior**:
  - If ambiguities remain: pipeline status becomes awaiting-user and prompt generation is deferred.
  - Else: pipeline proceeds to prompt generation.

---

### Stage : Prompt Generation Stage

Entry: `promptGenerationAgent(state)`

- **Role**: Generate final execution-ready code-generation prompt from intent + resolved template.
- **Input**:
  - `messages[0].content` (intent),
  - `final_filled_template_path`.
- **Output**:
  - `code_generation_prompt` (full text),
  - `code_generation_prompt_path` (saved `.txt` artifact).

- **Working flow**:
  1. Load final filled template JSON.
  2. Build strict meta-prompt with mandatory coverage contract.
  3. Generate long-form implementation prompt.
  4. Persist prompt artifact.

---

## 5) State Persistence and Run Tracking

`SqliteStateStore` persists run lifecycle in `outputs/session_state.sqlite`.

Tracked entities:

- **sessions**: run-level intent/status/template/prompt pointers.
- **stage_runs**: each pipeline stage status and outputs.
- **agent_runs**: per-agent execution status/input/output/errors.
- **code_validations**: post-implementation validation records.

This provides reproducibility, observability, and recovery context.

---

## 6) Client-Server Handshake Summary

1. IDE (as a MCP Client) sends intent to `generate_enriched_prompt`.
2. Server runs full multi-stage pipeline on the MCP Server.
3. If ambiguities exist, server returns ambiguity questionnaire payload.
4. Client sends answers via `resolve_self_learning_ambiguities`.
5. Server returns final draft prompt payload when resolved.
6. Client executes prompt in coding workflow.
7. Client/server branch sync and `validate_repo_code` after code push.

---

## 7) Primary Inputs, Outputs, and Artifacts

### Primary Input

- `intent` string from MCP client.

### Final Primary Output

- Final enriched code-generation prompt text + template.

### Major Artifacts Produced During Pipeline

- Feature validation output JSON
- Spec retrieval context JSON
- Code chunks JSON (per-message + combined)
- Spec-filled template JSON
- Final filled template JSON
- Final code-generation prompt `.txt`
- Session/agent/stage telemetry in SQLite

---

## 8) Runtime Dependencies (System-Level)

- **MCP/Server**: `fastmcp`
- **LLMs**:
  - Azure OpenAI (`langchain_openai`)
  - Google Gemini (`langchain_google_genai`) for template orchestration path
- **Retrieval/Graph/Vector**:
  - `networkx`, `faiss`, `numpy`
  - `langchain_huggingface` embeddings
- **Spec Processing**:
  - `pdfplumber`, optional PDF loader stack
- **Code KG Creation**:
  - tree-sitter based parser/chunk pipeline
- **Persistence**:
  - `sqlite3`

---