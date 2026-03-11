## Code_Generation_v0 – Agentic 5G Feature Code Generator

This project is an **agentic pipeline** that turns a high‑level 5G protocol feature request (e.g. *“Implement F1AP UE CONTEXT SETUP REQUEST Message Procedure of Inter‑gNB‑DU LTM Handover”*) into:

- **Structured specification understanding**
- **Specification + codebase knowledge graphs and vector indices**
- **Concrete multi‑source context (specs + code)**
- **A filled implementation template in JSON**
- **A very rich, enforcement‑heavy code‑generation prompt**
- (Optionally) **post‑implementation validation** of the modified codebase

The system is exposed to tools (e.g. Cursor MCP) via an MCP server so that IDE agents can orchestrate the pipeline and then run code generation / validation on top.

---

## High‑Level Architecture

At the center of the system is the `AIServiceGateway` orchestrator:

- **Entry point**: `gateway.py` → class `AIServiceGateway`
- **State container**: `GlobalState` (`GlobalState.py`) – a typed dictionary tracking session id, messages, specs, retrieval sources, contexts, template paths, prompts, and validation results.
- **Main pipeline** (simplified):
  1. **Feature Validation Agent** (`Feature_Validation/main.py`)
  2. **Knowledge Creation Agent** (`Knowledge_Retrieval/Knowldge_creations/knowledge_creator_agent.py`)
  3. **Knowledge Retrieval Agent** (`Knowledge_Retrieval/retriever_agent.py`)
  4. **Template Filler Agent** (`Template_Orchestrator/template_filler_agent.py`)
  5. *(Optional)* **Code Validation Agent** (`Code_Validation/code_validation_agent.py`)

The MCP server (`mcp_server.py`) wraps this pipeline in tools that IDEs can call:

- `generate_enriched_prompt(intent)` – runs the full orchestration and returns a **draft code‑generation prompt** plus metadata.
- `validate_repo_code(branch)` – runs the **validation pipeline** on a git branch after code changes are pushed.
- `commit_and_push_feature_branch(...)` – helper for client/server git‑sync.

---

## GlobalState (`GlobalState.py`)

`GlobalState` is a `TypedDict` used as the shared state between all agents.

- **Core session fields**
  - `session_id`: unique id per run
  - `messages`: LangGraph / LangChain message list (usually the initial `HumanMessage` with the user query)
- **Feature validation outputs**
  - `message_names`: extracted protocol message names
  - `protocol_classification`: protocol/interface/change‑type metadata
  - `specifications`: list of 3GPP specs to use (with numbers, titles, and downloaded PDF paths)
  - `selected_template_name`, `selected_template_path`: chosen feature template
  - `feature_intent`: normalized feature description
- **Knowledge retrieval**
  - `specs_retrieval_sources`: locations of spec KG + FAISS indices
  - `code_retrieval_sources`: locations of code KG + FAISS indices
  - `specs_context`: consolidated spec chunks
  - `specs_chunks_path`: saved specs chunks JSON
  - `code_artifacts_context`: consolidated code chunks
  - `code_artifacts_chunks_path`: saved code chunks JSON
- **Template / prompt**
  - `spec_filled_template_path`: filled specification‑only template
  - `final_filled_template_path`: final spec+code template
  - `code_generation_prompt`, `code_generation_prompt_path`: full enriched prompt and its file location
- **Validation**
  - `code_validation_results`: results from `CodeValidationAgent` (if run)

---

## Orchestrator – `AIServiceGateway` (`gateway.py`)

**Role**: Top‑level orchestrator that wires all agents together for a single user query.

- **Method**: `orchestrator(user_query: str, run_validation: bool = False, ...) -> GlobalState`
- **Steps**:
  1. Generate a `session_id` and initialize `GlobalState` with the initial `HumanMessage`.
  2. Call `FeatureValidationAgent.run(state)` to:
     - Extract message names from the query.
     - Classify protocol/interface and derive specs.
     - Download relevant 3GPP PDFs.
     - Select the right JSON template.
     - Build a normalized feature intent.
  3. Call `knowledge_creator_agent.createKnowledge(state)` to:
     - Produce spec + code knowledge graphs and vector indices.
  4. Call `retriever_agent.retrieverAgent(state)` to:
     - Retrieve multi‑source spec and code chunks using KG + FAISS.
     - Store them as `specs_context` and `code_artifacts_context`.
  5. Call `template_filler_agent.templateFillerAgent(state)` to:
     - Fill the spec template.
     - Fill the code template based on spec + code contexts.
     - Generate the final **code‑generation prompt** and save it.
  6. *(Optionally)* run `CodeValidationAgent` to validate a patched codebase.

The return value is the final `GlobalState`, including the generated prompt and all intermediate artefacts.

---

## Feature Validation Agent (`Feature_Validation/main.py`)

**Goal**: Turn a free‑form user query into a **validated feature definition**, selected template, and concrete spec set.

Implements a `StateGraph(GlobalState)` with the following nodes:

- **`extract_message`**
  - Uses Azure OpenAI to extract 5G message names from the user query.
  - Writes `message_names` into state.
- **`classify_protocol`**
  - Classifies primary protocol, interface, entities, and change type using LLM.
  - Uses config from `config/spec_registry.json`.
  - Writes `protocol_classification`.
- **`resolve_specs`**
  - Resolves which 3GPP specs apply:
    - Adds the primary protocol spec and architecture specs.
    - Uses LLM to expand to related specs (while constrained to `VALID_SPECS`).
  - Writes the `specifications` list (spec numbers, titles, placeholders for PDF paths).
- **`select_template`**
  - Loads JSON templates from `Feature_Validation/template_store`.
  - Uses LLM to pick the most suitable template for the classification.
  - Writes `selected_template_name`, `selected_template_json`, `selected_template_path`.
- **`build_feature_intent`**
  - Creates `feature_intent` (domain, technology, primary protocol, change type, original query).
- **`download_spec`**
  - For each spec:
    - Uses Serper API to search ETSI for the **latest PDF**.
    - Downloads the PDF into `./data/specs`.
    - Extracts a `doc_id` from the URL.
  - Updates each spec entry with `doc_id` and `downloaded_pdf_path`.

The assembled state is then passed to the knowledge creation agent.

---

## Knowledge Creation Agent (`Knowledge_Retrieval/Knowldge_creations/knowledge_creator_agent.py`)

**Goal**: Build knowledge graphs + vector databases for both **specifications** and **codebase**, and wire their paths into `GlobalState`.

### Specification Knowledge

- **Function**: `specKnowledgeCreatorForEachSpec(DOC_ID, SPEC_PATH, SPEC_NUM, RUN_KNOWLEDGE_CREATE=False)`
  - Uses `CreateSpecVectorKg` to:
    - Parse the spec PDF into sections and deepest‑level chunks.
    - Extract relationships:
      - Hierarchical (section tree).
      - Explicit references between clauses.
      - Semantic relations via FAISS‑based nearest neighbors.
    - Build a networkx knowledge graph and save it to:
      - `resources/Specs_Knowledge/{DOC_ID}/KnowledgeGraph/knowledge_graph.pkl`
    - Build FAISS index + metadata and save to:
      - `resources/Specs_Knowledge/{DOC_ID}/vector_db/faiss_index.index`
      - `resources/Specs_Knowledge/{DOC_ID}/vector_db/faiss_metadata.json`
  - Returns a small dict describing these paths.
- **Function**: `specKnowledgeCreator(state)`
  - Iterates over `state['specifications']`.
  - Calls `specKnowledgeCreatorForEachSpec` (with `RUN_KNOWLEDGE_CREATE` flag).
  - Stores the list as `state['specs_retrieval_sources']`.

### Codebase Knowledge

- **Function**: `codeKnowledgeCreator(state, RUN_KNOWLEDGE_CREATE=False)`
  - For the OAI codebase:
    - Parses the codebase (`parse_codebase`).
    - Extracts code chunks (`extract_chunks`).
    - Builds a code knowledge graph + FAISS index (`build_kg_vector_new`).
  - Paths are stored in:
    - `resources/Codebase_Knowledge/OAI/KnowledgeGraph/knowledge_graph.pkl`
    - `resources/Codebase_Knowledge/OAI/vector_db/faiss_index.index`
    - `resources/Codebase_Knowledge/OAI/vector_db/faiss_metadata.json`
  - Sets `state['code_retrieval_sources']` with codebase name, target dirs, and these paths.

- **Function**: `createKnowledge(state)`
  - Runs spec + code knowledge creation.
  - Returns updated `GlobalState`.

---

## Knowledge Retrieval Agent (`Knowledge_Retrieval/retriever_agent.py`)

**Goal**: Given the feature intent, spec/code knowledge sources, and templates, retrieve **rich, structured context** from both specifications and code.

### Spec Chunk Retrieval – `specChunkRetrieverAgent`

- Inputs:
  - `state['specs_retrieval_sources']` – built by knowledge creator.
  - `message_name` – top‑level message name from `state['message_names'][0]`.
  - `template_path` – path to the selected template.
- Flow:
  - Filters `specs_retrieval_sources` down to the required spec IDs (e.g. `ts_138473v180400p`, `ts_138401v180600p`).
  - Initializes `MultiSourceChunkRetriever` with:
    - All selected spec knowledge sources
    - A top‑chunk percentage and `top_k` per source
    - Template path (to align retrieval with template fields)
  - Steps:
    1. Retrieve initial chunks per source (semantic search).
    2. Save raw combined chunks to `./outputs/spec_chunks`.
    3. Discover main IE definitions (`discover_main_ie_definition`).
    4. Perform **agentic iterative IE expansion** (`expand_ies_agentically`) to recursively pull all relevant IE chunks.
    5. Combine everything into `final_chunks`.
  - Outputs:
    - `state['specs_context'] = final_chunks`
    - `state['specs_chunks_path'] = <path to chunks JSON>`

### Code Chunk Retrieval – `codeChunkRetrieverAgent`

- Inputs:
  - `state['code_retrieval_sources']` – KG + FAISS + metadata for the codebase.
  - `query` = `message_name` (feature‑specific message identifier).
- Flow:
  - Builds a config dict for `SemanticGraphRAG` with FAISS index, metadata, KG path, etc.
  - Uses `SemanticGraphRAG.retrieve` to:
    - Retrieve top semantic chunks.
    - Expand via KG traversal (e.g. `function_calls`, `function_uses_struct`) to follow code relationships.
  - Builds output JSON (semantic chunks, expanded chunks, metadata), saves to `./outputs/code_chunks/{feature}_chunks.json`.
  - Writes:
    - `state['code_artifacts_context']` – structured view of semantic and expanded code chunks.
    - `state['code_artifacts_chunks_path']` – JSON location on disk.

### Aggregator – `retrieverAgent(state)`

- Derives:
  - `message_name = state['message_names'][0]`
  - `user_intent = state['messages'][0].content`
- Steps:
  1. Run `specChunkRetrieverAgent` to build `specs_context`.
  2. Run `codeChunkRetrieverAgent` to build `code_artifacts_context`.
- Returns updated state, ready for the template filler.

---

## Template Filler Agent (`Template_Orchestrator/template_filler_agent.py`)

**Goal**: Fill the selected JSON template in two passes (spec + code), then build a **very rich, reusable code‑generation prompt**.

This module wires three components:

- `SpecTemplateFiller` (`spec_template_filler.py`)
- `CodeTemplateFiller` (`code_template_filler.py`)
- `promptGenerationAgent` (`prompt_generator.py`)

The module also configures the LLM stack:

- If `GOOGLE_API_KEY` is set:
  - Uses `ChatGoogleGenerativeAI` (`gemini-2.5-flash`) with low temperature.
- Else, if Azure OpenAI variables are set:
  - Uses `AzureChatOpenAI` (`gpt‑4o‑mini` or configured deployment).
- Else:
  - Raises `ValueError` (no LLM credentials).

### Spec Template Filling – `specTemplateFillerAgent(state)`

- Inputs:
  - `state['selected_template_path']` – template JSON file.
  - `state['specs_context']` – retrieved spec chunks.
  - User query (`state['messages'][0].content`).
- Steps:
  1. Instantiate `SpecTemplateFiller(template_file=TEMPLATE_FILE)`.
  2. Call `extract_information(query, chunks)` to derive structured information elements and constraints from specs.
  3. Call `fill_template(extracted_info, chunks)` to produce a **filled spec template**.
  4. Save the filled template under `./outputs/spec_filled_templates` and write path into:
     - `state['spec_filled_template_path']`.

### Code Template Filling – `codeTemplateFillerAgent(state)`

- Uses `CodeTemplateFiller(llm=llm)` and the spec‑filled template to:
  - Inject code‑specific knowledge (functions, structs, actors, files, etc.) based on `state['code_artifacts_context']`.
  - Produce a final **spec+code template** stored at:
    - `state['final_filled_template_path']`.

### Prompt Generation – `promptGenerationAgent(state)` (`Template_Orchestrator/prompt_generator.py`)

**Role**: Generate a **long‑form, highly structured, enforcement‑heavy** prompt that another LLM will use for real code generation.

- Inputs:
  - `state['messages'][0].content` – user query / feature intent.
  - `state['final_filled_template_path']` – filled template JSON.
- Behavior:
  1. Loads the filled template JSON.
  2. Constructs a **meta‑prompt** describing:
     - Required sections (role definition, task overview, implementation phases, checklists, output format, etc.).
     - Strict rules against partial implementations.
     - A detailed **Coverage Contract**:
       - Dependency closure report for IEs/structures.
       - Impact matrix linking each node to concrete code touch‑points.
       - Self‑audit coverage gate requiring full closure coverage.
  3. Calls Azure OpenAI to generate the final **code‑generation prompt text** (no code).
  4. Injects the Coverage Contract at the top, appends the filled template JSON at the end, and enforces the `"Here is the template :"` tail marker.
  5. Saves the prompt to `./outputs/code_generation_prompts/code_prompt_<timestamp>.txt`.
  6. Writes:
     - `state["code_generation_prompt"]`
     - `state["code_generation_prompt_path"]`.

### Orchestration – `templateFillerAgent(state)`

Pipeline:

1. `specTemplateFillerAgent(state)`
2. `codeTemplateFillerAgent(state)`
3. `promptGenerationAgent(state)`

Returns the fully enriched `GlobalState`.

---

## MCP Servers (`mcp_server.py` and `mcp_server copy 2.py`)

There are two MCP entrypoints; the primary one used by Cursor is `mcp_server.py`:

- **`generate_enriched_prompt(intent: str)`**
  - Lazily imports `AIServiceGateway`.
  - Calls `gateway.orchestrator(intent)`.
  - Stores the resulting `GlobalState` in a module‑level `_last_orchestrator_state`.
  - Returns:
    - `prompt`: the enriched code‑generation prompt.
    - `session_id`, `template_path` (if available).
    - `intent` echo.
    - A `success` flag and `type="draft_prompt"`.
- **`validate_repo_code(branch: str = "new_feature")`**
  - Uses `_last_orchestrator_state` (if any) to pull `user_intent` and `template_path` into `CodeValidationAgent`.
  - Performs:
    1. `git fetch`, `checkout`, and `pull` on the configured repo.
    2. Runs `git_diff.py` to export:
       - `Git_Diff` – snapshot of modified files.
       - `Difference_Folder` – patch‑like view of changes.
    3. Instantiates `CodeValidationAgent` pointing at these folders.
    4. Runs `run_noninteractive()` to execute:
       - Memory / leak scan (pattern + Cppcheck).
       - Mandatory IE presence check.
       - ASN.1–C struct dependency validation.
       - Functional checks based on tests generated from the same template and user intent.
  - Returns raw `results` plus friendly `status_messages` when all checks are clean.
- **`commit_and_push_feature_branch(...)`**
  - Phase‑split helper:
    - `phase="client_commands"`: returns shell commands for the IDE client to run (create/switch branch, add/commit/push).
    - `phase="server_pull"`: server‑side fetch/checkout/pull of the same branch.

`mcp_server copy 2.py` is a simpler prototype that exposes a static `code://{intent}` resource and a `prepare_code_generation` tool; it is not used in the main pipeline.

---

## Code Validation Agent (`Code_Validation/code_validation_agent.py`)

**Goal**: After code generation and developer edits, provide **post‑commit validation** of the modified codebase.

### Initialization

- Important parameters:
  - `OAI_PATH`: list of OAI repo roots.
  - `CODEBASE_PATH`: folder containing the **full** code snapshot for validation (usually from `Git_Diff`).
  - `CODE_DIFF_FOLDER`: folder with only changed files (usually `Difference_Folder`).
  - `TEMPLATE_PATH`: final filled template JSON used for feature spec.
  - `USER_INTENT`: original feature request.

### Phases

- **`memory_phase`**
  - Scans changed C files for allocation/free imbalance.
  - Runs `cppcheck` with leak‑oriented filters.
  - Stores leak lines in `results["memory"]`.
- **`mandatory_phase`**
  - Uses `TEMPLATE_PATH` to:
    - Load `Information_Elements`.
    - Ask the LLM to compute a **hierarchical list of feature‑level mandatory IEs**, allowing for optional IEs that are semantically mandatory.
    - Flatten the IE names and scan the OAI codebase for their presence (exact and variant forms).
  - Stores missing IEs per codebase path in `results["mandatory_ie"]`.
- **`dependency_phase`**
  - Validates **structural consistency between ASN.1 and updated C structs**:
    - Loads IE definitions (`Information_Elements`) from the template.
    - Extracts `typedef struct` definitions from headers in `CODE_DIFF_FOLDER`.
    - Uses LLM to build a semantic mapping `struct_name → IE_Name`.
    - Recursively validates each struct vs its ASN.1 counterpart using explicit rules:
      - Flattening allowed.
      - Optional fields may be omitted without failure.
      - Numeric ranges must be representable.
      - OCTET STRING semantics preserved.
    - Tracks invalid structures and reports them in `results["dependency"]`.
- **`functional_phase`**
  - Loads changed files from `CODE_DIFF_FOLDER`.
  - Loads template JSON for context.
  - Test case source:
    - Non‑interactive mode (`use_llm_only=True`): generates functional tests from the template.
    - Interactive mode: can accept developer‑written test cases.
  - For each test case:
    - Streams each modified file in chunks to the LLM with the test description.
    - Asks whether the code (structurally or logically) satisfies the test.
    - Aggregates PASS/FAIL and reasons.
  - Stores failed test cases in `results["functional"]`.

- **Convenience methods**
  - `run()` – interactive full run.
  - `run_noninteractive()` – non‑interactive run suitable for MCP server.
  - `run_with_state(state)` – binds template path and user intent from `GlobalState` and then runs.

---

## How to Run the Pipeline Manually

### 1. Direct script run (`gateway.py`)

From the project root (`Code_Generation_v0`):

```bash
python gateway.py
```

The `__main__` block:

- Creates `AIServiceGateway`.
- Uses a hard‑coded query (`F1AP UE CONTEXT SETUP RESPONSE ...` example).
- Runs `orchestrator(query)`.
- Prints a filtered view of the final `GlobalState` (without large contexts).

You can adapt the query string or wrap `AIServiceGateway` usage in your own script.

### 2. Via MCP server (`mcp_server.py`) – for IDE integration

- Ensure environment variables are set (see below).
- Configure Cursor to use `mcp_server.py` as an MCP server.
- From the IDE agent:
  - Call `generate_enriched_prompt(intent)` with a natural‑language feature request.
  - Display the `prompt` field to the user for review/approval.
  - Use the returned prompt as the **system/instruction prompt** for code generation.
  - After implementing code and pushing to a branch, call:
    - `validate_repo_code(branch="your_branch")` to run post‑commit validation.

---

## Environment Variables and External Dependencies

- **LLM providers**
  - Azure OpenAI (Feature Validation, Template Prompt Generation, Validation Agent):
    - `AZURE_OPENAI_API_KEY`
    - `AZURE_OPENAI_ENDPOINT`
    - `AZURE_OPENAI_API_VERSION`
    - `AZURE_OPENAI_MODEL_NAME`
  - Google Gemini (optional fallback for template filling):
    - `GOOGLE_API_KEY`
- **Search / spec download**
  - `SERPER_API_KEY` – used in Feature Validation to find latest ETSI PDFs.
- **Validation LLM**
  - Uses `AzureOpenAI` client (same `AZURE_*` envs).

Additional tools required on the host:

- `git`
- `cppcheck`
- Build toolchain for OAI (gcc/clang, CMake, etc.) if you enable the `build_phase`.

---

## Outputs and Artefacts

Key output folders (relative to project root):

- `./data/specs` – downloaded ETSI 3GPP PDFs.
- `./resources/Specs_Knowledge/<DOC_ID>/...` – spec knowledge graph + FAISS.
- `./resources/Codebase_Knowledge/OAI/...` – code knowledge graph + FAISS.
- `./outputs/spec_chunks` – raw and expanded spec chunks JSON.
- `./outputs/code_chunks` – semantic + KG‑expanded code chunks JSON.
- `./outputs/spec_filled_templates` – filled spec templates.
- `./outputs/final_filled_templates` – final spec+code templates (consumed by prompt generator).
- `./outputs/code_generation_prompts` – generated prompts ready for code generation.
- `./Git_Diff`, `./Difference_Folder` – created by `git_diff.py` during validation.

---

## Extensibility Notes

- **New protocols / features**
  - Extend `Feature_Validation/config/spec_registry.json` with new spec mappings.
  - Add new templates under `Feature_Validation/template_store`.
- **Alternative codebases**
  - Update paths in `knowledge_creator_agent.codeKnowledgeCreator` and `CodeValidationAgent.__init__` to point at a different repo.
  - Rebuild code KG + FAISS for that codebase.
- **Custom validation steps**
  - Extend `CodeValidationAgent` with new phases or additional LLM checks.
  - Wire them into `run()` / `run_noninteractive()` as needed.

