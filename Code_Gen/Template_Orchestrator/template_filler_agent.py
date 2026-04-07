import os
import logging
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from .spec_template_filler import SpecTemplateFiller
from .code_template_filler import CodeTemplateFiller
from .prompt_generator import promptGenerationAgent
from .spec_retrieval_context_adapter import (
    agentic_ie_retrieval_to_template_filler_inputs,
    normalize_chunks_for_spec_template_filler,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# LLM Configuration
# ----------------------------------------------------------

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o-mini")

# Google Gemini Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = None
if GOOGLE_API_KEY:
    try:
        llm = ChatGoogleGenerativeAI(
            api_key=GOOGLE_API_KEY,
            model="gemini-2.5-flash",
            temperature=0.2,
        )
        # print("Using Google Gemini LLM")
    except Exception as e:
        # logger.error("Failed to initialize Google Gemini: %s", e)
        llm = None

if llm is None and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
    try:
        llm = AzureChatOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=AZURE_DEPLOYMENT_NAME,
            temperature=0.3,
        )
        # print("Using Azure OpenAI LLM")
    except Exception as e:
        # logger.error("Failed to initialize Azure OpenAI: %s", e)
        llm = None

if llm is None:
    raise ValueError("No LLM credentials found. Please set GOOGLE_API_KEY or Azure OpenAI credentials.")


@dataclass
class Message:
    content: str


def _ensure_messages_from_intent(state):
    if state.get("messages"):
        return state
    intent = state.get("intent", "")
    if intent:
        state["messages"] = [Message(content=str(intent))]
    return state


def _resolve_template_path(state):
    direct = state.get("selected_template_path")
    if direct and os.path.exists(direct):
        return direct
    payload = state.get("specs_retrieval_payload")
    if isinstance(payload, dict):
        payload_template = payload.get("template_path")
        if payload_template and os.path.exists(payload_template):
            return payload_template
    repo_root = Path(__file__).resolve().parent.parent
    fallback = repo_root / "inputs" / "Template.json"
    return str(fallback)


def specTemplateFillerAgent(state):
    state = _ensure_messages_from_intent(state)
    OUTPUT_DIR = str(Path(__file__).resolve().parent.parent / "outputs" / "spec_filled_templates")
    TEMPLATE_FILE = _resolve_template_path(state)
    QUERY = state.get("intent") or state.get("messages")[0].content

    # New retriever stores full agentic payload; adapt+normalize when available.
    if isinstance(state.get("specs_retrieval_payload"), dict):
        adapted = agentic_ie_retrieval_to_template_filler_inputs(
            state["specs_retrieval_payload"]
        )
        SPEC_CHUNKS = adapted["chunks"]
        FEATURE_JSON_PATH = adapted.get("feature_json_path")
        if not QUERY:
            QUERY = adapted.get("query", "")
        if (
            (not state.get("selected_template_path"))
            and adapted.get("template_path")
            and os.path.exists(adapted["template_path"])
        ):
            TEMPLATE_FILE = adapted["template_path"]
    else:
        SPEC_CHUNKS = normalize_chunks_for_spec_template_filler(
            state.get("specs_context", []),
            assign_order_scores=True,
        )
        FEATURE_JSON_PATH = None


    # Specification Agentic Template Filler
    # print("=" * 60)
    # print("Specification Agentic Template Filling")
    # print("=" * 60)
    
    
    # Initialize template filler for IE discovery
    spec_template_filler = SpecTemplateFiller(template_file=TEMPLATE_FILE)

    # Step 7: Extract information using LLM
    # print("=" * 60)
    # print("Step 1: Extracting Information (Multi-Source Aware)")
    # print("=" * 60)
    extracted_info = spec_template_filler.extract_information(
        query=QUERY,
        chunks=SPEC_CHUNKS,
        feature_json_path=FEATURE_JSON_PATH,
    )
    
    # Step 8: Fill template
    # print("=" * 60)
    # print("Step 2: Filling Template")
    # print("=" * 60)
    filled_template = spec_template_filler.fill_template(
        extracted_info=extracted_info,
        chunks=SPEC_CHUNKS
    )
    
    # Step 9: Save filled template
    # print("=" * 60)
    print("Step 3: Saving Filled Template")
    # print("=" * 60)
    spec_template_path = spec_template_filler.save_output(
        filled_template=filled_template,
        query=QUERY,
        output_dir=OUTPUT_DIR
    )


    state['spec_filled_template_path'] = spec_template_path


    return state

    
def codeTemplateFillerAgent(state):
    state = _ensure_messages_from_intent(state)
    code_artifacts_filler = CodeTemplateFiller(llm=llm)

    SPEC_TEMPLATE_PATH = state.get("spec_filled_template_path")
    
    final_filled_template_path = code_artifacts_filler.template_filler(state, SPEC_TEMPLATE_PATH)
    state["final_filled_template_path"] = final_filled_template_path
    return state


def templateFillerAgent(state):
    state = _ensure_messages_from_intent(state)
    # print("Specification Template Filler")
    state = specTemplateFillerAgent(state)

    # print("Code Artifacts Template Filler")
    state = codeTemplateFillerAgent(state)

    # print("Prompt Generation")
    state = promptGenerationAgent(state)

    return state


