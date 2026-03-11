import os
import logging
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from .spec_template_filler import SpecTemplateFiller
from .code_template_filler import CodeTemplateFiller
from .prompt_generator import promptGenerationAgent

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


def specTemplateFillerAgent(state):
       
    OUTPUT_DIR = "./outputs/spec_filled_templates"
    # OUTPUT_DIR_CHUNKS = "../outputs/spec_chunks"
    TEMPLATE_FILE = state.get("selected_template_path")
    SPEC_CHUNKS = state.get("specs_context")
    QUERY = state.get("messages")[0].content


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
        chunks=SPEC_CHUNKS
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
    code_artifacts_filler = CodeTemplateFiller(llm=llm)

    SPEC_TEMPLATE_PATH = state.get("spec_filled_template_path")
    
    final_filled_template_path = code_artifacts_filler.template_filler(state, SPEC_TEMPLATE_PATH)
    state["final_filled_template_path"] = final_filled_template_path
    return state


def templateFillerAgent(state):
    # print("Specification Template Filler")
    state = specTemplateFillerAgent(state)

    # print("Code Artifacts Template Filler")
    state = codeTemplateFillerAgent(state)

    # print("Prompt Generation")
    state = promptGenerationAgent(state)

    return state


