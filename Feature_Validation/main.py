# feature_validation.py

import os
import json
import re
import requests
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


from GlobalState import GlobalState
# import logging

# logger = logging.getLogger(__name__)


# ==========================================================
# ENV SETUP
# ==========================================================

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION")
deployment = os.getenv("AZURE_OPENAI_MODEL_NAME")
serper_api_key = os.getenv("SERPER_API_KEY")

if not all([api_key, endpoint, api_version, deployment]):
    raise ValueError("Missing Azure OpenAI environment variables")

if not serper_api_key:
    raise ValueError("Missing SERPER_API_KEY")

llm = AzureChatOpenAI(
    azure_deployment=deployment,
    api_key=api_key,
    azure_endpoint=endpoint,
    api_version=api_version,
    temperature=0.0
)


# ==========================================================
# LOAD SPEC REGISTRY
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
SPEC_REGISTRY_PATH = BASE_DIR / "config" / "spec_registry.json"

def load_spec_registry():
    with open(SPEC_REGISTRY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return (
        data.get("VALID_SPECS", {}),
        data.get("PRIMARY_PROTOCOL_SPEC", {}),
        data.get("ARCHITECTURE_SPECS", [])
    )

VALID_SPECS, PRIMARY_PROTOCOL_SPEC, ARCHITECTURE_SPECS = load_spec_registry()


# ==========================================================
# 1️ MESSAGE EXTRACTION
# ==========================================================

def extract_message_node(state: GlobalState) -> GlobalState:

    user_query = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )

    prompt = f"""
Extract 5G NR / 5G Core protocol message names from:

{user_query}

Return JSON:
{{ "message_names": [] }}
"""

    response = llm.invoke(
        [HumanMessage(content=prompt)],
        response_format={"type": "json_object"}
    )

    data = json.loads(response.content)

    return {
        "message_names": list(set(data.get("message_names", [])))
    }


# ==========================================================
# 2️ PROTOCOL CLASSIFICATION
# ==========================================================

def classify_protocol_node(state: GlobalState) -> GlobalState:

    user_query = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    ).upper()

    detected_protocol = None
    for proto in PRIMARY_PROTOCOL_SPEC.keys():
        if proto in user_query:
            detected_protocol = proto
            break

    prompt = f"""
You are a 5G protocol classifier.

Query:
{user_query}

Return JSON:
{{
  "primary_protocol": "",
  "interface": "",
  "network_entities": [],
  "change_type": "",
  "confidence": 0.0
}}
"""

    response = llm.invoke(
        [HumanMessage(content=prompt)],
        response_format={"type": "json_object"}
    )

    classification = json.loads(response.content)

    if detected_protocol:
        classification["primary_protocol"] = detected_protocol
        classification["confidence"] = 1.0

    classification["primary_protocol"] = classification.get("primary_protocol", "").upper()

    return {"protocol_classification": classification}


# ==========================================================
# 3️ SPEC RESOLUTION
# ==========================================================

def resolve_specs_node(state: GlobalState) -> GlobalState:

    classification = state.get("protocol_classification", {})
    protocol = classification.get("primary_protocol")

    specs = set()

    primary_spec = PRIMARY_PROTOCOL_SPEC.get(protocol)
    if primary_spec:
        specs.add(primary_spec)

    specs.update(ARCHITECTURE_SPECS)

    # LLM-based related spec expansion
    prompt = f"""
For implementing this 5G feature:

{json.dumps(classification, indent=2)}

From this valid spec list:
{json.dumps(list(VALID_SPECS.keys()), indent=2)}

Return additional related 3GPP TS numbers required (if any).

Return JSON:
{{ "related_specs": [] }}
"""

    response = llm.invoke(
        [HumanMessage(content=prompt)],
        response_format={"type": "json_object"}
    )

    related_specs = json.loads(response.content).get("related_specs", [])
    related_specs = [s for s in related_specs if s in VALID_SPECS]

    specs.update(related_specs)

    spec_objects = [
        {
            "spec_number": spec,
            "title": VALID_SPECS.get(spec, ""),
            "clause": "",
            "description": "",
            "pdf_link": "",
            "downloaded_pdf_path": ""
        }
        for spec in sorted(specs)
    ]

    return {"specifications": spec_objects}


# ==========================================================
# 4️ TEMPLATE SELECTION
# ==========================================================

def load_template_registry(template_dir="template_store"):

    registry = []

    BASE_DIR = Path(__file__).resolve().parent
    template_path = BASE_DIR / "template_store"

    # template_path = Path(template_dir)
    # print("DEBUG : ",template_path)

    for file in template_path.glob("*.json"):
        if file.stat().st_size == 0:
            continue

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        registry.append({
            "template_name": file.stem,
            "protocol_layers": data.get("Design_Details", {}).get("Protocol_Layers", []),
            "protocol_interfaces": data.get("Design_Details", {}).get("Protocol_Interfaces", []),
            "asn_domains": [x.get("ASN_Name") for x in data.get("ASN_IES_File_Path", [])],
            "file_path": str(file)
        })

    return registry


def select_template_node(state: GlobalState) -> GlobalState:

    classification = state.get("protocol_classification", {})
    registry = load_template_registry()

    prompt = f"""
Feature Classification:
{json.dumps(classification, indent=2)}

Available Templates:
{json.dumps(registry, indent=2)}

Return JSON:
{{ "selected_template": "" }}
"""

    response = llm.invoke(
        [HumanMessage(content=prompt)],
        response_format={"type": "json_object"}
    )

    selected_name = json.loads(response.content).get("selected_template")

    selected_template = next(
        (t for t in registry if t["template_name"] == selected_name),
        None
    )

    if not selected_template:
        raise ValueError("No matching template found")

    with open(selected_template["file_path"], "r", encoding="utf-8") as f:
        template_json = json.load(f)

    return {
        "selected_template_name": selected_name,
        "selected_template_json": template_json,
        "selected_template_path": selected_template["file_path"]
    }


# ==========================================================
# 5️ FEATURE INTENT
# ==========================================================

def build_feature_intent_node(state: GlobalState) -> GlobalState:

    user_query = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )

    classification = state.get("protocol_classification", {})

    return {
        "feature_intent": {
            "original_query": user_query,
            "domain": "3GPP_RAN",
            "technology": "5G_NR",
            "release_version": "",
            "primary_protocol": classification.get("primary_protocol"),
            "change_type": classification.get("change_type")
        }
    }


# ==========================================================
# 6️ DOWNLOAD SPECS (ETSI Latest Version Logic)
# ==========================================================

def download_spec_node(state: GlobalState) -> GlobalState:

    specifications = state.get("specifications", [])
    out_dir = Path("./data/specs")
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in specifications:

        spec_number = spec["spec_number"]
        # print(f"\n--- Searching PDF for {spec_number} ---")

        search_query = f"{spec_number} latest release pdf site:etsi.org"

        headers = {
            "X-API-KEY": serper_api_key,
            "Content-Type": "application/json"
        }

        payload = {"q": search_query}

        response = requests.post(
            "https://google.serper.dev/search",
            headers=headers,
            json=payload
        )

        response.raise_for_status()
        organic_results = response.json().get("organic", [])

        etsi_pdf_links = []

        for result in organic_results:
            link = result.get("link", "")
            if "etsi.org" in link and link.endswith(".pdf"):
                match = re.search(r'(\d+)\.(\d+)\.(\d+)', link)
                if match:
                    version = tuple(map(int, match.groups()))
                    etsi_pdf_links.append((version, link))

        if not etsi_pdf_links:
            # print(f"! No ETSI PDF found for {spec_number}")
            continue

        etsi_pdf_links.sort(reverse=True)
        latest_pdf_url = etsi_pdf_links[0][1]

        #  NEW: Extract doc_id
        doc_id = Path(latest_pdf_url).stem
        spec["doc_id"] = doc_id

        pdf_path = out_dir / f"{spec_number.replace(' ', '_')}.pdf"

        if pdf_path.exists():
            spec["downloaded_pdf_path"] = str(pdf_path)
            continue

        with requests.get(latest_pdf_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(pdf_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)

        spec["downloaded_pdf_path"] = str(pdf_path)

    return {"specifications": specifications}

# ==========================================================
# GRAPH
# ==========================================================

graphBuilder = StateGraph(GlobalState)

graphBuilder.add_node("extract_message", extract_message_node)
graphBuilder.add_node("classify_protocol", classify_protocol_node)
graphBuilder.add_node("resolve_specs", resolve_specs_node)
graphBuilder.add_node("select_template", select_template_node)
graphBuilder.add_node("build_feature_intent", build_feature_intent_node)
graphBuilder.add_node("download_spec", download_spec_node)

graphBuilder.add_edge(START, "extract_message")
graphBuilder.add_edge("extract_message", "classify_protocol")
graphBuilder.add_edge("classify_protocol", "resolve_specs")
graphBuilder.add_edge("resolve_specs", "select_template")
graphBuilder.add_edge("select_template", "build_feature_intent")
graphBuilder.add_edge("build_feature_intent", "download_spec")
graphBuilder.add_edge("download_spec", END)

graph = graphBuilder.compile()


# ==========================================================
# FEATURE VALIDATION AGENT
# ==========================================================

class FeatureValidationAgent:

    def __init__(self):
        self.graph = graph

    def run(self, state: GlobalState) -> GlobalState:
        return self.graph.invoke(state)


# ==========================================================
# STANDALONE TEST
# ==========================================================

# if __name__ == "__main__":

#     import uuid

#     agent = FeatureValidationAgent()

#     state: GlobalState = {
#         "session_id": str(uuid.uuid4()),
#         "messages": [
#             HumanMessage(content="Implement F1AP UE CONTEXT SETUP REQUEST Message Procedure of Inter-gNB-DU LTM Handover")
#         ]
#     }

#     updated_state = agent.run(state)

#     print(json.dumps(updated_state, indent=2, default=str))
