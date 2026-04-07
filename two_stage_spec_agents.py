"""
Two-stage spec resolution (self-contained).

Agent 1 (architecture / procedure spec):
- Given a feature string, resolve the main procedure / architecture spec,
  locate the exact section containing the call-flow, and extract:
  - procedure_spec_info (spec_number, spec_version, section_id, spec_link, doc_id)
  - section_text
  - message_details (messages + feature_protocols)

Agent 2 (protocol-specific specs):
- Given the feature, procedure_spec_info and feature_protocols from Agent 1,
  search for the per-protocol 3GPP specs where the feature / handover is
  defined, and return:
  - spec_number (per protocol)
  - section_id (per spec where feature is defined)
  - latest Rel-18 ETSI PDF URL and version
  - local downloaded PDF path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any
import os
import re
import json
from datetime import datetime
import concurrent.futures

import requests
import pdfplumber
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_agent


load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_OPENAI_MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME")

llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_MODEL_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=OPENAI_API_VERSION,
    temperature=0.3,
    top_p=0.9,
)

BASE_DIR = Path(__file__).resolve().parent
SPECS_DIR = BASE_DIR / "specs"
FEATURE_RUNS_DIR = BASE_DIR / "feature_runs"
SPEC_REGISTRY_PATH = BASE_DIR / "config" / "spec_registry.json"
FEATURE_CATALOG_PATH = BASE_DIR / "config" / "feature_catalog.json"
TEMPLATES_DIR = BASE_DIR / "templates"


# ---------------------------------------------------------------------------
# Shared helpers (local copies, no cross-file dependencies)
# ---------------------------------------------------------------------------

def serperSearch(query: str) -> dict:
    """
    Web search using Serper API.
    Returns JSON results as provided by the API.
    """
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": 5}
    response = requests.post("https://google.serper.dev/search", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


def _parse_json_from_agent_result(results: dict) -> dict:
    """
    Extract JSON object from a LangChain agent result.
    Prefer 'output', then last message.content, then first {...} block.
    """
    if not isinstance(results, dict):
        raise ValueError(f"Unexpected agent result type: {type(results)}")

    raw_text = None
    output = results.get("output")
    if isinstance(output, str) and output.strip():
        raw_text = output

    if raw_text is None:
        messages = results.get("messages") or []
        if messages:
            last = messages[-1]
            content = getattr(last, "content", None)
            if content is None and isinstance(last, dict):
                content = last.get("content")
            if isinstance(content, str) and content.strip():
                raw_text = content

    if not raw_text:
        raise ValueError("Agent result does not contain non-empty 'output' or 'messages[-1].content'.")

    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        return json.loads(match.group(0))


def _extract_json_from_text(text: str) -> dict:
    """
    Utility to robustly extract a JSON object from raw LLM output.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("LLM returned empty content when JSON was expected.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        return json.loads(match.group(0))


def _looks_like_pdf(path: str) -> bool:
    """Quick validation: file exists and starts with %PDF- header."""
    try:
        p = Path(path)
        if not p.exists() or p.is_dir() or p.stat().st_size < 1024:
            return False
        with open(p, "rb") as f:
            header = f.read(5)
        return header == b"%PDF-"
    except Exception:
        return False


def _extract_pdf_url_from_text(text: str) -> str | None:
    if not text:
        return None
    match = re.search(r"https?://\S+?\.pdf(\?\S+)?", text)
    return match.group(0) if match else None


def find_latest_etsi_pdf_url(
    spec_number: str,
    serper_api_key: str,
    preferred_release_major: int = 18,
) -> dict | None:
    """
    Serper search -> collect ETSI pdf links with version -> pick latest.
    """
    search_query = f"{spec_number} latest release pdf site:etsi.org"
    headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}
    response = requests.post(
        "https://google.serper.dev/search",
        headers=headers,
        json={"q": search_query, "num": 10},
        timeout=30,
    )
    response.raise_for_status()
    organic_results = response.json().get("organic", [])

    etsi_pdf_links: list[tuple[tuple[int, int, int], str]] = []
    etsi_pdf_links_nover: list[str] = []
    for result in organic_results:
        link = (result.get("link", "") or "").strip()
        title = (result.get("title", "") or "").strip()
        snippet = (result.get("snippet", "") or "").strip()

        candidate = None
        if "etsi.org" in link and link.lower().endswith(".pdf"):
            candidate = link
        else:
            extracted = _extract_pdf_url_from_text(" ".join([link, title, snippet]))
            if extracted and "etsi.org" in extracted:
                candidate = extracted

        if not candidate:
            continue

        match = re.search(r"(\d+)\.(\d+)\.(\d+)", candidate)
        if match:
            version = tuple(map(int, match.groups()))
            etsi_pdf_links.append((version, candidate))
            continue

        match2 = re.search(r"v(\d{2})(\d{2})(\d{2})[a-z]?\.pdf(\?|$)", candidate, flags=re.IGNORECASE)
        if match2:
            version = tuple(map(int, match2.groups()[:3]))
            etsi_pdf_links.append((version, candidate))
            continue

        etsi_pdf_links_nover.append(candidate)

    if not etsi_pdf_links:
        if etsi_pdf_links_nover:
            return {"url": etsi_pdf_links_nover[0], "version": None}
        return None

    # Prefer a specific release major (e.g. Release 18 => 18.x.y) when available.
    preferred = [x for x in etsi_pdf_links if x[0][0] == int(preferred_release_major)]
    if preferred:
        preferred.sort(reverse=True)
        latest_version, latest_url = preferred[0]
        return {
            "url": latest_url,
            "version": latest_version,
            "preferred_release_major": int(preferred_release_major),
        }

    # Fallback: no preferred release found; use the highest overall.
    etsi_pdf_links.sort(reverse=True)
    latest_version, latest_url = etsi_pdf_links[0]
    return {
        "url": latest_url,
        "version": latest_version,
        "preferred_release_major": int(preferred_release_major),
    }


def download_etsi_latest_pdf(
    spec_number: str,
    out_dir: Path,
    serper_api_key: str,
    preferred_release_major: int = 18,
) -> dict | None:
    """
    Download latest ETSI PDF for spec_number into out_dir.
    Returns dict {downloaded_pdf_path, spec_link, doc_id} or None.
    """
    found = find_latest_etsi_pdf_url(
        spec_number,
        serper_api_key,
        preferred_release_major=preferred_release_major,
    )
    if not found:
        return None
    url = found["url"]
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_name = url.split("/")[-1]
    pdf_path = out_dir / spec_name

    if pdf_path.exists() and _looks_like_pdf(str(pdf_path)):
        return {"downloaded_pdf_path": str(pdf_path), "spec_link": url, "doc_id": Path(url).stem}

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(pdf_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
    if not _looks_like_pdf(str(pdf_path)):
        return None
    return {"downloaded_pdf_path": str(pdf_path), "spec_link": url, "doc_id": Path(url).stem}


def downloadSpec(spec_link: str) -> str:
    """
    Download the specification PDF and return local path.
    """
    SPECS_DIR.mkdir(parents=True, exist_ok=True)
    file_name = spec_link.split("/")[-1]
    file_path = SPECS_DIR / file_name

    if file_path.exists():
        return str(file_path)

    r = requests.get(spec_link)
    r.raise_for_status()

    with open(file_path, "wb") as f:
        f.write(r.content)

    return str(file_path)


def getSectionText(spec_path: str, section_id: str) -> str:
    """
    Get the text of the section from the specification PDF.
    """
    if not os.path.exists(spec_path):
        raise FileNotFoundError(f"Spec file not found: {spec_path}")

    section_id = section_id.strip()
    if not section_id:
        raise ValueError("section_id must be a non-empty string")

    page_chunks: list[str] = []
    with pdfplumber.open(spec_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            page_chunks.append(f"\n[PAGE {page_index}]\n")
            page_chunks.append(text)

    full_text = "".join(page_chunks)

    body_start_idx = 0
    toc_match = re.search(r"\b(Table of Contents|Contents)\b", full_text, flags=re.IGNORECASE)
    if toc_match:
        scope_match = re.search(r"(^|\n)\s*1\s+Scope\b", full_text[toc_match.end() :], flags=re.IGNORECASE)
        if scope_match:
            body_start_idx = toc_match.end() + scope_match.start()

    body_text = full_text[body_start_idx:]

    escaped_section = re.escape(section_id)
    section_start_pattern = re.compile(rf"(^|\n)\s*{escaped_section}(\s+|$)", flags=re.MULTILINE)

    start_match = None
    for match in section_start_pattern.finditer(body_text):
        line_start = body_text.rfind("\n", 0, match.start())
        if line_start == -1:
            line_start = 0
        else:
            line_start += 1
        line_end = body_text.find("\n", match.start())
        if line_end == -1:
            line_end = len(body_text)
        line_text = body_text[line_start:line_end]

        if re.search(r"\.{5,}\s*\d+\s*$", line_text):
            continue

        start_match = match
        break

    if not start_match:
        raise ValueError(
            f"Section id '{section_id}' not found in spec text (after TOC) "
            "or only appears in Table of Contents."
        )

    section_start_idx = start_match.start()

    next_heading_pattern = re.compile(r"(^|\n)\s*\d+(\.\d+)+\s+\S+", flags=re.MULTILINE)
    next_match = next_heading_pattern.search(body_text, pos=start_match.end())

    if next_match:
        section_end_idx = next_match.start()
    else:
        section_end_idx = len(body_text)

    section_text = body_text[section_start_idx:section_end_idx].strip()
    return section_text


def getMessageDetails(section_text: str) -> dict:
    """
    Interpret a spec section and extract call-flow messages + protocol layers.
    """
    if not section_text or not section_text.strip():
        raise ValueError("section_text must be a non-empty string")

    system_prompt = """
        You are a telecom protocol expert specializing in 3GPP specifications.

        Your task is to read a 3GPP spec section describing a signalling procedure
        and return a precise, strictly valid JSON description of the call-flow messages.

        CRITICAL CONSTRAINTS (NO EXCEPTIONS):
        - You MUST treat the numbered list of messages in the spec as the single source of truth.
        - For each numbered step that clearly represents a signalling message, you MUST:
          - keep the index exactly as shown in the spec (1, 2, 3, ...),
          - keep the message name exactly as written in the spec (or a lossless normalization,
            e.g. removing extra spaces or line breaks only),
          - keep the ordering identical to the spec.
        - You MUST NOT:
          - reorder messages,
          - insert new messages that are not explicitly present,
          - drop messages that are explicitly present.

        Direction and protocol mapping:
        - Determine sender and receiver ONLY from the spec text for that step
          (e.g. "UE sends", "source gNB-DU sends", "gNB-CU forwards to", etc.).
        - Use explicit roles in the direction string, e.g. "UE -> Source gNB-DU",
          "Source gNB-DU -> gNB-CU", "gNB-CU -> candidate gNB-DU", "target gNB-DU -> gNB-CU".
        - For protocol_layer:
          - If it is an RRC message name (e.g. MeasurementReport, RRCReconfiguration,
            RRCReconfigurationComplete), use "RRC".
          - If the text clearly corresponds to an F1 interface message between CU and DU,
            use "F1AP".
          - If the text clearly corresponds to NG interface signalling between gNB-CU and AMF,
            use "NGAP".
          - If the text states an L1 or physical measurement/report, use "PHY" or "L1"
            (but be consistent).
          - If the procedure includes a "Cell Switch Command" for LTM, that command is a
            lower-layer action and should be mapped to "MAC" (unless the spec explicitly
            states it is RRC).
          - If it is clearly a CU-DU control message not mapped to NGAP, use the most accurate
            protocol you can infer from the spec context (e.g. "F1AP"), and DO NOT default to
            "NGAP" unless the message is explicitly an NGAP message.
        - You MUST NOT use "NGAP" as a default protocol layer just because CU and DU are involved.

        At the end, determine the unique list of protocol layers that this
        feature/procedure belongs to as feature_protocols. This MUST be the set of
        all distinct protocol_layer values you actually used in the messages array.

        Additional rules:
        - The JSON MUST be syntactically valid: use double quotes, commas between fields/items,
          and numeric indices MUST be integers.
        - Do NOT include explanations outside the JSON. Respond with JSON ONLY.

        Output JSON format (no deviation):
        {
          "messages": [
            {
              "index": 1,
              "name": "<message name taken from spec>",
              "direction": "<sender with role> -> <receiver with role>",
              "protocol_layer": "<protocol or interface name>"
            }
          ],
          "feature_protocols": [
            "<protocol or interface name>",
            "... more if applicable ..."
          ]
        }
    """

    user_content = f"""Here is the 3GPP specification section text to analyze:
        \"\"\"text
        {section_text}
        \"\"\"
    """

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    )

    raw = getattr(response, "content", response)
    if isinstance(raw, list):
        raw_text = "".join(str(part) for part in raw)
    else:
        raw_text = str(raw)

    extracted = _extract_json_from_text(raw_text)
    return _post_process_message_details(extracted, section_text)


def _post_process_message_details(message_details: dict, section_text: str) -> dict:
    """
    Deterministic cleanup to avoid protocol-layer misclassification and ensure
    we don't miss lower-layer protocols for LTM (e.g., MAC for Cell Switch Command).

    Rules are intentionally simple and conservative:
    - Keep message order unchanged.
    - Only adjust protocol_layer when a message name clearly implies it.
    - Recompute feature_protocols as the set of protocol_layer values used.
    """
    if not isinstance(message_details, dict):
        return {"messages": [], "feature_protocols": []}

    messages = message_details.get("messages")
    if not isinstance(messages, list):
        messages = []

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    section_lc = (section_text or "").lower()
    mac_hint = "cell switch command" in section_lc

    for m in messages:
        if not isinstance(m, dict):
            continue
        name = m.get("name") or ""
        name_lc = _norm(str(name))
        pl = (m.get("protocol_layer") or "").strip()

        # Map L1/PHY measurement explicitly
        if "l1 measurement" in name_lc or "physical" in name_lc:
            m["protocol_layer"] = "PHY"
            continue

        # LTM Cell Switch Command is lower layer; map to MAC when hinted by section text.
        if "cell switch command" in name_lc and mac_hint:
            m["protocol_layer"] = "MAC"
            continue

        # If the message name itself contains RRC-specific identifiers, force RRC.
        if "rrc" in name_lc or "measurementreport" in name_lc:
            m["protocol_layer"] = "RRC"
            continue

        # Do not allow NGAP as a default label for CU-DU messages.
        # If LLM labeled NGAP but the message name is a CU-DU/DU-CU style transport/control,
        # keep it as "F1AP" (CU-DU interface) unless explicitly NG.
        if pl.upper() == "NGAP" and ("cu-du" in name_lc or "du-cu" in name_lc or "ue context" in name_lc):
            m["protocol_layer"] = "F1AP"

    # Recompute feature_protocols from final message list.
    protocols: List[str] = []
    seen = set()
    for m in messages:
        if not isinstance(m, dict):
            continue
        p = (m.get("protocol_layer") or "").strip()
        if not p:
            continue
        p_norm = normalize_protocol_name(p)
        if p_norm and p_norm not in seen:
            seen.add(p_norm)
            protocols.append(p_norm)

    return {
        "messages": messages,
        "feature_protocols": protocols,
    }


def getSpecDetails(feature: str) -> dict:
    """
    Agent to get spec_number, spec_version, section_id and spec_link for the given feature.
    """
    system_prompt = """
        You are a telecom protocol expert specialized in analyzing 3GPP specifications
        and extracting the architecture / procedure section that contains the call-flow
        messages for a given feature.

        VERY IMPORTANT SCOPE:
        - Your primary target is the **architecture-level specification** that defines
          the overall gNB / DU / CU architecture and procedures (for NR this is typically
          TS 38.401 or its direct successor).
        - When the feature relates to inter-gNB / DU / CU handover or mobility (e.g.
          "Inter-gNB-DU LTM handover procedure"), you MUST select the architecture spec
          (e.g. TS 38.401) as the main procedure/architecture spec.
        - Prefer the NR architecture spec with **Release 18** version (for example
          v18.6.0 or v18.7.0) when available on ETSI.

        Your objective is to:
        1. Identify the correct 3GPP **architecture / procedure** specification
           containing the signalling procedure for the given telecom feature.
        2. Determine the latest **Release 18** specification version where the
           procedure exists (for TS 38.401, a version such as v18.6.0 or later).
        3. Locate the exact section in the specification where the call-flow messages appear.

        If multiple releases exist (e.g. Rel‑17, Rel‑18, Rel‑19), you MUST prioritize
        Release 18. Within Release 18, pick the highest minor/patch version available
        (e.g. v18.7.0 over v18.6.0).

        Return STRICT JSON ONLY, in this format:
        {
          "spec_number": "TS xx.xxx",
          "spec_version": "vX.Y.Z",
          "section_id": "x.y.z",
          "spec_link": "https://www.etsi.org/..."
        }
    """

    agent = create_agent(
        model=llm,
        tools=[serperSearch],
        system_prompt=system_prompt,
    )
    return agent.invoke({"messages": [{"role": "user", "content": feature}]})


# ---------------------------------------------------------------------------
# Spec registry helpers (local copy)
# ---------------------------------------------------------------------------

VALID_SPECS: Dict[str, str] = {}
PRIMARY_PROTOCOL_SPEC: Dict[str, str] = {}
ARCHITECTURE_SPECS: List[str] = []


def load_spec_registry() -> None:
    """
    Load VALID_SPECS, PRIMARY_PROTOCOL_SPEC, ARCHITECTURE_SPECS from local JSON registry.
    """
    global VALID_SPECS, PRIMARY_PROTOCOL_SPEC, ARCHITECTURE_SPECS
    if not SPEC_REGISTRY_PATH.exists():
        VALID_SPECS, PRIMARY_PROTOCOL_SPEC, ARCHITECTURE_SPECS = {}, {}, []
        return
    with open(SPEC_REGISTRY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    VALID_SPECS = data.get("VALID_SPECS", {}) or {}
    PRIMARY_PROTOCOL_SPEC = data.get("PRIMARY_PROTOCOL_SPEC", {}) or {}
    ARCHITECTURE_SPECS = data.get("ARCHITECTURE_SPECS", []) or []


def get_valid_specs_registry(query: str = "") -> str:
    """
    Tool for the protocol-specs agent.
    Returns VALID_SPECS and ARCHITECTURE_SPECS as JSON string.
    """
    if not VALID_SPECS and not ARCHITECTURE_SPECS:
        load_spec_registry()
    payload = {"VALID_SPECS": VALID_SPECS, "ARCHITECTURE_SPECS": ARCHITECTURE_SPECS}
    return json.dumps(payload, indent=2)


def normalize_protocol_name(protocol: str) -> str:
    if not protocol:
        return ""
    p = str(protocol).strip()
    if not p:
        return ""
    p_upper = p.upper()
    p_upper = p_upper.replace(" ", "").replace("_", "").replace("/", "").replace("\\", "")
    aliases = {
        "L1": "PHY",
        "L2": "MAC",
        "L3": "RRC",
        "XN-AP": "XNAP",
        "XNAP": "XNAP",
    }
    return aliases.get(p_upper, p_upper)


def normalize_protocol_list(protocols: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for p in protocols or []:
        norm = normalize_protocol_name(p)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


CANONICAL_FEATURE_NAMES: List[str] = []


def load_feature_catalog() -> None:
    """
    Optional: load canonical feature labels from Feature_Validation/config/feature_catalog.json.

    Expected JSON format:
    { "CANONICAL_FEATURE_NAMES": ["Feature A", "Feature B", ...] }
    """
    global CANONICAL_FEATURE_NAMES
    if not FEATURE_CATALOG_PATH.exists():
        return
    try:
        with open(FEATURE_CATALOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        names = data.get("CANONICAL_FEATURE_NAMES", []) or []
        if isinstance(names, list):
            CANONICAL_FEATURE_NAMES = [str(x).strip() for x in names if str(x).strip()]
    except Exception:
        # Catalog is optional; ignore load errors to avoid breaking existing flows.
        return


def _normalize_feature_label(feature_name: str) -> str:
    """
    Map classifier output variants to canonical feature labels.
    """
    s = (feature_name or "").strip()
    if not s:
        return ""
    s_lc = s.lower()

    # Canonical mappings for known LTM/inter-DU handover intent variants.
    if (
        ("ltm" in s_lc and "handover" in s_lc)
        or ("inter-gnb-du" in s_lc and "handover" in s_lc)
        or ("inter gnb du" in s_lc and "handover" in s_lc)
    ):
        return "Inter-gNB-DU LTM handover procedure"

    # If output already matches canonical list (case-insensitive), preserve canonical casing.
    for c in CANONICAL_FEATURE_NAMES:
        if s_lc == c.lower():
            return c

    return s


def identify_feature_from_intent(user_intent: str) -> str:
    """
    Stage 0 classifier:
    Translate detailed conversational signalling intent into a concise
    high-level 3GPP feature/procedure name that downstream agents can use.
    """
    intent = (user_intent or "").strip()
    if not intent:
        raise ValueError("user_intent must be a non-empty string")

    if not CANONICAL_FEATURE_NAMES:
        load_feature_catalog()

    allowed_block = ""
    if CANONICAL_FEATURE_NAMES:
        allowed_features_json = json.dumps(CANONICAL_FEATURE_NAMES, ensure_ascii=True)
        allowed_block = f"""
Optional canonical feature labels (use if one clearly matches; do NOT force-fit):
{allowed_features_json}
"""

    system_prompt = f"""
You are a 3GPP systems architect.

Task:
Classify a user signalling-intent description into a high-level 5G feature/procedure.

You MUST return JSON ONLY with this exact schema:
{{
  "feature_name": "<string>",
  "confidence": <number between 0 and 1>,
  "is_generic": <true|false>,
  "evidence_terms": ["<short token>", "..."]
}}
{allowed_block}

Strict classification rules:
1) If an optional canonical label clearly matches, you may use it; otherwise output the best
   standardised 3GPP-style procedure name you can infer.
2) If the intent mentions mobility/handover context across source/candidate DU or includes
   combinations such as UE CONTEXT SETUP/MODIFICATION + RRCReconfiguration +
   DL/UL RRC MESSAGE TRANSFER + LTM semantics, classify as:
   "Inter-gNB-DU LTM handover procedure".
3) Do NOT output broad message-level labels (e.g. "UE Context Setup Procedure") when
   multi-step handover context is present.
4) If truly uncertain, set "is_generic": true and lower confidence (<0.75).
5) No text outside JSON.
"""

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": intent},
        ]
    )
    # print(system_prompt)

    raw = getattr(response, "content", response)
    if isinstance(raw, list):
        raw_text = "".join(str(part) for part in raw).strip()
    else:
        raw_text = str(raw).strip()

    feature_name = ""
    confidence = 0.0
    is_generic = True

    try:
        payload = _extract_json_from_text(raw_text)
        feature_name = str(payload.get("feature_name", "")).strip()
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        is_generic = bool(payload.get("is_generic", True))
    except Exception:
        # Keep backwards compatibility: if model ignored schema, treat as raw feature text.
        feature_name = raw_text

    # Defensive cleanup to keep downstream behaviour stable.
    feature_name = feature_name.strip().strip('"').strip("'")
    feature_name = re.sub(r"\s+", " ", feature_name).strip()
    feature_name = _normalize_feature_label(feature_name)

    # Deterministic intent-based guardrail for LTM/inter-DU handover patterns.
    intent_lc = intent.lower()
    # Normalize non-alphanumeric punctuation (handles cases like gNB‑CU with a non-standard dash).
    intent_norm = re.sub(r"[^a-z0-9]+", " ", intent_lc).strip()
    ue_ctx_mod_pair = (
        "ue context modification request" in intent_norm
        and "ue context modification response" in intent_norm
    )
    ltm_handover_signals = [
        "candidate gnb-du",
        "source gnb-du",
        "ue context setup request",
        "ue context setup response",
        "ue context modification request",
        "ue context modification response",
        "rrcreconfiguration",
        "rrcreconfigurationcomplete",
        "dl rrc message transfer",
        "ul rrc message transfer",
        "ltm",
    ]
    hits = sum(1 for k in ltm_handover_signals if k in intent_lc)
    # If the intent is explicitly a CU<->source gNB-DU context modification exchange
    # handled on gNB-CU (without necessarily using the word "LTM"), it still belongs
    # to the Inter-gNB-DU LTM call flow in 38.401.
    if ue_ctx_mod_pair and "source gnb du" in intent_norm and "gnb cu" in intent_norm:
        feature_name = "Inter-gNB-DU LTM handover procedure"
        confidence = max(confidence, 0.85)
        is_generic = False
    elif hits >= 2 and ("handover" in intent_lc or "ltm" in intent_lc or "candidate gnb-du" in intent_lc):
        feature_name = "Inter-gNB-DU LTM handover procedure"
        confidence = max(confidence, 0.9)
        is_generic = False

    # Fallback if classifier remains generic/low-confidence/empty.
    if not feature_name or is_generic or confidence < 0.75:
        feature_name = intent

    print(
        f"[IntentClassifier] Identified Feature: {feature_name} "
        f"(confidence={confidence:.2f}, is_generic={is_generic})"
    )
    return feature_name


def _extract_targets_from_intent(user_intent: str) -> List[Dict[str, str]]:
    """
    Extract the specific message names (and optional role hints) mentioned
    in the user intent.
    """
    text = (user_intent or "").strip()
    if not text:
        return []

    # Normalize for strict validation: message name extracted by the LLM must appear in the intent.
    intent_norm = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    system_prompt = """
You are an intent-to-primitive extractor for 3GPP signalling procedures.

Task:
From the given user intent, extract ONLY the message names that are explicitly listed in the intent.
Do not infer or add any other messages from surrounding procedure context.

Rules:
- A message name is "explicitly listed" if it is written as:
  - a quoted message name (inside double quotes), OR
  - a plain message name immediately followed by the word "message"
    (for example: UE CONTEXT SETUP RESPONSE message).
- Include each explicitly listed message name once.
- For each message, infer optional sender_role_hint and receiver_role_hint only from clear text like:
  - "gNB-CU sends ... to candidate gNB-DU"
  - "candidate gNB-DU responds with ... handled on gNB-CU"
  - "UE responds with ... to source gNB-DU"
  If you cannot infer, return empty strings for the hints.
- Sender/receiver role hints MUST be drawn from:
  "gNB-CU", "candidate gNB-DU", "source gNB-DU", "UE".

Output JSON ONLY with this schema:
{
  "targets": [
    {
      "name": "<message name, e.g. UE CONTEXT SETUP REQUEST>",
      "sender_role_hint": "<one of gNB-CU|candidate gNB-DU|source gNB-DU|UE or empty>",
      "receiver_role_hint": "<one of gNB-CU|candidate gNB-DU|source gNB-DU|UE or empty>"
    }
  ]
}
"""

    user_content = f"User intent:\n{text}\n"

    def _invoke_llm() -> Any:
        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        )
        return getattr(response, "content", response)

    # Hard timeout: extraction must not hang.
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(_invoke_llm)
    try:
        raw = fut.result(timeout=60)
    except concurrent.futures.TimeoutError:
        ex.shutdown(wait=False, cancel_futures=True)
        print("[IntentScoped] Timeout while extracting message targets from intent.")
        return []
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    raw_text = str(raw).strip()
    payload = _extract_json_from_text(raw_text) if "{" in raw_text else {}
    targets_raw = payload.get("targets", []) if isinstance(payload, dict) else []
    if not isinstance(targets_raw, list):
        return []

    validated: List[Dict[str, str]] = []
    seen: set = set()
    for t in targets_raw:
        if not isinstance(t, dict):
            continue
        name = str(t.get("name", "")).strip()
        if not name:
            continue
        name_norm = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()

        # Strict validation: extracted message name must literally appear in the intent.
        if not name_norm or name_norm not in intent_norm:
            continue

        sender_hint = str(t.get("sender_role_hint", "")).strip()
        receiver_hint = str(t.get("receiver_role_hint", "")).strip()
        key = f"{name_norm}|{sender_hint}|{receiver_hint}"
        if key in seen:
            continue
        seen.add(key)
        validated.append(
            {
                "name": name,
                **({"sender_role_hint": sender_hint} if sender_hint else {}),
                **({"receiver_role_hint": receiver_hint} if receiver_hint else {}),
            }
        )

    return validated


def _message_name_norm(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip()).lower()


def _message_key(m: Dict[str, Any]) -> str:
    return f"{m.get('index')}|{m.get('name')}|{m.get('direction')}|{m.get('protocol_layer')}"


def _filter_message_details_by_intent(message_details: Dict[str, Any], targets: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Keep only messages from message_details that match the intent targets
    by message name, and optionally by sender/receiver role hints.
    """
    messages = (message_details or {}).get("messages") or []
    if not messages or not targets:
        return {"messages": [], "feature_protocols": []}

    kept_keys: set = set()
    kept_messages: List[Dict[str, Any]] = []

    for t in targets:
        t_name = _message_name_norm(t.get("name", ""))
        sender_hint = (t.get("sender_role_hint") or "").strip().lower()
        receiver_hint = (t.get("receiver_role_hint") or "").strip().lower()

        best = None
        best_score = -1

        for m in messages:
            if _message_name_norm(m.get("name", "")) != t_name:
                continue

            direction_lc = (m.get("direction") or "").lower()
            score = 0
            if sender_hint:
                if sender_hint in direction_lc:
                    score += 2
            if receiver_hint:
                if receiver_hint in direction_lc:
                    score += 2

            # If we have no hints, keep the first occurrence (stable behaviour).
            if not sender_hint and not receiver_hint:
                score = 0.5

            if score > best_score:
                best_score = score
                best = m

        if best is not None:
            k = _message_key(best)
            if k not in kept_keys:
                kept_keys.add(k)
                kept_messages.append(best)

    # Preserve original ordering from message_details.
    kept_set = {(_message_key(m)) for m in kept_messages}
    ordered = [m for m in messages if _message_key(m) in kept_set]

    # Recompute feature_protocols from the filtered list.
    protocols: List[str] = []
    seen = set()
    for m in ordered:
        p = (m.get("protocol_layer") or "").strip()
        p_norm = normalize_protocol_name(p)
        if p_norm and p_norm not in seen:
            seen.add(p_norm)
            protocols.append(p_norm)

    return {"messages": ordered, "feature_protocols": protocols}


def _trim_section_text_by_step_numbers(section_text: str, step_numbers: List[int]) -> str:
    """
    Trim TS call-flow sections down to only the numbered steps that contain
    the messages we kept.
    """
    if not section_text:
        return ""
    step_set = {int(s) for s in step_numbers if isinstance(s, int) or str(s).isdigit()}
    if not step_set:
        return section_text

    starts: List[tuple[int, int]] = []
    # Steps usually start at line beginning with: "<n>." (e.g. "3. ...").
    for match in re.finditer(r"(?m)^\s*(\d+)\.\s+", section_text):
        n = int(match.group(1))
        starts.append((match.start(), n))

    if not starts:
        return section_text

    starts_sorted = sorted(starts, key=lambda x: x[0])
    # Map number -> ranges; pick the first occurrence range for each step.
    chosen_chunks: List[str] = []
    for i, (pos, n) in enumerate(starts_sorted):
        if n not in step_set:
            continue
        end_pos = starts_sorted[i + 1][0] if i + 1 < len(starts_sorted) else len(section_text)
        chunk = section_text[pos:end_pos].strip()
        if chunk:
            chosen_chunks.append(chunk)

    return "\n\n".join(chosen_chunks) if chosen_chunks else section_text


def _choose_template_info(feature_protocols: List[str], protocol_specs: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Select a template JSON based on protocol priority and hardcoded mapping.
    """
    protocol_to_template = {
        "NGAP": "Template_NR_NGAP.json",
        "NAS": "Template_NR_NAS.json",
        "RRC": "Template_NR_RRC.json",
        "MAC": "Template_NR_MAC.json",
        "PHY": "Template_NR_PHY.json",
        "L1": "Template_NR_PHY.json",
        "F1AP": "Template_common.json",
        "XNAP": "Template_common.json",
        "E1AP": "Template_common.json",
        "RLC": "Template_common.json",
        "PDCP": "Template_common.json",
    }
    protocol_priority = [
        "NGAP",
        "NAS",
        "RRC",
        "MAC",
        "PHY",
        "L1",
        "F1AP",
        "XNAP",
        "E1AP",
        "RLC",
        "PDCP",
    ]
    default_template = "Template_common.json"

    protocols_norm = normalize_protocol_list(feature_protocols or [])
    if not protocols_norm:
        protocols_norm = normalize_protocol_list(
            [
                str(ps.get("protocol", "")).strip()
                for ps in (protocol_specs or [])
                if isinstance(ps, dict)
            ]
        )

    selected_protocol = ""
    for p in protocol_priority:
        if p in protocols_norm:
            selected_protocol = p
            break
    if not selected_protocol and protocols_norm:
        selected_protocol = protocols_norm[0]

    template_name = protocol_to_template.get(selected_protocol, default_template)
    template_path = TEMPLATES_DIR / template_name

    if not template_path.exists():
        template_name = default_template
        template_path = TEMPLATES_DIR / default_template

    return {
        "protocol": selected_protocol,
        "template_name": template_name,
        "template_path": str(template_path),
    }


def run_with_intent(user_intent: str) -> Dict[str, Any]:
    """
    Stage 0 + intent-scoped two-stage pipeline:
    1) Identify a high-level feature name (for locating the correct architecture section).
    2) From the extracted call-flow, keep ONLY the messages explicitly mentioned in the intent.
    3) Run protocol-spec + message-section agents using only that filtered subset.
    """
    feature_name = identify_feature_from_intent(user_intent)

    targets = _extract_targets_from_intent(user_intent)

    # First attempt: use feature + context for better section disambiguation.
    arch = None
    try:
        arch = run_architecture_agent(f"{feature_name}\n\nContext:\n{user_intent}")
    except ValueError as e:
        # Agent 1 can sometimes output a section_id that does not exist in the
        # specific downloaded PDF when given free-form context.
        print(f"[IntentScoped] ArchitectureAgent failed with context, retrying with feature only. Error: {e}")
        try:
            arch = run_architecture_agent(feature_name)
        except ValueError as e2:
            print(f"[IntentScoped] ArchitectureAgent failed with feature-only as well. Error: {e2}")
            arch = {
                "procedure_spec_info": {},
                "section_text": "",
                "message_details": {"messages": [], "feature_protocols": []},
            }

    message_details_full = arch.get("message_details", {})
    if not targets:
        # If extraction failed, avoid returning an empty result set.
        filtered_message_details = message_details_full
        trimmed_section_text = arch.get("section_text", "")
    else:
        filtered_message_details = _filter_message_details_by_intent(message_details_full, targets)
    
        # If nothing matched in the chosen section, retry using the raw intent as anchor.
        if not filtered_message_details.get("messages"):
            # Do NOT re-run Agent 1 here: fallback section ids from free-form intent
            # can lead to missing/non-existent section_id extraction failures.
            # Instead, retry using name-only matching (ignore sender/receiver hints).
            targets_no_hints: List[Dict[str, str]] = [
                {"name": t.get("name", "")} for t in targets if isinstance(t, dict) and t.get("name")
            ]
            filtered_message_details = _filter_message_details_by_intent(message_details_full, targets_no_hints)

        # Recompute section_text to include only relevant steps.
        kept_step_numbers = [
            m.get("index")
            for m in filtered_message_details.get("messages", [])
            if isinstance(m.get("index"), int)
        ]
        if not kept_step_numbers:
            trimmed_section_text = ""
        else:
            trimmed_section_text = _trim_section_text_by_step_numbers(arch.get("section_text", ""), kept_step_numbers)

    procedure_spec_info = arch.get("procedure_spec_info", {})
    feature_protocols = filtered_message_details.get("feature_protocols", [])

    protocol_specs = find_protocol_specs(
        feature=feature_name,
        procedure_spec_info=procedure_spec_info,
        feature_protocols=feature_protocols,
    )
    message_sections = find_message_sections(
        feature=feature_name,
        protocol_specs=protocol_specs,
        message_details=filtered_message_details,
    )
    template_info = _choose_template_info(
        feature_protocols=filtered_message_details.get("feature_protocols", []),
        protocol_specs=protocol_specs,
    )

    final = {
        "procedure_spec_info": procedure_spec_info,
        "section_text": trimmed_section_text,
        "message_details": filtered_message_details,
        "protocol_specs": protocol_specs,
        "protocol_message_sections": message_sections,
        "template": template_info,
    }

    written = save_feature_run(feature_name, final)
    print(f"[IntentScopedTwoStage] Saved run output to: {written}")
    return final


# ---------------------------------------------------------------------------
# Run artifact storage
# ---------------------------------------------------------------------------

def _safe_filename(text: str, max_len: int = 120) -> str:
    """
    Convert an arbitrary feature string to a filesystem-safe filename stem.
    """
    s = (text or "").strip()
    s = re.sub(r"[^\w\s\-\.]", "", s)  # keep word chars, spaces, -, .
    s = re.sub(r"\s+", " ", s).strip().replace(" ", "_")
    s = s.strip("._-")
    if not s:
        s = "feature"
    return s[:max_len]


def save_feature_run(feature: str, result: Dict[str, Any]) -> str:
    """
    Save final result JSON to Feature_Validation/feature_runs with a timestamp.
    Returns the written file path as a string.
    """
    FEATURE_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = _safe_filename(feature)
    out_path = FEATURE_RUNS_DIR / f"{stem}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return str(out_path)


# ---------------------------------------------------------------------------
# Agent 1: architecture / procedure spec + call flow
# ---------------------------------------------------------------------------

def run_architecture_agent(feature: str) -> Dict[str, Any]:
    """
    Runs the first agent:
    - Resolve the main procedure / architecture spec for the feature.
    - Prefer the latest ETSI Rel-18 PDF for that spec.
    - Locate the exact section_id for the feature.
    - Extract section_text and message_details (including feature_protocols).
    """
    print(f"[ArchitectureAgent] Starting for feature: {feature}")
    os.chdir(BASE_DIR)
    SPECS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Use agent to get spec_number, section_id, and an initial spec_link.
    results = getSpecDetails(feature)
    print("[ArchitectureAgent] Raw getSpecDetails result received")
    procedure_spec_info = _parse_json_from_agent_result(results)

    spec_number = procedure_spec_info.get("spec_number")
    spec_link = procedure_spec_info.get("spec_link")
    print(f"[ArchitectureAgent] Parsed spec_number={spec_number}, spec_link={spec_link}")
    if not spec_number and not spec_link:
        raise ValueError("getSpecDetails did not return spec_number or spec_link")

    # 2) Prefer latest ETSI PDF by spec_number (Rel-18+) when available.
    spec_path = None
    if spec_number and SERPER_API_KEY:
        dl = download_etsi_latest_pdf(spec_number, SPECS_DIR, SERPER_API_KEY, preferred_release_major=18)
        if dl and dl.get("downloaded_pdf_path") and _looks_like_pdf(dl["downloaded_pdf_path"]):
            spec_path = dl["downloaded_pdf_path"]
            procedure_spec_info["spec_link"] = dl.get("spec_link", spec_link)
            # Derive spec_version from URL if possible.
            url = procedure_spec_info.get("spec_link", "")
            ver_match = re.search(r"(\d+)\.(\d+)\.(\d+)", url) or re.search(
                r"v(\d{2})(\d{2})(\d{2})", url, re.I
            )
            if ver_match:
                procedure_spec_info["spec_version"] = "v{}.{}.{}".format(
                    *map(int, ver_match.groups()[:3])
                )
            print(f"[ArchitectureAgent] Updated to latest ETSI link: {procedure_spec_info['spec_link']}")
            print(f"[ArchitectureAgent] Parsed spec_version={procedure_spec_info.get('spec_version')}")

    # 3) Fallback to agent-provided link if latest ETSI lookup failed.
    if spec_path is None and spec_link:
        spec_path = downloadSpec(spec_link)
        if not Path(spec_path).is_absolute():
            spec_path = str(BASE_DIR / spec_path)

    if spec_path is None:
        raise ValueError("Could not obtain procedure spec PDF (latest or agent link).")

    if not _looks_like_pdf(spec_path) and spec_number and SERPER_API_KEY:
        # Last attempt: re-download via ETSI helper.
        dl = download_etsi_latest_pdf(spec_number, SPECS_DIR, SERPER_API_KEY, preferred_release_major=18)
        if dl and dl.get("downloaded_pdf_path"):
            spec_path = dl["downloaded_pdf_path"]
            procedure_spec_info["spec_link"] = dl.get(
                "spec_link", procedure_spec_info.get("spec_link")
            )
            print(f"[ArchitectureAgent] Fallback ETSI download used. spec_path={spec_path}")

    # 4) Extract section text and call-flow details.
    section_id = (procedure_spec_info.get("section_id") or "").strip()
    print(f"[ArchitectureAgent] Using section_id={section_id} from procedure_spec_info")
    section_text = getSectionText(spec_path, section_id) if section_id else ""
    print(f"[ArchitectureAgent] Extracted section_text length={len(section_text)}")
    message_details = getMessageDetails(section_text) if section_text else {
        "messages": [],
        "feature_protocols": [],
    }

    # Normalize protocols for downstream usage.
    feature_protocols = normalize_protocol_list(
        message_details.get("feature_protocols", [])
    )
    message_details["feature_protocols"] = feature_protocols
    print(f"[ArchitectureAgent] Normalized feature_protocols={feature_protocols}")

    return {
        "procedure_spec_info": procedure_spec_info,
        "section_text": section_text,
        "message_details": message_details,
    }


# ---------------------------------------------------------------------------
# Agent 2: protocol-specific specs (per protocol, section_id, Rel-18 ETSI URL)
# ---------------------------------------------------------------------------

PROTOCOL_SPECS_SYSTEM_PROMPT = """
You are a senior 5G/4G telecom protocol expert.

Goal:
Given:
- A feature description (e.g. "Inter-gNB-DU LTM handover procedure"),
- The main procedure/architecture spec info (spec_number, spec_version, section_id),
- The list of protocol layers/interfaces involved in the call flow (e.g. ["RRC", "NGAP", "XnAP"]),

you MUST identify, for each protocol, the 3GPP specification that defines this feature's
procedure or signalling behaviour, and the exact section where the feature / handover is described.

You have access to tools:
- serperSearch(query): web search over the open Internet.
- get_valid_specs_registry(): returns the VALID_SPECS and ARCHITECTURE_SPECS registry as JSON.

============================================================
STRICT STEP-BY-STEP BEHAVIOUR
============================================================

For EACH protocol in the input list:

Step 1 – Understand the protocol role
- Interpret the protocol (e.g. "RRC", "NGAP", "XnAP", "F1AP") and its typical 3GPP spec family.
- Decide whether it is:
  - a radio protocol (e.g. RRC, MAC, PHY, PDCP),
  - a core/access interface protocol (e.g. NGAP, XnAP, F1AP, S1AP),
  - or something else.

Step 2 – Consult the spec registry
- Call get_valid_specs_registry() ONCE at the beginning to see:
  - VALID_SPECS: allowed spec_number -> title mapping
  - ARCHITECTURE_SPECS: list of specs that are architecture-oriented
- Use ONLY spec_numbers present in VALID_SPECS.
- Ignore any spec that is not in VALID_SPECS.

Step 3 – Plan 3–4 focused web queries PER protocol
- Before using serperSearch, mentally plan at least 3 distinct queries:
  1) "<feature> <protocol> 3GPP procedure"
  2) "<feature> <protocol> signalling 3GPP"
  3) "<feature> <protocol> section call flow"
  4) "<feature> <protocol> TS <candidate spec_number>" (if you already suspect a spec)
- Then actually USE serperSearch with several of these variations to gather evidence
  from 3gpp.org, etsi.org and other reliable sources.

Step 4 – Choose the correct 3GPP spec per protocol
- From search results AND the VALID_SPECS registry:
  - pick the spec_number that **defines the protocol for this feature**.
  - IMPORTANT: Do NOT return architecture-only specs here; pick the protocol-specific spec
    that contains message formats / procedures (e.g. NGAP spec, RRC spec).
- Reject:
  - specs outside VALID_SPECS,
  - specs that are only generic or unrelated to the feature.

Step 5 – Identify the exact section_id per spec
- Using search snippets and known 3GPP structure:
  - identify the section id (e.g. "8.2.1.5") where the procedure or message sequence
    for this feature is described for this protocol:
    - UE-side behaviour for RRC,
    - NG interface signalling for NGAP,
    - Xn interface signalling for XnAP, etc.
- Prefer the MOST SPECIFIC subsection that directly contains the call flow or detailed
  procedure, NOT only a broad parent section.
- If you cannot find an exact subsection, return the closest parent section that clearly
  and explicitly describes this feature for that protocol.

Step 6 – Output format (STRICT JSON ONLY)

Return JSON ONLY, no explanations before or after.

Output schema:
{
  "protocol_specs": [
    {
      "protocol": "<protocol name from input>",
      "spec_number": "TS xx.xxx",
      "section_id": "x.y.z",
      "reason": "<short reason why this spec+section is selected>"
    }
  ]
}

Rules:
- protocol MUST be copied from the input protocols list (case-insensitive normalization is ok).
- spec_number MUST exist in VALID_SPECS (from get_valid_specs_registry()).
- section_id MUST be a non-empty string when you return an entry.
- If you are not confident for a protocol, it is better to SKIP that protocol entirely
  (do not add an entry) rather than guess randomly.
- Do NOT include any commentary outside the JSON.
"""


def find_protocol_specs(
    feature: str,
    procedure_spec_info: Dict[str, Any],
    feature_protocols: List[str],
) -> List[Dict[str, Any]]:
    """
    Agent 2:
    Given the feature, procedure_spec_info, and normalized feature_protocols from Agent 1,
    use an agent (with serperSearch + get_valid_specs_registry) to identify, for each protocol,
    the relevant protocol spec and section_id. Then enrich with latest ETSI Rel-18 URL and
    local PDF path using find_latest_etsi_pdf_url / download_etsi_latest_pdf.
    """
    protocols_norm = normalize_protocol_list(feature_protocols or [])
    if not protocols_norm:
        return []
    print(f"[ProtocolSpecsAgent] Starting for protocols={protocols_norm}")

    # Prepare user content describing the context for the agent.
    context = {
        "feature": feature,
        "procedure_spec_info": {
            "spec_number": procedure_spec_info.get("spec_number"),
            "spec_version": procedure_spec_info.get("spec_version"),
            "section_id": procedure_spec_info.get("section_id"),
            "spec_link": procedure_spec_info.get("spec_link"),
        },
        "protocols_in_call_flow": protocols_norm,
    }
    user_content = json.dumps(context, indent=2)

    agent = create_agent(
        model=llm,
        tools=[serperSearch, get_valid_specs_registry],
        system_prompt=PROTOCOL_SPECS_SYSTEM_PROMPT,
    )
    # Avoid hanging indefinitely on slow/unresponsive network/LLM.
    def _invoke():
        return agent.invoke({"messages": [{"role": "user", "content": user_content}]})

    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(_invoke)
    try:
        agent_result = fut.result(timeout=180)
    except concurrent.futures.TimeoutError:
        print("[ProtocolSpecsAgent] Timeout while invoking agent.")
        ex.shutdown(wait=False, cancel_futures=True)
        return []
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    print("[ProtocolSpecsAgent] Agent invocation completed")

    data = _parse_json_from_agent_result(agent_result)
    protocol_specs_raw = data.get("protocol_specs", [])
    if not isinstance(protocol_specs_raw, list):
        return []
    print(f"[ProtocolSpecsAgent] protocol_specs_raw count={len(protocol_specs_raw)}")

    SPECS_DIR.mkdir(parents=True, exist_ok=True)

    enriched: List[Dict[str, Any]] = []
    for item in protocol_specs_raw:
        if not isinstance(item, dict):
            continue
        protocol = (item.get("protocol") or "").strip()
        spec_number = (item.get("spec_number") or "").strip()
        section_id = (item.get("section_id") or "").strip()
        reason = (item.get("reason") or "").strip()

        if not protocol or not spec_number or not section_id:
            continue
        if spec_number not in VALID_SPECS:
            continue

        # Resolve latest ETSI URL and download.
        spec_link = ""
        spec_version = ""
        downloaded_pdf_path = ""
        doc_id = ""

        if SERPER_API_KEY:
            found = find_latest_etsi_pdf_url(spec_number, SERPER_API_KEY, preferred_release_major=18)
            if found and found.get("url"):
                spec_link = found["url"]
                doc_id = Path(spec_link).stem
                dl = download_etsi_latest_pdf(spec_number, SPECS_DIR, SERPER_API_KEY, preferred_release_major=18)
                if dl and dl.get("downloaded_pdf_path") and _looks_like_pdf(
                    dl["downloaded_pdf_path"]
                ):
                    downloaded_pdf_path = dl["downloaded_pdf_path"]
                    spec_link = dl.get("spec_link", spec_link)

                # Try to parse version from URL.
                url = spec_link or found.get("url", "")
                ver_match = re.search(r"(\d+)\.(\d+)\.(\d+)", url) or re.search(
                    r"v(\d{2})(\d{2})(\d{2})", url, re.I
                )
                if ver_match:
                    spec_version = "v{}.{}.{}".format(
                        *map(int, ver_match.groups()[:3])
                    )

        print(
            f"[ProtocolSpecsAgent] Enriched spec for protocol={protocol}, "
            f"spec_number={spec_number}, section_id={section_id}, spec_version={spec_version}"
        )

        enriched.append(
            {
                "protocol": protocol,
                "spec_number": spec_number,
                "spec_title": VALID_SPECS.get(spec_number, ""),
                "section_id": section_id,
                "spec_version": spec_version,
                "spec_link": spec_link,
                "downloaded_pdf_path": downloaded_pdf_path,
                "doc_id": doc_id,
                "reason": reason,
            }
        )

    return enriched


# ---------------------------------------------------------------------------
# Agent 3: per-message sections inside protocol specs
# ---------------------------------------------------------------------------

MESSAGE_SECTIONS_SYSTEM_PROMPT = """
You are a senior 5G/4G telecom protocol expert.

Goal:
Given:
- A feature description (e.g. "Inter-gNB-DU LTM handover procedure"),
- The list of protocol specs already selected for this feature (spec_number, protocol),
- The ordered list of call-flow messages for this feature (name, direction, protocol_layer),

you MUST, for each protocol and each of its messages, identify 1–3 sections in the
corresponding 3GPP protocol spec where the message is defined or described in a way
that is RELEVANT for implementing this feature.

Strong Release 18 focus:
- Whenever possible, you MUST consider the Release 18 version of each protocol spec
  (e.g. v18.x.y) when reasoning about sections.
- If you see multiple releases, always align your section choices with Rel-18 content.

Section selection guidelines per message:
- For each (protocol, message_name) pair:
  - Look for:
    - the section where the message format / IE list is defined,
    - and/or the section where the message is used in the procedure for this feature,
    - and/or any behaviour/configuration section that explains how this message is
      processed for the feature.
  - Prefer the most specific subsections that directly mention the message name or
    clearly describe its role in the feature.
  - Limit to a SMALL set (1–3 sections per message) and avoid generic parent sections
    unless necessary.

Mandatory output format (JSON ONLY):
{
  "protocol_message_sections": [
    {
      "protocol": "<protocol name, e.g. RRC, NGAP, F1AP>",
      "spec_number": "TS xx.xxx",
      "messages": [
        {
          "message_name": "<exact message name from call flow>",
          "sections": [
            {
              "section_id": "x.y.z",
              "role": "procedure | message_format | behaviour",
              "reason": "short explanation why this section matters for this message"
            }
          ]
        }
      ]
    }
  ]
}

Rules:
- protocol MUST be one of the input protocol names.
- spec_number MUST match the spec_number provided for that protocol.
- section_id MUST be non-empty if you include it.
- You may SKIP a message if you truly cannot find any relevant section, but this should
  be rare for standardised messages.
- Do NOT include any commentary outside the JSON.
"""


def find_message_sections(
    feature: str,
    protocol_specs: List[Dict[str, Any]],
    message_details: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Agent 3:
    For each protocol + its messages from the call-flow, find sections inside the
    corresponding protocol spec where those messages are defined / used / described,
    with a strong focus on Release 18 content.

    Returns a structured list grouped by protocol and spec_number.
    """
    if not protocol_specs or not message_details:
        return []

    messages = message_details.get("messages") or []
    if not messages:
        return []

    # Build a compact context for the agent.
    context = {
        "feature": feature,
        "protocol_specs": [
            {
                "protocol": ps.get("protocol"),
                "spec_number": ps.get("spec_number"),
            }
            for ps in protocol_specs
            if ps.get("protocol") and ps.get("spec_number")
        ],
        "messages": [
            {
                "index": m.get("index"),
                "name": m.get("name"),
                "direction": m.get("direction"),
                "protocol_layer": m.get("protocol_layer"),
            }
            for m in messages
            if isinstance(m, dict)
        ],
    }
    user_content = json.dumps(context, indent=2)

    agent = create_agent(
        model=llm,
        tools=[serperSearch, get_valid_specs_registry],
        system_prompt=MESSAGE_SECTIONS_SYSTEM_PROMPT,
    )
    # Avoid hanging indefinitely on slow/unresponsive network/LLM.
    def _invoke():
        return agent.invoke({"messages": [{"role": "user", "content": user_content}]})

    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(_invoke)
    try:
        agent_result = fut.result(timeout=180)
    except concurrent.futures.TimeoutError:
        print("[MessageSectionsAgent] Timeout while invoking agent.")
        ex.shutdown(wait=False, cancel_futures=True)
        return []
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    print("[MessageSectionsAgent] Agent invocation completed")

    data = _parse_json_from_agent_result(agent_result)
    raw = data.get("protocol_message_sections", [])
    if not isinstance(raw, list):
        return []

    # Basic sanity filtering: only keep entries matching known protocol+spec combos.
    valid_pairs = {
        (ps.get("protocol"), ps.get("spec_number"))
        for ps in protocol_specs
        if ps.get("protocol") and ps.get("spec_number")
    }

    filtered: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        protocol = (item.get("protocol") or "").strip()
        spec_number = (item.get("spec_number") or "").strip()
        if not protocol or not spec_number:
            continue
        if (protocol, spec_number) not in valid_pairs:
            continue

        msgs = item.get("messages") or []
        cleaned_msgs: List[Dict[str, Any]] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            name = (m.get("message_name") or "").strip()
            sections = m.get("sections") or []
            if not name or not isinstance(sections, list):
                continue
            cleaned_secs: List[Dict[str, Any]] = []
            for s in sections:
                if not isinstance(s, dict):
                    continue
                sid = (s.get("section_id") or "").strip()
                role = (s.get("role") or "").strip() or "procedure"
                reason = (s.get("reason") or "").strip()
                if not sid:
                    continue
                cleaned_secs.append(
                    {
                        "section_id": sid,
                        "role": role,
                        "reason": reason,
                    }
                )
            if cleaned_secs:
                cleaned_msgs.append(
                    {
                        "message_name": name,
                        "sections": cleaned_secs,
                    }
                )
        if cleaned_msgs:
            filtered.append(
                {
                    "protocol": protocol,
                    "spec_number": spec_number,
                    "messages": cleaned_msgs,
                }
            )

    print(f"[MessageSectionsAgent] Completed. Protocol entries={len(filtered)}")
    return filtered


def run_two_stage(feature: str) -> Dict[str, Any]:
    """
    Convenience orchestration:
    - Run architecture/procedure agent (Agent 1).
    - Then run protocol-specific specs agent (Agent 2).
    """
    print(f"[TwoStage] Running full two-stage flow for feature: {feature}")
    arch = run_architecture_agent(feature)
    procedure_spec_info = arch.get("procedure_spec_info", {})
    message_details = arch.get("message_details", {})
    feature_protocols = message_details.get("feature_protocols", [])
    protocol_specs = find_protocol_specs(
        feature=feature,
        procedure_spec_info=procedure_spec_info,
        feature_protocols=feature_protocols,
    )
    print(f"[TwoStage] Completed protocol specs resolution. Count={len(protocol_specs)}")

    message_sections = find_message_sections(
        feature=feature,
        protocol_specs=protocol_specs,
        message_details=message_details,
    )
    print(f"[TwoStage] Completed per-message section resolution. Entries={len(message_sections)}")
    template_info = _choose_template_info(
        feature_protocols=message_details.get("feature_protocols", []),
        protocol_specs=protocol_specs,
    )

    final = {
        "procedure_spec_info": procedure_spec_info,
        "section_text": arch.get("section_text", ""),
        "message_details": message_details,
        "protocol_specs": protocol_specs,
        "protocol_message_sections": message_sections,
        "template": template_info,
    }

    written = save_feature_run(feature, final)
    print(f"[TwoStage] Saved run output to: {written}")
    return final


if __name__ == "__main__":
    # Example manual test run; adjust feature as needed.
    # test_feature = "Inter-gNB-DU LTM handover procedure"
    # result = run_two_stage(test_feature)
    # Example intent-driven run:
    user_intent= 'gNB-CU has to prepare and send F1AP "UE CONTEXT SETUP REQUEST" message to the candidate gNB-DU and candidate gNB-DU has to respond with F1AP "UE CONTEXT SETUP RESPONSE" message and this message has to be handled on gNB-CU'

    # user_intent = 'gNB-CU has to prepare and send F1AP "UE CONTEXT MODIFICATION REQUEST" message to the source gNB-DU and Source gNB-DU responds with a UE CONTEXT MODIFICATION RESPONSE message and this message has to be handled on gNB-CU'

    # user_intent = 'gNB-CU prepares the "RRCReconfiguration" RRC message with the LTM Configuration IEs inline to the LTM configuration data received in previous F1AP messages and sends the RRCReconfiguration message buffer piggy bagged in F1AP "DL RRC MESSAGE TRANSFER" message to source gNB-DU and source gNB-DU forwards this F1AP message to UE. Then, UE responds with "RRCReconfigurationComplete" RRC message to source gNB-DU and it forwards the message to gNB-CU piggy bagged in "UL RRC MESSAGE TRANSFER" message'

    result = run_with_intent(user_intent)
    # Output is already saved into feature_runs/ with a timestamp.
