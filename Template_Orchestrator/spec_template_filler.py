"""
Specification Agentic Template Filler with Multi-Source Awareness
- LLM-driven template filling with intelligent source attribution
- Handles cross-source dependencies (e.g., IEs in one spec, call flow in another)
- Makes decisions about which data to use from which source
"""

import json
import os
import sys
import time
import copy
import re
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
logger = logging.getLogger(__name__)

# Google Gemini Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# logger.debug("GOOGLE_API_KEY configured: %s", "***" if GOOGLE_API_KEY else "not set")
if not GOOGLE_API_KEY:
    raise ValueError(
        "Google API key not found. Please set GOOGLE_API_KEY environment variable."
    )

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    api_key=GOOGLE_API_KEY,
    model="gemini-2.5-flash",
    temperature=0.4,
    top_p=0.4
)


class SpecTemplateFiller:
    """
    Specification Agentic template filler that intelligently fills templates using multi-source context
    """
    
    def __init__(self, template_file: str):
        """
        Initialize Specification Agentic Template Filler
        
        Args:
            template_file: Path to template JSON file
        """
        # print("Initializing Specification Agentic Template Filler...")
        # print("-" * 60)
        
        # Load template
        if os.path.exists(template_file):
            with open(template_file, 'r') as f:
                self.template = json.load(f)
        else:
            self.template = {}
            # logger.warning("Template file not found: %s", template_file)
        
        # print("Specification Agentic Template Filler initialized")
    
    def build_multi_source_context(self, chunks: List[Dict[str, Any]], content_percentage: float = 0.60) -> str:
        """
        Build context string from chunks with source attribution
        
        Args:
            chunks: List of chunks with knowledge_source metadata
            content_percentage: Percentage of original content to keep (0.60 = 60%, default: 0.60)
            
        Returns:
            Formatted context string organized by source
        """
        # Organize chunks by source
        chunks_by_source = {}
        for chunk in chunks:
            source_id = chunk.get("knowledge_source", chunk.get("source_id", "unknown"))
            if source_id not in chunks_by_source:
                chunks_by_source[source_id] = []
            chunks_by_source[source_id].append(chunk)
        
        context_parts = []
        
        for source_id, source_chunks in chunks_by_source.items():
            context_parts.append(f"{'='*60}")
            context_parts.append(f"KNOWLEDGE SOURCE: {source_id}")
            context_parts.append(f"{'='*60}")
            context_parts.append("")
            
            # Sort chunks by rank if available
            sorted_chunks = sorted(
                source_chunks, 
                key=lambda x: x.get("rank", x.get("llm_rerank_score", x.get("semantic_score", 999)))
            )
            
            for i, chunk in enumerate(sorted_chunks, 1):
                section_id = chunk.get("section_id", "Unknown")
                section_title = chunk.get("section_title", "")
                content = chunk.get("content", "")
                resolved = chunk.get("resolved_from_reference", False)
                ref_type = chunk.get("reference_type", "")
                
                # Truncate content to specified percentage of original length
                original_length = len(content)
                if content and original_length > 0:
                    max_length = max(1, int(original_length * content_percentage))
                    if len(content) > max_length:
                        content = content[:max_length] + f"\n[... content truncated, {original_length - max_length} characters removed ({int((1 - content_percentage) * 100)}% of original) ...]"
                
                # Add section header
                header = f"[{i}] Section {section_id}: {section_title}"
                if resolved:
                    header += f" [RESOLVED FROM {ref_type.upper()}]"
                context_parts.append(header)
                context_parts.append("-" * 60)
                
                # Add content
                if content:
                    context_parts.append(content)
                else:
                    context_parts.append("[No content available]")
                
                context_parts.append("")  # Empty line between sections
        
        return "\n".join(context_parts)
    
    def _analyze_template_structure(self, template: Dict[str, Any], prefix: str = "") -> List[str]:
        """
        Recursively analyze template structure to generate dynamic instructions
        
        Args:
            template: Template dictionary
            prefix: Prefix for nested fields
            
        Returns:
            List of instruction strings
        """
        instructions = []
        
        for key, value in template.items():
            field_path = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Nested object
                nested_instructions = self._analyze_template_structure(value, field_path)
                instructions.extend(nested_instructions)
            elif isinstance(value, list) and len(value) > 0:
                # Array - check if it contains objects
                if isinstance(value[0], dict):
                    # Array of objects
                    instructions.append(f"- For {field_path}: Extract as an array of objects. Each object should have: {', '.join(value[0].keys())}")
                    # Analyze the object structure
                    nested_instructions = self._analyze_template_structure(value[0], f"{field_path}[item]")
                    instructions.extend(nested_instructions)
                else:
                    # Array of primitives
                    instructions.append(f"- For {field_path}: Extract as an array/list from the context")
            elif isinstance(value, str):
                # String field
                if not value:  # Empty string means it needs to be filled
                    # Generate smart instruction based on field name
                    key_lower = key.lower()
                    if 'id' in key_lower:
                        instructions.append(f"- For {field_path}: Generate a unique identifier based on the feature/procedure name")
                    elif 'name' in key_lower:
                        instructions.append(f"- For {field_path}: Extract or infer the name from the query and context")
                    elif 'description' in key_lower or 'info' in key_lower:
                        instructions.append(f"- For {field_path}: Extract detailed description/information from the context")
                    elif 'release' in key_lower or 'version' in key_lower:
                        instructions.append(f"- For {field_path}: Extract version/release information")
                    elif 'reference' in key_lower or 'spec' in key_lower or 'section' in key_lower:
                        instructions.append(f"- For {field_path}: Extract 3GPP specification reference (spec number, title, section)")
                    elif 'path' in key_lower or 'file' in key_lower:
                        instructions.append(f"- For {field_path}: Extract file paths if mentioned in the context")
                    elif 'mapping' in key_lower:
                        instructions.append(f"- For {field_path}: Extract mapping information if available")
                    elif 'ie_definition' in key_lower or ('definition' in key_lower and 'ie' in key_lower):
                        instructions.append(f"- For {field_path}: Extract the COMPLETE ASN.1 structure/type definition, NOT just a description. Include the full ASN.1 syntax (e.g., 'INTEGER (0..4095)', 'OCTET STRING (SIZE(1..32))', 'CHOICE {{ ... }}', 'SEQUENCE {{ ... }}', etc.). If ASN.1 syntax is not available, extract the complete type/structure definition from the specification.")
                    else:
                        instructions.append(f"- For {field_path}: Extract relevant information from the context based on the field name")
        
        return instructions
    
    def _invoke_llm_with_retry(self, prompt: str, max_retries: int = 3, initial_delay: float = 5.0) -> Any:
        """
        Invoke LLM with retry logic and exponential backoff for quota errors
        
        Args:
            prompt: Prompt to send to LLM
            max_retries: Maximum number of retries
            initial_delay: Initial delay in seconds (will be doubled on each retry)
            
        Returns:
            LLM response
            
        Raises:
            Exception: If all retries are exhausted
        """
        from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
        
        delay = initial_delay
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return llm.invoke(prompt)
            except ChatGoogleGenerativeAIError as e:
                error_str = str(e).lower()
                # Check if it's a quota/resource exhausted error
                if 'resource_exhausted' in error_str or '429' in error_str or 'quota' in error_str:
                    last_error = e
                    if attempt < max_retries - 1:
                        # Extract retry delay from error message if available
                        import re
                        retry_match = re.search(r'retry in ([\d.]+)s', error_str)
                        if retry_match:
                            delay = float(retry_match.group(1))
                            # Add a small buffer
                            delay = max(delay, 5.0)
                        else:
                            # Use exponential backoff
                            delay = initial_delay * (2 ** attempt)
                        
                        # logger.warning("Quota limit reached (attempt %d/%d). Retrying in %.1fs...", attempt + 1, max_retries, delay)
                        time.sleep(delay)
                        continue
                    else:
                        # logger.error("Quota limit exceeded after %d attempts. Please try again later.", max_retries)
                        raise
                else:
                    # Not a quota error, re-raise immediately
                    raise
            except Exception as e:
                # For other errors, only retry once
                if attempt < max_retries - 1 and attempt == 0:
                    # logger.warning("Error on attempt %d/%d: %s. Retrying...", attempt + 1, max_retries, str(e)[:100])
                    time.sleep(2.0)
                    continue
                else:
                    raise
        
        # Should not reach here, but just in case
        if last_error:
            raise last_error
        raise Exception("Failed to invoke LLM after retries")
    
    def extract_information(self, 
                          query: str, 
                          chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract structured information from multi-source chunks using LLM
        
        Args:
            query: User query string
            chunks: List of retrieved chunks from multiple sources
            
        Returns:
            Extracted information as dictionary matching template structure
        """
        # print("Extracting information using LLM (Multi-Source Aware)...")
        # print("-" * 60)
        
        # Deduplicate chunks based on section_id + source_id before template filling
        # print("Deduplicating chunks (before template filling)...")
        chunks_before_dedup = len(chunks)
        chunks = self._deduplicate_chunks_for_template_filling(chunks)
        chunks_after_dedup = len(chunks)
        if chunks_before_dedup != chunks_after_dedup:
            print("Removed %d duplicate chunks (%d : %d)", chunks_before_dedup - chunks_after_dedup, chunks_before_dedup, chunks_after_dedup)
            # pass
        else:
            print("No duplicates found (%d chunks)", chunks_after_dedup)
            # pass
        
        # Reduce chunks to top 70% based on scores
        # print("Reducing chunks to top 70%%...")
        chunks_before_reduction = len(chunks)
        
        # Sort chunks by score (prefer rank, then llm_rerank_score, then semantic_score)
        def get_sort_score(chunk):
            # Prefer rank if available
            if 'rank' in chunk and chunk.get('rank', 0) > 0:
                return chunk.get('rank', 0)
            # Then LLM rerank score
            if 'llm_rerank_score' in chunk and chunk.get('llm_rerank_score', 0) > 0:
                return chunk.get('llm_rerank_score', 0)
            # Fallback to semantic score
            return chunk.get('semantic_score', 0)
        
        sorted_chunks = sorted(chunks, key=get_sort_score, reverse=True)
        
        # Keep top 70%
        num_to_keep = max(1, int(len(sorted_chunks) * 0.70))
        chunks = sorted_chunks[:num_to_keep]
        
        chunks_after_reduction = len(chunks)
        if chunks_before_reduction != chunks_after_reduction:
            # print("Reduced to %d chunks (%d : %d, top 70%%)", chunks_after_reduction, chunks_before_reduction, chunks_after_reduction)
            pass
        else:
            # print("Using all %d chunks", chunks_after_reduction)
            pass
        
        # Build multi-source context
        context = self.build_multi_source_context(chunks)
        context_size = len(context)
        print("Context built: %s characters from %d chunks", f"{context_size:,}", len(chunks))
        
        # Count chunks by source
        chunks_by_source = {}
        for chunk in chunks:
            source_id = chunk.get("knowledge_source", chunk.get("source_id", "unknown"))
            chunks_by_source[source_id] = chunks_by_source.get(source_id, 0) + 1
        
        # print("Chunks by source:")
        for source_id, count in chunks_by_source.items():
            # print("   - %s: %d chunks", source_id, count)
            pass
        
        # Build prompt with dynamic template analysis
        template_str = json.dumps(self.template, indent=2)
        
        # Generate dynamic instructions based on template structure
        dynamic_instructions = self._analyze_template_structure(self.template)
        instructions_text = "\n".join(dynamic_instructions) if dynamic_instructions else "Extract all fields from the context."
        
        prompt = f"""You are an expert system analyst specializing in extracting and structuring information from 3GPP telecommunications specification documents. Your task is to extract comprehensive, accurate, and well-structured information from the provided context to fill the template completely.

        TASK:
        Extract all relevant information from the provided 3GPP specification context and fill the template structure with the most accurate and complete data possible.

        USER QUERY:
        {query}

        **CRITICAL INSTRUCTION FOR FEATURE-SPECIFIC QUERIES:**
        **FIRST STEP: IDENTIFY THE FEATURE/PROCEDURE NAME FROM THE QUERY**
        - Extract the specific feature/procedure name or keywords from the user query
        - Examples:
        * Query: "Implement F1AP UE Context Setup Request Message Procedure for Inter-gnB-DU LTM Handover"
            : Feature keywords: "LTM", "Inter-gnB-DU LTM Handover", "LTM Handover"
            : Extract ONLY IEs that contain "LTM" or are related to LTM (e.g., LTMInformation, LTMConfigurationIDMappingList, etc.)
            : DO NOT extract generic IEs like gNB-CU UE F1AP ID, gNB-DU UE F1AP ID, etc. that are not LTM-specific
        
        * Query: "Implement F1AP UE Context Setup Request for Conditional Handover"
            : Feature keywords: "Conditional Handover", "CHO"
            : Extract ONLY IEs related to Conditional Handover
            : DO NOT extract IEs related to other handover types or generic IEs

        - **MANDATORY RULE: Extract ONLY child IEs that match the feature keywords/procedure name**
        - For Information_Elements: Extract ONLY the feature-specific main IEs and their sub-structures that match the identified feature
        - DO NOT include generic message IEs that are not feature-specific
        - **CRITICAL FILTERING**: When examining the MAIN IE definition, only extract child IEs whose names contain the feature keywords or are explicitly mentioned in feature-specific sections
        - **If an IE does NOT contain the feature keywords AND is NOT mentioned in feature-specific procedure sections AND is NOT a sub-structure of a feature-specific main IE : DO NOT include it**

        RETRIEVED CONTEXT FROM 3GPP SPECIFICATION DOCUMENTS (Agentically Discovered):
        {context}

        TEMPLATE STRUCTURE:
        {template_str}

        CRITICAL - TWO-STEP FILTERING FOR INFORMATION_ELEMENTS:
        The user query is: "{query}"

        TWO-STEP PROCESS FOR IE EXTRACTION:

        STEP 1: IDENTIFY THE MAIN MESSAGE IE DEFINITION
        1. Identify the main message from the query:
        - Extract the message name from the query (e.g., "UE CONTEXT SETUP REQUEST", "RRC CONNECTION REJECT", "INITIAL UE MESSAGE")
        - This is the PRIMARY message for the procedure/feature

        2. Find the MAIN MESSAGE IE DEFINITION in the context:
        - Search for the message IE definition pattern: "MESSAGENAMEIEs PROTOCOL-IES ::= {{ ... }}"
        - The pattern follows: [MessageName]IEs [Protocol]-PROTOCOL-IES ::= {{ ... }}
        - This is the MAIN IE definition that contains all the IEs for the message
        - Example: "UEContextSetupRequestIEs F1AP-PROTOCOL-IES ::= {{ ID id-IE1 ... | ID id-IE2 ... | ... }}"

        3. Understand the structure:
        - The MAIN IE is the message IE definition itself (e.g., "UEContextSetupRequestIEs")
        - Within this MAIN IE definition, there are multiple child IEs listed
        - For feature-specific queries, identify which child IEs within this MAIN IE are feature-related
        - These feature-related child IEs will be extracted in Step 2 Phase 2A

        STEP 2: FILTER FOR FEATURE-SPECIFIC IEs (Hierarchical Extraction: Feature-Specific Child IEs First, Then Their Sub-IEs)
        1. **IDENTIFY THE FEATURE/PROCEDURE NAME FROM THE QUERY** (CRITICAL FIRST STEP):
        - Extract the specific feature/procedure name or keywords from the user query "{query}"
        - Identify feature keywords (e.g., "LTM", "Inter-gnB-DU LTM Handover", "Conditional Handover", "CHO", "DAPS", etc.)
        - Extract acronyms and abbreviations related to the feature
        - Example: Query "Implement F1AP UE Context Setup Request Message Procedure for Inter-gnB-DU LTM Handover"
            * Feature keywords identified: "LTM", "Inter-gnB-DU LTM Handover", "LTM Handover"
            * You MUST only extract IEs that contain "LTM" in their name or are explicitly related to LTM procedures
            * DO NOT extract generic IEs like "gNB-CU UE F1AP ID", "gNB-DU UE F1AP ID", "UEAggregateMaximumBitRate", etc.

        2. **PHASE 2A: Extract Feature-Specific Child IEs from the Main Message IE Definition**:
        - **UNDERSTAND THE HIERARCHY**:
            * The MAIN IE is the message IE definition (e.g., "UEContextSetupRequestIEs")
            * Within this MAIN IE definition, there are multiple child IEs listed
            * For feature-specific queries, extract ONLY the child IEs from within the MAIN IE that match the feature keywords
        
        - **CRITICAL FILTERING RULES**:
            * From the MAIN IE definition found in Step 1, examine each child IE listed
            * For each child IE, check if its name contains the feature keywords identified in Step 2.1
            * Example: If feature is "LTM", only extract child IEs like:
            - LTMInformation (contains "LTM")
            - LTMConfigurationIDMappingList (contains "LTM")
            - Any other IEs with "LTM" in the name
            * DO NOT extract child IEs like:
            - gNB-CU UE F1AP ID (does NOT contain "LTM")
            - gNB-DU UE F1AP ID (does NOT contain "LTM")
            - UEAggregateMaximumBitRate (does NOT contain "LTM")
            - Any generic IE not containing the feature keyword
        
        - **MATCHING CRITERIA**: A child IE is feature-specific if:
            * Its name contains the feature keyword(s) (e.g., "LTMInformation" contains "LTM")
            * OR it is explicitly mentioned in feature-specific procedure sections in the context
            * OR it is conditionally used only when the feature is involved
        
        - Extract ONLY these feature-specific child IEs FIRST with their ASN.1 structure
        - These are the child IEs within the MAIN IE definition that match the feature keywords
        
        - **STRICT EXCLUSION for feature-specific queries**:
            * **MANDATORY**: For queries with specific features, extract ONLY child IEs whose names contain the feature keywords
            * **DO NOT include generic message IEs** that do not contain the feature keywords
            * EXCLUDE child IEs that are:
            - Generic to the message but do NOT contain the feature keyword in their name
            - Used in other scenarios but not in this specific feature
            - Not mentioned in feature-related procedure sections
            - Not containing any of the identified feature keywords
            * **ONLY include**: Child IEs from the MAIN IE definition that contain the feature keywords AND their sub-structures

        3. **PHASE 2B: Recursively Extract Sub-Structures and Sub-IEs WITH ASN.1 DEFINITIONS** (After Feature-Specific Child IEs):
        - **CRITICAL: DO NOT WRITE DESCRIPTIONS - ONLY EXTRACT ASN.1 DEFINITIONS/STRUCTURES**
        - For EACH feature-specific child IE extracted in Phase 2A, recursively extract its sub-structures:
        
        - **RECURSIVE EXTRACTION PROCESS WITH COMPLETE DEFINITIONS**:
            * Start with feature-specific child IEs (already extracted in Phase 2A)
            * For each feature-specific child IE, examine its ASN.1 structure (SEQUENCE/CHOICE)
            * For each field in the structure that references a type (e.g., "grandParentField GrandParentIE1", "parentField ParentIE1"):
            - Find the type definition in the context (e.g., "GrandParentIE1 ::= SEQUENCE {{ parentField ParentIE1, ... }}")
            - Extract the COMPLETE ASN.1 definition/structure for that type, NOT a description
            - Extract it as a separate IE entry with full ASN.1 syntax
            - For nested types (e.g., if GrandParentIE1 contains ParentIE1, and ParentIE1 contains childIE1):
                : Extract GrandParentIE1's ASN.1 definition
                : Then extract ParentIE1's ASN.1 definition (referenced within GrandParentIE1)
                : Then extract childIE1's ASN.1 definition (referenced within ParentIE1)
                : Continue recursively until reaching primitive types (INTEGER, OCTET STRING, ENUMERATED with simple values)
            * Continue recursively for ALL nested structures until reaching primitive types or non-feature-related structures
            * Stop recursion when reaching types that are not feature-specific
        
        - **EXAMPLE: RECURSIVE EXTRACTION WITH COMPLETE DEFINITIONS**:
            If you have: GrandParentIE1 {{ parentField ParentIE1 }}
            And: ParentIE1 {{ childField childIE1 }}
            And: childIE1 ::= INTEGER (0..255)
            
            You MUST extract ALL THREE definitions recursively:
            1. GrandParentIE1: Extract complete ASN.1 definition "GrandParentIE1 ::= SEQUENCE {{ parentField ParentIE1, ... }}"
            2. ParentIE1: Extract complete ASN.1 definition "ParentIE1 ::= SEQUENCE {{ childField childIE1, ... }}"
            3. childIE1: Extract complete ASN.1 definition "childIE1 ::= INTEGER (0..255)"
            
            **DO NOT** write descriptions like "ParentIE1 is a structure that contains childIE1"
            **DO** extract the actual ASN.1 syntax: "ParentIE1 ::= SEQUENCE {{ childField childIE1 }}"
        
        - **CRITICAL RULE**: For every IE type referenced in a structure, you MUST:
            1. Find its ASN.1 definition in the context
            2. Extract the COMPLETE ASN.1 syntax/structure, NOT a textual description
            3. If it references other types, recursively extract those types' ASN.1 definitions as well
            4. Continue until all nested types have their ASN.1 definitions extracted

        3. **ORDERING REQUIREMENT**:
        - List main IEs FIRST in the Information_Elements array
        - Then list sub-IEs in hierarchical order (direct sub-IEs of main IEs, then nested sub-IEs)
        - Maintain logical grouping: group sub-IEs under their parent main IE conceptually
        - For feature-specific queries: Extract feature-specific child IEs first, then their sub-IEs hierarchically

        4. Verification checklist for FEATURE-SPECIFIC CHILD IEs (ALL must be YES):
        ✓ Is this IE a child IE within the MAIN message IE definition?
        ✓ **CRITICAL**: Does the IE name contain the feature keywords identified from the query?
            * Example: For "LTM Handover" query, does the IE name contain "LTM"? (e.g., "LTMInformation" : YES, "gNB-CU UE F1AP ID" : NO)
        ✓ Is this child IE directly related to the feature/procedure mentioned in the query?
        ✓ Is this child IE mentioned in feature-specific procedure sections?
        ✓ Does this child IE contain feature-related keywords or names in its ASN.1 definition or context?
        ✓ **IF THE IE NAME DOES NOT CONTAIN THE FEATURE KEYWORD : EXCLUDE IT IMMEDIATELY**
        ✓ If any answer is NO : EXCLUDE the IE

        5. Verification checklist for SUB-IEs (ALL must be YES):
        ✓ Is this sub-IE referenced within a feature-specific child IE?
        ✓ Is this sub-IE's type definition feature-related (has feature name/keywords or is feature-specific)?
        ✓ Is this sub-IE part of the feature's structure hierarchy?
        ✓ If any answer is NO : EXCLUDE the sub-IE

        REMEMBER: 
        - **STEP 0: Extract MAIN IE DEFINITION FIRST**:
        * The MAIN IE definition (e.g., "UEContextSetupRequestIEs F1AP-PROTOCOL-IES") MUST be the FIRST entry in Information_Elements
        * Extract its complete ASN.1 definition from the context
        * This shows all child IEs that can be included in the message
        
        - Step 1: Identify the MAIN message IE definition from the query
        - Step 2 Phase 2A: Extract feature-specific CHILD IEs from within the MAIN IE definition SECOND
        * Extract only the child IEs that are related to the specific feature/procedure mentioned in the query
        * These are child IEs within the MAIN IE definition that are feature-specific
        - Step 2 Phase 2B: Then recursively extract SUB-IEs from those feature-specific child IEs THIRD
        - **CRITICAL ORDER**: MAIN IE Definition : Feature-specific child IEs : Sub-IEs hierarchically
        - Quality over quantity: Include only IEs that are BOTH in the message AND feature-specific
        - Maintain hierarchical structure: MAIN IE Definition : Feature-Specific Child IEs : Direct Sub-IEs : Nested Sub-IEs

        EXTRACTION STRATEGY:
        1. CAREFULLY READ AND ANALYZE:
        - Understand the user query and what information is being requested
        - Thoroughly analyze the retrieved context sections (including agentically discovered chunks)
        - Identify all relevant information that maps to the template fields
        - Cross-reference information across different sections when available

        2. FIELD-SPECIFIC EXTRACTION GUIDELINES:
        {instructions_text}

        3. COMPREHENSIVE EXTRACTION RULES:
        
        A. IDENTIFIERS AND NAMES:
        - For fields with "ID", "identifier", "Feature_ID": Generate a clear, descriptive, UPPERCASE identifier based on the feature/procedure name (e.g., "RRC_CONNECTION_REJECT_PROCEDURE")
        - For fields with "name", "Name", "Feature_Name": Extract the exact feature/procedure name from context, or derive from query if context lacks explicit name
        - Ensure identifiers are unique, consistent, and follow naming conventions

        B. DESCRIPTIONS AND TEXT FIELDS:
        - Extract complete, detailed descriptions - do not truncate important information
        - Preserve technical terminology and specifications exactly as stated
        - Include all relevant details: what, how, when, why, where applicable
        - For multi-part descriptions, use array format with each part as a separate element
        - If description spans multiple paragraphs in context, combine them meaningfully

        C. REFERENCES AND SPECIFICATIONS:
        - Extract complete 3GPP specification references: spec number, title, and all relevant section numbers
        - Include section numbers in comma-separated format when multiple sections are referenced
        - Format: "TS 38.331" for spec number, full title for spec title
        - Extract section numbers accurately (e.g., "5.3.3.1, 5.3.3.5, 5.3.13.1")
        
        D. ARRAYS AND LISTS:
        - Extract ALL items when information is available - be comprehensive, not selective
        - **EXCEPTION FOR Information_Elements**: Be HIGHLY SELECTIVE - only include IEs directly related to the query
        - Maintain logical ordering when sequence matters
        - For arrays of objects: extract all fields for each object completely
        - Include all relevant items even if similar - each may have unique value

        E. STRUCTURED DATA (OBJECTS):
        - Fill ALL nested fields in objects - leave no field empty unnecessarily
        - Maintain relationships between fields within the same object
        - Ensure consistency across related fields

        F. PROCEDURE STEPS AND IMPLEMENTATION DETAILS:
        - Extract ALL steps in the procedure/implementation
        - Preserve step numbering and hierarchy (e.g., 1, 2, 3, or a, b, c for sub-steps)
        - Include complete step descriptions with all technical details
        - Maintain the exact sequence and relationships between steps

        G. TECHNICAL ELEMENTS - INFORMATION ELEMENTS (IEs):
        - **CRITICAL: TWO-STEP FILTERING PROCESS** - Follow the two-step approach described above
        
        **STEP 1: Extract Main Message IEs**
        - Find the message definition section (e.g., "UE CONTEXT SETUP REQUEST ::= SEQUENCE {{ ... }}")
        - Extract ALL IEs listed in that message's SEQUENCE/CHOICE structure
        - This is your base set of IEs from the message
        
        **STEP 2: Filter for Feature-Specific IEs (Hierarchical: Main IEs First, Then Sub-IEs)**
        
        **PHASE 2A: Extract Main Feature-Specific IEs**
        - From Step 1 IEs, identify MAIN IEs that are feature-specific
        - Extract these MAIN IEs FIRST with their ASN.1 structure
        - Identify feature-specific IEs by checking if they contain feature keywords, are mentioned in feature sections, or are conditionally used for the feature
        - EXCLUDE main IEs that are generic to the message but not specific to the feature
        
        **PHASE 2B: Recursively Extract Sub-Structures WITH ASN.1 DEFINITIONS (After Main IEs)**
        - **CRITICAL: DO NOT WRITE DESCRIPTIONS - ONLY EXTRACT ASN.1 DEFINITIONS/STRUCTURES**
        - For EACH main IE from Phase 2A, extract its sub-structures recursively with COMPLETE ASN.1 DEFINITIONS
        - Process: Main IE : Direct Sub-IEs : Nested Sub-IEs : Deeper Nested Sub-IEs
        - For each field in a SEQUENCE/CHOICE that references a type:
            * If the type is feature-related, extract it as a separate IE entry
            * Extract the COMPLETE ASN.1 definition/structure for that type from the context
            * **DO NOT** write descriptions - extract actual ASN.1 syntax (e.g., "TypeName ::= SEQUENCE {{ field1 Type1, field2 Type2 }}")
            * If the extracted type references other types, recursively extract those types' ASN.1 definitions as well
            * Continue recursively for ALL nested structures until reaching primitive types
            * Stop at primitive types (INTEGER, OCTET STRING, ENUMERATED with simple values) or non-feature-related structures
        
        - **RECURSIVE EXAMPLE**: 
            If GrandParentIE1 {{ parentField ParentIE1 }} references ParentIE1 {{ childField childIE1 }} which references childIE1 ::= INTEGER (0..255):
            : Extract ALL THREE: GrandParentIE1's ASN.1, ParentIE1's ASN.1, and childIE1's ASN.1 definitions recursively
            : DO NOT describe them - extract the actual ASN.1 syntax for each
        
        **ORDERING:**
        - List main IEs FIRST in Information_Elements array
        - Then list sub-IEs hierarchically (grouped by their parent main IE)
        - Maintain logical structure: Main : Sub : Sub-Sub : etc.
        
        **Example extraction order:**
        - First: [FeatureMainIE1] (main IE)
        - Then: [SubIE1], [SubIE2], [SubIE3] (sub-IEs of FeatureMainIE1)
        - Then: [NestedSubIE1], [NestedSubIE2] (sub-IEs of SubIE2)
        - Continue recursively for all nested structures...
        
        **MANDATORY VERIFICATION PROCESS**: For each IE, you MUST verify:
            1. Is this IE a child IE within the MAIN message IE definition (e.g., within UEContextSetupRequestIEs) OR a sub-IE of a feature-specific child IE? (Step 1 check)
            2. Is this IE related to the specific feature in the query? (Step 2 check)
            3. If BOTH are YES : Include it (with ASN.1 structure)
            4. If either is NO : EXCLUDE it immediately
        
        **ONLY INCLUDE IEs that are**:
            * FEATURE-SPECIFIC CHILD IEs: Child IEs within the MAIN message IE definition (Step 1) AND directly related to the feature (Step 2 Phase 2A), OR
            * SUB-IEs: Referenced as sub-structures within feature-specific child IEs that are feature-related (Step 2 Phase 2B)
            * Mentioned in feature-specific procedure sections
            * Part of feature-specific nested structures
            * Types referenced by main IEs if those types are feature-specific
        
        **CRITICAL ORDERING REQUIREMENT**:
        - **STEP 0: Extract and list the MAIN IE DEFINITION FIRST**:
            * The MAIN IE definition (e.g., "UEContextSetupRequestIEs F1AP-PROTOCOL-IES") MUST be the FIRST entry in Information_Elements
            * Extract the complete ASN.1 definition of the MAIN IE (e.g., "UEContextSetupRequestIEs F1AP-PROTOCOL-IES ::= {{ ID id-LTMInformation-Setup ... | ID id-LTMConfigurationIDMappingList ... | ... }}")
            * This MAIN IE definition shows all child IEs that can be included in the message
        
        - **STEP 1: Extract and list FEATURE-SPECIFIC CHILD IEs SECOND**:
            * After the MAIN IE definition, list the feature-specific child IEs
            * List ONLY the feature-specific child IEs that are related to the query
            * DO NOT include generic child IEs that are not specific to the feature/procedure mentioned in the query
        
        - **STEP 2: Extract and list SUB-IEs THIRD**:
            * Then extract and list SUB-IEs after feature-specific child IEs, maintaining hierarchical order
            * Group sub-IEs logically: list sub-IEs of the first feature-specific child IE, then sub-IEs of the second feature-specific child IE, etc.
            * For nested structures: list direct sub-IEs before deeper nested sub-IEs
        
        - **Example CORRECT order**:
            1. [MessageName]IEs (MAIN IE definition - MUST be FIRST entry)
                - IE_Name: "[MessageName]IEs"
                - IE_Definition: "[MessageName]IEs PROTOCOL-IES ::= {{ ID id-FeatureIE1 ... | ID id-FeatureIE2 ... | ... }}"
            2. [FeatureIE1] (feature-specific child IE - SECOND)
            3. [FeatureIE2] (feature-specific child IE - THIRD)
            4. [FeatureIE3] (feature-specific child IE - FOURTH)
            5. [SubIE1] (sub-IE of FeatureIE1)
            6. [SubIE2] (sub-IE of FeatureIE1)
            7. [SubIE3] (sub-IE of FeatureIE2)
            8. [NestedSubIE1] (nested sub-IE of SubIE2)
            9. ... (continue for all nested structures)
        - **WRONG order (DO NOT DO THIS)**: Starting with generic message IEs that are not feature-specific - these should NOT be included for feature-specific queries
        
        **STRICTLY EXCLUDE IEs that are**:
            * NOT a child IE within the MAIN message IE definition AND NOT a sub-IE of a feature-specific child IE (fails Step 1)
            * In the message but NOT related to the feature (fails Step 2)
            * Generic to the message but not feature-specific
            * Only mentioned in other message definitions (RESPONSE, FAILURE, etc.)
            * Mentioned in unrelated sections or procedures
        
        **VERIFICATION CHECKLIST** (BOTH must be YES before including):
            ✓ STEP 1: Is this IE in the message definition section for the queried message?
            ✓ STEP 2: Is this IE directly related to the specific feature/scenario in the query?
            ✓ If either answer is NO : EXCLUDE the IE
        
        **When in doubt, EXCLUDE** - Better to have fewer, highly relevant IEs than many irrelevant ones
        - **CRITICAL FOR IE_Definition FIELD**: Extract the COMPLETE ASN.1 structure/type definition recursively, NOT descriptions
        - **MANDATORY RULE**: The IE_Definition field MUST contain the actual ASN.1 syntax/type definition, NEVER descriptive text
        - **DO NOT WRITE DESCRIPTIONS** - Only extract ASN.1 definitions/structures recursively
        - **RECURSIVE EXTRACTION REQUIRED**: 
            * When an IE references another type (e.g., GrandParentIE1 contains ParentIE1, ParentIE1 contains childIE1):
            * You MUST extract ALL definitions recursively:
            : Extract GrandParentIE1's complete ASN.1 definition
            : Extract ParentIE1's complete ASN.1 definition (referenced in GrandParentIE1)
            : Extract childIE1's complete ASN.1 definition (referenced in ParentIE1)
            : Continue recursively for ALL nested types until reaching primitive types
            * Each extracted IE MUST have its complete ASN.1 definition in the IE_Definition field
        
        - Include the full ASN.1 syntax such as:
            * Type definitions: "INTEGER (0..4095)", "OCTET STRING (SIZE(1..32))", "BIT STRING (SIZE(1..8))"
            * Complex types: "CHOICE {{ option1 Type1, option2 Type2 }}", "SEQUENCE {{ field1 Type1, field2 Type2 }}"
            * Enumerated types: "ENUMERATED {{ value1, value2, value3 }}"
            * Optional/Mandatory indicators: "OPTIONAL", "MANDATORY"
            * Complete definitions: "TypeName ::= SEQUENCE {{ field1 Type1, field2 Type2 }}"
            * References to other types: When a type references another type, extract both definitions recursively
        
        - If ASN.1 syntax is present in the context, extract it EXACTLY as written (preserve formatting, brackets, etc.)
        - If only a type name is given (e.g., "ParentIE1"), search the context for its ASN.1 definition and extract that
        - If that type references other types (e.g., "childIE1"), recursively search for and extract those ASN.1 definitions as well
        - Look for sections titled "Information Element Definitions", "ASN.1 definitions", or similar
        - **STRICTLY PROHIBITED**: 
            * **DO NOT** fill IE_Definition with descriptions like "Identifies the type..." or "The ParentIE1 is used to..."
            * **DO NOT** fill IE_Definition with explanatory text like "ParentIE1 contains childIE1"
            * **DO NOT** write about what the IE does - ONLY extract the ASN.1 syntax
        - **ONLY ALLOWED**: Extract actual ASN.1 definitions/structures recursively
        - If ASN.1 definition is not found in context, leave IE_Definition as empty string "" rather than filling with description
        - **SELECTIVITY CHECK**: Before including an IE, verify it is mentioned in the context as part of the specific procedure/message described in the user query
        - Quality over quantity: Include only the most relevant IEs (typically 10-30 IEs for a specific procedure, not all possible IEs)

        4. QUALITY STANDARDS:
        - ACCURACY: Extract information exactly as stated in context, do not paraphrase technical specifications
        - COMPLETENESS: Fill every field in the template - extract all available information
        - CONSISTENCY: Ensure related fields are consistent (e.g., Feature_ID and Feature_Name should align)
        - PRECISION: Use exact terminology, numbers, and references from the context
        - STRUCTURE: Maintain exact JSON structure and nesting as specified in template
        - RELEVANCE: Focus on information directly related to the user query

        5. HANDLING MISSING INFORMATION:
        - If a field cannot be filled from context: use empty string "" for strings, [] for arrays, {{}} for objects
        - Do NOT invent, infer, or guess information not present in the context
        - For optional fields: only fill if information is available, otherwise use appropriate empty value
        - Be conservative: better to leave empty than to provide incorrect information

        6. DATA TYPE REQUIREMENTS:
        - Strings: Always use string type, even for numbers if template specifies string
        - Arrays: Always use array format, even for single items (unless template specifies otherwise)
        - Objects: Always include all nested fields, even if some are empty
        - Booleans: Use true/false (not strings) if template specifies boolean
        - Numbers: Use appropriate numeric type if template specifies number

        CRITICAL REQUIREMENTS:
        - The output JSON MUST exactly match the template structure - same fields, same nesting, same data types
        - ALL fields from the template MUST be present in the output
        - NO additional fields beyond the template structure
        - NO missing fields from the template structure
        - Output MUST be valid, parseable JSON
        - Use proper JSON formatting (quotes, commas, brackets, braces)
        - Do NOT include markdown formatting, code blocks, or explanatory text
        - Return ONLY the JSON object

        FINAL VERIFICATION FOR Information_Elements:
        Before finalizing the output, perform this verification:

        STEP 1 VERIFICATION: Message Definition Check
        - For FEATURE-SPECIFIC CHILD IEs: Verify "Is this IE explicitly listed as a child IE within the MAIN message IE definition (e.g., within UEContextSetupRequestIEs)?"
        - For SUB-IEs: Verify "Is this sub-IE referenced within a feature-specific child IE?"
        - If NO : REMOVE that IE immediately

        STEP 2 VERIFICATION: Feature-Specific Check
        - **CRITICAL**: First, identify the feature keywords from the query "{query}" (e.g., "LTM", "CHO", "DAPS", etc.)
        - For FEATURE-SPECIFIC CHILD IEs: Verify "Does the IE name contain the feature keywords identified from the query?"
        * Example: For "LTM Handover" query, check if IE name contains "LTM"
        * If IE name does NOT contain the feature keyword : REMOVE IT IMMEDIATELY
        - For MAIN IEs: Verify "Is this IE directly related to the specific feature/scenario in the query?"
        - For SUB-IEs: Verify "Is this sub-IE's type definition feature-related (contains feature keywords or is referenced by a feature-specific child IE)?"
        - Check each IE - if it does NOT relate to the feature mentioned in the query AND is NOT a sub-IE of a feature-specific main IE, REMOVE it
        - **EXPLICIT EXCLUSION CHECK**: For feature-specific queries, REMOVE IEs that are:
        * Generic to the message but do NOT contain the feature keyword in their name
        * Not mentioned in feature-related procedure sections
        * Not containing feature-related keywords or names (especially in the IE name itself)
        * Example for "LTM Handover": REMOVE "gNB-CU UE F1AP ID", "gNB-DU UE F1AP ID", "UEAggregateMaximumBitRate" and other generic IEs that don't contain "LTM"
        - If the IE name does NOT match the feature keywords : REMOVE that IE

        STEP 3 VERIFICATION: Hierarchical Ordering Check
        - **CRITICAL**: Verify that the MAIN IE DEFINITION is listed FIRST in the Information_Elements array
        - Verify that feature-specific child IEs are listed SECOND (after MAIN IE definition)
        - Verify that ONLY feature-specific child IEs are listed second (not generic message IEs)
        - Verify that NO generic IEs are included before or mixed with the feature-specific IEs
        - Verify that SUB-IEs are listed THIRD (after feature-specific child IEs)
        - Verify that sub-IEs are grouped logically (sub-IEs of first feature-specific child IE, then sub-IEs of second feature-specific child IE, etc.)
        - Verify hierarchical order: direct sub-IEs before deeper nested sub-IEs
        - **Example CORRECT order**:
        1. [MessageName]IEs (MAIN IE definition - MUST be FIRST entry)
        2. [FeatureIE1] (feature-specific child IE - SECOND)
        3. [FeatureIE2] (feature-specific child IE - THIRD)
        4. [SubIE1] (sub-IE - AFTER feature-specific child IEs)
        5. [SubIE2] (sub-IE - AFTER feature-specific child IEs)
        6. [NestedSubIE1] (nested sub-IE - AFTER direct sub-IEs)
        7. ... (continue hierarchically)
        - Example WRONG order (DO NOT DO THIS):
        1. [GenericIE1] (WRONG - not a feature-specific IE)
        2. [GenericIE2] (WRONG - not a feature-specific IE)
        3. [FeatureIE1] (correct but in wrong position)

        FINAL REVIEW:
        1. Count the IEs in Information_Elements array
        2. Verify ordering: Main IEs first, then sub-IEs hierarchically
        3. If you have more than 20-30 IEs : Review and remove the least feature-specific ones
        4. Remember: It's better to have few highly relevant IEs than many irrelevant ones
        5. Each IE must pass BOTH Step 1 (message definition/sub-structure) AND Step 2 (feature-specific) checks
        6. Ordering must follow: Main IEs : Direct Sub-IEs : Nested Sub-IEs

        OUTPUT FORMAT:
        Return a single, valid JSON object that exactly matches the template structure. The JSON should be properly formatted and ready for parsing."""

        try:
            # print("Sending prompt to LLM...")
            start_time = time.time()
            response = self._invoke_llm_with_retry(prompt)
            elapsed = time.time() - start_time
            # print("LLM response received in %.2fs", elapsed)
            
            response_text = response.content.strip()
            
            # Clean JSON response
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            # print("Parsing JSON response...")
            extracted_info = json.loads(response_text)
            
            # print("Information extracted successfully")
            
            return extracted_info
            
        except json.JSONDecodeError as e:
            # logger.warning("JSON parsing error: %s", str(e))
            # logger.debug("Response preview: %s", response_text[:500] if response_text else "")
            raise
        except Exception as e:
            # logger.warning("Error during extraction: %s", str(e))
            raise
    
    def fill_template(self, 
                     extracted_info: Dict[str, Any], 
                     chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fill template with extracted information and add knowledge hints
        
        Args:
            extracted_info: Extracted information from LLM
            chunks: List of retrieved chunks
            
        Returns:
            Filled template dictionary
        """
        # print("Filling template...")
        # print("-" * 60)
        
        # Deep copy template
        filled_template = copy.deepcopy(self.template)
        
        # Fill template with extracted information
        for key, value in extracted_info.items():
            if key in filled_template:
                filled_template[key] = value
        
        # Add Knowledge_Hints with section IDs and sources from chunks
        if "Knowledge_Hints" in filled_template:
            knowledge_hints = []
            
            # Get unique section IDs with source attribution
            seen_sections = set()
            for chunk in chunks:
                section_id = chunk.get("section_id")
                section_title = chunk.get("section_title", "")
                source_id = chunk.get("knowledge_source", chunk.get("source_id", ""))
                
                if section_id and section_id not in seen_sections:
                    seen_sections.add(section_id)
                    if source_id:
                        hint = f"{section_id} ({source_id}): {section_title}"
                    else:
                        hint = f"{section_id}: {section_title}" if section_title else section_id
                    knowledge_hints.append(hint)
            
            filled_template["Knowledge_Hints"] = knowledge_hints
        
        # Ensure all required fields are present
        self._ensure_required_fields(filled_template, self.template)
        
        # print("Template filled successfully")
        
        return filled_template
    
    def _ensure_required_fields(self, template: Dict[str, Any], reference_template: Dict[str, Any] = None):
        """
        Recursively ensure all required fields have default values if missing
        """
        if reference_template is None:
            reference_template = self.template
        
        for key, ref_value in reference_template.items():
            if key not in template:
                # Field missing, add default based on type
                if isinstance(ref_value, dict):
                    template[key] = {}
                    self._ensure_required_fields(template[key], ref_value)
                elif isinstance(ref_value, list):
                    template[key] = []
                elif isinstance(ref_value, str):
                    template[key] = ""
                else:
                    template[key] = ref_value
            else:
                # Field exists, check nested structures
                if isinstance(ref_value, dict) and isinstance(template[key], dict):
                    self._ensure_required_fields(template[key], ref_value)
                elif isinstance(ref_value, list) and isinstance(template[key], list):
                    # Ensure list items match structure if list contains objects
                    if len(ref_value) > 0 and isinstance(ref_value[0], dict):
                        for item in template[key]:
                            if isinstance(item, dict):
                                self._ensure_required_fields(item, ref_value[0])
    
    
    
    
    def _deduplicate_chunks_for_template_filling(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate chunks based on section_id + source_id/knowledge_source combination
        
        Simple logic: Create a unique key from section_id + source_id, and keep only the first
        occurrence of each unique combination.
        
        Args:
            chunks: List of chunks to deduplicate
            
        Returns:
            List of deduplicated chunks
        """
        seen_keys = set()
        unique_chunks = []
        
        for chunk in chunks:
            section_id = chunk.get('section_id', '')
            source_id = chunk.get('knowledge_source', chunk.get('source_id', ''))
            
            # Create unique key from section_id + source_id
            unique_key = f"{section_id}|||{source_id}"
            
            # Only keep first occurrence
            if unique_key not in seen_keys:
                seen_keys.add(unique_key)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def save_output(self, 
                   filled_template: Dict[str, Any], 
                   query: str,
                   output_dir: str) -> str:
        """
        Save filled template to output file
        
        Args:
            filled_template: Filled template dictionary
            query: User query string
            output_dir: Output directory (default: backend/resources per user preference)
            
        Returns:
            Output file path
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize query for filename
        safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
        safe_query = safe_query.replace(' ', '_')
        filename = f"filled_template_{safe_query}_{timestamp}.json"
        output_path = os.path.join(output_dir, filename)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filled_template, f, indent=2, ensure_ascii=False)
        
        # print("Saved filled template to: %s", output_path)
        
        return output_path
