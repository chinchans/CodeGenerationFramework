"""
GraphRAG v1: Knowledge Graph + FAISS Vector DB for 3GPP Specifications
- Chunks are deepest-level subsections (e.g., 5.1.1.1.2)
- Knowledge Graph: Chunks as nodes, relationships as edges
- FAISS Vector DB: HNSW index for fast similarity search
- Azure OpenAI (gpt-4o-mini) for semantic relationship extraction
- Semantic search-based candidate selection for accurate relationship discovery
"""

import json
import re
import pickle
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
import networkx as nx
import faiss
from dotenv import load_dotenv

import logging

logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
# FAISS_INDEX_FILE = "./ts_138401v180600p/vector_db/faiss_index.index"
# FAISS_METADATA_FILE = "./ts_138401v180600p/vector_db/faiss_metadata.json"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o-mini")




doc_id = None

# Initialize Azure OpenAI LLM
if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
    raise ValueError(

        "Azure OpenAI credentials not found. Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables."
    )



llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    temperature=0.5,
    timeout=120,  # 2 minute timeout per request
    max_retries=2,  # Retry up to 2 times on failure
)

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"
)


class SectionNode:
    """Represents a section in the hierarchical structure"""
    def __init__(self, section_id: str, title: str, level: int, page_num: int = None):
        self.section_id = section_id
        self.title = title
        self.level = level
        self.page_num = page_num  # Page where section title appears
        self.page_numbers = set([page_num] if page_num else [])  # All pages where content exists
        self.content_lines = []
        self.children = []
        self.parent = None
        self.has_direct_text = False
        self.is_leaf = False

    def add_content_line(self, line: str, page_num: int = None):
        """Add a line of content to this section"""
        self.content_lines.append(line)
        if page_num:
            self.page_numbers.add(page_num)
        if line.strip() and not re.match(r'^\d+(\.\d+)+', line.strip()):
            self.has_direct_text = True

    def get_full_content(self) -> str:
        """Get full content including section number and title"""
        content = f"{self.section_id} {self.title}\n"
        content += "\n".join(self.content_lines)
        return content.strip()


class CreateSpecVectorKg():
    def parse_section_number(self,section_str: str) -> Tuple[Optional[List[int]], str]:
        """
        Parse section number string (e.g., "5.1.1.1.2") into list of integers
        Returns: (section_numbers, remaining_text)
        """
        match = re.match(r'^(\d+(?:\.\d+)*)', section_str.strip())
        if match:
            section_part = match.group(1)
            numbers = [int(x) for x in section_part.split('.')]
            remaining = section_str[match.end():].strip()
            return numbers, remaining
        return None, section_str


    def is_parent_of(self,parent: List[int], child: List[int]) -> bool:
        """Check if parent is a direct parent of child"""
        if len(parent) >= len(child):
            return False
        return child[:len(parent)] == parent and len(child) == len(parent) + 1


    def load_and_parse_pdf(self,pdf_path: str, doc_id1: str, skip_start: int = 10, skip_from: int = 156) -> Dict[str, SectionNode]:
        """
        Load PDF and parse into hierarchical section structure
        Returns: Dictionary of section_id -> SectionNode
        """

        global doc_id
        doc_id = doc_id1
        start_time = time.time()
        # print(f" Loading PDF from: {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        end_index = skip_from - 1
        filtered_pages = pages[skip_start:end_index] if end_index > skip_start else pages[skip_start:]

        # print(f"   Loaded {len(pages)} total pages")
        # print(f"   Skipping pages 1-{skip_start} and {skip_from} to end")
        # print(f"  Processing pages {skip_start+1} to {min(end_index, len(pages))} ({len(filtered_pages)} pages)")
        # print(f"  Parsing sections and building hierarchical tree...")

        # Enhanced regex to match deeper nesting (up to 6 levels: 5.1.1.1.1.1)
        section_regex = re.compile(r'^(\d+(?:\.\d+){0,5})\s+(.+)$')

        sections = {}  # section_id -> SectionNode
        current_section_stack = []  # Stack to track current section hierarchy
        sections_by_level = defaultdict(int)

        for page_idx, page in enumerate(filtered_pages):
            page_num = skip_start + page_idx + 1
            text = page.page_content
            lines = text.split('\n')

            if (page_idx + 1) % 100 == 0:
                pass  # print(f"      Processing page {page_num}... ({page_idx + 1}/{len(filtered_pages)})", end='\r')
                

            for line in lines:
                line_stripped = line.strip()

                # Check if this is a section header
                match = section_regex.match(line_stripped) if line_stripped else None

                if match:
                    section_id = match.group(1)
                    title = match.group(2).strip()

                    # Parse section number
                    section_nums, _ = self.parse_section_number(section_id)
                    if section_nums is None:
                        # Not a valid section, treat as content
                        if current_section_stack:
                            current_section_id = current_section_stack[-1][1]
                            sections[current_section_id].add_content_line(line, page_num)
                        continue

                    level = len(section_nums)
                    sections_by_level[level] += 1

                    # Create section node
                    section_node = SectionNode(section_id, title, level, page_num)
                    sections[section_id] = section_node

                    # Update stack - remove sections that are not parents of current
                    while current_section_stack:
                        parent_nums, parent_id = current_section_stack[-1]
                        if self.is_parent_of(parent_nums, section_nums):
                            # Current section is a child of the top of stack
                            parent_node = sections[parent_id]
                            parent_node.children.append(section_id)
                            section_node.parent = parent_id
                            break
                        else:
                            current_section_stack.pop()

                    # Add current section to stack
                    current_section_stack.append((section_nums, section_id))

                else:
                    # This line is content, add it to the current section
                    # Include empty lines to preserve content structure (they might be meaningful spacing)
                    if current_section_stack:
                        current_section_id = current_section_stack[-1][1]
                        sections[current_section_id].add_content_line(line, page_num)

        # Mark leaf nodes
        leaf_count = 0
        for section_id, node in sections.items():
            if not node.children:
                node.is_leaf = True
                leaf_count += 1

        elapsed_time = time.time() - start_time
        # print(f"    Parsing completed in {elapsed_time:.2f} seconds")
        # print(f"    Statistics:")
        # print(f"      - Total sections found: {len(sections)}")
        # print(f"      - Leaf sections (deepest level): {leaf_count}")
        # print(f"      - Parent sections: {len(sections) - leaf_count}")

        # Show section level distribution
        level_dist = {}
        for section_id, node in sections.items():
            level_dist[node.level] = level_dist.get(node.level, 0) + 1

        # print(f"    Section level distribution:")
        for level in sorted(level_dist.keys()):
            pass  # print(f"      - Level {level}: {level_dist[level]} sections")
            

        return sections


    def _get_parent_path(self,section_id: str, sections: Dict[str, SectionNode]) -> List[str]:
        """Get full parent path from root to section"""
        path = []
        current_id = section_id
        while current_id:
            path.insert(0, current_id)
            if current_id in sections and sections[current_id].parent:
                current_id = sections[current_id].parent
            else:
                break
        return path


    def extract_deepest_chunks(self,sections: Dict[str, SectionNode]) -> List[Dict[str, Any]]:
        """
        Extract chunks for deepest-level sections and parent sections with direct text
        Returns: List of chunk dictionaries
        """
        start_time = time.time()
        chunks = []

        # First, collect all leaf nodes (deepest level)
        leaf_sections = [sec_id for sec_id, node in sections.items() if node.is_leaf]

        # print(f"   Found {len(leaf_sections)} leaf sections (deepest level)")
        # print(f"   Processing leaf sections...")

        # Process leaf sections
        for section_id in leaf_sections:
            node = sections[section_id]
            chunk = {
                "section_id": section_id,
                "section_title": node.title,
                "content": node.get_full_content(),
                "metadata": {
                    "doc_id": doc_id,
                    "section_number": section_id,
                    "section_title": node.title,
                    "parent_section_id": node.parent,
                    "parent_path": self._get_parent_path(section_id, sections),
                    "level": node.level,
                    "child_section_ids": [],
                    "page_numbers": sorted(list(node.page_numbers)) if node.page_numbers else [],
                    "has_children": False,
                    "is_leaf": True,
                    "direct_text_only": False
                }
            }
            chunks.append(chunk)

        # Process parent sections that have direct text
        processed_parents = set()

        def process_parent(section_id: str):
            """Recursively process parent sections"""
            if section_id in processed_parents or section_id not in sections:
                return

            node = sections[section_id]

            # Only process if it has direct text and children
            if node.has_direct_text and node.children:
                # Extract only direct text (lines before first child section)
                direct_text_lines = []
                in_direct_text = True

                for line in node.content_lines:
                    # Check if this line starts a child section
                    is_child_section = False
                    for child_id in node.children:
                        child_node = sections[child_id]
                        if line.strip().startswith(child_node.section_id):
                            is_child_section = True
                            break

                    if is_child_section:
                        in_direct_text = False
                        break

                    if in_direct_text:
                        direct_text_lines.append(line)

                if direct_text_lines:
                    direct_content = f"{node.section_id} {node.title}\n"
                    direct_content += "\n".join(direct_text_lines)

                    chunk = {
                        "section_id": section_id,
                        "section_title": node.title,
                        "content": direct_content.strip(),
                        "metadata": {
                            "doc_id": doc_id,
                            "section_number": section_id,
                            "section_title": node.title,
                            "parent_section_id": node.parent,
                            "parent_path": self._get_parent_path(section_id, sections),
                            "level": node.level,
                            "child_section_ids": node.children,
                            "page_numbers": sorted(list(node.page_numbers)) if node.page_numbers else [],
                            "has_children": True,
                            "is_leaf": False,
                            "direct_text_only": True
                        }
                    }
                    chunks.append(chunk)
                    processed_parents.add(section_id)

            # Recursively process parent
            if node.parent:
                process_parent(node.parent)

        # Process all parents of leaf nodes
        for section_id in leaf_sections:
            node = sections[section_id]
            if node.parent:
                process_parent(node.parent)

        # print(f" Created {len(chunks)} chunks ({len(leaf_sections)} leaf + {len(chunks) - len(leaf_sections)} parent with direct text)")
        return chunks


    def extract_hierarchical_relationships(self,chunks: List[Dict[str, Any]], sections: Dict[str, SectionNode]) -> List[Tuple[str, str, str]]:
        """
        Extract hierarchical relationships (PARENT_OF, CHILD_OF)
        Returns: List of (source, relationship, target) tuples
        """
        start_time = time.time()
        # print(f"   Extracting parent-child relationships from hierarchical structure...")

        relationships = []
        parent_rels = 0
        child_rels = 0

        for chunk in chunks:
            section_id = chunk["section_id"]
            parent_id = chunk["metadata"]["parent_section_id"]
            child_ids = chunk["metadata"]["child_section_ids"]

            # Add parent relationship
            if parent_id:
                relationships.append((section_id, "CHILD_OF", parent_id))
                relationships.append((parent_id, "PARENT_OF", section_id))
                parent_rels += 1

            # Add child relationships
            for child_id in child_ids:
                relationships.append((section_id, "PARENT_OF", child_id))
                relationships.append((child_id, "CHILD_OF", section_id))
                child_rels += 1

        # Remove duplicates
        relationships = list(set(relationships))
        elapsed = time.time() - start_time
        # print(f"    Extracted {len(relationships)} hierarchical relationships in {elapsed:.3f}s")
        # print(f"      - Parent relationships: {parent_rels}")
        # print(f"      - Child relationships: {child_rels}")
        return relationships


    def extract_explicit_references(self, chunks: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
        """
        Extract explicit section references from chunk content
        Returns: List of (source, relationship, target) tuples
        """
        start_time = time.time()
        # print(f"    Scanning chunk content for explicit section references...")

        relationships = []

        # Pattern to match section references
        reference_patterns = [
            r'section\s+(\d+(?:\.\d+)+)',
            r'Section\s+(\d+(?:\.\d+)+)',
            r'SECTION\s+(\d+(?:\.\d+)+)',
            r'see\s+(\d+(?:\.\d+)+)',
            r'See\s+(\d+(?:\.\d+)+)',
            r'as\s+defined\s+in\s+(\d+(?:\.\d+)+)',
            r'defined\s+in\s+(\d+(?:\.\d+)+)',
            r'according\s+to\s+(\d+(?:\.\d+)+)',
            r'per\s+(\d+(?:\.\d+)+)',
            r'as\s+per+(\d+(?:\.\d+)+)'
        ]

        # Create set of valid section IDs for lookup
        valid_section_ids = {chunk["section_id"] for chunk in chunks}
        # print(f"      Valid section IDs: {len(valid_section_ids)}")

        chunks_with_refs = 0
        total_refs_found = 0

        for idx, chunk in enumerate(chunks):
            if (idx + 1) % 100 == 0:
                pass  # print(f"      Scanning chunk {idx + 1}/{len(chunks)}...", end='\r')
                

            source_id = chunk["section_id"]
            content = chunk["content"]
            chunk_refs = 0

            for pattern in reference_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    referenced_id = match.group(1)
                    # Check if referenced section exists in our chunks
                    if referenced_id in valid_section_ids and referenced_id != source_id:
                        relationships.append((source_id, "REFERENCES", referenced_id))
                        relationships.append((referenced_id, "REFERENCED_BY", source_id))
                        chunk_refs += 1
                        total_refs_found += 1

            if chunk_refs > 0:
                chunks_with_refs += 1

        # print()  # New line after progress indicator

        # Remove duplicates
        relationships = list(set(relationships))
        elapsed = time.time() - start_time
        # print(f"    Extracted {len(relationships)} explicit reference relationships in {elapsed:.2f}s")
        # print(f"      - Chunks with references: {chunks_with_refs}/{len(chunks)}")
        # print(f"      - Total references found: {total_refs_found}")
        # print(f"      - Unique relationships: {len(relationships) // 2} (with reverse)")

        return relationships


    def create_faiss_index(self,chunks: List[Dict[str, Any]]) -> Tuple[faiss.Index, Dict[int, str], np.ndarray]:
        """
        Create FAISS HNSW index for chunk embeddings
        Returns: (FAISS index, id_to_section_id mapping, embeddings_array)
        """
        start_time = time.time()
        # print(f"    Generating embeddings for {len(chunks)} chunks...")
        # print(f"      Using model: BAAI/bge-large-en-v1.5")

        # Generate embeddings
        chunk_texts = [chunk["content"] for chunk in chunks]
        # print(f"      Embedding chunks (this may take a while)...")

        embed_start = time.time()
        embeddings_list = embeddings.embed_documents(chunk_texts)
        embed_time = time.time() - embed_start

        embeddings_array = np.array(embeddings_list).astype('float32')

        # print(f"       Generated embeddings in {embed_time:.2f}s")
        # print(f"         Shape: {embeddings_array.shape}")
        # print(f"         Dimension: {embeddings_array.shape[1]}")

        # Create HNSW index
        # print(f"    Building FAISS HNSW index...")
        dimension = embeddings_array.shape[1]
        index = faiss.IndexHNSWFlat(dimension, 64)  # M=64 for good balance

        # Set ef_construction and ef_search parameters
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 50

        # Add vectors to index
        index_start = time.time()
        index.add(embeddings_array)
        index_time = time.time() - index_start

        # print(f"       Index built in {index_time:.2f}s")
        # print(f"         Total vectors: {index.ntotal}")
        # print(f"         Index type: HNSW (M=64, efConstruction=200, efSearch=50)")

        # Create mapping from FAISS ID to section_id
        id_to_section = {i: chunks[i]["section_id"] for i in range(len(chunks))}

        elapsed = time.time() - start_time
        # print(f"    FAISS index creation completed in {elapsed:.2f}s")

        return index, id_to_section, embeddings_array


    def extract_semantic_relationships(
        self,
        chunks: List[Dict[str, Any]],
        faiss_index: faiss.Index,
        id_to_section: Dict[int, str],
        top_k: int = 30,
        final_k: int = 20,
        batch_size: int = 5 ) -> List[Tuple[str, str, str]]:
        """
        Extract semantic relationships using Azure OpenAI LLM with FAISS semantic search

        Args:
            chunks: List of chunk dictionaries
            faiss_index: FAISS HNSW index with chunk embeddings
            id_to_section: Mapping from FAISS ID to section_id
            top_k: Number of similar chunks to retrieve from FAISS
            final_k: Number of candidates to send to LLM (after filtering)
            batch_size: Batch size for processing

        Returns: List of (source, relationship, target) tuples
        """
        relationships = []

        # Create section ID to chunk mapping
        section_to_chunk = {chunk["section_id"]: chunk for chunk in chunks}
        all_section_ids = list(section_to_chunk.keys())

        # print(f"    Extracting semantic relationships using Azure OpenAI with FAISS semantic search...")
        # print(f"      Configuration:")
        # print(f"         - Model: {AZURE_DEPLOYMENT_NAME}")
        # print(f"         - FAISS top_k: {top_k} (similar chunks to retrieve)")
        # print(f"         - Final candidates: {final_k} (after filtering)")
        # print(f"         - Batch size: {batch_size}")
        # print(f"      Processing {len(chunks)} chunks...")

        # Process in batches
        total_batches = (len(chunks) - 1) // batch_size + 1
        processed = 0
        relationships_found = 0
        llm_calls = 0
        llm_errors = 0
        batch_start_time = time.time()

        for batch_idx in range(0, len(chunks), batch_size):
            batch = chunks[batch_idx:batch_idx+batch_size]
            current_batch = batch_idx // batch_size + 1

            for chunk in batch:
                processed += 1
                source_id = chunk.get("section_id")
                if not source_id:
                    continue  # Skip chunks without section_id

                source_content = chunk.get("content", "")[:3000]  # Limit content length
                section_title = chunk.get("section_title", "Unknown")[:50]

                # Print progress more frequently
                if processed % 5 == 0 or processed == 1:
                    pass  # print(f"      [{processed}/{len(chunks)}] Processing: {source_id} - {section_title}...", flush=True)

                    

                # Step 1: Generate embedding for source chunk
                chunk_content = chunk.get("content", "")
                if not chunk_content:
                    continue  # Skip empty chunks

                try:
                    source_embedding = embeddings.embed_query(chunk_content)
                except Exception as e:
                    if llm_errors < 3:
                        pass  # print(f"\n        Embedding error for {source_id}: {str(e)[:100]}")
                        
                    llm_errors += 1
                    continue
                source_embedding_array = np.array([source_embedding]).astype('float32')

                # Step 2: Query FAISS for top-K similar chunks
                k = top_k
                faiss_start = time.time()
                distances, indices = faiss_index.search(source_embedding_array, k)
                faiss_time = time.time() - faiss_start

                # Step 3: Map FAISS indices to section IDs
                candidate_indices = indices[0] if len(indices) > 0 else []
                candidate_ids = [
                    id_to_section[idx]
                    for idx in candidate_indices
                    if idx >= 0 and idx < len(chunks) and idx in id_to_section  # Safety checks
                ]

                # Step 4: Filter candidates (exclude self and immediate family)
                parent_id = chunk.get("metadata", {}).get("parent_section_id")
                child_ids = chunk.get("metadata", {}).get("child_section_ids", [])
                filtered_candidates = [
                    sid for sid in candidate_ids
                    if sid != source_id
                    and sid != parent_id
                    and sid not in child_ids
                    and sid in section_to_chunk  # Ensure candidate exists
                ]

                # Step 5: Keep top final_k candidates
                candidate_ids = filtered_candidates[:final_k]

                if not candidate_ids:
                    continue

                # Step 6: Prepare candidate info for LLM
                candidate_info = "\n".join([
                    f"- {sid}: {section_to_chunk.get(sid, {}).get('section_title', 'Unknown')}"
                    for sid in candidate_ids
                    if sid in section_to_chunk
                ])

                # Step 7: Prepare LLM prompt
                prompt = f"""
                    Analyze the following section from a 3GPP specification document and identify semantic relationships with other sections.

                    Source Section:
                    Section ID: {source_id}
                    Title: {section_title}
                    Content: {source_content}

                    Candidate Sections (semantically similar):
                    {candidate_info}

                    Identify semantic relationships between the source section and candidate sections. Consider:
                    - DEPENDS_ON: Source section depends on concepts/definitions from target
                    - USES: Source section uses mechanisms/procedures defined in target
                    - DEFINES: Source section defines terms/concepts used in target
                    - RELATED_TO: General semantic relationship
                    - PREREQUISITE_FOR: Source section is prerequisite knowledge for target

                    Return ONLY a valid JSON array of relationships. Each relationship should have:
                    - "target_section_id": section ID
                    - "relationship_type": one of DEPENDS_ON, USES, DEFINES, RELATED_TO, PREREQUISITE_FOR
                    - "confidence": "high" or "medium" or "low"

                    Format:
                    [
                    {{"target_section_id": "5.1.2", "relationship_type": "DEPENDS_ON", "confidence": "high"}},
                    ...
                    ]

                    If no relationships found, return empty array: []
                """

                try:
                    # Step 8: Call LLM with timeout handling
                    llm_start = time.time()
                    llm_calls += 1
                    if processed % 5 == 0 or processed == 1:
                        pass  # print(f"      [{processed}/{len(chunks)}] Calling LLM for {source_id}...", flush=True)
                        

                    try:
                        response = llm.invoke(prompt)
                    except Exception as api_error:
                        error_type = type(api_error).__name__
                        error_msg = str(api_error)[:200]
                        pass  # print(f"\n        LLM API error for {source_id}: {error_type} - {error_msg}", flush=True)
                        llm_errors += 1
                        if llm_errors <= 10:
                            import traceback
                            pass  # print(f"      Full error: {traceback.format_exc()[:500]}", flush=True)
                        continue

                    llm_time = time.time() - llm_start
                    if processed % 5 == 0 or processed == 1:
                        pass  # print(f"      [{processed}/{len(chunks)}] LLM response received in {llm_time:.1f}s for {source_id}", flush=True)
                        

                    response_text = response.content.strip()

                    # Step 9: Clean JSON response
                    if response_text.startswith("```json"):
                        response_text = response_text[7:]
                    if response_text.startswith("```"):
                        response_text = response_text[3:]
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()

                    # Step 10: Parse JSON
                    try:
                        rels = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        if llm_errors < 3:  # Only log first few JSON errors
                            pass  # print(f"\n        JSON parse error for {source_id}: {str(e)[:100]}")
                            
                        llm_errors += 1
                        continue

                    # Ensure rels is a list
                    if not isinstance(rels, list):
                        if llm_errors < 3:
                            pass  # print(f"\n        Expected list, got {type(rels).__name__} for {source_id}")
                            
                        llm_errors += 1
                        continue

                    # Step 11: Process relationships
                    chunk_rels = 0
                    for rel in rels:
                        if not isinstance(rel, dict):
                            continue

                        target_id = rel.get("target_section_id")
                        rel_type = rel.get("relationship_type")
                        confidence = rel.get("confidence", "medium")

                        # Validate target exists and is in candidates
                        if target_id and rel_type and target_id in candidate_ids and target_id in section_to_chunk:
                            relationships.append((source_id, rel_type, target_id))
                            chunk_rels += 1
                            relationships_found += 1
                            # Add reverse relationship for some types
                            if rel_type == "DEPENDS_ON":
                                relationships.append((target_id, "PREREQUISITE_FOR", source_id))
                                relationships_found += 1
                            elif rel_type == "PREREQUISITE_FOR":
                                relationships.append((target_id, "DEPENDS_ON", source_id))
                                relationships_found += 1

                    if processed % 5 == 0:
                        avg_llm_time = (time.time() - batch_start_time) / max(llm_calls, 1)
                        pass  # print(f"      [{processed}/{len(chunks)}] Avg LLM time: {avg_llm_time:.1f}s | Relationships: {relationships_found}", flush=True)

                except json.JSONDecodeError:
                    # Already handled above
                    pass
                    
                except KeyError as e:
                    llm_errors += 1
                    if llm_errors <= 5:
                        pass  # print(f"\n        KeyError for {source_id}: {str(e)}", flush=True)
                        
                    continue
                except Exception as e:
                    # Skip on error (log if needed)
                    llm_errors += 1
                    if llm_errors <= 10:  # Show more errors for debugging
                        error_msg = str(e)[:200] if str(e) else "Unknown error"
                        pass  # print(f"\n        Error processing {source_id}: {error_msg}", flush=True)
                        pass  # print(f"      Error type: {type(e).__name__}", flush=True)
                    continue

        # print()  # New line after progress indicators
        elapsed = time.time() - batch_start_time

        # Remove duplicates
        relationships = list(set(relationships))

        # print(f"    Semantic relationship extraction completed in {elapsed:.1f}s")
        # print(f"      - Chunks processed: {processed}/{len(chunks)}")
        # print(f"      - LLM calls: {llm_calls}")
        # print(f"      - LLM errors: {llm_errors}")
        # print(f"      - Relationships found: {relationships_found}")
        # print(f"      - Unique relationships: {len(relationships)}")
        if llm_calls > 0:
            pass  # print(f"      - Avg LLM time: {elapsed / llm_calls:.2f}s per chunk")
            

        return relationships

        
    def build_knowledge_graph(self,chunks: List[Dict[str, Any]], relationships: List[Tuple[str, str, str]]) -> nx.DiGraph:
        """
        Build NetworkX knowledge graph with chunks as nodes
        """
        start_time = time.time()
        # print(f"    Building NetworkX knowledge graph...")
        # print(f"      Adding {len(chunks)} nodes...")

        graph = nx.DiGraph()

        # Add chunks as nodes
        for chunk in chunks:
            section_id = chunk.get("section_id")
            if not section_id:
                continue  # Skip chunks without section_id

            # Combine metadata with content (metadata already has section_id, section_title, etc.)
            metadata = chunk.get("metadata", {})
            content = chunk.get("content", "")

            node_attributes = {
                **metadata,
                "content": content
            }
            graph.add_node(section_id, **node_attributes)

        # print(f"      Adding {len(relationships)} edges...")

        # Add relationships as edges
        edges_added = 0
        edges_updated = 0
        invalid_edges = 0

        for source, rel_type, target in relationships:
            # Validate relationship tuple
            if not source or not target or not rel_type:
                invalid_edges += 1
                continue

            if graph.has_node(source) and graph.has_node(target):
                # Check if edge exists
                if graph.has_edge(source, target):
                    # Add relationship type to existing edge attributes
                    existing_rels = graph[source][target].get("relationship_types", [])
                    if not isinstance(existing_rels, list):
                        existing_rels = [existing_rels] if existing_rels else []
                    if rel_type not in existing_rels:
                        existing_rels.append(rel_type)
                        graph[source][target]["relationship_types"] = existing_rels
                        edges_updated += 1
                else:
                    graph.add_edge(source, target, relationship_types=[rel_type])
                    edges_added += 1
            else:
                invalid_edges += 1

        if invalid_edges > 0:
            pass  # print(f"        Skipped {invalid_edges} invalid edges (missing nodes)")
            

        elapsed = time.time() - start_time
        # print(f"    Knowledge graph built in {elapsed:.3f}s")
        # print(f"      - Nodes: {graph.number_of_nodes()}")
        # print(f"      - Edges: {graph.number_of_edges()}")
        # print(f"      - New edges added: {edges_added}")
        # print(f"      - Existing edges updated: {edges_updated}")

        return graph


    def save_faiss_index(self,index: faiss.Index, id_to_section: Dict[int, str], chunks: List[Dict[str, Any]],FAISS_INDEX_FILE: str,FAISS_METADATA_FILE:str):
        """Save FAISS index and metadata"""
        start_time = time.time()
        # print(f"      Saving FAISS index to {FAISS_INDEX_FILE}...")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)

        # Save FAISS index
        faiss.write_index(index, FAISS_INDEX_FILE)
        index_size = os.path.getsize(FAISS_INDEX_FILE) / (1024 * 1024)  # Size in MB

        # print(f"      Saving metadata to {FAISS_METADATA_FILE}...")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(FAISS_METADATA_FILE), exist_ok=True)

        # Save metadata mapping
        metadata = {
            "id_to_section": id_to_section,
            "section_to_chunk": {
                chunk.get("section_id", ""): {
                    "section_title": chunk.get("section_title", "Unknown"),
                    "metadata": chunk.get("metadata", {})
                }
                for chunk in chunks
                if chunk.get("section_id")  # Only include chunks with valid section_id
            }
        }

        with open(FAISS_METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        metadata_size = os.path.getsize(FAISS_METADATA_FILE) / (1024 * 1024)  # Size in MB

        elapsed = time.time() - start_time
        # print(f"    FAISS index and metadata saved in {elapsed:.2f}s")
        # print(f"      - Index file: {FAISS_INDEX_FILE} ({index_size:.2f} MB)")
        # print(f"      - Metadata file: {FAISS_METADATA_FILE} ({metadata_size:.2f} MB)")


    def save_kg(self,kg_file_path,graph: nx.DiGraph):
        """Save knowledge graph to disk"""
        start_time = time.time()
        # print(f"      Saving to {kg_file_path}...")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(kg_file_path), exist_ok=True)

        with open(kg_file_path, 'wb') as f:
            pickle.dump(graph, f)

        file_size = os.path.getsize(kg_file_path) / (1024 * 1024)  # Size in MB
        elapsed = time.time() - start_time

        # print(f"    Knowledge graph saved in {elapsed:.2f}s")
        # print(f"      - File: {kg_file_path}")
        # print(f"      - Size: {file_size:.2f} MB")
        # print(f"      - Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")


    def load_kg(self,kg_file_path) -> Optional[nx.DiGraph]:
        """Load knowledge graph from disk"""
        if not os.path.exists(kg_file_path):
            return None
        with open(kg_file_path, 'rb') as f:
            return pickle.load(f)

