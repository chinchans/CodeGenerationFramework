# GlobalState.py

from typing import TypedDict, Dict, Any, Optional, List, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.store.base import Op


class GlobalState(TypedDict, total=False):
    # Core Session Info
    session_id: str

    # LangGraph Message Handling
    messages: Annotated[List[BaseMessage], add_messages]

    # Feature Validation Data
    message_names: Optional[List[str]]
    protocol_classification: Optional[Dict[str, Any]]
    specifications: Optional[List[Dict[str, Any]]]
    selected_template_name: Optional[str]
    selected_template_path: Optional[str]
    feature_intent: Optional[Dict[str, Any]]


    # Knowledge Retrieval Data
    specs_retrieval_sources: List[Dict[str,Any]]
    code_retrieval_sources : Dict[str,Any]
    specs_context:List[Any]
    specs_chunks_path:str
    code_artifacts_context:Dict[str,Any]
    code_artifacts_chunks_path:str

    # Template Filler Data
    spec_filled_template_path:str
    final_filled_template_path:str
    code_generation_prompt: str
    code_generation_prompt_path: str

    # Code Validation (post-commit / optional)
    code_validation_results: Dict[str, Any]
