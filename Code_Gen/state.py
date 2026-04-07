from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class CodeGenState(TypedDict, total=False):
    # Core run metadata
    input_feature_json: str
    input_intent: str
    intent: str
    template_path_used: str

    # Feature Validation outputs
    message_names: List[str]
    protocol_classification: Dict[str, Any]
    specifications: List[Dict[str, Any]]
    selected_template_name: str
    selected_template_path: str
    feature_intent: Dict[str, Any]

    # Knowledge creation / retrieval
    spec_kg_status: Dict[str, Any]
    code_retrieval_sources: Dict[str, Any]
    spec_retrieval_context: Dict[str, Any]
    spec_retrieval_context_path: str
    code_artifacts_context: Dict[str, Any]
    code_artifacts_chunks_path: str

    # Template orchestration outputs
    spec_filled_template_path: str
    final_filled_template_path: str
    code_generation_prompt: str
    code_generation_prompt_path: str

    # Self Learning validation outputs
    self_learning_matched_rule: Optional[str]
    self_learning_has_ambiguities: bool
    self_learning_ambiguities: List[Dict[str, Any]]
    self_learning_resolution_applied: bool
    self_learning_resolved_template_path: str
    self_learning_validation_policy: Dict[str, str]
    self_learning_llm_ambiguity_review: Dict[str, Any]
    self_learning_deterministic_ambiguity_count: int
    self_learning_human_resolution_required: bool
    self_learning_next_step_if_ambiguous: str

    # Tracking
    run_manifest_path: Optional[str]
