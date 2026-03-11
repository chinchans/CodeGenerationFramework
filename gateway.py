# gateway.py

import sys
import logging
from pathlib import Path
import json
import uuid
from typing import Optional

# logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Add project root to Python path
# ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# ----------------------------------------------------------
# Imports
# ----------------------------------------------------------
from GlobalState import GlobalState
from langchain_core.messages import HumanMessage



class AIServiceGateway:
    def __init__(self):
        from Feature_Validation.main import FeatureValidationAgent

        self.feature_validation_agent = FeatureValidationAgent()

    def orchestrator(
        self,
        user_query: str,
        run_validation: bool = False,
        codebase_path: Optional[str] = None,
        oai_path: Optional[str] = None,
    ) -> GlobalState:

        # --------------------------------------------------
        # STEP 1: Create Session ID
        # --------------------------------------------------
        session_id = str(uuid.uuid4())
        # logger.info("Created session: %s", session_id)

        # --------------------------------------------------
        # STEP 2: Initialize GlobalState
        # --------------------------------------------------
        state: GlobalState = {
            "session_id": session_id,
            "messages": [HumanMessage(content=user_query)]
        }

        # --------------------------------------------------
        # STEP 3: Feature Validation Agent
        # --------------------------------------------------
        state = self.feature_validation_agent.run(state)

        # logger.info("Feature Validation Agent")
        # logger.info(state)
        
        # ------------------------------------------------------------------------------------
        # STEP 4: Pass State to Knowledge Retrieval Agent(For Vector DB + Knowldge Graph Creation for both the specification and code artifact)
        # ------------------------------------------------------------------------------------
        from Knowledge_Retrieval.Knowldge_creations import knowledge_creator_agent

        state = knowledge_creator_agent.createKnowledge(state)

        # logger.info("State after knowledge creator")
        # logger.info(state)

        # -----------------------------------------------------------------------------------------------
        # STEP 5: Pass State to Knowledge Retrieval Agent(For Retrieving the context from the specification and code artifacts)
        # -----------------------------------------------------------------------------------------------
        from Knowledge_Retrieval import retriever_agent

        state = retriever_agent.retrieverAgent(state)

        
        # -------------------------------------------------------------------------------------------
        # STEP 6: Pass State to Template Filler Agent(Both for Specification Context + Code Artifacts)
        # -------------------------------------------------------------------------------------------
        from Template_Orchestrator import template_filler_agent

        state = template_filler_agent.templateFillerAgent(state)

        # -------------------------------------------------------------------------------------------
        # STEP 7: Code Validation Agent – run when requested
        # -------------------------------------------------------------------------------------------
        from Code_Validation.code_validation_agent import CodeValidationAgent

        # if run_validation and codebase_path:
        #     try:
        #         template_path = state.get("final_filled_template_path")

        #         # Prefer intent from state messages if available, otherwise fall back to user_query
        #         user_intent = user_query
        #         messages = state.get("messages") or []
        #         if messages and hasattr(messages[0], "content"):
        #             user_intent = messages[0].content

        #         oai_list = [oai_path] if oai_path else [codebase_path]

        #         code_validation_agent = CodeValidationAgent(
        #             oai_path=oai_list,
        #             codebase_path=codebase_path,
        #             template_path=template_path,
        #             user_intent=user_intent,
        #         )
        #         validation_results = code_validation_agent.run()
        #         state["code_validation_results"] = validation_results
        #     except Exception as e:
        #         state["code_validation_results"] = {"error": str(e)}

        return state


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # print("query")
    
    gateway = AIServiceGateway()

    query = "Implement F1AP UE CONTEXT SETUP Response Message Procedure of Inter-gNB-DU LTM Handover"
    print(query)
    final_state = gateway.orchestrator(query)

    print("Done ")
    filtered_state = {key: value for key, value in final_state.items() if key not in ['specs_context', 'code_artifacts_context','code_generation_prompt']}

    # Print the filtered state
    print("Final global state: %s", json.dumps(filtered_state, indent=2, default=str))

    # logger.info("Final global state: %s", json.dumps(final_state, indent=2, default=str))
