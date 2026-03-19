from app.models.conversation import ChatMessageRequest, ConversationMessage


def build_evidence_extraction_node(llm_client):
    def evidence_extraction_node(state: dict) -> dict:
        request = ChatMessageRequest.model_validate(state["request"])
        history = [ConversationMessage.model_validate(item) for item in state.get("history", [])]
        merged = llm_client.extract_evidence(request=request, history=history)

        return {
            "merged_evidence_pack": merged.model_dump(mode="json", exclude_none=True),
            "missing_fields": merged.missing_core_fields(),
            "missing_artifacts": merged.missing_best_effort_artifacts(),
            "history": [],
            "current_phase": "evidence_extraction",
            "escalation_active": state.get("escalation_active", False),
        }

    return evidence_extraction_node
