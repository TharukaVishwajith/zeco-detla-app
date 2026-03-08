from app.models.conversation import ChatMessageRequest, ConversationMessage, SupportScopeStatus
from app.models.evidence import EvidencePack


def _build_scope_question(missing_scope_fields: list[str]) -> str:
    ordered = ", ".join(missing_scope_fields) if missing_scope_fields else "site_type, system_size_kw, user_role, ownership_verified"
    return f"Before troubleshooting, please confirm the following site eligibility details: {ordered}."


def _latest_evidence_snapshot(history: list[ConversationMessage]) -> EvidencePack:
    for message in reversed(history):
        if message.evidence_snapshot:
            return message.evidence_snapshot
    return EvidencePack()


def build_intake_node(llm_client):
    def intake_node(state: dict) -> dict:
        request = ChatMessageRequest.model_validate(state["request"])
        history = [ConversationMessage.model_validate(item) for item in state.get("history", [])]
        request_for_classification = request.model_copy(
            update={"evidence_pack": _latest_evidence_snapshot(history).merge(request.evidence_pack)}
        )
        classification = llm_client.classify_intent(
            request=request_for_classification,
            history=history,
        )
        output = {
            "user_query": classification.user_query or request.message,
            "classification": classification.model_dump(mode="json"),
            "current_phase": "intake",
            "support_scope_status": classification.support_scope_status.value,
            "unsupported_reason": classification.unsupported_reason.value if classification.unsupported_reason else None,
            "missing_scope_fields": classification.missing_scope_fields,
        }
        system_message = classification.system_message
        if classification.support_scope_status == SupportScopeStatus.unknown and not system_message:
            system_message = _build_scope_question(classification.missing_scope_fields)
        if system_message:
            output["system_message"] = system_message
            output["response_text"] = system_message
            output["next_action"] = "ask_question"
            output["citations"] = []
        return output

    return intake_node
