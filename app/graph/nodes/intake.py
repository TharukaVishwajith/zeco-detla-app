from app.core.conversation_context import latest_escalation_state, merge_evidence_from_conversation
from app.models.conversation import ChatMessageRequest, ConversationMessage, IntentType, SupportScopeStatus
from app.models.evidence import format_markdown_field_list


def _build_scope_question(missing_scope_fields: list[str]) -> str:
    fields = missing_scope_fields or ["site_type", "system_size_kw", "user_role", "ownership_verified"]
    return (
        "## Site Eligibility Check\n\n"
        "Before troubleshooting, please confirm:\n"
        f"{format_markdown_field_list(fields)}"
    )

def build_intake_node(llm_client):
    def intake_node(state: dict) -> dict:
        request = ChatMessageRequest.model_validate(state["request"])
        history = [ConversationMessage.model_validate(item) for item in state.get("history", [])]
        merged_evidence = merge_evidence_from_conversation(
            current_message=request.message,
            request_evidence=request.evidence_pack,
            history=history,
        )
        request_for_classification = request.model_copy(update={"evidence_pack": merged_evidence})
        classification = llm_client.classify_intent(
            request=request_for_classification,
            history=history,
        )
        escalation_active = False
        if request.issue_resolved:
            classification = classification.model_copy(update={"intent": IntentType.troubleshoot, "system_message": None})
        else:
            escalation_active = latest_escalation_state(history) or request.request_ticket or classification.intent == IntentType.escalate
        if escalation_active and classification.intent != IntentType.escalate:
            classification = classification.model_copy(update={"intent": IntentType.escalate, "system_message": None})
        output = {
            "user_query": classification.user_query or request.message,
            "classification": classification.model_dump(mode="json"),
            "current_phase": "intake",
            "support_scope_status": classification.support_scope_status.value,
            "unsupported_reason": classification.unsupported_reason.value if classification.unsupported_reason else None,
            "missing_scope_fields": classification.missing_scope_fields,
            "escalation_active": escalation_active,
        }
        system_message = classification.system_message
        if escalation_active:
            system_message = None
        elif classification.support_scope_status == SupportScopeStatus.unknown and not system_message:
            system_message = _build_scope_question(classification.missing_scope_fields)
        if system_message:
            output["system_message"] = system_message
            output["response_text"] = system_message
            output["next_action"] = "ask_question"
            output["citations"] = []
        return output

    return intake_node
