from app.core.conversation_context import latest_escalation_state, latest_evidence_snapshot
from app.models.conversation import ChatMessageRequest, ConversationMessage, IntentType, SupportScopeStatus


def build_intake_node(llm_client):
    def intake_node(state: dict) -> dict:
        request = ChatMessageRequest.model_validate(state["request"])
        history = [ConversationMessage.model_validate(item) for item in state.get("history", [])]
        previous_escalation_active = latest_escalation_state(history) if not request.issue_resolved else False
        baseline_evidence = latest_evidence_snapshot(history).merge(request.evidence_pack)
        request_for_classification = request.model_copy(update={"evidence_pack": baseline_evidence})
        classification = llm_client.classify_intent(
            request=request_for_classification,
            history=history,
        )
        escalation_active = False
        if request.issue_resolved:
            classification = classification.model_copy(update={"intent": IntentType.troubleshoot, "system_message": None})
        else:
            escalation_active = previous_escalation_active or request.request_ticket or classification.intent == IntentType.escalate
        if escalation_active and classification.intent != IntentType.escalate:
            classification = classification.model_copy(update={"intent": IntentType.escalate, "system_message": None})
        merged_evidence = classification.evidence_pack.merge(baseline_evidence)
        output = {
            "user_query": classification.user_query or request.message,
            "classification": classification.model_dump(mode="json"),
            "current_phase": "intake",
            "history": [],
            "merged_evidence_pack": merged_evidence.model_dump(mode="json", exclude_none=True),
            "missing_fields": merged_evidence.missing_core_fields(),
            "missing_artifacts": merged_evidence.missing_best_effort_artifacts(),
            "support_scope_status": classification.support_scope_status.value,
            "unsupported_reason": classification.unsupported_reason.value if classification.unsupported_reason else None,
            "missing_scope_fields": classification.missing_scope_fields,
            "escalation_active": escalation_active,
            "previous_escalation_active": previous_escalation_active,
        }
        system_message = classification.system_message
        if escalation_active:
            system_message = None
        elif (
            classification.intent != IntentType.general_question
            and classification.support_scope_status == SupportScopeStatus.unknown
        ):
            system_message = None
        if system_message:
            output["system_message"] = system_message
            output["response_text"] = system_message
            output["next_action"] = "ask_question"
            output["citations"] = []
        return output

    return intake_node
