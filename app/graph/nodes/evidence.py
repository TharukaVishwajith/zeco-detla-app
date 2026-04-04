from app.models.conversation import ChatMessageRequest
from app.models.conversation import ConversationMessage
from app.models.conversation import IntentClassification
from app.models.evidence import EvidencePack, format_markdown_field_list


MINIMUM_TICKET_EVIDENCE_RATIO = 0.7


def build_evidence_collection_node(llm_client):
    def evidence_collection_node(state: dict) -> dict:
        request = ChatMessageRequest.model_validate(state["request"])
        classification = IntentClassification.model_validate(state["classification"])
        source_history = [ConversationMessage.model_validate(item) for item in state.get("source_history", [])]
        merged_evidence = EvidencePack.model_validate(state.get("merged_evidence_pack", {}))
        missing_fields = merged_evidence.missing_core_fields()
        missing_artifacts = merged_evidence.missing_best_effort_artifacts()
        evidence_completion_ratio = merged_evidence.core_completion_ratio()
        safety_assessment = state.get("safety_assessment", {})
        if not safety_assessment.get("escalate_immediately") and classification.risk_flags:
            safety_assessment = {
                **safety_assessment,
                "escalate_immediately": True,
                "reason": safety_assessment.get("reason") or classification.risk_flags[0],
            }
        support_scope_status = state.get("support_scope_status")
        previous_escalation_active = bool(state.get("previous_escalation_active"))
        ready_for_ticket = previous_escalation_active or not missing_fields or evidence_completion_ratio >= MINIMUM_TICKET_EVIDENCE_RATIO

        if not ready_for_ticket:
            response_text = llm_client.generate_evidence_collection_response(
                request=request,
                classification=classification,
                history=source_history,
                merged_evidence=merged_evidence,
                missing_fields=missing_fields,
                support_scope_status=support_scope_status,
                safety_assessment=safety_assessment,
            )
            next_action = "collect_evidence"
        else:
            response_text = "## Evidence Ready\n\nI have enough information to create the support ticket now."
            if missing_fields:
                response_text += (
                    "\n\nI will note these remaining missing details in the escalation:\n"
                    f"{format_markdown_field_list(missing_fields)}"
                )
            if missing_artifacts:
                response_text += (
                    "\n\nI will note these unavailable or missing artifacts in the escalation:\n"
                    f"{format_markdown_field_list(missing_artifacts)}"
                )
            next_action = "create_ticket"

        return {
            "missing_fields": missing_fields,
            "missing_artifacts": missing_artifacts,
            "evidence_completion_ratio": evidence_completion_ratio,
            "ticket_ready": ready_for_ticket,
            "response_text": response_text,
            "next_action": next_action,
            "current_phase": "evidence_collection",
            "escalation_active": True,
        }

    return evidence_collection_node
