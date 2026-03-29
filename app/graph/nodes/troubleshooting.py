from app.models.conversation import (
    ChatMessageRequest,
    IntentClassification,
    RetrievedDocument,
    TroubleshootingAction,
    TroubleshootingResponse,
)


def _ensure_support_ticket_offer(response_text: str) -> str:
    ticket_offer = "If the issue persists, would you like me to help create a support ticket?"
    if ticket_offer.lower() in response_text.lower():
        return response_text
    stripped = response_text.rstrip()
    if not stripped:
        return ticket_offer
    return f"{stripped}\n\n{ticket_offer}"


def build_troubleshooting_node(llm_client, validation_service):
    def troubleshooting_node(state: dict) -> dict:
        request = ChatMessageRequest.model_validate(state["request"])
        classification = IntentClassification.model_validate(state["classification"])
        user_query = state.get("user_query") or request.message
        documents = [RetrievedDocument.model_validate(item) for item in state.get("retrieved_docs", [])]

        if request.issue_resolved:
            response = llm_client.generate_resolved_troubleshooting_response()
            is_valid, errors = True, []
        else:
            response = llm_client.generate_troubleshooting_response(
                message=user_query,
                retrieved_docs=documents,
                classification=classification,
            )
            if not request.request_ticket and response.next_action != TroubleshootingAction.resolved:
                response = response.model_copy(update={"response_text": _ensure_support_ticket_offer(response.response_text)})
            is_valid, errors = validation_service.validate_troubleshooting_response(response=response, retrieved_docs=documents)
        if not is_valid:
            if request.issue_resolved:
                response = llm_client.generate_resolved_troubleshooting_response()
            else:
                response = llm_client._grounded_fallback_response(  # noqa: SLF001 - best-effort fallback for invalid output
                    message=user_query,
                    retrieved_docs=documents,
                    classification=classification,
                )
                if not request.request_ticket and response.next_action != TroubleshootingAction.resolved:
                    response = response.model_copy(update={"response_text": _ensure_support_ticket_offer(response.response_text)})

        if request.request_ticket and response.next_action != TroubleshootingAction.resolved:
            response.next_action = TroubleshootingAction.collect_evidence

        return {
            "troubleshooting_response": response.model_dump(mode="json"),
            "response_text": response.response_text,
            "citations": response.citations,
            "next_action": response.next_action.value,
            "current_phase": "troubleshooting",
            "escalation_active": response.next_action in {TroubleshootingAction.collect_evidence, TroubleshootingAction.escalate},
            "errors": errors if not is_valid else [],
        }

    return troubleshooting_node
