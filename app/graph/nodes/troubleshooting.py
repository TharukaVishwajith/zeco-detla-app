from app.models.conversation import (
    ChatMessageRequest,
    IntentClassification,
    RetrievedDocument,
    TroubleshootingAction,
    TroubleshootingResponse,
)


def build_troubleshooting_node(llm_client, validation_service):
    def troubleshooting_node(state: dict) -> dict:
        request = ChatMessageRequest.model_validate(state["request"])
        classification = IntentClassification.model_validate(state["classification"])
        user_query = state.get("user_query") or request.message
        documents = [RetrievedDocument.model_validate(item) for item in state.get("retrieved_docs", [])]
        skip_validation = False

        if request.issue_resolved:
            response = TroubleshootingResponse(
                response_text="Issue marked as resolved. No ticket will be created.",
                citations=[],
                next_action=TroubleshootingAction.resolved,
            )
        else:
            response = llm_client.generate_troubleshooting_response(
                message=user_query,
                retrieved_docs=documents,
                classification=classification,
                validation_service=validation_service,
            )

        if skip_validation:
            is_valid, errors = True, []
        else:
            is_valid, errors = validation_service.validate_troubleshooting_response(response=response, retrieved_docs=documents)
        if not is_valid:
            fallback_text = (
                "## Safe next step\n\n"
                "I could not complete a reliable support answer this turn.\n\n"
                "Please reply with the exact device display text or the latest visible symptom, or ask me to escalate the issue."
            )
            response = TroubleshootingResponse(
                response_text=fallback_text,
                citations=[],
                next_action=TroubleshootingAction.ask_question,
            )

        if request.request_ticket and response.next_action != TroubleshootingAction.resolved:
            response.next_action = TroubleshootingAction.collect_evidence

        return {
            "troubleshooting_response": response.model_dump(mode="json"),
            "response_text": response.response_text,
            "citations": response.citations,
            "next_action": response.next_action.value,
            "current_phase": "troubleshooting",
            "escalation_active": response.next_action in {TroubleshootingAction.collect_evidence, TroubleshootingAction.escalate},
            "response_source": response.response_source.value,
            "errors": errors if not is_valid else [],
        }

    return troubleshooting_node
