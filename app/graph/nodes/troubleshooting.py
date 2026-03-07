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
            )

        if skip_validation:
            is_valid, errors = True, []
        else:
            is_valid, errors = validation_service.validate_troubleshooting_response(response=response, retrieved_docs=documents)
        if not is_valid:
            fallback_text = (
                "I could not produce a fully grounded answer from the retrieved Delta knowledge-base content. "
                "Please provide the exact model number and fault text, or request escalation."
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
            "errors": errors if not is_valid else [],
        }

    return troubleshooting_node
