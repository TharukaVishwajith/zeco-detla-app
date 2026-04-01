from app.models.conversation import (
    ChatMessageRequest,
    ConversationMessage,
    ConversationRole,
    IntentClassification,
    RetrievedDocument,
    TroubleshootingAction,
)


MAX_TROUBLESHOOTING_ROUNDS = 5
ROUND_COUNTED_ACTIONS = {
    TroubleshootingAction.ask_question,
    TroubleshootingAction.continue_troubleshooting,
}


def build_troubleshooting_node(llm_client, validation_service):
    def troubleshooting_node(state: dict) -> dict:
        request = ChatMessageRequest.model_validate(state["request"])
        classification = IntentClassification.model_validate(state["classification"])
        user_query = state.get("user_query") or request.message
        documents = [RetrievedDocument.model_validate(item) for item in state.get("retrieved_docs", [])]
        source_history = [ConversationMessage.model_validate(item) for item in state.get("source_history", [])]
        troubleshooting_rounds = _count_troubleshooting_rounds(source_history)

        if not request.issue_resolved and troubleshooting_rounds >= MAX_TROUBLESHOOTING_ROUNDS:
            return {
                "troubleshooting_response": {
                    "response_text": "",
                    "citations": [],
                    "next_action": TroubleshootingAction.escalate.value,
                },
                "response_text": "",
                "citations": [],
                "next_action": TroubleshootingAction.escalate.value,
                "current_phase": "troubleshooting",
                "escalation_active": True,
                "troubleshooting_rounds": troubleshooting_rounds,
                "force_ticket_creation": True,
                "errors": [],
            }

        if request.issue_resolved:
            response = llm_client.generate_resolved_troubleshooting_response()
            is_valid, errors = True, []
        else:
            response = llm_client.generate_troubleshooting_response(
                message=user_query,
                retrieved_docs=documents,
                classification=classification,
                history=source_history,
            )
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

        if request.request_ticket and response.next_action != TroubleshootingAction.resolved:
            response.next_action = TroubleshootingAction.collect_evidence

        if not request.issue_resolved and response.next_action in ROUND_COUNTED_ACTIONS:
            troubleshooting_rounds += 1

        return {
            "troubleshooting_response": response.model_dump(mode="json"),
            "response_text": response.response_text,
            "citations": response.citations,
            "next_action": response.next_action.value,
            "current_phase": "troubleshooting",
            "escalation_active": response.next_action in {TroubleshootingAction.collect_evidence, TroubleshootingAction.escalate},
            "troubleshooting_rounds": troubleshooting_rounds,
            "force_ticket_creation": False,
            "errors": errors if not is_valid else [],
        }

    return troubleshooting_node


def _count_troubleshooting_rounds(history: list[ConversationMessage]) -> int:
    return sum(
        1
        for message in history
        if message.role == ConversationRole.assistant and message.next_action in ROUND_COUNTED_ACTIONS
    )
