import re

from app.models.conversation import (
    ChatMessageRequest,
    ConversationMessage,
    IntentClassification,
    IntentType,
    RetrievedDocument,
    TroubleshootingAction,
)


MAX_TROUBLESHOOTING_ROUNDS = 5


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
                    "counts_as_troubleshooting_round": False,
                },
                "response_text": "",
                "citations": [],
                "next_action": TroubleshootingAction.escalate.value,
                "current_phase": "troubleshooting",
                "escalation_active": True,
                "troubleshooting_rounds": troubleshooting_rounds,
                "force_ticket_creation": True,
                "counts_as_troubleshooting_round": False,
                "errors": [],
            }

        if request.issue_resolved:
            response = llm_client.generate_resolved_troubleshooting_response()
            response = _normalize_troubleshooting_response(response, classification)
            is_valid, errors = True, []
        else:
            response = llm_client.generate_troubleshooting_response(
                message=user_query,
                retrieved_docs=documents,
                classification=classification,
                history=source_history,
            )
            response = _normalize_troubleshooting_response(response, classification)
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
            response = _normalize_troubleshooting_response(response, classification)

        if request.request_ticket and response.next_action != TroubleshootingAction.resolved:
            response.next_action = TroubleshootingAction.collect_evidence
            response = _normalize_troubleshooting_response(response, classification)

        if not request.issue_resolved and response.counts_as_troubleshooting_round:
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
            "counts_as_troubleshooting_round": response.counts_as_troubleshooting_round,
            "errors": errors if not is_valid else [],
        }

    return troubleshooting_node


def _count_troubleshooting_rounds(history: list[ConversationMessage]) -> int:
    for message in reversed(history):
        if message.troubleshooting_rounds is not None:
            return message.troubleshooting_rounds
    return sum(1 for message in history if message.counts_as_troubleshooting_round is True)


def _normalize_troubleshooting_response(response, classification: IntentClassification):
    should_count = _should_count_as_troubleshooting_round(response=response, classification=classification)
    if response.counts_as_troubleshooting_round == should_count:
        return response
    return response.model_copy(update={"counts_as_troubleshooting_round": should_count})


def _should_count_as_troubleshooting_round(*, response, classification: IntentClassification) -> bool:
    if classification.intent != IntentType.troubleshoot:
        return False
    if response.next_action != TroubleshootingAction.continue_troubleshooting:
        return False
    return _has_actionable_numbered_steps(response.response_text)


def _has_actionable_numbered_steps(response_text: str) -> bool:
    actionable_prefixes = (
        "check ",
        "look ",
        "confirm ",
        "turn ",
        "wait ",
        "verify ",
        "perform ",
        "restart ",
        "acknowledge ",
        "inspect ",
        "record ",
        "note ",
        "ensure ",
        "review ",
        "power ",
        "press ",
        "switch ",
        "disconnect ",
        "reconnect ",
        "compare ",
        "observe ",
        "reset ",
        "monitor ",
    )
    for raw_line in response_text.splitlines():
        match = re.match(r"^\s*\d+\.\s+(.*\S)\s*$", raw_line)
        if not match:
            continue
        step_text = match.group(1).strip("`*_ ").lower()
        if step_text.startswith(actionable_prefixes):
            return True
    return False
