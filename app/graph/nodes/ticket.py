import re

from app.models.conversation import (
    ChatMessageRequest,
    ConversationMessage,
    ConversationRole,
    IntentClassification,
    TroubleshootingAction,
)


TROUBLESHOOTING_HISTORY_ACTIONS = {
    TroubleshootingAction.ask_question,
    TroubleshootingAction.continue_troubleshooting,
    TroubleshootingAction.resolved,
}


def build_ticket_creation_node(ticket_service, llm_client):
    def ticket_creation_node(state: dict) -> dict:
        request = ChatMessageRequest.model_validate(state["request"])
        classification = IntentClassification.model_validate(state["classification"])
        source_history = [ConversationMessage.model_validate(item) for item in state.get("source_history", [])]
        troubleshooting_steps = _collect_troubleshooting_notes(state)
        escalation_reason = state.get("safety_assessment", {}).get("reason")
        ticket_response = ticket_service.create_from_graph(
            request=request,
            issue_summary=_build_issue_summary(state, request),
            troubleshooting_steps=troubleshooting_steps,
            escalation_reason=escalation_reason,
            merged_evidence_pack=state.get("merged_evidence_pack", {}),
            unsupported_reason=state.get("unsupported_reason"),
            missing_artifacts=state.get("missing_artifacts", []),
        )
        response_text = _build_ticket_creation_response(
            state=state,
            ticket_id=ticket_response.ticket_id,
            llm_client=llm_client,
            request=request,
            classification=classification,
            history=source_history,
        )
        return {
            "ticket_response": ticket_response.model_dump(mode="json"),
            "response_text": response_text,
            "next_action": "escalate",
            "current_phase": "ticket_creation",
            "escalation_active": False,
        }

    return ticket_creation_node


def _collect_troubleshooting_notes(state: dict) -> list[str]:
    notes: list[str] = []
    seen: set[str] = set()

    history = [ConversationMessage.model_validate(item) for item in state.get("source_history", [])]
    for message in history:
        if message.role != ConversationRole.assistant or message.next_action not in TROUBLESHOOTING_HISTORY_ACTIONS:
            continue
        _append_note(notes, seen, message.content)

    troubleshooting_response = state.get("troubleshooting_response")
    if troubleshooting_response and troubleshooting_response.get("next_action") in {
        action.value for action in TROUBLESHOOTING_HISTORY_ACTIONS
    }:
        _append_note(notes, seen, troubleshooting_response.get("response_text"))
    return notes


def _build_ticket_creation_response(
    *,
    state: dict,
    ticket_id: str,
    llm_client,
    request: ChatMessageRequest,
    classification: IntentClassification,
    history: list[ConversationMessage],
) -> str:
    intro_text = (state.get("ticket_response_intro_text") or "").strip()
    if not intro_text:
        intro_text = llm_client.generate_ticket_creation_intro(
            request=request,
            classification=classification,
            history=history,
            troubleshooting_rounds=state.get("troubleshooting_rounds", 0),
            support_scope_status=state.get("support_scope_status"),
            escalate_immediately=bool(state.get("safety_assessment", {}).get("escalate_immediately")),
            force_ticket_creation=bool(state.get("force_ticket_creation")),
        ).strip()
    confirmation = f"Support ticket `{ticket_id}` has been created successfully."
    if not intro_text:
        return confirmation
    return f"{intro_text}\n\n{confirmation}"


def _append_note(notes: list[str], seen: set[str], content: str | None) -> None:
    if not content:
        return
    normalized = _normalize_note(content)
    if not normalized:
        return
    dedupe_key = normalized.casefold()
    if dedupe_key in seen:
        return
    seen.add(dedupe_key)
    notes.append(content.strip())


def _normalize_note(content: str | None) -> str | None:
    if not content:
        return None
    normalized = re.sub(r"\s+", " ", content).strip()
    return normalized or None


def _build_issue_summary(state: dict, request: ChatMessageRequest) -> str:
    current_message = _normalize_note(request.message) or request.message
    if not _is_generic_escalation_request(current_message):
        return state.get("user_query") or current_message

    history = [ConversationMessage.model_validate(item) for item in state.get("source_history", [])]
    for message in reversed(history):
        if message.role != ConversationRole.user:
            continue
        candidate = _normalize_note(message.content)
        if not candidate or _is_generic_escalation_request(candidate) or _is_evidence_only_reply(candidate):
            continue
        return candidate

    return state.get("user_query") or current_message


def _is_generic_escalation_request(text: str) -> bool:
    normalized = re.sub(r"[.!?]+$", "", text.strip().lower())
    patterns = (
        r"please create (?:the )?ticket",
        r"create (?:the )?ticket",
        r"please escalate(?: this| it)?",
        r"escalate(?: this| it)?",
        r"log (?:a )?(?:case|ticket)",
        r"go ahead",
        r"do it",
    )
    return any(re.fullmatch(pattern, normalized) for pattern in patterns)


def _is_evidence_only_reply(text: str) -> bool:
    lowered = text.lower()
    evidence_terms = (
        "serial number",
        "firmware version",
        "timestamp",
        "backup loads",
        "recent changes",
        "app version",
        "portal version",
    )
    issue_terms = (
        "fault",
        "error",
        "alarm",
        "trip",
        "restart",
        "display",
        "inverter",
        "battery",
        "monitor",
    )
    return any(term in lowered for term in evidence_terms) and not any(term in lowered for term in issue_terms)
