from collections.abc import Mapping
from typing import Any

from app.models.conversation import ConversationState, TroubleshootingAction


def derive_conversation_state(state: Mapping[str, Any]) -> ConversationState:
    next_action = state.get("next_action")
    current_phase = state.get("current_phase")

    if next_action == TroubleshootingAction.resolved.value:
        return ConversationState.resolved
    if state.get("ticket_response"):
        return ConversationState.ticket_created
    if current_phase == "evidence_collection" or next_action in {
        TroubleshootingAction.collect_evidence.value,
        TroubleshootingAction.escalate.value,
        "create_ticket",
    }:
        return ConversationState.awaiting_evidence
    if current_phase == "intake" and state.get("system_message"):
        return ConversationState.needs_clarification
    return ConversationState.troubleshooting
