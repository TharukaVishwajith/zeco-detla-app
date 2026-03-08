from app.models.conversation import ChatMessageRequest


def build_ticket_creation_node(ticket_service):
    def ticket_creation_node(state: dict) -> dict:
        request = ChatMessageRequest.model_validate(state["request"])
        troubleshooting_steps = []
        troubleshooting_response = state.get("troubleshooting_response")
        if troubleshooting_response:
            troubleshooting_steps.append(troubleshooting_response["response_text"])
        escalation_reason = state.get("safety_assessment", {}).get("reason")
        ticket_response = ticket_service.create_from_graph(
            request=request,
            troubleshooting_steps=troubleshooting_steps,
            escalation_reason=escalation_reason,
            merged_evidence_pack=state.get("merged_evidence_pack", {}),
            unsupported_reason=state.get("unsupported_reason"),
            missing_artifacts=state.get("missing_artifacts", []),
        )
        response_text = f"Support ticket {ticket_response.ticket_id} created successfully."
        return {
            "ticket_response": ticket_response.model_dump(mode="json"),
            "response_text": response_text,
            "next_action": "escalate",
            "current_phase": "ticket_creation",
        }

    return ticket_creation_node
