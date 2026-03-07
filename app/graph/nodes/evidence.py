from app.models.conversation import ChatMessageRequest


def build_evidence_collection_node():
    def evidence_collection_node(state: dict) -> dict:
        request = ChatMessageRequest.model_validate(state["request"])
        missing_fields = request.evidence_pack.missing_required_fields()
        safety_assessment = state.get("safety_assessment", {})

        if missing_fields:
            if safety_assessment.get("escalate_immediately"):
                response_text = (
                    "A safety hazard was detected. Do not continue operating the equipment. "
                    "Before I can create the support ticket, please provide: "
                    + ", ".join(missing_fields)
                    + "."
                )
            else:
                response_text = "To create the support ticket, please provide: " + ", ".join(missing_fields) + "."
            next_action = "collect_evidence"
        else:
            response_text = "All required evidence is present. Creating the support ticket now."
            next_action = "create_ticket"

        return {
            "missing_fields": missing_fields,
            "response_text": response_text,
            "next_action": next_action,
            "current_phase": "evidence_collection",
        }

    return evidence_collection_node

