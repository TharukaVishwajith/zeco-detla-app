from app.models.conversation import ChatMessageRequest


def build_intake_node(llm_client):
    def intake_node(state: dict) -> dict:
        request = ChatMessageRequest.model_validate(state["request"])
        classification = llm_client.classify_intent(message=request.message, device_info=request.device_info)
        return {
            "classification": classification.model_dump(mode="json"),
            "current_phase": "intake",
        }

    return intake_node

