from app.models.conversation import ChatMessageRequest, ConversationMessage


def build_intake_node(llm_client):
    def intake_node(state: dict) -> dict:
        request = ChatMessageRequest.model_validate(state["request"])
        history = [ConversationMessage.model_validate(item) for item in state.get("history", [])]
        classification = llm_client.classify_intent(
            message=request.message,
            device_info=request.device_info,
            history=history,
        )
        output = {
            "user_query": classification.user_query or request.message,
            # History is consumed in intake for contextual query generation; downstream nodes should use user_query.
            "history": [],
            "classification": classification.model_dump(mode="json"),
            "current_phase": "intake",
        }
        system_message = classification.system_message
        if system_message:
            output["system_message"] = system_message
            output["response_text"] = system_message
            output["next_action"] = "ask_question"
            output["citations"] = []
        return output

    return intake_node
