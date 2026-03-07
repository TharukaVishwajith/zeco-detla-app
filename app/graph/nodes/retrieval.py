from app.models.conversation import ChatMessageRequest, IntentClassification


def build_retrieval_node(retrieval_service, default_top_k: int):
    def retrieval_node(state: dict) -> dict:
        request = ChatMessageRequest.model_validate(state["request"])
        classification = IntentClassification.model_validate(state["classification"])
        top_k = request.top_k or default_top_k
        documents = retrieval_service.retrieve(query=request.message, classification=classification, top_k=top_k)
        return {
            "retrieved_docs": [document.model_dump(mode="json") for document in documents],
            "current_phase": "retrieval",
        }

    return retrieval_node

