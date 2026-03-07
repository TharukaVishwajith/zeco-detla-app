from app.models.conversation import IntentClassification, RetrievedDocument


class RetrievalService:
    def __init__(self, adapter):
        self.adapter = adapter

    def retrieve(
        self,
        query: str,
        classification: IntentClassification,
        top_k: int,
    ) -> list[RetrievedDocument]:
        filters = {}
        if classification.device_type.value != "unknown":
            filters["product"] = classification.device_type.value
        if classification.model_number:
            filters["model"] = classification.model_number
        if classification.error_code:
            filters["error_code"] = classification.error_code
        return self.adapter.search(query=query, size=top_k)

