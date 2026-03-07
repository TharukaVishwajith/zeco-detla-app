from app.models.conversation import RetrievedDocument, TroubleshootingResponse


UNSAFE_INSTRUCTION_TERMS = {"open electrical enclosures", "bypass breakers", "rewire", "disable protections"}


class ValidationService:
    def validate_troubleshooting_response(
        self,
        response: TroubleshootingResponse,
        retrieved_docs: list[RetrievedDocument],
    ) -> tuple[bool, list[str]]:
        errors: list[str] = []
        valid_doc_ids = {doc.doc_id for doc in retrieved_docs}
        if not response.response_text.strip():
            errors.append("response_text is empty")
        if retrieved_docs and not response.citations:
            errors.append("grounded response must include citations")
        if any(citation not in valid_doc_ids for citation in response.citations):
            errors.append("response contains citations not present in retrieved documents")
        lowered = response.response_text.lower()
        if any(term in lowered for term in UNSAFE_INSTRUCTION_TERMS):
            errors.append("response contains unsafe operational guidance")
        return (len(errors) == 0, errors)

