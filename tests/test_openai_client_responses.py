import unittest

from app.adapters.openai_client import OpenAIClient
from app.models.conversation import (
    DeviceType,
    EvidencePack,
    IntentClassification,
    IntentType,
    RetrievedDocument,
    SupportScopeStatus,
    TroubleshootingAction,
)


class OpenAIClientResponseTests(unittest.TestCase):
    def setUp(self):
        self.client = OpenAIClient(
            api_key=None,
            chat_model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
        )

    def test_heuristic_system_message_is_simple_and_user_friendly(self):
        message = self.client._heuristic_system_message("hi")  # noqa: SLF001 - validating helper output directly

        self.assertIn("## Tell me a bit more", message)
        self.assertIn("What issue you are seeing", message)
        self.assertIn("The exact alarm or error text", message)
        self.assertNotIn("technical details for", message)

    def test_grounded_fallback_without_docs_asks_for_minimum_details(self):
        classification = IntentClassification(
            intent=IntentType.troubleshoot,
            device_type=DeviceType.inverter,
            support_scope_status=SupportScopeStatus.unknown,
        )

        response = self.client._grounded_fallback_response(  # noqa: SLF001 - validating helper output directly
            message="not working",
            retrieved_docs=[],
            classification=classification,
        )

        self.assertEqual(response.next_action, TroubleshootingAction.ask_question)
        self.assertIn("## Let’s narrow this down", response.response_text)
        self.assertIn("1. The model number", response.response_text)
        self.assertIn("2. The exact error code or fault text", response.response_text)
        self.assertIn("Reply with those details", response.response_text)

    def test_fallback_evidence_collection_response_keeps_evidence_optional(self):
        response_text = self.client._fallback_evidence_collection_response(  # noqa: SLF001 - validating helper output directly
            merged_evidence=EvidencePack(error_code="E031"),
            missing_fields=["serial_number", "timestamp"],
            support_scope_status=SupportScopeStatus.unknown.value,
            safety_assessment={"escalate_immediately": False},
        )

        self.assertIn("I can create the support ticket for you.", response_text)
        self.assertIn("If you have any of these additional details", response_text)
        self.assertIn("If not, tell me and I will proceed with the information already gathered.", response_text)

    def test_grounded_fallback_with_docs_uses_simple_step_format(self):
        classification = IntentClassification(
            intent=IntentType.troubleshoot,
            device_type=DeviceType.inverter,
            error_code="E031",
            support_scope_status=SupportScopeStatus.supported,
        )
        retrieved_docs = [
            RetrievedDocument(
                doc_id="doc-1",
                title="Troubleshooting Guide",
                section_title="Restart Procedure",
                content="Check the display, acknowledge the alarm, and run the restart sequence",
            )
        ]

        response = self.client._grounded_fallback_response(  # noqa: SLF001 - validating helper output directly
            message="Still showing E031",
            retrieved_docs=retrieved_docs,
            classification=classification,
        )

        self.assertEqual(response.next_action, TroubleshootingAction.continue_troubleshooting)
        self.assertEqual(response.citations, ["doc-1"])
        self.assertIn("## First, check the E031 condition", response.response_text)
        self.assertIn("1. Check the display, acknowledge the alarm, and run the restart sequence.", response.response_text)
        self.assertIn("Reply with the exact display message or LED state", response.response_text)


if __name__ == "__main__":
    unittest.main()
