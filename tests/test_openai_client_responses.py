import unittest

from app.adapters.openai_client import OpenAIClient
from app.models.conversation import (
    ChatMessageRequest,
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

    def test_non_greeting_question_passes_through_without_clarification(self):
        classification = self.client.classify_intent(
            ChatMessageRequest(message="What does standby mode mean?")
        )

        self.assertEqual(classification.intent, IntentType.general_question)
        self.assertIsNone(classification.system_message)
        self.assertNotIn("issue_or_question_details", classification.missing_info)

    def test_greeting_only_message_gets_clarification_prompt(self):
        classification = self.client.classify_intent(ChatMessageRequest(message="Hi"))

        self.assertEqual(classification.intent, IntentType.general_question)
        self.assertIn("issue_or_question_details", classification.missing_info)
        self.assertIn("## Tell me a bit more", classification.system_message or "")

    def test_grounded_fallback_without_docs_answers_directly(self):
        classification = IntentClassification(
            intent=IntentType.troubleshoot,
            device_type=DeviceType.inverter,
            support_scope_status=SupportScopeStatus.unknown,
            error_code="E031",
        )

        response = self.client._grounded_fallback_response(  # noqa: SLF001 - validating helper output directly
            message="not working",
            retrieved_docs=[],
            classification=classification,
        )

        self.assertEqual(response.next_action, TroubleshootingAction.continue_troubleshooting)
        self.assertTrue(response.counts_as_troubleshooting_round)
        self.assertIn("## First, check the E031 condition", response.response_text)
        self.assertIn("1. Confirm the device is powered", response.response_text)
        self.assertNotIn("Reply with the exact display text or LED state after these checks.", response.response_text)

    def test_fallback_evidence_collection_response_keeps_evidence_optional(self):
        response_text = self.client._fallback_evidence_collection_response(  # noqa: SLF001 - validating helper output directly
            merged_evidence=EvidencePack(error_code="E031"),
            missing_fields=["serial_number", "timestamp"],
            support_scope_status=SupportScopeStatus.unknown.value,
            safety_assessment={"escalate_immediately": False},
        )

        self.assertIn("## Support Ticket Details", response_text)
        self.assertIn("I am ready to create the support ticket.", response_text)
        self.assertIn("If available, please send these remaining details", response_text)
        self.assertIn("If you do not have them, tell me and I will continue with the information already gathered.", response_text)

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
        self.assertTrue(response.counts_as_troubleshooting_round)
        self.assertIn("## First, check the E031 condition", response.response_text)
        self.assertIn("1. Check the display, acknowledge the alarm, and run the restart sequence.", response.response_text)
        self.assertNotIn("Reply with the exact display message or LED state after this step.", response.response_text)

    def test_resolved_troubleshooting_response_closes_conversation(self):
        response = self.client.generate_resolved_troubleshooting_response()

        self.assertEqual(response.next_action, TroubleshootingAction.resolved)
        self.assertEqual(response.citations, [])
        self.assertFalse(response.counts_as_troubleshooting_round)
        self.assertIn("Glad to hear the issue is resolved", response.response_text)
        self.assertNotIn("support ticket", response.response_text.lower())

    def test_fallback_ticket_creation_intro_mentions_escalation_and_ticket(self):
        request = ChatMessageRequest(message="The inverter still fails after several attempts")
        classification = IntentClassification(
            intent=IntentType.troubleshoot,
            device_type=DeviceType.inverter,
            support_scope_status=SupportScopeStatus.supported,
        )

        response_text = self.client.generate_ticket_creation_intro(
            request=request,
            classification=classification,
            history=[],
            troubleshooting_rounds=5,
            support_scope_status=SupportScopeStatus.supported.value,
            escalate_immediately=False,
            force_ticket_creation=True,
        )

        self.assertIn("## Support Escalation", response_text)
        self.assertIn("troubleshooting steps", response_text.lower())
        self.assertIn("support ticket", response_text.lower())


if __name__ == "__main__":
    unittest.main()
