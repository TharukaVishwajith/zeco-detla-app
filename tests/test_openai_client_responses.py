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
    TroubleshootingResponse,
    TroubleshootingResponseSource,
)
from app.services.validation_service import ValidationService


class ScriptedOpenAIClient(OpenAIClient):
    def __init__(self, payloads):
        super().__init__(
            api_key=None,
            chat_model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
        )
        self.client = object()
        self.payloads = payloads
        self.seen_agents: list[str] = []

    def _invoke_agent_json(self, agent_name: str, system_prompt: str, user_prompt: str) -> dict | None:  # noqa: ARG002
        self.seen_agents.append(agent_name)
        values = self.payloads.get(agent_name, [])
        if not values:
            raise RuntimeError(f"unexpected agent invocation: {agent_name}")
        value = values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


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

    def test_grounded_fallback_without_docs_is_last_resort_safe_message(self):
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
        self.assertIn("## Safe next step", response.response_text)
        self.assertIn("I could not complete a reliable support answer this turn.", response.response_text)
        self.assertEqual(response.response_source, TroubleshootingResponseSource.grounded_kb)

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

    def test_grounded_agent_handoff_uses_internal_fallback_answer(self):
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
        client = ScriptedOpenAIClient(
            payloads={
                "troubleshooting": [
                    {
                        "response_text": "KB coverage insufficient.",
                        "citations": [],
                        "next_action": "ask_question",
                        "handoff_to_fallback": True,
                        "response_source": "grounded_kb",
                    }
                ],
                "fallback_troubleshooting": [
                    {
                        "response_text": "## Check the restart state\n\n1. Wait 60 seconds.\n2. Restart the inverter once.\n\nReply with the display text after restart.",
                        "citations": ["doc-1"],
                        "next_action": "continue_troubleshooting",
                        "handoff_to_fallback": False,
                        "response_source": "internal_fallback",
                    }
                ],
            }
        )
        response = client.generate_troubleshooting_response(
            message="Still showing E031",
            retrieved_docs=retrieved_docs,
            classification=classification,
            validation_service=ValidationService(),
        )

        self.assertEqual(response.next_action, TroubleshootingAction.continue_troubleshooting)
        self.assertEqual(response.response_source, TroubleshootingResponseSource.internal_fallback)
        self.assertEqual(response.citations, [])
        self.assertIn("## Check the restart state", response.response_text)
        self.assertEqual(client.seen_agents, ["troubleshooting", "fallback_troubleshooting"])

    def test_retrieval_empty_uses_internal_fallback_when_available(self):
        classification = IntentClassification(
            intent=IntentType.troubleshoot,
            device_type=DeviceType.inverter,
            support_scope_status=SupportScopeStatus.unknown,
        )
        client = ScriptedOpenAIClient(
            payloads={
                "fallback_troubleshooting": [
                    {
                        "response_text": "## Check the front panel\n\n1. Confirm the inverter display is on.\n2. Note any fault text.\n\nReply with the exact text shown.",
                        "citations": [],
                        "next_action": "ask_question",
                        "handoff_to_fallback": False,
                        "response_source": "internal_fallback",
                    }
                ]
            }
        )

        response = client.generate_troubleshooting_response(
            message="not working",
            retrieved_docs=[],
            classification=classification,
            validation_service=ValidationService(),
        )

        self.assertEqual(response.response_source, TroubleshootingResponseSource.internal_fallback)
        self.assertEqual(client.seen_agents, ["fallback_troubleshooting"])

    def test_fallback_failure_returns_last_resort_response(self):
        classification = IntentClassification(
            intent=IntentType.troubleshoot,
            device_type=DeviceType.inverter,
            support_scope_status=SupportScopeStatus.unknown,
        )
        client = ScriptedOpenAIClient(payloads={"fallback_troubleshooting": [RuntimeError("boom")]})

        response = client.generate_troubleshooting_response(
            message="not working",
            retrieved_docs=[],
            classification=classification,
            validation_service=ValidationService(),
        )

        self.assertEqual(response.next_action, TroubleshootingAction.ask_question)
        self.assertIn("## Safe next step", response.response_text)

    def test_validation_allows_uncited_internal_fallback_but_rejects_uncited_grounded(self):
        validation_service = ValidationService()
        docs = [
            RetrievedDocument(
                doc_id="doc-1",
                content="Restart the inverter safely.",
            )
        ]

        grounded = TroubleshootingResponse(
            response_text="Use the restart procedure.",
            citations=[],
            next_action=TroubleshootingAction.continue_troubleshooting,
            response_source=TroubleshootingResponseSource.grounded_kb,
        )
        fallback = TroubleshootingResponse(
            response_text="Use the restart procedure.",
            citations=[],
            next_action=TroubleshootingAction.continue_troubleshooting,
            response_source=TroubleshootingResponseSource.internal_fallback,
        )

        grounded_valid, grounded_errors = validation_service.validate_troubleshooting_response(grounded, docs)
        fallback_valid, fallback_errors = validation_service.validate_troubleshooting_response(fallback, docs)

        self.assertFalse(grounded_valid)
        self.assertIn("grounded response must include citations", grounded_errors)
        self.assertTrue(fallback_valid)
        self.assertEqual(fallback_errors, [])

    def test_validation_rejects_grounded_insufficient_info_without_handoff(self):
        validation_service = ValidationService()
        docs = [
            RetrievedDocument(
                doc_id="doc-1",
                content="Restart the inverter safely.",
            )
        ]

        grounded = TroubleshootingResponse(
            response_text="I need more information before I can determine the next step.",
            citations=["doc-1"],
            next_action=TroubleshootingAction.ask_question,
            handoff_to_fallback=False,
            response_source=TroubleshootingResponseSource.grounded_kb,
        )

        grounded_valid, grounded_errors = validation_service.validate_troubleshooting_response(grounded, docs)

        self.assertFalse(grounded_valid)
        self.assertIn("grounded response admits insufficient information without fallback handoff", grounded_errors)


if __name__ == "__main__":
    unittest.main()
