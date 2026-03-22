import unittest

from app.core.conversation_context import merge_evidence_from_conversation
from app.graph.workflow import WorkflowDependencies, build_workflow
from app.models.conversation import (
    ChatMessageRequest,
    ConversationMessage,
    ConversationRole,
    DeviceInfo,
    DeviceType,
    EvidencePack,
    IntentClassification,
    IntentType,
    RetrievedDocument,
    SupportScopeStatus,
    TroubleshootingAction,
    TroubleshootingResponse,
    UnsupportedReason,
)
from app.models.ticket import TicketResponse
from app.services.retrieval_service import RetrievalService
from app.services.ticket_service import TicketService
from app.services.validation_service import ValidationService


class FakeLLMClient:
    def __init__(self):
        self.extract_evidence_calls = 0

    def classify_intent(self, request, history=None):
        message = request.message
        device_info = request.device_info
        lowered = message.lower()
        user_query = message
        if history:
            recent_context = " ".join(item.content for item in history[-2:])
            user_query = f"{message} | context: {recent_context}"
        history_text = " ".join(item.content.lower() for item in (history or []))
        missing_info = []
        system_message = None
        has_domain_context = any(
            term in f"{history_text} {lowered}" for term in ("inverter", "battery", "pv", "monitor", "error", "fault")
        )
        is_brief = 0 < len(lowered.split()) <= 4
        support_scope_status = SupportScopeStatus.unknown
        unsupported_reason = None
        if any(term in lowered for term in ("80 kw", "80kw", "over 30 kw", "greater than 30 kw", "above 30 kw")):
            support_scope_status = SupportScopeStatus.unsupported
            unsupported_reason = UnsupportedReason.site_capacity_exceeded
        elif "industrial" in lowered or "major commercial" in lowered:
            support_scope_status = SupportScopeStatus.unsupported
            unsupported_reason = UnsupportedReason.industrial_site
        elif any(term in lowered for term in ("utility-scale", "utility scale", "embedded network")):
            support_scope_status = SupportScopeStatus.unsupported
            unsupported_reason = UnsupportedReason.utility_scale_or_embedded_network
        elif any(term in lowered for term in ("home", "residential", "home use", "my system", "our system", "owner", "customer owner")):
            support_scope_status = SupportScopeStatus.supported
        if "ticket" in lowered:
            intent = IntentType.escalate
        elif is_brief and not has_domain_context:
            intent = IntentType.general_question
            missing_info.append("issue_or_question_details")
            system_message = (
                "## Delta Support\n\n"
                f"Please share the technical details for: {message}\n\n"
                "- Issue description\n"
                "- Alarm or error text\n"
                "- Model number"
            )
        else:
            intent = IntentType.troubleshoot
        return IntentClassification(
            intent=intent,
            device_type=device_info.device_type if device_info else DeviceType.inverter,
            user_query=user_query,
            error_code="E031" if "E031" in message else None,
            model_number=device_info.model_number if device_info else None,
            risk_flags=[],
            missing_info=missing_info,
            support_scope_status=support_scope_status,
            unsupported_reason=unsupported_reason,
            missing_scope_fields=[],
            system_message=system_message,
        )

    def generate_troubleshooting_response(self, message, retrieved_docs, classification):
        citations = [doc.doc_id for doc in retrieved_docs[:1]]
        response_text = "Follow the documented restart sequence from the retrieved Delta KB article."
        if "still the same" in message.lower() and "context:" in message:
            response_text = (
                "Based on the prior conversation, continue from the documented restart sequence and share the new fault state."
            )
        return TroubleshootingResponse(
            response_text=response_text,
            citations=citations,
            next_action=TroubleshootingAction.continue_troubleshooting,
        )

    def create_embedding(self, text, dimensions=None):
        return None

    def extract_evidence(self, request, history=None):
        self.extract_evidence_calls += 1
        return merge_evidence_from_conversation(
            current_message=request.message,
            request_evidence=request.evidence_pack,
            history=history or [],
        )


class FakeSearchAdapter:
    def __init__(self):
        self.last_query = None
        self.last_filters = None

    def search(self, query, size=5, filters=None):
        self.last_query = query
        self.last_filters = filters or {}
        return [
            RetrievedDocument(
                doc_id="doc-1",
                title="Troubleshooting Guide",
                product="inverter",
                model=filters.get("model") if filters else None,
                error_code=filters.get("error_code") if filters else None,
                section_title="Restart Procedure",
                content="Verify the inverter display, acknowledge the alarm, and perform the documented restart sequence.",
                score=0.82,
            )
        ]


class FakeTicketAdapter:
    configured = False

    def __init__(self):
        self.last_payload = None

    def create_ticket(self, payload):
        self.last_payload = payload
        return TicketResponse(
            ticket_id="MOCK-12345678",
            status="mock_created",
            message="Ticket created in test mode.",
        )


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.llm_client = FakeLLMClient()
        self.search_adapter = FakeSearchAdapter()
        self.ticket_adapter = FakeTicketAdapter()
        dependencies = WorkflowDependencies(
            llm_client=self.llm_client,
            retrieval_service=RetrievalService(self.search_adapter),
            validation_service=ValidationService(),
            ticket_service=TicketService(self.ticket_adapter),
            retrieval_top_k=5,
        )
        self.workflow = build_workflow(dependencies)

    def test_troubleshooting_path_returns_grounded_response(self):
        request = ChatMessageRequest(
            message="My inverter shows E031 after restart",
            device_info=DeviceInfo(device_type=DeviceType.inverter, model_number="M100A"),
            evidence_pack=EvidencePack(
                user_role="customer_owner",
                ownership_verified=True,
            ),
        )
        state = self.workflow.invoke({"request": request.model_dump(mode="json")})
        self.assertEqual(state["user_query"], "My inverter shows E031 after restart")
        self.assertEqual(state["current_phase"], "troubleshooting")
        self.assertEqual(state["next_action"], "continue_troubleshooting")
        self.assertEqual(state["citations"], ["doc-1"])
        self.assertEqual(self.llm_client.extract_evidence_calls, 0)

    def test_unknown_scope_continues_to_troubleshooting(self):
        request = ChatMessageRequest(
            message="My inverter shows E031 after restart",
            device_info=DeviceInfo(device_type=DeviceType.inverter, model_number="M100A"),
        )
        state = self.workflow.invoke({"request": request.model_dump(mode="json")})
        self.assertEqual(state["current_phase"], "troubleshooting")
        self.assertEqual(state["next_action"], "continue_troubleshooting")
        self.assertNotIn("system_message", state)
        self.assertEqual(state["missing_scope_fields"], [])

    def test_initial_message_scope_evidence_skips_site_eligibility_prompt(self):
        request = ChatMessageRequest(
            message="H10E delta invertor home use 10kw has error",
        )
        state = self.workflow.invoke({"request": request.model_dump(mode="json")})
        self.assertEqual(state["current_phase"], "troubleshooting")
        self.assertNotIn("## Site Eligibility Check", state["response_text"])
        self.assertEqual(state["support_scope_status"], "supported")
        self.assertEqual(state["next_action"], "continue_troubleshooting")

    def test_safety_path_collects_missing_evidence(self):
        request = ChatMessageRequest(
            message="There is smoke coming from the inverter enclosure",
            device_info=DeviceInfo(device_type=DeviceType.inverter, model_number="M100A"),
            evidence_pack=EvidencePack(
                user_role="customer_owner",
                ownership_verified=True,
            ),
        )
        state = self.workflow.invoke({"request": request.model_dump(mode="json")})
        self.assertEqual(state["current_phase"], "evidence_collection")
        self.assertIn("serial_number", state["missing_fields"])
        self.assertEqual(state["next_action"], "collect_evidence")

    def test_escalation_with_complete_evidence_creates_ticket(self):
        request = ChatMessageRequest(
            message="Please create a ticket for inverter fault E031",
            request_ticket=True,
            device_info=DeviceInfo(device_type=DeviceType.inverter, model_number="M100A"),
            evidence_pack=EvidencePack(
                user_role="customer_owner",
                ownership_verified=True,
                inverter_model="M100A",
                serial_number="SN12345",
                firmware_version="1.0.4",
                error_code="E031",
                timestamp="2026-03-07T09:30:00Z",
                backup_loads_present=False,
                recent_changes="No recent changes",
            ),
        )
        state = self.workflow.invoke({"request": request.model_dump(mode="json")})
        self.assertEqual(state["current_phase"], "ticket_creation")
        self.assertEqual(state["ticket_response"]["status"], "mock_created")
        self.assertTrue(self.ticket_adapter.last_payload.message_html.startswith("<div>"))
        self.assertIn("Evidence pack", self.ticket_adapter.last_payload.message_html)
        self.assertIn("Serial number", self.ticket_adapter.last_payload.message_html)
        self.assertEqual(self.llm_client.extract_evidence_calls, 1)

    def test_removed_fields_are_ignored(self):
        evidence = EvidencePack.model_validate(
            {
                "legacy_scope_field": "residential",
                "legacy_system_field": 10,
                "legacy_topology_field": "ac_coupled",
                "legacy_phase_field": "single_phase",
                "legacy_environment_field": "Dry, 32C ambient",
                "user_role": "customer_owner",
            }
        )
        self.assertEqual(evidence.user_role, "customer_owner")
        self.assertNotIn("legacy_scope_field", evidence.model_dump())
        self.assertNotIn("legacy_system_field", evidence.model_dump())
        self.assertNotIn("legacy_topology_field", evidence.model_dump())
        self.assertNotIn("legacy_phase_field", evidence.model_dump())
        self.assertNotIn("legacy_environment_field", evidence.model_dump())

    def test_explicit_unsupported_site_collects_evidence_without_retrieval(self):
        request = ChatMessageRequest(
            message="Please create a ticket for the 80 kW industrial inverter site with fault E031",
            request_ticket=True,
            device_info=DeviceInfo(device_type=DeviceType.inverter, model_number="M100A"),
            evidence_pack=EvidencePack(
                user_role="licensed_installer",
                ownership_verified=True,
            ),
        )
        state = self.workflow.invoke({"request": request.model_dump(mode="json")})
        self.assertEqual(state["current_phase"], "evidence_collection")
        self.assertEqual(state["support_scope_status"], "unsupported")
        self.assertNotIn("retrieved_docs", state)

    def test_message_only_unsupported_site_skips_troubleshooting(self):
        request = ChatMessageRequest(
            message="This is an 80 kW industrial inverter site with fault E031",
            device_info=DeviceInfo(device_type=DeviceType.inverter, model_number="M100A"),
            evidence_pack=EvidencePack(
                user_role="licensed_installer",
                ownership_verified=True,
            ),
        )
        state = self.workflow.invoke({"request": request.model_dump(mode="json")})
        self.assertEqual(state["current_phase"], "evidence_collection")
        self.assertEqual(state["support_scope_status"], "unsupported")

    def test_greeting_message_returns_friendly_domain_redirect(self):
        request = ChatMessageRequest(message="Hi")
        state = self.workflow.invoke({"request": request.model_dump(mode="json")})
        self.assertEqual(state["current_phase"], "intake")
        self.assertEqual(state["next_action"], "ask_question")
        self.assertEqual(state["citations"], [])
        self.assertIn("## Delta Support", state["response_text"])
        self.assertIn("technical details for: Hi", state["response_text"])
        self.assertIn("system_message", state)
        self.assertEqual(state["user_query"], "Hi")
        self.assertNotIn("retrieved_docs", state)
        self.assertNotIn("ticket_response", state)

    def test_follow_up_message_uses_prior_history_context(self):
        request = ChatMessageRequest(message="Still the same after the restart")
        history = [
            ConversationMessage(
                role=ConversationRole.user,
                content="My inverter shows E031 after restart",
            ),
            ConversationMessage(
                role=ConversationRole.assistant,
                content="Follow the documented restart sequence from the retrieved Delta KB article.",
                escalation_active=False,
                evidence_snapshot=EvidencePack(
                    user_role="customer_owner",
                    ownership_verified=True,
                    inverter_model="M100A",
                    serial_number="SN12345",
                    firmware_version="1.0.4",
                    error_code="E031",
                    timestamp="2026-03-07T09:30:00Z",
                    backup_loads_present=False,
                    recent_changes="No recent changes",
                ),
            ),
        ]
        state = self.workflow.invoke(
            {
                "request": request.model_dump(mode="json"),
                "history": [message.model_dump(mode="json") for message in history],
            }
        )
        self.assertEqual(state["current_phase"], "troubleshooting")
        self.assertEqual(state["next_action"], "continue_troubleshooting")
        self.assertIn("prior conversation", state["response_text"])
        self.assertIn("context:", state["user_query"])
        self.assertEqual(state.get("history"), [])
        self.assertIn("My inverter shows E031 after restart", self.search_adapter.last_query)
        self.assertEqual(state["merged_evidence_pack"]["serial_number"], "SN12345")

    def test_escalation_follow_up_stays_on_evidence_collection(self):
        request = ChatMessageRequest(message="Serial number SN12345")
        history = [
            ConversationMessage(
                role=ConversationRole.user,
                content="Please escalate this inverter issue",
            ),
            ConversationMessage(
                role=ConversationRole.assistant,
                content="To create the support ticket, please provide the missing evidence.",
                intent=IntentType.escalate,
                next_action=TroubleshootingAction.collect_evidence,
                escalation_active=True,
                evidence_snapshot=EvidencePack(
                    user_role="customer_owner",
                    ownership_verified=True,
                    error_code="E031",
                ),
            ),
        ]
        state = self.workflow.invoke(
            {
                "request": request.model_dump(mode="json"),
                "history": [message.model_dump(mode="json") for message in history],
            }
        )
        self.assertEqual(state["current_phase"], "evidence_collection")
        self.assertEqual(state["classification"]["intent"], IntentType.escalate.value)
        self.assertTrue(state["escalation_active"])
        self.assertTrue(state["previous_escalation_active"])
        self.assertNotIn("serial_number", state["missing_fields"])
        self.assertEqual(state["next_action"], "collect_evidence")

    def test_escalation_follow_up_uses_history_before_asking_again(self):
        request = ChatMessageRequest(message="Please create the ticket")
        history = [
            ConversationMessage(
                role=ConversationRole.user,
                content="Here are the missing details: serial number SN12345, firmware version 1.0.4, timestamp 2026-03-07T09:30:00Z",
            ),
            ConversationMessage(
                role=ConversationRole.assistant,
                content="I still need a few more items before I can create the ticket.",
                intent=IntentType.escalate,
                next_action=TroubleshootingAction.collect_evidence,
                escalation_active=True,
                evidence_snapshot=EvidencePack(
                    user_role="customer_owner",
                    ownership_verified=True,
                    inverter_model="M100A",
                    error_code="E031",
                    backup_loads_present=False,
                    recent_changes="No recent changes",
                ),
            ),
        ]
        state = self.workflow.invoke(
            {
                "request": request.model_dump(mode="json"),
                "history": [message.model_dump(mode="json") for message in history],
            }
        )
        self.assertEqual(state["current_phase"], "ticket_creation")
        self.assertEqual(state["ticket_response"]["status"], "mock_created")
        self.assertFalse(state["escalation_active"])

    def test_escalation_creates_ticket_when_evidence_ratio_reaches_threshold(self):
        request = ChatMessageRequest(message="Serial number SN12345")
        history = [
            ConversationMessage(
                role=ConversationRole.user,
                content="Please escalate this inverter issue",
            ),
            ConversationMessage(
                role=ConversationRole.assistant,
                content="Please provide the remaining evidence so I can create the ticket.",
                intent=IntentType.escalate,
                next_action=TroubleshootingAction.collect_evidence,
                escalation_active=True,
                evidence_snapshot=EvidencePack(
                    user_role="customer_owner",
                    ownership_verified=True,
                    inverter_model="M100A",
                    firmware_version="1.0.4",
                    error_code="E031",
                    timestamp="2026-03-07T09:30:00Z",
                    recent_changes="No recent changes",
                ),
            ),
        ]
        state = self.workflow.invoke(
            {
                "request": request.model_dump(mode="json"),
                "history": [message.model_dump(mode="json") for message in history],
            }
        )
        self.assertEqual(state["current_phase"], "ticket_creation")
        self.assertGreaterEqual(state["evidence_completion_ratio"], 0.7)
        self.assertEqual(state["ticket_response"]["status"], "mock_created")


if __name__ == "__main__":
    unittest.main()
