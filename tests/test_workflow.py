import unittest

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
        missing_scope_fields = []
        has_domain_context = any(
            term in f"{history_text} {lowered}" for term in ("inverter", "battery", "pv", "monitor", "error", "fault")
        )
        is_brief = 0 < len(lowered.split()) <= 4
        support_scope_status = SupportScopeStatus.supported
        unsupported_reason = None
        if request.evidence_pack.system_size_kw == "80" or "80 kw" in lowered:
            support_scope_status = SupportScopeStatus.unsupported
            unsupported_reason = UnsupportedReason.site_capacity_exceeded
        elif "industrial" in lowered:
            support_scope_status = SupportScopeStatus.unsupported
            unsupported_reason = UnsupportedReason.industrial_site
        elif not all(
            (
                request.evidence_pack.site_type,
                request.evidence_pack.system_size_kw,
                request.evidence_pack.user_role,
                request.evidence_pack.ownership_verified is not None,
            )
        ):
            support_scope_status = SupportScopeStatus.unknown
            if not request.evidence_pack.site_type:
                missing_scope_fields.append("site_type")
            if not request.evidence_pack.system_size_kw:
                missing_scope_fields.append("system_size_kw")
            if not request.evidence_pack.user_role:
                missing_scope_fields.append("user_role")
            if request.evidence_pack.ownership_verified is None:
                missing_scope_fields.append("ownership_verified")
        if "ticket" in lowered:
            intent = IntentType.escalate
        elif is_brief and not has_domain_context:
            intent = IntentType.general_question
            missing_info.append("issue_or_question_details")
            system_message = f"I can help with Delta technical support. Please share technical details for: {message}"
        else:
            intent = IntentType.troubleshoot
        if support_scope_status == SupportScopeStatus.unknown and intent != IntentType.general_question:
            system_message = (
                "Before troubleshooting, please confirm the following site eligibility details: "
                + ", ".join(missing_scope_fields)
                + "."
            )
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
            missing_scope_fields=missing_scope_fields,
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
        self.search_adapter = FakeSearchAdapter()
        self.ticket_adapter = FakeTicketAdapter()
        dependencies = WorkflowDependencies(
            llm_client=FakeLLMClient(),
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
                site_type="residential",
                system_size_kw="10",
                user_role="customer_owner",
                ownership_verified=True,
            ),
        )
        state = self.workflow.invoke({"request": request.model_dump(mode="json")})
        self.assertEqual(state["user_query"], "My inverter shows E031 after restart")
        self.assertEqual(state["current_phase"], "troubleshooting")
        self.assertEqual(state["next_action"], "continue_troubleshooting")
        self.assertEqual(state["citations"], ["doc-1"])

    def test_unknown_scope_asks_only_scope_questions(self):
        request = ChatMessageRequest(
            message="My inverter shows E031 after restart",
            device_info=DeviceInfo(device_type=DeviceType.inverter, model_number="M100A"),
        )
        state = self.workflow.invoke({"request": request.model_dump(mode="json")})
        self.assertEqual(state["current_phase"], "intake")
        self.assertEqual(state["next_action"], "ask_question")
        self.assertIn("site eligibility details", state["response_text"])
        self.assertEqual(state["missing_scope_fields"], ["site_type", "system_size_kw", "user_role", "ownership_verified"])

    def test_safety_path_collects_missing_evidence(self):
        request = ChatMessageRequest(
            message="There is smoke coming from the inverter enclosure",
            device_info=DeviceInfo(device_type=DeviceType.inverter, model_number="M100A"),
            evidence_pack=EvidencePack(
                site_type="residential",
                system_size_kw="10",
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
                site_type="residential",
                system_size_kw="10",
                user_role="customer_owner",
                ownership_verified=True,
                inverter_model="M100A",
                serial_number="SN12345",
                firmware_version="1.0.4",
                error_code="E031",
                timestamp="2026-03-07T09:30:00Z",
                system_topology="ac_coupled",
                phase_type="single_phase",
                backup_loads_present=False,
                recent_changes="No recent changes",
                environmental_conditions="Dry, 32C ambient",
            ),
        )
        state = self.workflow.invoke({"request": request.model_dump(mode="json")})
        self.assertEqual(state["current_phase"], "ticket_creation")
        self.assertEqual(state["ticket_response"]["status"], "mock_created")
        self.assertIn("Unsafe instructions given: no", self.ticket_adapter.last_payload.escalation_summary)

    def test_explicit_unsupported_site_collects_evidence_without_retrieval(self):
        request = ChatMessageRequest(
            message="Please create a ticket for inverter fault E031",
            request_ticket=True,
            device_info=DeviceInfo(device_type=DeviceType.inverter, model_number="M100A"),
            evidence_pack=EvidencePack(
                site_type="commercial",
                system_size_kw="80",
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
                site_type="industrial",
                system_size_kw="80",
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
                evidence_snapshot=EvidencePack(
                    site_type="residential",
                    system_size_kw="10",
                    user_role="customer_owner",
                    ownership_verified=True,
                    inverter_model="M100A",
                    serial_number="SN12345",
                    firmware_version="1.0.4",
                    error_code="E031",
                    timestamp="2026-03-07T09:30:00Z",
                    system_topology="ac_coupled",
                    phase_type="single_phase",
                    backup_loads_present=False,
                    recent_changes="No recent changes",
                    environmental_conditions="Dry, 32C ambient",
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


if __name__ == "__main__":
    unittest.main()
