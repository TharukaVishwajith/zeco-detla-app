import unittest

from app.graph.workflow import WorkflowDependencies, build_workflow
from app.models.conversation import (
    ChatMessageRequest,
    DeviceInfo,
    DeviceType,
    EvidencePack,
    IntentClassification,
    IntentType,
    RetrievedDocument,
    TroubleshootingAction,
    TroubleshootingResponse,
)
from app.models.ticket import TicketResponse
from app.services.retrieval_service import RetrievalService
from app.services.ticket_service import TicketService
from app.services.validation_service import ValidationService


class FakeLLMClient:
    def classify_intent(self, message, device_info=None):
        if "ticket" in message.lower():
            intent = IntentType.escalate
        else:
            intent = IntentType.troubleshoot
        return IntentClassification(
            intent=intent,
            device_type=device_info.device_type if device_info else DeviceType.inverter,
            error_code="E031" if "E031" in message else None,
            model_number=device_info.model_number if device_info else None,
            risk_flags=[],
            missing_info=[],
        )

    def generate_troubleshooting_response(self, message, retrieved_docs, classification):
        citations = [doc.doc_id for doc in retrieved_docs[:1]]
        return TroubleshootingResponse(
            response_text="Follow the documented restart sequence from the retrieved Delta KB article.",
            citations=citations,
            next_action=TroubleshootingAction.continue_troubleshooting,
        )

    def create_embedding(self, text, dimensions=None):
        return None


class FakeSearchAdapter:
    def search(self, query, size=5, filters=None):
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

    def create_ticket(self, payload):
        return TicketResponse(
            ticket_id="MOCK-12345678",
            status="mock_created",
            message="Ticket created in test mode.",
        )


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        dependencies = WorkflowDependencies(
            llm_client=FakeLLMClient(),
            retrieval_service=RetrievalService(FakeSearchAdapter()),
            validation_service=ValidationService(),
            ticket_service=TicketService(FakeTicketAdapter()),
            retrieval_top_k=5,
        )
        self.workflow = build_workflow(dependencies)

    def test_troubleshooting_path_returns_grounded_response(self):
        request = ChatMessageRequest(
            message="My inverter shows E031 after restart",
            device_info=DeviceInfo(device_type=DeviceType.inverter, model_number="M100A"),
        )
        state = self.workflow.invoke({"request": request.model_dump(mode="json")})
        self.assertEqual(state["current_phase"], "troubleshooting")
        self.assertEqual(state["next_action"], "continue_troubleshooting")
        self.assertEqual(state["citations"], ["doc-1"])

    def test_safety_path_collects_missing_evidence(self):
        request = ChatMessageRequest(
            message="There is smoke coming from the inverter enclosure",
            device_info=DeviceInfo(device_type=DeviceType.inverter, model_number="M100A"),
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
                site_type="commercial",
                system_size_kw="80",
                inverter_model="M100A",
                serial_number="SN12345",
                firmware_version="1.0.4",
                error_code="E031",
                timestamp="2026-03-07T09:30:00Z",
                recent_changes="No recent changes",
                environmental_conditions="Dry, 32C ambient",
            ),
        )
        state = self.workflow.invoke({"request": request.model_dump(mode="json")})
        self.assertEqual(state["current_phase"], "ticket_creation")
        self.assertEqual(state["ticket_response"]["status"], "mock_created")


if __name__ == "__main__":
    unittest.main()

