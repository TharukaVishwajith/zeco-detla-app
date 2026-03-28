import unittest

from app.core.conversation_context import extract_message_evidence, merge_evidence_from_conversation
from app.core.conversation_state import derive_conversation_state
from app.graph.workflow import WorkflowDependencies, build_workflow
from app.models.conversation import (
    ChatMessageRequest,
    ConversationState,
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
    TroubleshootingResponseSource,
    UnsupportedReason,
)
from app.models.ticket import TicketResponse
from app.services.retrieval_service import RetrievalService
from app.services.ticket_service import TicketService
from app.services.validation_service import ValidationService


def build_fake_evidence_collection_response(
    merged_evidence: EvidencePack,
    missing_fields: list[str],
    support_scope_status,
    safety_assessment,
) -> str:
    field_list = "\n".join(f"- {field.replace('_', ' ')}" for field in missing_fields)
    provided = sorted(merged_evidence.provided_fields().keys())
    progress_text = f"I already have: {', '.join(provided[:4])}.\n\n" if provided else ""
    if safety_assessment.get("escalate_immediately"):
        return (
            "## Immediate Safety Escalation\n\n"
            "A safety hazard was detected. Do not continue operating the equipment.\n\n"
            f"{progress_text}"
            "I can help create the support ticket.\n\n"
            "If you have any of these additional details, send them in one reply:\n"
            f"{field_list}\n\n"
            "If not, tell me and I will proceed with the information already gathered."
        )
    if support_scope_status == "unsupported":
        return (
            "## Unsupported Site Escalation\n\n"
            "This site is outside Delta AI support scope.\n\n"
            f"{progress_text}"
            "I can still help collect what is needed for the escalation ticket.\n\n"
            "If you have any of these additional details, send them in one reply:\n"
            f"{field_list}\n\n"
            "If not, tell me and I will proceed with the information already gathered."
        )
    return (
        "## Ticket Information Needed\n\n"
        f"{progress_text}"
        "I can create the support ticket for you.\n\n"
        "If you have any of these additional details, send them in one reply:\n"
        f"{field_list}\n\n"
        "If not, tell me and I will proceed with the information already gathered."
    )


class FakeLLMClient:
    def __init__(self):
        self.classify_intent_calls = 0

    def classify_intent(self, request, history=None):
        self.classify_intent_calls += 1
        message = request.message
        device_info = request.device_info
        lowered = message.lower()
        user_query = message
        evidence = merge_evidence_from_conversation(
            current_message=request.message,
            request_evidence=request.evidence_pack,
            history=history or [],
        )
        if history:
            recent_context = " ".join(item.content for item in history[-2:])
            user_query = f"{message} | context: {recent_context}"
        history_text = " ".join(item.content.lower() for item in (history or []))
        risk_flags = [term for term in ("smoke", "fire", "sparking") if term in f"{history_text} {lowered}"]
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
        if risk_flags or "ticket" in lowered:
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
        evidence_collection_response_text = None
        active_escalation = request.request_ticket or any(item.escalation_active for item in (history or []) if item.role == ConversationRole.assistant)
        if active_escalation or intent == IntentType.escalate:
            evidence_collection_response_text = build_fake_evidence_collection_response(
                merged_evidence=evidence,
                missing_fields=evidence.missing_core_fields(),
                support_scope_status=support_scope_status.value,
                safety_assessment={"escalate_immediately": bool(risk_flags)},
            )
        return IntentClassification(
            intent=intent,
            device_type=device_info.device_type if device_info else DeviceType.inverter,
            user_query=user_query,
            error_code="E031" if "E031" in message else None,
            model_number=device_info.model_number if device_info else None,
            evidence_pack=evidence,
            evidence_collection_response_text=evidence_collection_response_text,
            risk_flags=risk_flags,
            missing_info=missing_info,
            support_scope_status=support_scope_status,
            unsupported_reason=unsupported_reason,
            missing_scope_fields=[],
            system_message=system_message,
        )

    def generate_troubleshooting_response(self, message, retrieved_docs, classification, validation_service=None):  # noqa: ARG002
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
            response_source=TroubleshootingResponseSource.grounded_kb,
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

    def _build_workflow_with_client(self, llm_client):
        dependencies = WorkflowDependencies(
            llm_client=llm_client,
            retrieval_service=RetrievalService(self.search_adapter),
            validation_service=ValidationService(),
            ticket_service=TicketService(self.ticket_adapter),
            retrieval_top_k=5,
        )
        return build_workflow(dependencies)

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
        self.assertEqual(self.llm_client.classify_intent_calls, 1)

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
        self.assertIn("I can help create the support ticket", state["response_text"])
        self.assertNotIn("## Evidence Required", state["response_text"])

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
                additional_info="Issue still present after restart and basic checks.",
            ),
        )
        state = self.workflow.invoke({"request": request.model_dump(mode="json")})
        self.assertEqual(state["current_phase"], "ticket_creation")
        self.assertEqual(state["ticket_response"]["status"], "mock_created")
        self.assertTrue(self.ticket_adapter.last_payload.message_html.startswith("<div>"))
        self.assertIn("Evidence pack", self.ticket_adapter.last_payload.message_html)
        self.assertIn("Serial number", self.ticket_adapter.last_payload.message_html)
        self.assertIn("Additional info", self.ticket_adapter.last_payload.message_html)
        self.assertEqual(self.llm_client.classify_intent_calls, 1)

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
        self.assertIn("I can still help collect what is needed for the escalation ticket.", state["response_text"])

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

    def test_internal_fallback_continue_troubleshooting_stays_in_troubleshooting(self):
        class FallbackContinueClient(FakeLLMClient):
            def generate_troubleshooting_response(self, message, retrieved_docs, classification, validation_service=None):  # noqa: ARG002
                return TroubleshootingResponse(
                    response_text="## Check the inverter state\n\n1. Note the fault text.\n2. Restart once.\n\nReply with the updated display text.",
                    citations=[],
                    next_action=TroubleshootingAction.continue_troubleshooting,
                    response_source=TroubleshootingResponseSource.internal_fallback,
                )

        workflow = self._build_workflow_with_client(FallbackContinueClient())
        request = ChatMessageRequest(message="My inverter shows E031 after restart")

        state = workflow.invoke({"request": request.model_dump(mode="json")})

        self.assertEqual(state["current_phase"], "troubleshooting")
        self.assertEqual(state["next_action"], "continue_troubleshooting")
        self.assertEqual(state["response_source"], TroubleshootingResponseSource.internal_fallback.value)

    def test_internal_fallback_escalate_routes_to_evidence_collection(self):
        class FallbackEscalateClient(FakeLLMClient):
            def generate_troubleshooting_response(self, message, retrieved_docs, classification, validation_service=None):  # noqa: ARG002
                return TroubleshootingResponse(
                    response_text="## Safety check\n\n1. Stop operating the inverter.\n2. Note the exact fault text.\n\nReply with the fault text so I can collect escalation details.",
                    citations=[],
                    next_action=TroubleshootingAction.escalate,
                    response_source=TroubleshootingResponseSource.internal_fallback,
                )

        workflow = self._build_workflow_with_client(FallbackEscalateClient())
        request = ChatMessageRequest(message="My inverter shows E031 after restart")

        state = workflow.invoke({"request": request.model_dump(mode="json")})

        self.assertEqual(state["current_phase"], "evidence_collection")
        self.assertEqual(state["response_source"], TroubleshootingResponseSource.internal_fallback.value)

    def test_merge_evidence_preserves_explicit_additional_info(self):
        merged = merge_evidence_from_conversation(
            current_message="Still the same after the restart, and it now trips every evening.",
            request_evidence=EvidencePack(additional_info="Issue persists after restart and trips every evening."),
            history=[
                ConversationMessage(
                    role=ConversationRole.user,
                    content="My inverter shows E031 after restart.",
                ),
                ConversationMessage(
                    role=ConversationRole.user,
                    content="We had a grid outage yesterday before this started.",
                ),
                ConversationMessage(
                    role=ConversationRole.user,
                    content="Serial number SN12345",
                ),
            ],
        )

        self.assertEqual(merged.serial_number, "SN12345")
        self.assertEqual(merged.additional_info, "Issue persists after restart and trips every evening.")

    def test_extract_message_evidence_normalizes_user_role_to_installer_or_customer(self):
        installer_evidence = extract_message_evidence("I am the installer for this inverter site.")
        customer_evidence = extract_message_evidence("I am the customer and this is my system.")

        self.assertEqual(installer_evidence.user_role, "Installer")
        self.assertEqual(customer_evidence.user_role, "customer")

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
        self.assertEqual(state["current_phase"], "ticket_creation")
        self.assertEqual(state["classification"]["intent"], IntentType.escalate.value)
        self.assertFalse(state["escalation_active"])
        self.assertTrue(state["previous_escalation_active"])
        self.assertEqual(state["ticket_response"]["status"], "mock_created")
        self.assertEqual(self.ticket_adapter.last_payload.evidence_pack["serial_number"], "SN12345")

    def test_escalation_follow_up_without_more_evidence_still_creates_ticket(self):
        request = ChatMessageRequest(message="I do not have any more details")
        history = [
            ConversationMessage(
                role=ConversationRole.user,
                content="Please escalate this inverter issue",
            ),
            ConversationMessage(
                role=ConversationRole.assistant,
                content="I can create the support ticket for you. If you have any additional details, send them.",
                intent=IntentType.escalate,
                next_action=TroubleshootingAction.collect_evidence,
                escalation_active=True,
                evidence_snapshot=EvidencePack(
                    user_role="customer_owner",
                    ownership_verified=True,
                    inverter_model="M100A",
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
        self.assertEqual(state["current_phase"], "ticket_creation")
        self.assertEqual(state["ticket_response"]["status"], "mock_created")
        self.assertFalse(state["escalation_active"])

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

    def test_ticket_payload_includes_summarized_troubleshooting_history(self):
        request = ChatMessageRequest(
            message="Please create the ticket",
            request_ticket=True,
        )
        history = [
            ConversationMessage(
                role=ConversationRole.user,
                content="My inverter shows E031 after restart",
            ),
            ConversationMessage(
                role=ConversationRole.assistant,
                content=(
                    "## First, check the E031 condition\n\n"
                    "1. Check the display.\n"
                    "2. Acknowledge the alarm.\n"
                    "3. Run the restart sequence.\n\n"
                    "Reply with the exact display message after this step."
                ),
                next_action=TroubleshootingAction.continue_troubleshooting,
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

        self.assertEqual(state["current_phase"], "ticket_creation")
        self.assertIn(
            "Check the display; Acknowledge the alarm; Run the restart sequence",
            self.ticket_adapter.last_payload.escalation_summary,
        )
        self.assertNotIn("## First", self.ticket_adapter.last_payload.message_html)
        self.assertIn(
            "<li>Check the display; Acknowledge the alarm; Run the restart sequence</li>",
            self.ticket_adapter.last_payload.message_html,
        )

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


class ConversationStateMappingTests(unittest.TestCase):
    def test_maps_intake_message_to_needs_clarification(self):
        self.assertEqual(
            derive_conversation_state(
                {
                    "current_phase": "intake",
                    "system_message": "Please share your Delta issue or model number.",
                    "next_action": TroubleshootingAction.ask_question.value,
                }
            ),
            ConversationState.needs_clarification,
        )

    def test_maps_standard_troubleshooting_to_troubleshooting(self):
        self.assertEqual(
            derive_conversation_state(
                {
                    "current_phase": "troubleshooting",
                    "next_action": TroubleshootingAction.continue_troubleshooting.value,
                }
            ),
            ConversationState.troubleshooting,
        )

    def test_maps_evidence_collection_to_awaiting_evidence(self):
        self.assertEqual(
            derive_conversation_state(
                {
                    "current_phase": "evidence_collection",
                    "next_action": TroubleshootingAction.collect_evidence.value,
                }
            ),
            ConversationState.awaiting_evidence,
        )

    def test_maps_ticket_response_to_ticket_created(self):
        self.assertEqual(
            derive_conversation_state(
                {
                    "current_phase": "ticket_creation",
                    "next_action": TroubleshootingAction.escalate.value,
                    "ticket_response": {"ticket_id": "MOCK-123"},
                }
            ),
            ConversationState.ticket_created,
        )

    def test_maps_resolved_action_to_resolved(self):
        self.assertEqual(
            derive_conversation_state(
                {
                    "current_phase": "troubleshooting",
                    "next_action": TroubleshootingAction.resolved.value,
                }
            ),
            ConversationState.resolved,
        )


if __name__ == "__main__":
    unittest.main()
