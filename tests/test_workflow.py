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
            "A safety hazard was detected. Please do not continue operating the equipment.\n\n"
            f"{progress_text}"
            "I am preparing the support ticket now.\n\n"
            "If available, please send these remaining details:\n"
            f"{field_list}\n\n"
            "If you do not have them, tell me and I will continue with the information already gathered."
        )
    if support_scope_status == "unsupported":
        return (
            "## Unsupported Site Escalation\n\n"
            "This site is outside Delta AI support scope and needs customer service review.\n\n"
            f"{progress_text}"
            "I can still collect a few details for the escalation ticket.\n\n"
            "If available, please send these remaining details:\n"
            f"{field_list}\n\n"
            "If you do not have them, tell me and I will continue with the information already gathered."
        )
    return (
        "## Support Ticket Details\n\n"
        f"{progress_text}"
        "I am ready to create the support ticket.\n\n"
        "If available, please send these remaining details:\n"
        f"{field_list}\n\n"
        "If you do not have them, tell me and I will continue with the information already gathered."
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
        normalized = " ".join(lowered.split()).rstrip("?.!")
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
        needs_clarification = (
            not history
            and "?" not in message
            and device_info.device_type == DeviceType.unknown
            and not device_info.model_number
            and not any(character.isdigit() for character in normalized)
            and 0 < len(normalized.split()) <= 2
            and len(normalized) <= 10
        )
        if risk_flags or "ticket" in lowered:
            intent = IntentType.escalate
        elif "?" in message:
            intent = IntentType.general_question
        elif needs_clarification:
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

    def generate_troubleshooting_response(self, message, retrieved_docs, classification, history=None):
        citations = [doc.doc_id for doc in retrieved_docs[:1]]
        if not retrieved_docs:
            response_text = (
                "Try the usual checks for this issue, starting with a safe restart if the equipment instructions allow it."
            )
            if classification.error_code:
                response_text = (
                    f"## First, check the {classification.error_code} condition\n\n"
                    "Try the usual checks for this issue, starting with a safe restart if the equipment instructions allow it."
                )
        elif "still the same" in message.lower() and "context:" in message:
            response_text = (
                "Based on the prior conversation, continue from the documented restart sequence and share the new fault state."
            )
        else:
            response_text = "Follow the documented restart sequence from the retrieved Delta KB article."
        return TroubleshootingResponse(
            response_text=response_text,
            citations=citations,
            next_action=TroubleshootingAction.continue_troubleshooting,
            counts_as_troubleshooting_round=True,
        )

    def generate_resolved_troubleshooting_response(self):
        return TroubleshootingResponse(
            response_text=(
                "## Resolved\n\n"
                "Glad to hear the issue is resolved. I will close this here. If anything changes, send a new message and I can help again."
            ),
            citations=[],
            next_action=TroubleshootingAction.resolved,
            counts_as_troubleshooting_round=False,
        )

    def generate_ticket_creation_intro(
        self,
        *,
        request,
        classification,
        history=None,
        troubleshooting_rounds=0,
        support_scope_status=None,
        escalate_immediately=False,
        force_ticket_creation=False,
    ):
        if escalate_immediately:
            return (
                "## Immediate Safety Escalation\n\n"
                "For safety, I need to escalate this issue right away and create a support ticket for you.\n\n"
                "Please do not continue operating the equipment while the case is being reviewed."
            )
        if force_ticket_creation or troubleshooting_rounds >= 5:
            return (
                "## Support Escalation\n\n"
                "I'm sorry the troubleshooting steps didn't resolve the issue.\n\n"
                "Since the problem is still present, I will escalate this to our customer service team and create a support ticket for you now."
            )
        if request.request_ticket or classification.intent == IntentType.escalate:
            return (
                "## Support Escalation\n\n"
                "I understand you want this escalated, and I will create a support ticket for you now.\n\n"
                "Our customer service team will review the case and provide further assistance."
            )
        return (
            "## Support Escalation\n\n"
            "I will create a support ticket for you now.\n\n"
            "Our customer service team will review the case and provide further assistance."
        )

    def generate_evidence_collection_response(
        self,
        *,
        request,
        classification,
        history=None,
        merged_evidence,
        missing_fields,
        support_scope_status,
        safety_assessment,
    ):
        return build_fake_evidence_collection_response(
            merged_evidence=merged_evidence,
            missing_fields=missing_fields,
            support_scope_status=support_scope_status,
            safety_assessment=safety_assessment,
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


class EmptySearchAdapter:
    def __init__(self):
        self.last_query = None
        self.last_filters = None

    def search(self, query, size=5, filters=None):
        self.last_query = query
        self.last_filters = filters or {}
        return []


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
        self.assertEqual(self.llm_client.classify_intent_calls, 1)

    def test_troubleshooting_without_kb_docs_still_answers_directly(self):
        dependencies = WorkflowDependencies(
            llm_client=self.llm_client,
            retrieval_service=RetrievalService(EmptySearchAdapter()),
            validation_service=ValidationService(),
            ticket_service=TicketService(self.ticket_adapter),
            retrieval_top_k=5,
        )
        workflow = build_workflow(dependencies)

        request = ChatMessageRequest(
            message="My inverter shows E031 after restart",
            device_info=DeviceInfo(device_type=DeviceType.inverter, model_number="M100A"),
            evidence_pack=EvidencePack(
                user_role="customer_owner",
                ownership_verified=True,
            ),
        )
        state = workflow.invoke({"request": request.model_dump(mode="json")})

        self.assertEqual(state["current_phase"], "troubleshooting")
        self.assertEqual(state["next_action"], "continue_troubleshooting")
        self.assertEqual(state["citations"], [])
        self.assertIn("try the usual checks", state["response_text"].lower())

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
        self.assertIn("I am preparing the support ticket now.", state["response_text"])
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
        self.assertIn("I can still collect a few details for the escalation ticket.", state["response_text"])

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

    def test_non_greeting_question_moves_past_intake_and_gets_answered(self):
        request = ChatMessageRequest(message="What does standby mode mean?")
        state = self.workflow.invoke({"request": request.model_dump(mode="json")})

        self.assertEqual(state["current_phase"], "troubleshooting")
        self.assertEqual(state["next_action"], "continue_troubleshooting")
        self.assertNotIn("system_message", state)
        self.assertEqual(state["citations"], ["doc-1"])

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

    def test_fifth_completed_troubleshooting_round_auto_creates_ticket(self):
        request = ChatMessageRequest(
            message="It is still not working after all of that.",
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
        history: list[ConversationMessage] = []
        for round_number in range(1, 6):
            history.extend(
                [
                    ConversationMessage(
                        role=ConversationRole.user,
                        content=f"Round {round_number}: the issue is still present.",
                    ),
                    ConversationMessage(
                        role=ConversationRole.assistant,
                        content=f"Try troubleshooting step {round_number} and tell me what changes.",
                        next_action=TroubleshootingAction.continue_troubleshooting,
                        escalation_active=False,
                        counts_as_troubleshooting_round=True,
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
            )

        state = self.workflow.invoke(
            {
                "request": request.model_dump(mode="json"),
                "history": [message.model_dump(mode="json") for message in history],
            }
        )

        self.assertEqual(state["current_phase"], "ticket_creation")
        self.assertEqual(state["troubleshooting_rounds"], 5)
        self.assertEqual(state["ticket_response"]["status"], "mock_created")
        self.assertIn("troubleshooting steps didn't resolve the issue", state["response_text"])
        self.assertIn("Support ticket `MOCK-12345678` has been created successfully.", state["response_text"])
        self.assertNotIn(
            "troubleshooting steps didn't resolve the issue",
            self.ticket_adapter.last_payload.escalation_summary,
        )

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

    def test_issue_resolved_returns_closing_response(self):
        request = ChatMessageRequest(
            message="The issue is resolved now",
            issue_resolved=True,
            device_info=DeviceInfo(device_type=DeviceType.inverter, model_number="M100A"),
        )
        state = self.workflow.invoke({"request": request.model_dump(mode="json")})

        self.assertEqual(state["current_phase"], "troubleshooting")
        self.assertEqual(state["next_action"], "resolved")
        self.assertEqual(derive_conversation_state(state), ConversationState.resolved)
        self.assertIn("Glad to hear the issue is resolved", state["response_text"])
        self.assertNotIn("support ticket", state["response_text"].lower())

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
                counts_as_troubleshooting_round=True,
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

    def test_non_counted_reply_does_not_increment_troubleshooting_rounds(self):
        class NonCountingLLMClient(FakeLLMClient):
            def generate_troubleshooting_response(self, message, retrieved_docs, classification, history=None):
                return TroubleshootingResponse(
                    response_text="## Clarify one detail\n\nPlease confirm the exact wording on the display.",
                    citations=[],
                    next_action=TroubleshootingAction.ask_question,
                    counts_as_troubleshooting_round=False,
                )

        workflow = build_workflow(
            WorkflowDependencies(
                llm_client=NonCountingLLMClient(),
                retrieval_service=RetrievalService(self.search_adapter),
                validation_service=ValidationService(),
                ticket_service=TicketService(self.ticket_adapter),
                retrieval_top_k=5,
            )
        )

        history = [
            ConversationMessage(role=ConversationRole.user, content="Round 1: still the same."),
            ConversationMessage(
                role=ConversationRole.assistant,
                content="Try step 1 and tell me what happens.",
                next_action=TroubleshootingAction.continue_troubleshooting,
                counts_as_troubleshooting_round=True,
            ),
            ConversationMessage(role=ConversationRole.user, content="Round 2: still the same."),
            ConversationMessage(
                role=ConversationRole.assistant,
                content="Try step 2 and tell me what happens.",
                next_action=TroubleshootingAction.continue_troubleshooting,
                counts_as_troubleshooting_round=True,
            ),
            ConversationMessage(role=ConversationRole.user, content="Round 3: still the same."),
            ConversationMessage(
                role=ConversationRole.assistant,
                content="Try step 3 and tell me what happens.",
                next_action=TroubleshootingAction.continue_troubleshooting,
                counts_as_troubleshooting_round=True,
            ),
            ConversationMessage(role=ConversationRole.user, content="Round 4: still the same."),
            ConversationMessage(
                role=ConversationRole.assistant,
                content="Try step 4 and tell me what happens.",
                next_action=TroubleshootingAction.continue_troubleshooting,
                counts_as_troubleshooting_round=True,
            ),
        ]

        state = workflow.invoke(
            {
                "request": ChatMessageRequest(
                    message="Still not resolved.",
                    device_info=DeviceInfo(device_type=DeviceType.inverter, model_number="M100A"),
                ).model_dump(mode="json"),
                "history": [message.model_dump(mode="json") for message in history],
            }
        )

        self.assertEqual(state["current_phase"], "troubleshooting")
        self.assertEqual(state["next_action"], "ask_question")
        self.assertEqual(state["troubleshooting_rounds"], 4)
        self.assertFalse(state["counts_as_troubleshooting_round"])

    def test_actionable_numbered_steps_override_false_round_flag(self):
        class MisflaggedLLMClient(FakeLLMClient):
            def generate_troubleshooting_response(self, message, retrieved_docs, classification, history=None):
                return TroubleshootingResponse(
                    response_text=(
                        "## Try this next\n\n"
                        "1. Check the display for the exact fault text.\n"
                        "2. Restart the inverter using the documented sequence.\n\n"
                        "Reply with the exact display text after these steps."
                    ),
                    citations=["doc-1"],
                    next_action=TroubleshootingAction.continue_troubleshooting,
                    counts_as_troubleshooting_round=False,
                )

        workflow = build_workflow(
            WorkflowDependencies(
                llm_client=MisflaggedLLMClient(),
                retrieval_service=RetrievalService(self.search_adapter),
                validation_service=ValidationService(),
                ticket_service=TicketService(self.ticket_adapter),
                retrieval_top_k=5,
            )
        )

        state = workflow.invoke(
            {
                "request": ChatMessageRequest(
                    message="My inverter still shows E031.",
                    device_info=DeviceInfo(device_type=DeviceType.inverter, model_number="M100A"),
                ).model_dump(mode="json"),
            }
        )

        self.assertEqual(state["current_phase"], "troubleshooting")
        self.assertEqual(state["next_action"], "continue_troubleshooting")
        self.assertEqual(state["troubleshooting_rounds"], 1)
        self.assertTrue(state["counts_as_troubleshooting_round"])

    def test_non_actionable_continue_reply_override_true_round_flag(self):
        class MisflaggedLLMClient(FakeLLMClient):
            def generate_troubleshooting_response(self, message, retrieved_docs, classification, history=None):
                return TroubleshootingResponse(
                    response_text="## Clarify one detail\n\nPlease confirm the exact wording shown on the display.",
                    citations=[],
                    next_action=TroubleshootingAction.continue_troubleshooting,
                    counts_as_troubleshooting_round=True,
                )

        workflow = build_workflow(
            WorkflowDependencies(
                llm_client=MisflaggedLLMClient(),
                retrieval_service=RetrievalService(self.search_adapter),
                validation_service=ValidationService(),
                ticket_service=TicketService(self.ticket_adapter),
                retrieval_top_k=5,
            )
        )

        state = workflow.invoke(
            {
                "request": ChatMessageRequest(
                    message="My inverter still shows E031.",
                    device_info=DeviceInfo(device_type=DeviceType.inverter, model_number="M100A"),
                ).model_dump(mode="json"),
            }
        )

        self.assertEqual(state["current_phase"], "troubleshooting")
        self.assertEqual(state["next_action"], "continue_troubleshooting")
        self.assertEqual(state["troubleshooting_rounds"], 0)
        self.assertFalse(state["counts_as_troubleshooting_round"])

    def test_latest_persisted_round_count_prevents_mid_conversation_reset(self):
        history = [
            ConversationMessage(
                role=ConversationRole.user,
                content="Old troubleshooting context that already fell outside the visible window.",
            ),
            ConversationMessage(
                role=ConversationRole.assistant,
                content="Try step 5 and tell me what happens.",
                next_action=TroubleshootingAction.continue_troubleshooting,
                counts_as_troubleshooting_round=True,
                troubleshooting_rounds=5,
            ),
        ]

        state = self.workflow.invoke(
            {
                "request": ChatMessageRequest(
                    message="Still not resolved.",
                    device_info=DeviceInfo(device_type=DeviceType.inverter, model_number="M100A"),
                ).model_dump(mode="json"),
                "history": [message.model_dump(mode="json") for message in history],
            }
        )

        self.assertEqual(state["current_phase"], "ticket_creation")
        self.assertEqual(state["troubleshooting_rounds"], 5)


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
