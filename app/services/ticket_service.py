from app.models.conversation import ChatMessageRequest, TicketCreateRequest
from app.models.evidence import EvidencePack
from app.models.ticket import TicketPayload, TicketResponse


class TicketService:
    def __init__(self, adapter):
        self.adapter = adapter

    def create_from_graph(
        self,
        request: ChatMessageRequest,
        troubleshooting_steps: list[str],
        escalation_reason: str | None,
        merged_evidence_pack: dict,
        unsupported_reason: str | None,
        missing_artifacts: list[str],
    ) -> TicketResponse:
        evidence = EvidencePack.model_validate(merged_evidence_pack)
        payload = TicketPayload(
            customer_info=request.customer_info,
            device_info=request.device_info.model_dump(),
            issue_summary=request.message,
            troubleshooting_steps=troubleshooting_steps,
            attachments=[*evidence.photos, *evidence.logs],
            escalation_reason=escalation_reason or unsupported_reason,
            escalation_summary=self._build_escalation_summary(
                issue_summary=request.message,
                troubleshooting_steps=troubleshooting_steps,
                evidence=evidence,
                escalation_reason=escalation_reason,
                unsupported_reason=unsupported_reason,
                missing_artifacts=missing_artifacts,
            ),
            missing_artifacts=missing_artifacts,
            unsafe_instructions_given=False,
            evidence_pack=evidence.model_dump(exclude_none=True),
        )
        return self.adapter.create_ticket(payload)

    def create_from_request(self, request: TicketCreateRequest) -> TicketResponse:
        payload = TicketPayload(
            customer_info=request.customer_info,
            device_info=request.device_info.model_dump(),
            issue_summary=request.issue_summary,
            troubleshooting_steps=request.troubleshooting_steps,
            attachments=request.attachments,
            escalation_reason=request.escalation_reason,
            evidence_pack=request.evidence_pack.model_dump(exclude_none=True),
        )
        return self.adapter.create_ticket(payload)

    def _build_escalation_summary(
        self,
        issue_summary: str,
        troubleshooting_steps: list[str],
        evidence: EvidencePack,
        escalation_reason: str | None,
        unsupported_reason: str | None,
        missing_artifacts: list[str],
    ) -> str:
        summary_lines = [
            f"Symptoms: {issue_summary}",
            "Steps already attempted: " + (", ".join(troubleshooting_steps) if troubleshooting_steps else "None recorded"),
            "Escalation reason: " + ", ".join(
                reason for reason in (escalation_reason, unsupported_reason) if reason
            )
            if escalation_reason or unsupported_reason
            else "Escalation reason: none provided",
            "Evidence collected: " + ", ".join(sorted(evidence.provided_fields().keys())),
            "Missing or unavailable artifacts: " + (", ".join(missing_artifacts) if missing_artifacts else "None"),
            "Unsafe instructions given: no",
        ]
        return "\n".join(summary_lines)
