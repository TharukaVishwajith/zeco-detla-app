from app.models.conversation import ChatMessageRequest, TicketCreateRequest
from app.models.ticket import TicketPayload, TicketResponse


class TicketService:
    def __init__(self, adapter):
        self.adapter = adapter

    def create_from_graph(
        self,
        request: ChatMessageRequest,
        troubleshooting_steps: list[str],
        escalation_reason: str | None,
    ) -> TicketResponse:
        payload = TicketPayload(
            customer_info=request.customer_info,
            device_info=request.device_info.model_dump(),
            issue_summary=request.message,
            troubleshooting_steps=troubleshooting_steps,
            escalation_reason=escalation_reason,
            evidence_pack=request.evidence_pack.model_dump(exclude_none=True),
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

