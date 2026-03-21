import uuid

import httpx

from app.models.ticket import ContactFormSubmission, TicketPayload, TicketResponse


DEFAULT_TICKET_TYPE = "Sales - Marshall"


class TicketApiClient:
    def __init__(self, base_url: str | None, timeout_seconds: float):
        self.base_url = base_url.rstrip("/") if base_url else None
        self.timeout_seconds = timeout_seconds

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    def create_ticket(self, payload: TicketPayload) -> TicketResponse:
        if not self.base_url:
            return TicketResponse(
                ticket_id=f"MOCK-{uuid.uuid4().hex[:8].upper()}",
                status="mock_created",
                message="Ticket created in mock mode because no external ticket API is configured.",
            )

        response = httpx.post(
            self.base_url,
            json=ContactFormSubmission(
                type=DEFAULT_TICKET_TYPE,
                firstName=payload.customer_info.first_name or "",
                lastName=payload.customer_info.last_name or "",
                email=payload.customer_info.email or "",
                phone=payload.customer_info.phone or "",
                message=payload.message_html or "<div></div>",
            ).model_dump(by_alias=True),
            headers={"Content-Type": "application/json"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return self._parse_ticket_response(response.json())

    def _parse_ticket_response(self, response_json: object) -> TicketResponse:
        if not isinstance(response_json, dict):
            raise ValueError("Ticket API response must be a JSON object.")

        response_data = response_json.get("data")
        payload = response_data if isinstance(response_data, dict) else response_json
        ticket_id = payload.get("ticket_id") or payload.get("ticketId") or payload.get("id")
        message = response_json.get("message") or payload.get("message") or "Ticket created."
        status = self._normalize_ticket_status(
            response_json=response_json,
            payload=payload,
            message=message,
        )

        if ticket_id is None or status is None:
            raise ValueError(
                "Ticket API response missing required ticket fields. "
                f"Received keys: {sorted(response_json.keys())}"
            )

        return TicketResponse(
            ticket_id=str(ticket_id),
            status=str(status),
            message=str(message),
            data=response_data if isinstance(response_data, dict) else None,
        )

    def _normalize_ticket_status(
        self,
        response_json: dict[str, object],
        payload: dict[str, object],
        message: object,
    ) -> str | None:
        top_level_status = response_json.get("status")
        if isinstance(top_level_status, str) and top_level_status.strip():
            return top_level_status.strip()

        payload_status = payload.get("status")
        if isinstance(payload_status, str) and payload_status.strip():
            return payload_status.strip()

        if isinstance(message, str) and message.strip():
            normalized_message = message.strip().lower()
            if normalized_message in {"success", "ok"}:
                return "created"
            return normalized_message.replace(" ", "_")

        if payload.get("id") is not None:
            return "created"

        return None
