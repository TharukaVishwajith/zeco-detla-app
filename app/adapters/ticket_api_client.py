import uuid

import httpx

from app.models.ticket import TicketPayload, TicketResponse


class TicketApiClient:
    def __init__(self, base_url: str | None, api_key: str | None, timeout_seconds: float):
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key
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

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = httpx.post(
            f"{self.base_url}/tickets",
            json=payload.model_dump(),
            headers=headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return TicketResponse.model_validate(response.json())

