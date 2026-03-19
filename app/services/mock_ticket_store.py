import uuid
from datetime import datetime, timezone
from threading import Lock

from app.models.ticket import ContactFormSubmission, MockTicketRecord, TicketResponse


class InMemoryMockTicketStore:
    def __init__(self):
        self._records: list[MockTicketRecord] = []
        self._lock = Lock()

    def create_ticket(
        self,
        submission: ContactFormSubmission,
    ) -> tuple[MockTicketRecord, TicketResponse]:
        record = MockTicketRecord(
            ticketId=f"MOCK-{uuid.uuid4().hex[:8].upper()}",
            status="mock_created",
            receivedAt=datetime.now(timezone.utc).isoformat(),
            **submission.model_dump(by_alias=True),
        )
        response = TicketResponse(
            ticket_id=record.ticket_id,
            status=record.status,
            message="Mock contact form accepted and stored in memory.",
        )
        with self._lock:
            self._records.insert(0, record)
        return record, response

    def list_tickets(self) -> list[MockTicketRecord]:
        with self._lock:
            return list(self._records)
