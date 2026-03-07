from fastapi import APIRouter, Depends, Request

from app.models.conversation import TicketCreateRequest
from app.models.ticket import TicketResponse


router = APIRouter(prefix="/ticket", tags=["ticket"])


def get_ticket_service(request: Request):
    return request.app.state.ticket_service


@router.post("/create", response_model=TicketResponse)
async def create_ticket(payload: TicketCreateRequest, ticket_service=Depends(get_ticket_service)) -> TicketResponse:
    return ticket_service.create_from_request(payload)

