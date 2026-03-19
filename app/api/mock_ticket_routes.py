from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse

from app.models.ticket import ContactFormSubmission, MockTicketRecord, TicketResponse


router = APIRouter(tags=["mock-ticket"])
mock_ticket_ui_path = Path(__file__).resolve().parents[2] / "mock_ticket_viewer.html"


def get_mock_ticket_store(request: Request):
    return request.app.state.mock_ticket_store


@router.post("/public/website/contactForm", response_model=TicketResponse)
async def mock_contact_form_submission(
    payload: ContactFormSubmission,
    mock_ticket_store=Depends(get_mock_ticket_store),
) -> TicketResponse:
    _, response = mock_ticket_store.create_ticket(payload)
    return response


@router.get("/ticket/mock/submissions", response_model=list[MockTicketRecord])
async def list_mock_contact_form_submissions(
    mock_ticket_store=Depends(get_mock_ticket_store),
) -> list[MockTicketRecord]:
    return mock_ticket_store.list_tickets()


@router.get("/ticket/mock/status")
async def mock_ticket_status(
    request: Request,
    mock_ticket_store=Depends(get_mock_ticket_store),
) -> dict[str, str | int | bool | None]:
    adapter = request.app.state.ticket_service.adapter
    resolved_target_url = adapter.contact_form_url if adapter.base_url else None
    using_local_mock_endpoint = bool(
        resolved_target_url
        and (
            resolved_target_url.startswith("http://127.0.0.1")
            or resolved_target_url.startswith("http://localhost")
            or resolved_target_url.startswith("https://127.0.0.1")
            or resolved_target_url.startswith("https://localhost")
        )
    )
    return {
        "ticket_api_base_url": adapter.base_url,
        "resolved_target_url": resolved_target_url,
        "mock_endpoint": "/public/website/contactForm",
        "using_local_mock_endpoint": using_local_mock_endpoint,
        "stored_submission_count": len(mock_ticket_store.list_tickets()),
    }


@router.get("/ticket/mock-ui", tags=["ui"])
async def mock_ticket_ui() -> FileResponse:
    return FileResponse(mock_ticket_ui_path)
