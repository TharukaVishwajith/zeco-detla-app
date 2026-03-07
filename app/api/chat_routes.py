from uuid import uuid4

from fastapi import APIRouter, Depends, Request

from app.models.conversation import ChatMessageRequest, ChatMessageResponse, IntentClassification, TicketResponse


router = APIRouter(prefix="/chat", tags=["chat"])


def get_workflow(request: Request):
    return request.app.state.workflow


def get_conversation_history_service(request: Request):
    return request.app.state.conversation_history_service


@router.post("/message", response_model=ChatMessageResponse)
async def chat_message(
    payload: ChatMessageRequest,
    workflow=Depends(get_workflow),
    conversation_history_service=Depends(get_conversation_history_service),
) -> ChatMessageResponse:
    request_id = payload.request_id or str(uuid4())
    request_payload = payload.model_copy(update={"request_id": request_id})
    history = conversation_history_service.load_history(request_id)
    state = workflow.invoke(
        {
            "request": request_payload.model_dump(mode="json"),
            "history": [message.model_dump(mode="json") for message in history],
        }
    )
    classification = IntentClassification.model_validate(state["classification"])
    ticket = TicketResponse.model_validate(state["ticket_response"]) if state.get("ticket_response") else None
    response = ChatMessageResponse(
        request_id=request_id,
        current_phase=state.get("current_phase", "unknown"),
        intent=classification.intent,
        device_type=classification.device_type,
        response_text=state.get("response_text", ""),
        system_message=state.get("system_message"),
        citations=state.get("citations", []),
        next_action=state.get("next_action", "ask_question"),
        missing_fields=state.get("missing_fields", []),
        safety_flags=state.get("safety_flags", []),
        ticket=ticket,
        retrieved_documents=state.get("retrieved_docs", []),
    )
    conversation_history_service.persist_turn(
        request=request_payload,
        response_text=response.response_text,
        classification=classification,
        citations=response.citations,
        next_action=response.next_action.value,
        system_message=response.system_message,
    )
    return response
