from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.adapters.dynamodb_conversation_repository import DynamoConversationRepository
from app.adapters.elastic_client import OpenSearchHybridClient
from app.adapters.openai_client import OpenAIClient
from app.adapters.ticket_api_client import TicketApiClient
from app.api.chat_routes import router as chat_router
from app.api.ticket_routes import router as ticket_router
from app.core.agent_models import build_agent_model_config
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.graph.workflow import WorkflowDependencies, build_workflow
from app.models.conversation import HealthResponse
from app.services.conversation_history_service import ConversationHistoryService
from app.services.retrieval_service import RetrievalService
from app.services.ticket_service import TicketService
from app.services.validation_service import ValidationService


settings = get_settings()
configure_logging(settings.log_level)
test_client_path = Path(__file__).resolve().parent.parent / "test_client.html"

agent_model_config = build_agent_model_config(
    default_model=settings.openai_chat_model,
    raw_overrides=settings.openai_agent_models,
)
openai_client = OpenAIClient(
    api_key=settings.openai_api_key,
    chat_model=settings.openai_chat_model,
    embedding_model=settings.openai_embedding_model,
    agent_model_config=agent_model_config,
)
opensearch_client = OpenSearchHybridClient(
    host=settings.opensearch_host,
    index_name=settings.opensearch_index,
    region=settings.opensearch_region,
    port=settings.opensearch_port,
    username=settings.opensearch_username,
    password=settings.opensearch_password,
    verify_certs=settings.opensearch_verify_certs,
    vector_field=settings.opensearch_vector_field,
    embedding_dimensions=settings.openai_embedding_dimensions,
    llm_client=openai_client,
)
retrieval_service = RetrievalService(adapter=opensearch_client)
ticket_service = TicketService(
    adapter=TicketApiClient(
        base_url=settings.ticket_api_base_url,
        api_key=settings.ticket_api_key,
        timeout_seconds=settings.ticket_api_timeout_seconds,
    )
)
validation_service = ValidationService()
conversation_history_service = ConversationHistoryService(
    repository=DynamoConversationRepository(
        table_name=settings.dynamodb_table_name,
        region_name=settings.dynamodb_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    ),
    max_messages=settings.conversation_history_max_messages,
)
workflow = build_workflow(
    WorkflowDependencies(
        llm_client=openai_client,
        retrieval_service=retrieval_service,
        validation_service=validation_service,
        ticket_service=ticket_service,
        retrieval_top_k=settings.retrieval_top_k,
    )
)

app = FastAPI(title=settings.app_name, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.workflow = workflow
app.state.ticket_service = ticket_service
app.state.conversation_history_service = conversation_history_service

app.include_router(chat_router)
app.include_router(ticket_router)


@app.get("/", tags=["ui"])
async def test_client() -> FileResponse:
    return FileResponse(test_client_path)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm_configured=openai_client.enabled,
        opensearch_configured=opensearch_client.configured,
        ticket_api_configured=ticket_service.adapter.configured,
    )
