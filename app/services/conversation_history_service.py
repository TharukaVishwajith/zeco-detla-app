import logging

from app.models.conversation import ChatMessageRequest, ConversationMessage, ConversationRole, IntentClassification


logger = logging.getLogger(__name__)


class ConversationHistoryService:
    def __init__(self, repository, max_messages: int = 12):
        self.repository = repository
        self.max_messages = max_messages

    @property
    def configured(self) -> bool:
        return bool(self.repository and self.repository.configured)

    def load_history(self, request_id: str | None) -> list[ConversationMessage]:
        if not request_id or not self.configured:
            return []

        try:
            messages = self.repository.load_messages(request_id)
        except Exception as exc:  # pragma: no cover - storage/runtime path
            logger.warning("Failed to load conversation history for %s: %s", request_id, exc)
            return []
        if self.max_messages > 0:
            return messages[-self.max_messages :]
        return messages

    def persist_turn(
        self,
        request: ChatMessageRequest,
        response_text: str,
        classification: IntentClassification,
        citations: list[str],
        next_action: str,
        system_message: str | None = None,
    ) -> None:
        if not request.request_id or not self.configured:
            return

        messages = [
            ConversationMessage(
                role=ConversationRole.user,
                content=request.message,
                request_id=request.request_id,
                user_id=request.user_id,
            ),
            ConversationMessage(
                role=ConversationRole.assistant,
                content=response_text,
                request_id=request.request_id,
                system_message=system_message,
                intent=classification.intent,
                citations=citations,
                next_action=next_action,
            ),
        ]

        try:
            self.repository.save_messages(request.request_id, messages)
        except Exception as exc:  # pragma: no cover - storage/runtime path
            logger.warning("Failed to persist conversation history for %s: %s", request.request_id, exc)
