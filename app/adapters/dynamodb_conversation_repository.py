import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.models.conversation import ConversationMessage


logger = logging.getLogger(__name__)


class DynamoConversationRepository:
    def __init__(
        self,
        table_name: str | None,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
    ):
        self.table_name = table_name
        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self._table = None

    @property
    def configured(self) -> bool:
        return bool(self.table_name)

    def load_messages(self, request_id: str) -> list[ConversationMessage]:
        if not self.configured:
            return []

        table = self._get_table()
        if table is None:
            return []

        try:
            from boto3.dynamodb.conditions import Key
        except ImportError:  # pragma: no cover - local dependency issue
            logger.warning("boto3 is not installed; conversation history is disabled")
            return []

        items: list[dict[str, Any]] = []
        query_kwargs: dict[str, Any] = {
            "KeyConditionExpression": Key("pk1").eq(self._conversation_pk(request_id)),
            "ScanIndexForward": True,
        }

        while True:
            response = table.query(**query_kwargs)
            items.extend(response.get("Items", []))
            last_evaluated_key = response.get("LastEvaluatedKey")
            if not last_evaluated_key:
                break
            query_kwargs["ExclusiveStartKey"] = last_evaluated_key

        messages: list[ConversationMessage] = []
        for item in items:
            try:
                messages.append(self._deserialize_message(item))
            except Exception as exc:  # pragma: no cover - defensive path
                logger.warning("Skipping invalid conversation item for %s: %s", request_id, exc)
        return messages

    def save_messages(self, request_id: str, messages: list[ConversationMessage]) -> None:
        if not self.configured or not messages:
            return

        table = self._get_table()
        if table is None:
            return

        with table.batch_writer() as batch:
            for message in messages:
                batch.put_item(Item=self._serialize_message(request_id=request_id, message=message))

    def _get_table(self):
        if self._table is not None:
            return self._table

        if not self.table_name:
            return None

        try:
            import boto3
        except ImportError:  # pragma: no cover - local dependency issue
            logger.warning("boto3 is not installed; conversation history is disabled")
            return None

        resource_kwargs = self._resource_kwargs()

        try:
            dynamodb = boto3.resource("dynamodb", **resource_kwargs)
            self._table = dynamodb.Table(self.table_name)
        except Exception as exc:  # pragma: no cover - AWS credential/runtime path
            logger.warning("Failed to initialize DynamoDB table %s: %s", self.table_name, exc)
            return None
        return self._table

    def _serialize_message(self, request_id: str, message: ConversationMessage) -> dict[str, Any]:
        created_at = message.created_at or self._created_at()
        sort_timestamp = self._sort_timestamp(created_at)
        message_uuid = uuid4().hex
        message_id = message.message_id or f"m_{message_uuid}_{sort_timestamp}"
        item: dict[str, Any] = {
            "pk1": self._conversation_pk(request_id),
            "sk1": f"MSG#{sort_timestamp}#{message_uuid}",
            "message_id": message_id,
            "request_id": request_id,
            "role": message.role.value,
            "content": message.content,
            "created_at": created_at,
        }
        if message.user_id:
            item["user_id"] = message.user_id
        if message.system_message:
            item["system_message"] = message.system_message
        if message.intent:
            item["intent"] = message.intent.value
        if message.citations:
            item["citations"] = message.citations
        if message.next_action:
            item["next_action"] = message.next_action.value
        if message.support_scope_status:
            item["support_scope_status"] = message.support_scope_status.value
        if message.unsupported_reason:
            item["unsupported_reason"] = message.unsupported_reason.value
        if message.evidence_snapshot:
            item["evidence_snapshot"] = message.evidence_snapshot.model_dump(exclude_none=True)
        return item

    def _deserialize_message(self, item: dict[str, Any]) -> ConversationMessage:
        payload = {
            "message_id": item.get("message_id"),
            "request_id": item.get("request_id"),
            "role": item.get("role"),
            "content": item.get("content", ""),
            "created_at": item.get("created_at"),
            "user_id": item.get("user_id"),
            "system_message": item.get("system_message"),
            "intent": item.get("intent"),
            "citations": item.get("citations", []),
            "next_action": item.get("next_action"),
            "support_scope_status": item.get("support_scope_status"),
            "unsupported_reason": item.get("unsupported_reason"),
            "evidence_snapshot": item.get("evidence_snapshot"),
        }
        return ConversationMessage.model_validate(payload)

    def _conversation_pk(self, request_id: str) -> str:
        return f"CONV#{request_id}"

    def _resource_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.region_name:
            kwargs["region_name"] = self.region_name

        has_access_key = bool(self.aws_access_key_id)
        has_secret_key = bool(self.aws_secret_access_key)

        if has_access_key and has_secret_key:
            kwargs["aws_access_key_id"] = self.aws_access_key_id
            kwargs["aws_secret_access_key"] = self.aws_secret_access_key
            if self.aws_session_token:
                kwargs["aws_session_token"] = self.aws_session_token
        elif has_access_key or has_secret_key:
            logger.warning("Incomplete AWS key configuration; falling back to default credential chain")

        return kwargs

    def _created_at(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")

    def _sort_timestamp(self, created_at: str) -> str:
        normalized = created_at.replace("Z", "+00:00")
        timestamp = datetime.fromisoformat(normalized)
        return timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
