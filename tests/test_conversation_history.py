import unittest

from app.adapters.dynamodb_conversation_repository import DynamoConversationRepository
from app.models.conversation import (
    ChatMessageRequest,
    ConversationMessage,
    ConversationRole,
    DeviceType,
    IntentType,
    TroubleshootingAction,
)
from app.services.conversation_history_service import ConversationHistoryService

try:
    from app.api.chat_routes import chat_message
except ModuleNotFoundError:
    chat_message = None


class InMemoryConversationRepository:
    configured = True

    def __init__(self):
        self.messages: dict[str, list[ConversationMessage]] = {}

    def load_messages(self, request_id: str) -> list[ConversationMessage]:
        return list(self.messages.get(request_id, []))

    def save_messages(self, request_id: str, messages: list[ConversationMessage]) -> None:
        self.messages.setdefault(request_id, []).extend(messages)


class FakeWorkflow:
    def __init__(self):
        self.seen_histories: list[list[dict]] = []

    def invoke(self, state: dict) -> dict:
        history = state.get("history", [])
        self.seen_histories.append(history)
        request = state["request"]
        message = request["message"]

        if message.lower() == "hi":
            return {
                "classification": {
                    "intent": IntentType.general_question.value,
                    "device_type": DeviceType.unknown.value,
                    "error_code": None,
                    "model_number": None,
                    "risk_flags": [],
                    "missing_info": ["issue_or_question_details"],
                    "system_message": "Please share your Delta issue or model number.",
                },
                "current_phase": "intake",
                "response_text": "Please share your Delta issue or model number.",
                "system_message": "Please share your Delta issue or model number.",
                "citations": [],
                "next_action": TroubleshootingAction.ask_question.value,
            }

        prior_assistant = history[-1]["content"] if history else "none"
        return {
            "classification": {
                "intent": IntentType.troubleshoot.value,
                "device_type": DeviceType.inverter.value,
                "error_code": "E031" if "E031" in message or "E031" in prior_assistant else None,
                "model_number": None,
                "risk_flags": [],
                "missing_info": [],
                "system_message": None,
            },
            "current_phase": "troubleshooting",
            "response_text": f"history={len(history)} prior={prior_assistant}",
            "citations": ["doc-1"],
            "next_action": TroubleshootingAction.continue_troubleshooting.value,
        }


@unittest.skipIf(chat_message is None, "fastapi is not installed in this environment")
class ConversationHistoryRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repository = InMemoryConversationRepository()
        self.service = ConversationHistoryService(repository=self.repository, max_messages=12)
        self.workflow = FakeWorkflow()

    async def test_first_message_generates_request_id_and_persists_turn(self):
        response = await chat_message(
            ChatMessageRequest(message="My inverter shows E031"),
            workflow=self.workflow,
            conversation_history_service=self.service,
        )

        self.assertIsNotNone(response.request_id)
        stored_messages = self.repository.load_messages(response.request_id)
        self.assertEqual([message.role for message in stored_messages], [ConversationRole.user, ConversationRole.assistant])
        self.assertEqual(stored_messages[0].content, "My inverter shows E031")
        self.assertEqual(stored_messages[1].content, response.response_text)

    async def test_follow_up_message_loads_prior_history(self):
        first_response = await chat_message(
            ChatMessageRequest(message="My inverter shows E031"),
            workflow=self.workflow,
            conversation_history_service=self.service,
        )

        second_response = await chat_message(
            ChatMessageRequest(request_id=first_response.request_id, message="Still the same"),
            workflow=self.workflow,
            conversation_history_service=self.service,
        )

        self.assertEqual(len(self.workflow.seen_histories[0]), 0)
        self.assertEqual(len(self.workflow.seen_histories[1]), 2)
        self.assertEqual(self.workflow.seen_histories[1][0]["role"], ConversationRole.user.value)
        self.assertEqual(self.workflow.seen_histories[1][1]["role"], ConversationRole.assistant.value)
        self.assertIn("history=2", second_response.response_text)
        self.assertEqual(len(self.repository.load_messages(first_response.request_id)), 4)

    async def test_system_message_response_is_persisted_with_metadata(self):
        response = await chat_message(
            ChatMessageRequest(message="Hi"),
            workflow=self.workflow,
            conversation_history_service=self.service,
        )

        stored_messages = self.repository.load_messages(response.request_id)
        assistant_message = stored_messages[1]
        self.assertEqual(assistant_message.content, response.response_text)
        self.assertEqual(assistant_message.system_message, response.system_message)


class DynamoConversationRepositoryTests(unittest.TestCase):
    def test_sort_key_uses_timestamp_prefix(self):
        repository = DynamoConversationRepository(table_name="zeco_delta_table", region_name="us-east-1")
        item = repository._serialize_message(  # noqa: SLF001 - validating key format directly
            request_id="req-123",
            message=ConversationMessage(
                role=ConversationRole.user,
                content="hello",
                created_at="2026-03-07T10:15:30.123456Z",
            ),
        )

        self.assertEqual(item["pk1"], "CONV#req-123")
        self.assertTrue(item["sk1"].startswith("MSG#20260307T101530123456Z#"))
        self.assertTrue(item["message_id"].endswith("_20260307T101530123456Z"))

    def test_resource_kwargs_include_explicit_aws_keys(self):
        repository = DynamoConversationRepository(
            table_name="zeco_delta_table",
            region_name="ap-southeast-2",
            aws_access_key_id="AKIA_TEST",
            aws_secret_access_key="SECRET_TEST",
            aws_session_token="TOKEN_TEST",
        )

        kwargs = repository._resource_kwargs()  # noqa: SLF001 - validating credentials mapping directly
        self.assertEqual(kwargs["region_name"], "ap-southeast-2")
        self.assertEqual(kwargs["aws_access_key_id"], "AKIA_TEST")
        self.assertEqual(kwargs["aws_secret_access_key"], "SECRET_TEST")
        self.assertEqual(kwargs["aws_session_token"], "TOKEN_TEST")


if __name__ == "__main__":
    unittest.main()
