import json
import logging
import re
from pathlib import Path

from langchain_core.messages import AIMessage
from openai import OpenAI

from app.core.agent_models import (
    INTENT_AGENT_NAME,
    TROUBLESHOOTING_AGENT_NAME,
    AgentModelConfig,
    build_agent_model_config,
)
from app.models.conversation import (
    DeviceInfo,
    DeviceType,
    IntentClassification,
    IntentType,
    RetrievedDocument,
    TroubleshootingAction,
    TroubleshootingResponse,
)


logger = logging.getLogger(__name__)

try:
    from langchain.agents import create_agent
except ImportError:  # pragma: no cover - dependency mismatch path
    create_agent = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - dependency mismatch path
    ChatOpenAI = None

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
SAFETY_TERMS = {
    "fire",
    "smoke",
    "burning smell",
    "sparking",
    "water damage",
    "electrical hazard",
    "overheating",
    "hot enclosure",
}


class OpenAIClient:
    def __init__(
        self,
        api_key: str | None,
        chat_model: str,
        embedding_model: str,
        agent_model_config: AgentModelConfig | None = None,
    ):
        self.api_key = api_key
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.agent_model_config = agent_model_config or build_agent_model_config(default_model=chat_model, raw_overrides=None)
        self.client = OpenAI(api_key=api_key) if api_key else None
        self._agent_cache: dict[tuple[str, str], object] = {}

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def create_embedding(self, text: str, dimensions: int | None = None) -> list[float] | None:
        if not self.client:
            return None

        kwargs = {"model": self.embedding_model, "input": text}
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        response = self.client.embeddings.create(**kwargs)
        return response.data[0].embedding

    def classify_intent(self, message: str, device_info: DeviceInfo | None = None) -> IntentClassification:
        # fallback = self._heuristic_classification(message=message, device_info=device_info)
        # if not self.client:
        #     return fallback

        prompt = self._load_prompt("intent_prompt.txt")
        user_prompt = (
            f"User message:\n{message}\n\n"
            f"Known device info:\n{device_info.model_dump_json() if device_info else '{}'}\n\n"
            "Return JSON only."
        )

        try:
            payload = self._invoke_agent_json(
                agent_name=INTENT_AGENT_NAME,
                system_prompt=prompt,
                user_prompt=user_prompt,
            )
            if payload is None:
                raise RuntimeError("LangChain agent unavailable for intent classification")
            return IntentClassification.model_validate(payload)
        except Exception as exc:  # pragma: no cover - network/API failure path
            logger.warning("OpenAI classification failed, using heuristic fallback: %s", exc)
            fallback = self._heuristic_classification(message=message, device_info=device_info)
            if not self.client:
                return fallback
            return fallback

    def generate_troubleshooting_response(
        self,
        message: str,
        retrieved_docs: list[RetrievedDocument],
        classification: IntentClassification,
    ) -> TroubleshootingResponse:
        fallback = self._grounded_fallback_response(message, retrieved_docs, classification)
        if not self.client or not retrieved_docs:
            return fallback

        prompt = self._load_prompt("troubleshooting_prompt.txt")
        documents_block = "\n\n".join(
            f"Doc ID: {doc.doc_id}\nTitle: {doc.title}\nSection: {doc.section_title}\nContent: {doc.content}"
            for doc in retrieved_docs
        )
        user_prompt = (
            f"User message:\n{message}\n\n"
            f"Classification:\n{classification.model_dump_json()}\n\n"
            f"Retrieved Delta KB documents:\n{documents_block}\n\n"
            "Return JSON only."
        )

        try:
            payload = self._invoke_agent_json(
                agent_name=TROUBLESHOOTING_AGENT_NAME,
                system_prompt=prompt,
                user_prompt=user_prompt,
            )
            if payload is None:
                raise RuntimeError("LangChain agent unavailable for troubleshooting generation")
            return TroubleshootingResponse.model_validate(payload)
        except Exception as exc:  # pragma: no cover - network/API failure path
            logger.warning("OpenAI troubleshooting generation failed, using fallback: %s", exc)
            return fallback

    def _grounded_fallback_response(
        self,
        message: str,
        retrieved_docs: list[RetrievedDocument],
        classification: IntentClassification,
    ) -> TroubleshootingResponse:
        if not retrieved_docs:
            missing = ["model number", "exact error code", "recent changes"]
            response_text = (
                "I could not find a matching Delta knowledge-base article from the details provided. "
                f"Please share the {', '.join(missing)} so I can continue, or request escalation."
            )
            return TroubleshootingResponse(
                response_text=response_text,
                citations=[],
                next_action=TroubleshootingAction.ask_question,
            )

        primary_doc = retrieved_docs[0]
        excerpt = primary_doc.content.strip().replace("\n", " ")
        excerpt = excerpt[:320].rstrip()
        response_text = (
            f"Based on Delta KB document {primary_doc.doc_id}"
            + (f" ({primary_doc.section_title})" if primary_doc.section_title else "")
            + f", start with this documented guidance: {excerpt}."
        )
        if classification.error_code:
            response_text += f" This appears related to error code {classification.error_code}."
        response_text += " If the issue persists after that step, reply with the observed LED state or exact fault text."
        return TroubleshootingResponse(
            response_text=response_text,
            citations=[doc.doc_id for doc in retrieved_docs[:3]],
            next_action=TroubleshootingAction.continue_troubleshooting,
        )

    def _heuristic_classification(self, message: str, device_info: DeviceInfo | None) -> IntentClassification:
        lowered = message.lower()
        risk_flags = [term for term in SAFETY_TERMS if term in lowered]
        intent = IntentType.troubleshoot
        if any(term in lowered for term in ("ticket", "escalate", "support case", "technician")):
            intent = IntentType.escalate
        elif any(term in lowered for term in ("how do", "what is", "where can", "manual")):
            intent = IntentType.general_question

        device_type = DeviceType.unknown
        if "inverter" in lowered:
            device_type = DeviceType.inverter
        elif "battery" in lowered:
            device_type = DeviceType.battery
        elif "pv" in lowered or "panel" in lowered or "solar" in lowered:
            device_type = DeviceType.pv
        elif "monitor" in lowered or "gateway" in lowered or "meter" in lowered:
            device_type = DeviceType.monitoring
        elif device_info:
            device_type = device_info.device_type

        error_match = re.search(r"\b([A-Z]{1,4}[- ]?\d{2,5})\b", message)
        model_number = device_info.model_number if device_info else None
        missing_info = []
        if not model_number:
            missing_info.append("model_number")
        if not error_match:
            missing_info.append("error_code")

        return IntentClassification(
            intent=intent,
            device_type=device_type,
            error_code=error_match.group(1).replace(" ", "-") if error_match else None,
            model_number=model_number,
            risk_flags=risk_flags,
            missing_info=missing_info,
        )

    def _load_prompt(self, filename: str) -> str:
        return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()

    def _invoke_agent_json(self, agent_name: str, system_prompt: str, user_prompt: str) -> dict | None:
        if not self.api_key or not create_agent or not ChatOpenAI:
            return None

        model_name = self.agent_model_config.model_for(agent_name)
        cache_key = (agent_name, system_prompt)
        agent = self._agent_cache.get(cache_key)
        if agent is None:
            chat_model = ChatOpenAI(model=model_name, api_key=self.api_key, temperature=0)
            agent = create_agent(
                model=chat_model,
                tools=[],
                system_prompt=system_prompt,
            )
            self._agent_cache[cache_key] = agent

        result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
        content = self._extract_agent_text(result)
        return self._parse_json_payload(content)

    def _extract_agent_text(self, result: dict) -> str:
        messages = result.get("messages", []) if isinstance(result, dict) else []
        if not messages:
            return ""

        last_message = messages[-1]
        if isinstance(last_message, AIMessage):
            content = last_message.content
        elif isinstance(last_message, dict):
            content = last_message.get("content", "")
        else:
            content = getattr(last_message, "content", "")

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return str(content)

    def _parse_json_payload(self, content: str) -> dict:
        text = content.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise
