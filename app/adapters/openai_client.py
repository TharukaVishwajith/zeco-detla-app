import json
import logging
import re
from pathlib import Path

from langchain_core.messages import AIMessage
from openai import OpenAI

from app.core.conversation_context import latest_escalation_state, merge_evidence_from_conversation
from app.core.agent_models import (
    INTENT_AGENT_NAME,
    TROUBLESHOOTING_AGENT_NAME,
    AgentModelConfig,
    build_agent_model_config,
)
from app.models.conversation import (
    ChatMessageRequest,
    ConversationMessage,
    DeviceType,
    IntentClassification,
    IntentType,
    RetrievedDocument,
    SupportScopeStatus,
    TroubleshootingAction,
    TroubleshootingResponse,
    UnsupportedReason,
)
from app.models.evidence import EvidencePack, format_markdown_field_list, humanize_evidence_field


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

    def classify_intent(
        self,
        request: ChatMessageRequest,
        history: list[ConversationMessage] | None = None,
    ) -> IntentClassification:
        history = history or []
        message = request.message
        normalized_message = re.sub(r"\s+", " ", message).strip()
        active_escalation = latest_escalation_state(history) and not request.issue_resolved
        heuristic_evidence = merge_evidence_from_conversation(
            current_message=request.message,
            request_evidence=request.evidence_pack,
            history=history,
        )
        heuristic_missing_fields = heuristic_evidence.missing_core_fields()
        heuristic_missing_artifacts = heuristic_evidence.missing_best_effort_artifacts()
        heuristic_missing_field_labels = [humanize_evidence_field(field) for field in heuristic_missing_fields]
        heuristic_missing_artifact_labels = [humanize_evidence_field(field) for field in heuristic_missing_artifacts]
        required_ticket_field_labels = [humanize_evidence_field(field) for field in heuristic_evidence.required_core_fields()]
        prompt = self._load_prompt("intent_prompt.txt")
        history_block = self._format_history(history)
        user_prompt = (
            f"Conversation history (oldest first):\n{history_block}\n\n"
            f"Current user message (highest priority):\n{normalized_message}\n\n"
            f"Request ticket:\n{json.dumps(request.request_ticket)}\n\n"
            f"Prior escalation active:\n{json.dumps(active_escalation)}\n\n"
            f"Known device info:\n{request.device_info.model_dump_json()}\n\n"
            f"Customer info:\n{request.customer_info.model_dump_json()}\n\n"
            f"Structured evidence already provided:\n{request.evidence_pack.model_dump_json()}\n\n"
            f"Heuristic merged evidence so far:\n{heuristic_evidence.model_dump_json()}\n\n"
            f"Required ticket evidence fields:\n{json.dumps(required_ticket_field_labels)}\n\n"
            f"Heuristic missing core field labels:\n{json.dumps(heuristic_missing_field_labels)}\n\n"
            f"Heuristic missing artifact labels:\n{json.dumps(heuristic_missing_artifact_labels)}\n\n"
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
            classification = IntentClassification.model_validate(payload)
            classification.evidence_pack = heuristic_evidence.merge(classification.evidence_pack)
            classification.user_query = self._prioritize_current_message(
                current_message=normalized_message,
                user_query=classification.user_query,
                history=history,
            )
            classification.evidence_collection_response_text = self._normalize_evidence_collection_response_text(
                classification=classification,
                heuristic_evidence=heuristic_evidence,
                request=request,
                history=history,
            )
            return classification
        except Exception as exc:  # pragma: no cover - network/API failure path
            logger.warning("OpenAI classification failed, using heuristic fallback: %s", exc)
            fallback = self._heuristic_classification(request=request, history=history)
            fallback.evidence_pack = heuristic_evidence
            fallback.evidence_collection_response_text = self._fallback_evidence_collection_response(
                merged_evidence=heuristic_evidence,
                missing_fields=heuristic_missing_fields,
                support_scope_status=fallback.support_scope_status.value,
                safety_assessment={"escalate_immediately": bool(fallback.risk_flags)},
            ) if (request.request_ticket or active_escalation or fallback.intent == IntentType.escalate or fallback.risk_flags) else None
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
        if not self.client:
            return fallback

        prompt = self._load_prompt("troubleshooting_prompt.txt")
        if retrieved_docs:
            documents_block = "\n\n".join(
                f"Doc ID: {doc.doc_id}\nTitle: {doc.title}\nSection: {doc.section_title}\nContent: {doc.content}"
                for doc in retrieved_docs
            )
        else:
            documents_block = "No retrieved Delta KB documents were available for this turn."
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

    def generate_resolved_troubleshooting_response(self) -> TroubleshootingResponse:
        return TroubleshootingResponse(
            response_text=(
                "## Resolved\n\n"
                "Glad to hear the issue is resolved. I will close this here. "
                "If anything changes, send a new message and I can help again."
            ),
            citations=[],
            next_action=TroubleshootingAction.resolved,
        )

    def _grounded_fallback_response(
        self,
        message: str,
        retrieved_docs: list[RetrievedDocument],
        classification: IntentClassification,
    ) -> TroubleshootingResponse:
        if not retrieved_docs:
            opening = "## Try these next steps"
            if classification.error_code:
                opening = f"## First, check the {classification.error_code} condition"
            response_text = (
                f"{opening}\n\n"
                "Try the usual checks below:\n\n"
                "1. Confirm the device is powered and the fault text is still present.\n"
                "2. Perform a safe restart or reset if the equipment instructions allow it.\n"
                "3. Note any recent changes, then reply with the exact display text or LED state.\n\n"
            )
            if classification.error_code:
                response_text += f"This matches the reported code: `{classification.error_code}`.\n\n"
            response_text += "If the issue persists, would you like me to help create a support ticket?"
            return TroubleshootingResponse(
                response_text=response_text,
                citations=[],
                next_action=TroubleshootingAction.continue_troubleshooting,
            )

        primary_doc = retrieved_docs[0]
        excerpt = primary_doc.content.strip().replace("\n", " ")
        excerpt = excerpt[:320].rstrip()
        opening = "## First, try this"
        if classification.error_code:
            opening = f"## First, check the {classification.error_code} condition"

        response_text = (
            f"{opening}\n\n"
            "Follow this Delta-documented step:\n\n"
            "1. "
            f"{excerpt}.\n\n"
        )
        if classification.error_code:
            response_text += f"This matches the reported code: `{classification.error_code}`.\n\n"
        response_text += "Reply with the exact display message or LED state after this step.\n\n"
        response_text += "If the issue persists, would you like me to help create a support ticket?"
        return TroubleshootingResponse(
            response_text=response_text,
            citations=[doc.doc_id for doc in retrieved_docs[:3]],
            next_action=TroubleshootingAction.continue_troubleshooting,
        )

    def _fallback_evidence_collection_response(
        self,
        *,
        merged_evidence: EvidencePack,
        missing_fields: list[str],
        support_scope_status: str | None,
        safety_assessment: dict,
    ) -> str:
        field_list = format_markdown_field_list(missing_fields)
        provided_labels = sorted(humanize_evidence_field(name) for name in merged_evidence.provided_fields())
        progress_text = ""
        if provided_labels:
            preview = ", ".join(provided_labels[:4])
            if len(provided_labels) > 4:
                preview += ", ..."
            progress_text = f"I already have: {preview}.\n\n"

        if safety_assessment.get("escalate_immediately"):
            return (
                "## Immediate Safety Escalation\n\n"
                "A safety hazard was detected. Do not continue operating the equipment.\n\n"
                f"{progress_text}"
                "I can help create the support ticket.\n\n"
                "If you have any of these additional details, send them in one reply:\n"
                f"{field_list}\n\n"
                "If not, tell me and I will proceed with the information already gathered."
            )
        if support_scope_status == "unsupported":
            return (
                "## Unsupported Site Escalation\n\n"
                "This site is outside Delta AI support scope.\n\n"
                f"{progress_text}"
                "I can still help collect what is needed for the escalation ticket.\n\n"
                "If you have any of these additional details, send them in one reply:\n"
                f"{field_list}\n\n"
                "If not, tell me and I will proceed with the information already gathered."
            )
        return (
            "## Ticket Information Needed\n\n"
            f"{progress_text}"
            "I can create the support ticket for you.\n\n"
            "If you have any of these additional details, send them in one reply:\n"
            f"{field_list}\n\n"
            "If not, tell me and I will proceed with the information already gathered."
        )

    def _normalize_evidence_collection_response_text(
        self,
        *,
        classification: IntentClassification,
        heuristic_evidence: EvidencePack,
        request: ChatMessageRequest,
        history: list[ConversationMessage],
    ) -> str | None:
        active_escalation = latest_escalation_state(history) and not request.issue_resolved
        if not (request.request_ticket or active_escalation or classification.intent == IntentType.escalate or classification.risk_flags):
            return None
        response_text = (classification.evidence_collection_response_text or "").strip()
        if response_text:
            return response_text
        return self._fallback_evidence_collection_response(
            merged_evidence=classification.evidence_pack or heuristic_evidence,
            missing_fields=(classification.evidence_pack or heuristic_evidence).missing_core_fields(),
            support_scope_status=classification.support_scope_status.value,
            safety_assessment={"escalate_immediately": bool(classification.risk_flags)},
        )

    def _heuristic_classification(
        self,
        request: ChatMessageRequest,
        history: list[ConversationMessage] | None = None,
    ) -> IntentClassification:
        history = history or []
        message = request.message
        lowered = message.lower()
        recent_history = [item.content for item in history[-6:] if item.content]
        combined_text = "\n".join([*recent_history, message])
        combined_lowered = combined_text.lower()
        active_escalation = latest_escalation_state(history) and not request.issue_resolved
        risk_flags = [term for term in SAFETY_TERMS if term in combined_lowered]

        device_type = DeviceType.unknown
        if "inverter" in combined_lowered:
            device_type = DeviceType.inverter
        elif "battery" in combined_lowered:
            device_type = DeviceType.battery
        elif "pv" in combined_lowered or "panel" in combined_lowered or "solar" in combined_lowered:
            device_type = DeviceType.pv
        elif "monitor" in combined_lowered or "gateway" in combined_lowered or "meter" in combined_lowered:
            device_type = DeviceType.monitoring
        else:
            device_type = request.device_info.device_type

        error_match = re.search(r"\b([A-Z]{1,4}[- ]?\d{2,5})\b", message) or re.search(
            r"\b([A-Z]{1,4}[- ]?\d{2,5})\b",
            combined_text,
        )
        model_number = request.device_info.model_number
        has_domain_context = self._has_domain_context(
            lowered_message=combined_lowered,
            device_type=device_type,
            model_number=model_number,
            has_error_code=bool(error_match),
        )

        intent = IntentType.troubleshoot
        if risk_flags or active_escalation or any(term in lowered for term in ("ticket", "escalate", "support case", "technician")):
            intent = IntentType.escalate
        elif any(term in lowered for term in ("how do", "what is", "where can", "manual")) or "?" in message:
            intent = IntentType.general_question
        elif not has_domain_context and self._is_brief_message(lowered):
            intent = IntentType.general_question

        missing_info = []
        if not model_number:
            missing_info.append("model_number")
        if not error_match:
            missing_info.append("error_code")
        if intent == IntentType.general_question and not has_domain_context:
            missing_info.append("issue_or_question_details")
        system_message = None if intent == IntentType.escalate else (
            self._heuristic_system_message(message) if "issue_or_question_details" in missing_info else None
        )
        user_query = self._heuristic_user_query(message, history)
        support_scope_status, unsupported_reason = self._heuristic_support_scope(request=request, combined_lowered=combined_lowered)

        return IntentClassification(
            intent=intent,
            device_type=device_type,
            user_query=user_query,
            error_code=error_match.group(1).replace(" ", "-") if error_match else None,
            model_number=model_number,
            evidence_pack=request.evidence_pack,
            risk_flags=risk_flags,
            missing_info=missing_info,
            support_scope_status=support_scope_status,
            unsupported_reason=unsupported_reason,
            missing_scope_fields=[],
            system_message=system_message,
        )

    def _heuristic_support_scope(
        self,
        request: ChatMessageRequest,
        combined_lowered: str,
    ) -> tuple[SupportScopeStatus, UnsupportedReason | None]:
        ownership_verified = request.evidence_pack.ownership_verified
        if any(term in combined_lowered for term in ("home", "residential", "home use", "my system", "our system", "owner", "customer owner")):
            if ownership_verified is not False:
                return SupportScopeStatus.supported, None
        if any(term in combined_lowered for term in ("80 kw", "80kw", "over 30 kw", "greater than 30 kw", "above 30 kw")):
            return SupportScopeStatus.unsupported, UnsupportedReason.site_capacity_exceeded
        if any(term in combined_lowered for term in ("industrial", "major commercial")):
            return SupportScopeStatus.unsupported, UnsupportedReason.industrial_site
        if any(term in combined_lowered for term in ("utility-scale", "utility scale", "embedded network")):
            return SupportScopeStatus.unsupported, UnsupportedReason.utility_scale_or_embedded_network
        if ownership_verified is False or any(term in combined_lowered for term in ("unknown owner", "not sure who owns")):
            return SupportScopeStatus.unsupported, UnsupportedReason.ownership_unverifiable
        return SupportScopeStatus.unknown, None

    def _is_brief_message(self, lowered_message: str) -> bool:
        words = re.findall(r"[a-z0-9]+", lowered_message)
        return 0 < len(words) <= 4

    def _has_domain_context(
        self,
        lowered_message: str,
        device_type: DeviceType,
        model_number: str | None,
        has_error_code: bool,
    ) -> bool:
        if device_type != DeviceType.unknown or model_number or has_error_code:
            return True
        domain_terms = (
            "inverter",
            "battery",
            "pv",
            "panel",
            "solar",
            "monitor",
            "gateway",
            "meter",
            "fault",
            "alarm",
            "error",
            "trip",
            "shutdown",
        )
        return any(term in lowered_message for term in domain_terms)

    def _heuristic_system_message(self, user_message: str) -> str:
        snippet = re.sub(r"\s+", " ", user_message).strip()
        if len(snippet) > 80:
            snippet = f"{snippet[:77].rstrip()}..."
        if snippet:
            return (
                "## Tell me a bit more\n\n"
                f"You mentioned: \"{snippet}\"\n\n"
                "Please send one of these so I can help:\n"
                "- What issue you are seeing\n"
                "- The exact alarm or error text\n"
                "- The model number"
            )
        return (
            "## Tell me a bit more\n\n"
            "Please send one of these so I can help:\n"
            "- What issue you are seeing\n"
            "- The exact alarm or error text\n"
            "- The model number"
        )

    def _heuristic_user_query(self, current_message: str, history: list[ConversationMessage]) -> str:
        if not history:
            return current_message

        context_lines: list[str] = []
        for message in history:
            content = re.sub(r"\s+", " ", message.content).strip()
            if not content:
                continue
            context_lines.append(f"{message.role.value}: {content}")

        if not context_lines:
            return current_message
        return f"{current_message} | context: {' ; '.join(context_lines)}"

    def _prioritize_current_message(
        self,
        current_message: str,
        user_query: str | None,
        history: list[ConversationMessage],
    ) -> str:
        if not current_message:
            return ""

        normalized_current = re.sub(r"\s+", " ", current_message).strip()
        normalized_user_query = re.sub(r"\s+", " ", (user_query or "")).strip()
        if not normalized_user_query:
            return self._heuristic_user_query(normalized_current, history)

        if normalized_current.lower() in normalized_user_query.lower():
            return normalized_user_query

        return f"{normalized_current} | context: {normalized_user_query}"

    def _format_history(self, history: list[ConversationMessage]) -> str:
        if not history:
            return "(none)"

        lines = []
        for message in history:
            content = re.sub(r"\s+", " ", message.content).strip()
            if not content:
                continue
            metadata = []
            if message.intent:
                metadata.append(f"intent={message.intent.value}")
            if message.next_action:
                metadata.append(f"next_action={message.next_action.value}")
            if message.escalation_active is not None:
                metadata.append(f"escalation_active={'true' if message.escalation_active else 'false'}")
            label = message.role.value.upper()
            if metadata:
                label = f"{label} [{' '.join(metadata)}]"
            lines.append(f"{label}: {content}")
        return "\n".join(lines) if lines else "(none)"

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
