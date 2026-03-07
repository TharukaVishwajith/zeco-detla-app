import json
import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)

INTENT_AGENT_NAME = "intent_classifier"
TROUBLESHOOTING_AGENT_NAME = "troubleshooting"

DEFAULT_AGENT_MODEL_BY_NAME = {
    INTENT_AGENT_NAME: "gpt-5.2",
    TROUBLESHOOTING_AGENT_NAME: "gpt-5.2",
}


@dataclass(slots=True, frozen=True)
class AgentModelConfig:
    default_model: str
    model_by_agent: dict[str, str]

    def model_for(self, agent_name: str) -> str:
        return self.model_by_agent.get(agent_name, self.default_model)


def build_agent_model_config(default_model: str, raw_overrides: str | None) -> AgentModelConfig:
    model_by_agent = dict(DEFAULT_AGENT_MODEL_BY_NAME)
    if raw_overrides:
        try:
            overrides = json.loads(raw_overrides)
            if isinstance(overrides, dict):
                model_by_agent.update({str(key): str(value) for key, value in overrides.items()})
            else:
                logger.warning("OPENAI_AGENT_MODELS is not a JSON object; ignoring override.")
        except json.JSONDecodeError:
            logger.warning("OPENAI_AGENT_MODELS is not valid JSON; ignoring override.")

    return AgentModelConfig(default_model=default_model, model_by_agent=model_by_agent)
