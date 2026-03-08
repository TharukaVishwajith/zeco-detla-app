from typing import Any

from typing_extensions import NotRequired, TypedDict


class SupportGraphState(TypedDict, total=False):
    request: dict[str, Any]
    history: list[dict[str, Any]]
    user_query: str
    classification: dict[str, Any]
    merged_evidence_pack: dict[str, Any]
    safety_assessment: dict[str, Any]
    retrieved_docs: list[dict[str, Any]]
    troubleshooting_response: dict[str, Any]
    missing_fields: list[str]
    missing_artifacts: list[str]
    ticket_response: dict[str, Any]
    current_phase: str
    response_text: str
    system_message: str
    citations: list[str]
    next_action: str
    safety_flags: list[str]
    support_scope_status: str
    unsupported_reason: str
    missing_scope_fields: list[str]
    errors: NotRequired[list[str]]
