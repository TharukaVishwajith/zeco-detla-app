from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.models.evidence import EvidencePack
from app.models.ticket import CustomerInfo, TicketResponse


class IntentType(str, Enum):
    troubleshoot = "troubleshoot"
    escalate = "escalate"
    general_question = "general_question"


class DeviceType(str, Enum):
    inverter = "inverter"
    battery = "battery"
    pv = "pv"
    monitoring = "monitoring"
    unknown = "unknown"


class DeviceInfo(BaseModel):
    device_type: DeviceType = DeviceType.unknown
    model_number: str | None = None
    serial_number: str | None = None
    firmware_version: str | None = None
    product_family: str | None = None


class IntentClassification(BaseModel):
    intent: IntentType
    device_type: DeviceType = DeviceType.unknown
    error_code: str | None = None
    model_number: str | None = None
    risk_flags: list[str] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)
    system_message: str | None = None


class SafetyAssessment(BaseModel):
    escalate_immediately: bool = False
    reason: str | None = None
    safety_flags: list[str] = Field(default_factory=list)


class RetrievedDocument(BaseModel):
    doc_id: str
    title: str | None = None
    product: str | None = None
    model: str | None = None
    firmware: str | None = None
    error_code: str | None = None
    doc_type: str | None = None
    section_title: str | None = None
    page_number: int | None = None
    content: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TroubleshootingAction(str, Enum):
    ask_question = "ask_question"
    continue_troubleshooting = "continue_troubleshooting"
    collect_evidence = "collect_evidence"
    escalate = "escalate"
    resolved = "resolved"


class TroubleshootingResponse(BaseModel):
    response_text: str
    citations: list[str] = Field(default_factory=list)
    next_action: TroubleshootingAction


class ChatMessageRequest(BaseModel):
    request_id: str | None = None
    user_id: str | None = None
    message: str
    customer_info: CustomerInfo = Field(default_factory=CustomerInfo)
    device_info: DeviceInfo = Field(default_factory=DeviceInfo)
    evidence_pack: EvidencePack = Field(default_factory=EvidencePack)
    request_ticket: bool = False
    issue_resolved: bool = False
    top_k: int | None = None


class ChatMessageResponse(BaseModel):
    request_id: str | None = None
    current_phase: str
    intent: IntentType
    device_type: DeviceType
    response_text: str
    system_message: str | None = None
    citations: list[str] = Field(default_factory=list)
    next_action: TroubleshootingAction
    missing_fields: list[str] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    ticket: TicketResponse | None = None
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)


class TicketCreateRequest(BaseModel):
    customer_info: CustomerInfo = Field(default_factory=CustomerInfo)
    device_info: DeviceInfo = Field(default_factory=DeviceInfo)
    issue_summary: str
    troubleshooting_steps: list[str] = Field(default_factory=list)
    escalation_reason: str | None = None
    evidence_pack: EvidencePack = Field(default_factory=EvidencePack)
    attachments: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool
    opensearch_configured: bool
    ticket_api_configured: bool
