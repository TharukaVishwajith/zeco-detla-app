from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CustomerInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    first_name: str | None = Field(default=None, alias="firstName")
    last_name: str | None = Field(default=None, alias="lastName")
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    site_id: str | None = None

    @model_validator(mode="after")
    def populate_name_from_parts(self) -> "CustomerInfo":
        if self.name:
            return self

        first_name = (self.first_name or "").strip()
        last_name = (self.last_name or "").strip()
        combined_name = " ".join(part for part in (first_name, last_name) if part)
        if combined_name:
            self.name = combined_name
        return self


class TicketPayload(BaseModel):
    customer_info: CustomerInfo = Field(default_factory=CustomerInfo)
    device_info: dict[str, Any] = Field(default_factory=dict)
    issue_summary: str
    troubleshooting_steps: list[str] = Field(default_factory=list)
    attachments: list[str] = Field(default_factory=list)
    escalation_reason: str | None = None
    evidence_pack: dict[str, Any] = Field(default_factory=dict)


class TicketResponse(BaseModel):
    ticket_id: str
    status: str
    message: str
