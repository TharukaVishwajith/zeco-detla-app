from typing import Any

from pydantic import BaseModel, Field


OPTIONAL_EVIDENCE_FIELDS = {"battery_model", "system_topology", "photos", "logs"}


class EvidencePack(BaseModel):
    site_type: str | None = None
    system_size_kw: str | None = None
    inverter_model: str | None = None
    serial_number: str | None = None
    firmware_version: str | None = None
    battery_model: str | None = None
    error_code: str | None = None
    timestamp: str | None = None
    system_topology: str | None = None
    photos: list[str] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
    recent_changes: str | None = None
    environmental_conditions: str | None = None

    def provided_fields(self) -> dict[str, Any]:
        payload = self.model_dump()
        return {key: value for key, value in payload.items() if value not in (None, "", [], {})}

    def missing_required_fields(self) -> list[str]:
        provided = self.provided_fields()
        missing = []
        for field_name in self.__class__.model_fields:
            if field_name in OPTIONAL_EVIDENCE_FIELDS:
                continue
            if field_name not in provided:
                missing.append(field_name)
        return missing
