from typing import Any

from pydantic import BaseModel, Field


CORE_EVIDENCE_FIELDS = {
    "site_type",
    "system_size_kw",
    "user_role",
    "ownership_verified",
    "inverter_model",
    "serial_number",
    "firmware_version",
    "error_code",
    "timestamp",
    "system_topology",
    "phase_type",
    "backup_loads_present",
    "recent_changes",
    "environmental_conditions",
}
BEST_EFFORT_ARTIFACT_FIELDS = {
    "photos",
    "logs",
    "screenshot_provided",
    "app_or_portal_version",
    "photo_checklist_completed",
    "log_export_steps_provided",
    "log_time_period",
}
PHOTO_CHECKLIST_FIELDS = {
    "equipment_rating_labels",
    "installation_overview",
    "meter_ct_orientation",
    "indicator_leds_or_display",
}


class EvidencePack(BaseModel):
    site_type: str | None = None
    system_size_kw: str | None = None
    inverter_model: str | None = None
    serial_number: str | None = None
    firmware_version: str | None = None
    battery_model: str | None = None
    battery_firmware_version: str | None = None
    error_code: str | None = None
    timestamp: str | None = None
    system_topology: str | None = None
    user_role: str | None = None
    ownership_verified: bool | None = None
    phase_type: str | None = None
    backup_loads_present: bool | None = None
    app_or_portal_version: str | None = None
    screenshot_available: bool | None = None
    screenshot_provided: bool | None = None
    photo_checklist_completed: list[str] = Field(default_factory=list)
    log_export_steps_provided: bool | None = None
    log_time_period: str | None = None
    photos: list[str] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
    recent_changes: str | None = None
    environmental_conditions: str | None = None

    def provided_fields(self) -> dict[str, Any]:
        payload = self.model_dump()
        return {key: value for key, value in payload.items() if value not in (None, "", [], {})}

    def merge(self, other: "EvidencePack | None") -> "EvidencePack":
        if other is None:
            return self

        merged = self.model_dump()
        for key, value in other.model_dump().items():
            if value in (None, "", [], {}):
                continue
            if isinstance(value, list):
                combined = [*merged.get(key, []), *value]
                merged[key] = list(dict.fromkeys(combined))
                continue
            merged[key] = value
        return EvidencePack.model_validate(merged)

    def missing_core_fields(self) -> list[str]:
        provided = self.provided_fields()
        missing = [field_name for field_name in CORE_EVIDENCE_FIELDS if field_name not in provided]
        battery_present = bool(self.battery_model)
        if battery_present and "battery_firmware_version" not in provided:
            missing.append("battery_firmware_version")
        return sorted(missing)

    def missing_best_effort_artifacts(self) -> list[str]:
        provided = self.provided_fields()
        missing = [field_name for field_name in BEST_EFFORT_ARTIFACT_FIELDS if field_name not in provided]
        if self.screenshot_available is False:
            missing = [field_name for field_name in missing if field_name != "screenshot_provided"]
        if self.photo_checklist_completed:
            completed = set(self.photo_checklist_completed)
            for required_item in PHOTO_CHECKLIST_FIELDS - completed:
                missing.append(f"photo_checklist:{required_item}")
        return sorted(set(missing))
