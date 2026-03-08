import re

from app.models.conversation import ChatMessageRequest, ConversationMessage
from app.models.evidence import EvidencePack


SITE_TYPE_PATTERNS = {
    "residential": "residential",
    "home": "residential",
    "small commercial": "small_commercial",
    "light commercial": "small_commercial",
    "commercial": "commercial",
    "industrial": "industrial",
    "utility-scale": "utility_scale",
    "utility scale": "utility_scale",
    "embedded network": "embedded_network",
}
USER_ROLE_PATTERNS = {
    "licensed installer": "licensed_installer",
    "installer": "licensed_installer",
    "electrician": "licensed_installer",
    "customer": "customer_owner",
    "owner": "customer_owner",
    "homeowner": "customer_owner",
}
SYSTEM_TOPOLOGY_PATTERNS = {
    "ac-coupled": "ac_coupled",
    "ac coupled": "ac_coupled",
    "dc-coupled": "dc_coupled",
    "dc coupled": "dc_coupled",
}
PHASE_TYPE_PATTERNS = {
    "single phase": "single_phase",
    "three phase": "three_phase",
}
TRUE_PATTERNS = ("yes", "present", "available", "provided", "true")
FALSE_PATTERNS = ("no", "not available", "unavailable", "false")


def _latest_evidence_snapshot(history: list[ConversationMessage]) -> EvidencePack:
    for message in reversed(history):
        if message.evidence_snapshot:
            return message.evidence_snapshot
    return EvidencePack()


def _parse_bool(text: str, positive_terms: tuple[str, ...], negative_terms: tuple[str, ...]) -> bool | None:
    lowered = text.lower()
    if any(term in lowered for term in positive_terms):
        return True
    if any(term in lowered for term in negative_terms):
        return False
    return None


def _match_mapping(text: str, mapping: dict[str, str]) -> str | None:
    lowered = text.lower()
    for term, normalized in mapping.items():
        if term in lowered:
            return normalized
    return None


def _extract_message_evidence(text: str) -> EvidencePack:
    size_match = re.search(r"\b(\d+(?:\.\d+)?)\s?k[wW]\b", text)
    inverter_match = re.search(r"\binverter model[:\s]+([A-Za-z0-9_-]+)\b", text, re.IGNORECASE)
    battery_match = re.search(r"\bbattery model[:\s]+([A-Za-z0-9_-]+)\b", text, re.IGNORECASE)
    serial_match = re.search(r"\bserial(?: number)?[:\s]+([A-Za-z0-9_-]+)\b", text, re.IGNORECASE)
    firmware_match = re.search(r"\bfirmware(?: version)?[:\s]+([A-Za-z0-9._-]+)\b", text, re.IGNORECASE)
    battery_firmware_match = re.search(r"\bbattery firmware(?: version)?[:\s]+([A-Za-z0-9._-]+)\b", text, re.IGNORECASE)
    error_match = re.search(r"\b([A-Z]{1,4}[- ]?\d{2,5})\b", text)
    timestamp_match = re.search(r"\b(\d{4}-\d{2}-\d{2}[T ][0-9:]{5,8}Z?)\b", text)
    app_match = re.search(r"\b(?:app|portal) version[:\s]+([A-Za-z0-9._-]+)\b", text, re.IGNORECASE)
    ownership_verified = None
    if "not sure who owns" in text.lower() or "unknown owner" in text.lower():
        ownership_verified = False
    elif any(term in text.lower() for term in ("my system", "my site", "our site", "our system")):
        ownership_verified = True
    backup_loads_present = None
    if "backup loads" in text.lower():
        backup_loads_present = _parse_bool(text, TRUE_PATTERNS, FALSE_PATTERNS)

    screenshot_available = None
    screenshot_provided = None
    if "screenshot" in text.lower():
        screenshot_available = True
        screenshot_provided = _parse_bool(text, ("attached", "uploaded", "provided"), ("not available", "unable", "cannot"))

    return EvidencePack(
        site_type=_match_mapping(text, SITE_TYPE_PATTERNS),
        system_size_kw=size_match.group(1) if size_match else None,
        inverter_model=inverter_match.group(1) if inverter_match else None,
        serial_number=serial_match.group(1) if serial_match else None,
        firmware_version=firmware_match.group(1) if firmware_match else None,
        battery_model=battery_match.group(1) if battery_match else None,
        battery_firmware_version=battery_firmware_match.group(1) if battery_firmware_match else None,
        error_code=error_match.group(1).replace(" ", "-") if error_match else None,
        timestamp=timestamp_match.group(1) if timestamp_match else None,
        system_topology=_match_mapping(text, SYSTEM_TOPOLOGY_PATTERNS),
        user_role=_match_mapping(text, USER_ROLE_PATTERNS),
        ownership_verified=ownership_verified,
        phase_type=_match_mapping(text, PHASE_TYPE_PATTERNS),
        backup_loads_present=backup_loads_present,
        app_or_portal_version=app_match.group(1) if app_match else None,
        screenshot_available=screenshot_available,
        screenshot_provided=screenshot_provided,
    )


def build_evidence_extraction_node():
    def evidence_extraction_node(state: dict) -> dict:
        request = ChatMessageRequest.model_validate(state["request"])
        history = [ConversationMessage.model_validate(item) for item in state.get("history", [])]
        baseline = _latest_evidence_snapshot(history)
        extracted_from_history = EvidencePack()
        for message in history:
            if message.role.value != "user":
                continue
            extracted_from_history = extracted_from_history.merge(_extract_message_evidence(message.content))
        extracted_from_message = _extract_message_evidence(request.message)
        merged = baseline.merge(extracted_from_history).merge(request.evidence_pack).merge(extracted_from_message)

        return {
            "merged_evidence_pack": merged.model_dump(mode="json", exclude_none=True),
            "missing_fields": merged.missing_core_fields(),
            "missing_artifacts": merged.missing_best_effort_artifacts(),
            "history": [],
            "current_phase": "evidence_extraction",
        }

    return evidence_extraction_node
