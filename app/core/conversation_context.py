import re

from app.models.conversation import ConversationMessage, ConversationRole, TroubleshootingAction
from app.models.evidence import EvidencePack


TRUE_PATTERNS = ("yes", "present", "available", "provided", "true")
FALSE_PATTERNS = ("no", "not available", "unavailable", "false")


def latest_evidence_snapshot(history: list[ConversationMessage]) -> EvidencePack:
    for message in reversed(history):
        if message.evidence_snapshot:
            return message.evidence_snapshot
    return EvidencePack()


def latest_escalation_state(history: list[ConversationMessage]) -> bool:
    for message in reversed(history):
        if message.role != ConversationRole.assistant:
            continue
        if message.escalation_active is not None:
            return message.escalation_active
        if message.next_action in {TroubleshootingAction.collect_evidence, TroubleshootingAction.escalate}:
            return True
        if message.next_action in {
            TroubleshootingAction.ask_question,
            TroubleshootingAction.continue_troubleshooting,
            TroubleshootingAction.resolved,
        }:
            return False
        if message.intent == "escalate":
            return True
        if message.intent in {"troubleshoot", "general_question"}:
            return False
    return False


def merge_evidence_from_conversation(
    *,
    current_message: str,
    request_evidence: EvidencePack,
    history: list[ConversationMessage],
) -> EvidencePack:
    baseline = latest_evidence_snapshot(history)
    extracted_from_history = EvidencePack()
    for message in history:
        if message.role != ConversationRole.user:
            continue
        extracted_from_history = extracted_from_history.merge(extract_message_evidence(message.content))
    extracted_from_message = extract_message_evidence(current_message)
    return baseline.merge(extracted_from_history).merge(request_evidence).merge(extracted_from_message)


def extract_message_evidence(text: str) -> EvidencePack:
    inverter_match = re.search(
        r"\b(?:inverter model|model number|model)\s*(?:is|=|:)?\s*([A-Za-z0-9_-]+)\b",
        text,
        re.IGNORECASE,
    )
    battery_match = re.search(r"\bbattery model\s*(?:is|=|:)?\s*([A-Za-z0-9_-]+)\b", text, re.IGNORECASE)
    serial_match = re.search(r"\bserial(?: number)?\s*(?:is|=|:)?\s*([A-Za-z0-9_-]+)\b", text, re.IGNORECASE)
    firmware_match = re.search(r"\bfirmware(?: version)?\s*(?:is|=|:)?\s*([A-Za-z0-9._-]+)\b", text, re.IGNORECASE)
    battery_firmware_match = re.search(
        r"\bbattery firmware(?: version)?\s*(?:is|=|:)?\s*([A-Za-z0-9._-]+)\b",
        text,
        re.IGNORECASE,
    )
    error_match = re.search(r"\b([A-Z]{1,4}[- ]?\d{2,5})\b", text)
    timestamp_match = re.search(r"\b(\d{4}-\d{2}-\d{2}[T ][0-9:]{5,8}Z?)\b", text)
    app_match = re.search(r"\b(?:app|portal) version\s*(?:is|=|:)?\s*([A-Za-z0-9._-]+)\b", text, re.IGNORECASE)
    ownership_verified = None
    lowered = text.lower()
    if "not sure who owns" in lowered or "unknown owner" in lowered:
        ownership_verified = False
    elif any(
        term in lowered
        for term in (
            "my system",
            "my site",
            "our site",
            "our system",
            "i am the owner",
            "i'm the owner",
            "owner here",
            "i am responsible for this site",
            "i'm responsible for this site",
        )
    ):
        ownership_verified = True
    backup_loads_present = None
    if "backup loads" in lowered:
        backup_loads_present = _parse_bool(text, TRUE_PATTERNS, FALSE_PATTERNS)
    elif "no backup" in lowered:
        backup_loads_present = False
    elif "with backup" in lowered or "has backup" in lowered:
        backup_loads_present = True

    screenshot_available = None
    screenshot_provided = None
    if "screenshot" in lowered:
        screenshot_available = True
        screenshot_provided = _parse_bool(text, ("attached", "uploaded", "provided"), ("not available", "unable", "cannot"))

    return EvidencePack(
        inverter_model=inverter_match.group(1) if inverter_match else None,
        serial_number=serial_match.group(1) if serial_match else None,
        firmware_version=firmware_match.group(1) if firmware_match else None,
        battery_model=battery_match.group(1) if battery_match else None,
        battery_firmware_version=battery_firmware_match.group(1) if battery_firmware_match else None,
        error_code=error_match.group(1).replace(" ", "-") if error_match else None,
        timestamp=timestamp_match.group(1) if timestamp_match else None,
        user_role=_extract_user_role(text),
        ownership_verified=ownership_verified,
        backup_loads_present=backup_loads_present,
        app_or_portal_version=app_match.group(1) if app_match else None,
        screenshot_available=screenshot_available,
        screenshot_provided=screenshot_provided,
        recent_changes=_extract_recent_changes(text),
    )


def _parse_bool(text: str, positive_terms: tuple[str, ...], negative_terms: tuple[str, ...]) -> bool | None:
    lowered = text.lower()
    if any(term in lowered for term in positive_terms):
        return True
    if any(term in lowered for term in negative_terms):
        return False
    return None


def _extract_recent_changes(text: str) -> str | None:
    lowered = text.lower()
    phrases = (
        "after",
        "since",
        "before this started",
        "started when",
        "happened after",
        "changed after",
        "grid outage",
        "power outage",
        "storm",
        "update",
        "installation",
        "maintenance",
    )
    if not any(phrase in lowered for phrase in phrases):
        return None
    normalized = " ".join(text.split()).strip()
    return normalized or None


def _extract_user_role(text: str) -> str | None:
    lowered = text.lower()
    installer_terms = (
        "installer",
        "technician",
        "electrician",
        "contractor",
        "engineer",
    )
    customer_terms = (
        "customer",
        "owner",
        "homeowner",
        "end user",
        "end-user",
        "user",
    )

    role_patterns = (
        r"\b(?:user role|my role|role)\s*(?:is|=|:)\s*([A-Za-z][A-Za-z /-]{1,40})",
        r"\b(?:i am|i'm|im)\s+(?:the\s+|a\s+|an\s+)?([A-Za-z][A-Za-z /-]{1,40})",
    )
    for pattern in role_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        role_text = re.split(r"[,.;\n]", match.group(1), maxsplit=1)[0].strip().lower()
        if any(term in role_text for term in installer_terms):
            return "Installer"
        if any(term in role_text for term in customer_terms):
            return "customer"

    if any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in installer_terms):
        return "Installer"
    if any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in customer_terms):
        return "customer"
    return None
