from app.models.conversation import ChatMessageRequest, SafetyAssessment


SAFETY_TERMS = {
    "fire": "Potential fire hazard",
    "smoke": "Smoke reported",
    "burning smell": "Burning smell reported",
    "scorch": "Scorch or burn marks reported",
    "burn mark": "Scorch or burn marks reported",
    "thermal shutdown": "Thermal shutdown reported",
    "water damage": "Possible water ingress",
    "water ingress": "Possible water ingress",
    "flood": "Flooding reported",
    "condensation": "Condensation inside equipment reported",
    "physical damage": "Physical damage reported",
    "corrosion": "Corrosion reported",
    "impact": "Impact damage reported",
    "breaker trip": "Repeated protection device trips reported",
    "breaker trips": "Repeated protection device trips reported",
    "protection device": "Protection device issue reported",
    "bypass": "Bypassed protection device reported",
    "non-compliant wiring": "Unsafe wiring practice reported",
    "diy": "DIY electrical work reported",
    "ct relocation": "Installer-only CT relocation required",
    "open enclosure": "Installer-only enclosure access required",
    "wiring change": "Installer-only wiring change required",
    "electrical hazard": "Electrical hazard reported",
    "overheating": "Overheating reported",
    "sparking": "Sparking reported",
}


def build_safety_guard_node():
    def safety_guard_node(state: dict) -> dict:
        request = ChatMessageRequest.model_validate(state["request"])
        classification = state.get("classification", {})
        lowered = request.message.lower()
        flags = [term for term in SAFETY_TERMS if term in lowered]
        flags.extend(classification.get("risk_flags", []))
        unique_flags = sorted(set(flags))
        reason = None
        if unique_flags:
            first_flag = unique_flags[0]
            reason = SAFETY_TERMS.get(first_flag, "Safety hazard detected")
        assessment = SafetyAssessment(
            escalate_immediately=bool(unique_flags),
            reason=reason,
            safety_flags=unique_flags,
        )
        return {
            "safety_assessment": assessment.model_dump(mode="json"),
            "safety_flags": assessment.safety_flags,
            "current_phase": "safety_guardrails",
        }

    return safety_guard_node
