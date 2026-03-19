from app.models.evidence import EvidencePack, format_markdown_field_list


MINIMUM_TICKET_EVIDENCE_RATIO = 0.7


def build_evidence_collection_node():
    def evidence_collection_node(state: dict) -> dict:
        merged_evidence = EvidencePack.model_validate(state.get("merged_evidence_pack", {}))
        missing_fields = merged_evidence.missing_core_fields()
        missing_artifacts = merged_evidence.missing_best_effort_artifacts()
        evidence_completion_ratio = merged_evidence.core_completion_ratio()
        safety_assessment = state.get("safety_assessment", {})
        support_scope_status = state.get("support_scope_status")
        ready_for_ticket = not missing_fields or evidence_completion_ratio >= MINIMUM_TICKET_EVIDENCE_RATIO

        if not ready_for_ticket:
            field_list = format_markdown_field_list(missing_fields)
            if safety_assessment.get("escalate_immediately"):
                response_text = (
                    "## Immediate Safety Escalation\n\n"
                    "A safety hazard was detected. Do not continue operating the equipment.\n\n"
                    "Before I can create the support ticket, please provide:\n"
                    f"{field_list}"
                )
            elif support_scope_status == "unsupported":
                response_text = (
                    "## Unsupported Site Escalation\n\n"
                    "This site is outside Delta AI support scope.\n\n"
                    "Before I can create the escalation ticket, please provide:\n"
                    f"{field_list}"
                )
            else:
                response_text = (
                    "## Evidence Required\n\n"
                    "To create the support ticket, please provide:\n"
                    f"{field_list}"
                )
            next_action = "collect_evidence"
        else:
            response_text = "## Evidence Ready\n\nI have enough information to create the support ticket now."
            if missing_fields:
                response_text += (
                    "\n\nI will note these remaining missing details in the escalation:\n"
                    f"{format_markdown_field_list(missing_fields)}"
                )
            if missing_artifacts:
                response_text += (
                    "\n\nI will note these unavailable or missing artifacts in the escalation:\n"
                    f"{format_markdown_field_list(missing_artifacts)}"
                )
            next_action = "create_ticket"

        return {
            "missing_fields": missing_fields,
            "missing_artifacts": missing_artifacts,
            "evidence_completion_ratio": evidence_completion_ratio,
            "ticket_ready": ready_for_ticket,
            "response_text": response_text,
            "next_action": next_action,
            "current_phase": "evidence_collection",
            "escalation_active": True,
        }

    return evidence_collection_node
