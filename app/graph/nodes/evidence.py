from app.models.evidence import EvidencePack, format_markdown_field_list


def build_evidence_collection_node():
    def evidence_collection_node(state: dict) -> dict:
        merged_evidence = EvidencePack.model_validate(state.get("merged_evidence_pack", {}))
        missing_fields = merged_evidence.missing_core_fields()
        missing_artifacts = merged_evidence.missing_best_effort_artifacts()
        safety_assessment = state.get("safety_assessment", {})
        support_scope_status = state.get("support_scope_status")

        if missing_fields:
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
            response_text = "## Evidence Complete\n\nAll required evidence is present. Creating the support ticket now."
            if missing_artifacts:
                response_text += (
                    "\n\nI will note these unavailable or missing artifacts in the escalation:\n"
                    f"{format_markdown_field_list(missing_artifacts)}"
                )
            next_action = "create_ticket"

        return {
            "missing_fields": missing_fields,
            "missing_artifacts": missing_artifacts,
            "response_text": response_text,
            "next_action": next_action,
            "current_phase": "evidence_collection",
        }

    return evidence_collection_node
