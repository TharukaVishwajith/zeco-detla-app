from app.models.evidence import EvidencePack


def build_evidence_collection_node():
    def evidence_collection_node(state: dict) -> dict:
        merged_evidence = EvidencePack.model_validate(state.get("merged_evidence_pack", {}))
        missing_fields = merged_evidence.missing_core_fields()
        missing_artifacts = merged_evidence.missing_best_effort_artifacts()
        safety_assessment = state.get("safety_assessment", {})
        support_scope_status = state.get("support_scope_status")

        if missing_fields:
            if safety_assessment.get("escalate_immediately"):
                response_text = (
                    "A safety hazard was detected. Do not continue operating the equipment. "
                    "Before I can create the support ticket, please provide: "
                    + ", ".join(missing_fields)
                    + "."
                )
            elif support_scope_status == "unsupported":
                response_text = (
                    "This site is outside Delta AI support scope. Before I can create the escalation ticket, please provide: "
                    + ", ".join(missing_fields)
                    + "."
                )
            else:
                response_text = "To create the support ticket, please provide: " + ", ".join(missing_fields) + "."
            next_action = "collect_evidence"
        else:
            response_text = "All required evidence is present. Creating the support ticket now."
            if missing_artifacts:
                response_text += " I will note these unavailable or missing artifacts in the escalation: " + ", ".join(missing_artifacts) + "."
            next_action = "create_ticket"

        return {
            "missing_fields": missing_fields,
            "missing_artifacts": missing_artifacts,
            "response_text": response_text,
            "next_action": next_action,
            "current_phase": "evidence_collection",
        }

    return evidence_collection_node
