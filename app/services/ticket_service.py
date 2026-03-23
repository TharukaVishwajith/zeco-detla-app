import re
from html import escape

from app.models.conversation import ChatMessageRequest, TicketCreateRequest
from app.models.evidence import EvidencePack, humanize_evidence_field
from app.models.ticket import TicketPayload, TicketResponse


class TicketService:
    def __init__(self, adapter):
        self.adapter = adapter

    def create_from_graph(
        self,
        request: ChatMessageRequest,
        issue_summary: str,
        troubleshooting_steps: list[str],
        escalation_reason: str | None,
        merged_evidence_pack: dict,
        unsupported_reason: str | None,
        missing_artifacts: list[str],
    ) -> TicketResponse:
        evidence = EvidencePack.model_validate(merged_evidence_pack)
        summarized_troubleshooting_steps = self._summarize_troubleshooting_steps(troubleshooting_steps)
        payload = TicketPayload(
            customer_info=request.customer_info,
            device_info=request.device_info.model_dump(),
            issue_summary=issue_summary,
            message_html=self._build_ticket_message_html(
                issue_summary=issue_summary,
                troubleshooting_steps=summarized_troubleshooting_steps,
                evidence=evidence,
                escalation_reason=escalation_reason,
                unsupported_reason=unsupported_reason,
                missing_artifacts=missing_artifacts,
            ),
            troubleshooting_steps=summarized_troubleshooting_steps,
            attachments=[*evidence.photos, *evidence.logs],
            escalation_reason=escalation_reason or unsupported_reason,
            escalation_summary=self._build_escalation_summary(
                issue_summary=issue_summary,
                troubleshooting_steps=summarized_troubleshooting_steps,
                evidence=evidence,
                escalation_reason=escalation_reason,
                unsupported_reason=unsupported_reason,
                missing_artifacts=missing_artifacts,
            ),
            missing_artifacts=missing_artifacts,
            unsafe_instructions_given=False,
            evidence_pack=evidence.model_dump(exclude_none=True),
        )
        return self.adapter.create_ticket(payload)

    def create_from_request(self, request: TicketCreateRequest) -> TicketResponse:
        summarized_troubleshooting_steps = self._summarize_troubleshooting_steps(request.troubleshooting_steps)
        payload = TicketPayload(
            customer_info=request.customer_info,
            device_info=request.device_info.model_dump(),
            issue_summary=request.issue_summary,
            message_html=self._build_ticket_message_html(
                issue_summary=request.issue_summary,
                troubleshooting_steps=summarized_troubleshooting_steps,
                evidence=request.evidence_pack,
                escalation_reason=request.escalation_reason,
                unsupported_reason=None,
                missing_artifacts=[],
            ),
            troubleshooting_steps=summarized_troubleshooting_steps,
            attachments=request.attachments,
            escalation_reason=request.escalation_reason,
            evidence_pack=request.evidence_pack.model_dump(exclude_none=True),
        )
        return self.adapter.create_ticket(payload)

    def _build_escalation_summary(
        self,
        issue_summary: str,
        troubleshooting_steps: list[str],
        evidence: EvidencePack,
        escalation_reason: str | None,
        unsupported_reason: str | None,
        missing_artifacts: list[str],
    ) -> str:
        summary_lines = [
            f"Symptoms: {issue_summary}",
            "Steps already attempted: " + (", ".join(troubleshooting_steps) if troubleshooting_steps else "None recorded"),
            "Escalation reason: " + ", ".join(
                reason for reason in (escalation_reason, unsupported_reason) if reason
            )
            if escalation_reason or unsupported_reason
            else "Escalation reason: none provided",
            "Evidence collected: " + ", ".join(sorted(evidence.provided_fields().keys())),
            "Missing or unavailable artifacts: " + (", ".join(missing_artifacts) if missing_artifacts else "None"),
            "Unsafe instructions given: no",
        ]
        return "\n".join(summary_lines)

    def _build_ticket_message_html(
        self,
        issue_summary: str,
        troubleshooting_steps: list[str],
        evidence: EvidencePack,
        escalation_reason: str | None,
        unsupported_reason: str | None,
        missing_artifacts: list[str],
    ) -> str:
        reason_values = [reason for reason in (escalation_reason, unsupported_reason) if reason]
        evidence_items = self._build_evidence_list_items(evidence)
        troubleshooting_items = "".join(
            f"<li>{escape(step)}</li>" for step in troubleshooting_steps if step
        ) or "<li>None recorded</li>"
        missing_items = "".join(
            f"<li>{escape(humanize_evidence_field(field_name))}</li>" for field_name in missing_artifacts
        ) or "<li>None</li>"
        reasons_html = "".join(f"<li>{escape(reason)}</li>" for reason in reason_values) or "<li>None provided</li>"

        return (
            "<div>"
            f"<p><strong>Issue summary:</strong> {escape(issue_summary)}</p>"
            "<p><strong>Escalation reasons:</strong></p>"
            f"<ul>{reasons_html}</ul>"
            "<p><strong>Troubleshooting steps already attempted:</strong></p>"
            f"<ul>{troubleshooting_items}</ul>"
            "<p><strong>Evidence pack:</strong></p>"
            f"<ul>{evidence_items}</ul>"
            "<p><strong>Missing or unavailable artifacts:</strong></p>"
            f"<ul>{missing_items}</ul>"
            "</div>"
        )

    def _build_evidence_list_items(self, evidence: EvidencePack) -> str:
        provided = evidence.provided_fields()
        if not provided:
            return "<li>None provided</li>"

        items: list[str] = []
        for field_name in sorted(provided):
            label = humanize_evidence_field(field_name)
            value = provided[field_name]
            if isinstance(value, list):
                rendered_value = ", ".join(str(item) for item in value)
            elif isinstance(value, bool):
                rendered_value = "Yes" if value else "No"
            else:
                rendered_value = str(value)
            items.append(f"<li><strong>{escape(label)}:</strong> {escape(rendered_value)}</li>")
        return "".join(items)

    def _summarize_troubleshooting_steps(self, troubleshooting_steps: list[str]) -> list[str]:
        summaries: list[str] = []
        seen: set[str] = set()
        for step in troubleshooting_steps:
            summary = self._summarize_troubleshooting_step(step)
            if not summary:
                continue
            dedupe_key = summary.casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            summaries.append(summary)
        return summaries

    def _summarize_troubleshooting_step(self, content: str) -> str | None:
        has_explicit_steps = any(re.match(r"^\s*(?:\d+\.|[-*])\s+", line) for line in content.splitlines())
        cleaned_lines: list[str] = []
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            line = re.sub(r"^\d+\.\s*", "", line)
            line = re.sub(r"^[-*]\s*", "", line)
            line = re.sub(r"`([^`]*)`", r"\1", line)
            line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
            line = " ".join(line.split()).strip()
            if not line or line.endswith(":"):
                continue
            lowered = line.lower()
            if lowered.startswith(
                (
                    "reply with",
                    "please reply with",
                    "please send",
                    "send whatever remaining details",
                    "if not, tell me",
                    "if you have any of these additional details",
                )
            ):
                continue
            cleaned_lines.append(line.rstrip("."))

        if not cleaned_lines:
            return None

        lowered_content = content.lower()
        if not has_explicit_steps and any(phrase in lowered_content for phrase in ("please send", "reply with", "i need one more detail")):
            requested_details = ", ".join(cleaned_lines[:4])
            return f"Requested additional troubleshooting details: {requested_details}"

        if len(cleaned_lines) == 1:
            return cleaned_lines[0]

        return "; ".join(cleaned_lines[:4])
