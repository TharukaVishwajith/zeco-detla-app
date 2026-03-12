# Chat API Integration Guide

This document explains how to integrate the troubleshooting chat API from this service into an external web, mobile, or backend application.

## Base URL

Use the deployed FastAPI base URL for your environment.

Example:

```text
https://your-domain.example.com
```

Local development default:

```text
http://127.0.0.1:8000
```

## Available Endpoints

### `GET /health`

Use this as a startup and readiness check before sending chat traffic.

Example response:

```json
{
  "status": "ok",
  "llm_configured": true,
  "opensearch_configured": true,
  "ticket_api_configured": false
}
```

### `POST /chat/message`

Main endpoint for chat-based troubleshooting.

- Starts a new conversation when `request_id` is omitted.
- Continues an existing conversation when `request_id` is included.
- Returns the assistant reply plus workflow metadata that your external app can use to drive UI state.

## Authentication and Exposure Notes

This service does not implement API authentication in the application layer today.

If this API will be exposed to an external app, put it behind an API gateway, reverse proxy, or backend-for-frontend that provides:

- TLS
- authentication and authorization
- rate limiting
- request logging
- IP restrictions if needed

The app currently enables permissive CORS (`*`), so browser-based clients can call it if the endpoint is reachable.

## Conversation Flow

1. Send the first user message to `POST /chat/message` without `request_id`.
2. Read the returned `request_id`.
3. Store that `request_id` in the external app as the conversation/session key.
4. Send the same `request_id` on every follow-up message.
5. Optionally set `request_ticket: true` when the user wants escalation.
6. Optionally set `issue_resolved: true` when the issue has been resolved.

Important behavior:

- If DynamoDB conversation storage is configured, the service loads prior turns for the same `request_id`.
- If conversation storage is not configured, the API still works, but prior chat context will not be restored between requests.

## Chat Request Contract

### Request body

```json
{
  "request_id": "optional-existing-session-id",
  "user_id": "optional-user-or-account-id",
  "message": "My inverter shows E031 and stopped producing power.",
  "customer_info": {
    "firstName": "Jane",
    "lastName": "Doe",
    "name": "Jane Doe",
    "email": "jane@example.com",
    "phone": "+1-555-0100",
    "site_id": "SITE-1001"
  },
  "device_info": {
    "device_type": "inverter",
    "model_number": "M50A",
    "serial_number": "INV-001122",
    "firmware_version": "1.4.2",
    "product_family": "Delta RPI"
  },
  "evidence_pack": {
    "site_type": "residential",
    "system_size_kw": "8.4",
    "inverter_model": "M50A",
    "serial_number": "INV-001122",
    "firmware_version": "1.4.2",
    "error_code": "E031",
    "timestamp": "2026-03-12T08:30:00Z",
    "system_topology": "single inverter",
    "user_role": "owner",
    "ownership_verified": true,
    "phase_type": "single_phase",
    "backup_loads_present": false,
    "app_or_portal_version": "2.18.0",
    "screenshot_available": true,
    "screenshot_provided": true,
    "photo_checklist_completed": [
      "equipment_rating_labels",
      "installation_overview"
    ],
    "log_export_steps_provided": true,
    "log_time_period": "last 24 hours",
    "photos": [
      "https://files.example.com/photo-1.jpg"
    ],
    "logs": [
      "https://files.example.com/log-1.csv"
    ],
    "recent_changes": "Grid outage yesterday",
    "environmental_conditions": "Heavy rain overnight"
  },
  "request_ticket": false,
  "issue_resolved": false,
  "top_k": 10
}
```

### Required fields

- `message`

Everything else is optional, but sending customer, device, and evidence data materially improves troubleshooting quality and escalation quality.

### Field notes

- `request_id`: Omit on the first turn. Reuse the returned value for the rest of the conversation.
- `user_id`: Optional correlation key from the external app. It is not used as authentication.
- `customer_info.firstName` and `customer_info.lastName`: Accepted in camelCase. The backend also derives `name` automatically when possible.
- `device_info.device_type`: Supported values are `inverter`, `battery`, `pv`, `monitoring`, `unknown`.
- `request_ticket`: Signals that the user wants escalation through the chat workflow. If the AI workflow creates a ticket, it is returned inside the chat response.
- `issue_resolved`: Signals that the issue is resolved.
- `top_k`: Optional retrieval override for knowledge-base search depth.

## Chat Response Contract

Example response:

```json
{
  "request_id": "6f8b3d2a-2d38-4fb8-b7d4-6f2f3fbc9e07",
  "current_phase": "troubleshooting",
  "intent": "troubleshoot",
  "device_type": "inverter",
  "response_text": "Please confirm whether the inverter display is still showing E031 and share a photo of the screen if available.",
  "system_message": null,
  "citations": [
    "doc-123"
  ],
  "next_action": "collect_evidence",
  "missing_fields": [
    "serial_number"
  ],
  "missing_scope_fields": [],
  "safety_flags": [],
  "support_scope_status": "supported",
  "unsupported_reason": null,
  "ticket": null,
  "retrieved_documents": [
    {
      "doc_id": "doc-123",
      "title": "E031 Troubleshooting Guide",
      "product": "inverter",
      "model": "M50A",
      "firmware": "1.4.2",
      "error_code": "E031",
      "doc_type": "manual",
      "section_title": "Alarm Recovery",
      "page_number": 14,
      "content": "Check the inverter status LEDs and confirm recent grid events.",
      "score": 0.92,
      "metadata": {}
    }
  ]
}
```

### Response fields your app should use

- `request_id`: Persist this immediately after the first reply.
- `response_text`: Render as the assistant message.
- `current_phase`: Useful for analytics or UI labels.
- `intent`: Current intent classification.
- `next_action`: Tells your UI what type of step is expected next.
- `missing_fields`: Prompt the user for missing troubleshooting data.
- `missing_scope_fields`: Prompt the user for missing support-scope verification data.
- `safety_flags`: Surface prominently in the UI.
- `support_scope_status`: `supported`, `unsupported`, or `unknown`.
- `unsupported_reason`: Reason when support scope is not supported.
- `ticket`: Present when the AI workflow escalated and created or returned a ticket.
- `retrieved_documents`: Optional grounding data for support tooling, audit, or agent-assist UI.

### Enum values

`intent`

- `troubleshoot`
- `escalate`
- `general_question`

`next_action`

- `ask_question`
- `continue_troubleshooting`
- `collect_evidence`
- `escalate`
- `resolved`

`support_scope_status`

- `supported`
- `unsupported`
- `unknown`

## Frontend Integration Pattern

Recommended client behavior:

1. Call `GET /health` when the app starts or before enabling chat.
2. Create a local conversation object in the external app.
3. On the first user message, call `POST /chat/message` without `request_id`.
4. Save the returned `request_id` with the conversation.
5. On each later turn, send the same `request_id`.
6. Display `response_text` as chat output.
7. If `missing_fields` or `missing_scope_fields` are returned, prompt the user for those values and include them on the next request.
8. If `ticket` is returned, show the ticket ID and status in the UI. Do not call a separate ticket endpoint.
9. If `safety_flags` is not empty, elevate the warning in the UI and consider blocking unsafe self-service guidance.

## JavaScript Example

```js
const API_BASE_URL = "https://your-domain.example.com";

let requestId = null;

async function sendChatMessage(message, context = {}) {
  const payload = {
    request_id: requestId,
    user_id: context.userId ?? null,
    message,
    customer_info: context.customerInfo ?? {},
    device_info: context.deviceInfo ?? {},
    evidence_pack: context.evidencePack ?? {},
    request_ticket: Boolean(context.requestTicket),
    issue_resolved: Boolean(context.issueResolved),
    top_k: context.topK ?? null
  };

  const response = await fetch(`${API_BASE_URL}/chat/message`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  const body = await response.json();
  requestId = body.request_id || requestId;
  return body;
}
```

## cURL Example

```bash
curl -X POST "http://127.0.0.1:8000/chat/message" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "My inverter shows E031",
    "customer_info": {
      "firstName": "Jane",
      "lastName": "Doe",
      "email": "jane@example.com"
    },
    "device_info": {
      "device_type": "inverter",
      "model_number": "M50A"
    },
    "evidence_pack": {
      "error_code": "E031",
      "ownership_verified": true
    }
  }'
```

Follow-up request:

```bash
curl -X POST "http://127.0.0.1:8000/chat/message" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "6f8b3d2a-2d38-4fb8-b7d4-6f2f3fbc9e07",
    "message": "The code is still visible after reboot."
  }'
```

## Escalation Through Chat

This integration only needs `POST /chat/message`.

If the user asks for escalation, or the workflow decides escalation is required, send the next chat turn with `request_ticket: true`.

Example:

```json
{
  "request_id": "existing-session-id",
  "message": "Please escalate this case.",
  "request_ticket": true
}
```

If the AI workflow creates a ticket, the chat response includes a `ticket` object. Your external app should read that object from the chat response and present it to the user.

## Error Handling

Expect these classes of errors:

- `422 Unprocessable Entity`: Invalid JSON shape or invalid enum values.
- `500` or `502` class failures: Upstream LLM, retrieval, or external ticket API issues.
- network timeout failures from the client side.

Recommended client behavior:

- log request and response metadata with `request_id`
- retry only idempotent reads and safe cases
- do not blindly retry chat requests that may trigger escalation without deduplication
- show a fallback message if the API is unavailable

FastAPI validation errors typically look like:

```json
{
  "detail": [
    {
      "loc": ["body", "device_info", "device_type"],
      "msg": "Input should be ...",
      "type": "enum"
    }
  ]
}
```

## Backend Configuration Dependencies

These environment variables affect external integration behavior:

- `OPENAI_API_KEY`: required for LLM responses
- `OPENSEARCH_HOST`: enables retrieval and populated `retrieved_documents`
- `TICKET_API_BASE_URL`: enables real ticket creation when the chat workflow escalates
- `TICKET_API_KEY`: optional bearer token for the external ticket API used by the chat workflow
- `DYNAMODB_TABLE_NAME`: enables persisted conversation history by `request_id`
- `CONVERSATION_HISTORY_MAX_MESSAGES`: max stored turns loaded into each request

Operational impact:

- Without `OPENSEARCH_HOST`, the chat API still works but returns no retrieval results.
- Without `TICKET_API_BASE_URL`, chat-triggered ticket creation falls back to mock responses.
- Without `DYNAMODB_TABLE_NAME`, `request_id` continuity is client-side only and server history is not restored.

## Integration Checklist

- Deploy the API behind a secured public endpoint.
- Verify `GET /health` returns `status: ok`.
- Persist `request_id` per conversation in the external app.
- Send structured `customer_info`, `device_info`, and `evidence_pack` when available.
- Render `response_text`, `missing_fields`, `safety_flags`, and `ticket` in the UI.
- Handle `422` validation failures cleanly.
- Add auth, rate limiting, and observability before exposing the API publicly.
