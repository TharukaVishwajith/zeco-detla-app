# Zeco Delta Troubleshooting API

FastAPI service for AI-assisted Delta troubleshooting. The API accepts chat messages, retrieves supporting knowledge when configured, maintains conversation state by `request_id`, and can escalate through the chat workflow when needed.

## What This Service Exposes

- `GET /health` for readiness checks
- `POST /chat/message` for the troubleshooting chat workflow
- a local browser test client at `/`

## API Integration

For external app integration, use the chat integration guide:

- [CHAT_API_INTEGRATION.md](./CHAT_API_INTEGRATION.md)

That document is the source of truth for:

- chat request and response payloads
- `request_id` conversation handling
- escalation through chat using `request_ticket`
- frontend integration examples
- operational and environment configuration notes

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API locally:

```bash
python main.py
```

Default local base URL:

```text
http://127.0.0.1:8000
```

## Main Environment Variables

- `OPENAI_API_KEY`
- `OPENSEARCH_HOST`
- `TICKET_API_BASE_URL`
- `DYNAMODB_TABLE_NAME`
- `CONVERSATION_HISTORY_MAX_MESSAGES`

## Notes

- External clients should integrate only with the chat endpoint unless they are intentionally extending backend behavior.
- If you expose this service publicly, place it behind authentication, TLS, and rate limiting.
