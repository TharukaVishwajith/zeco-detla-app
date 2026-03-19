import unittest
from unittest.mock import Mock, patch

from app.adapters.ticket_api_client import DEFAULT_TICKET_TYPE, TicketApiClient
from app.models.ticket import CustomerInfo, TicketPayload


class TicketApiClientTests(unittest.TestCase):
    @patch("app.adapters.ticket_api_client.httpx.post")
    def test_create_ticket_posts_external_payload_shape(self, mock_post):
        response = Mock()
        response.json.return_value = {
            "ticket_id": "T-100",
            "status": "created",
            "message": "created",
        }
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        client = TicketApiClient(base_url="https://example.com", api_key=None, timeout_seconds=10)
        payload = TicketPayload(
            customer_info=CustomerInfo(
                firstName="Jane",
                lastName="Doe",
                email="jane@example.com",
                phone="+1-555-0100",
            ),
            issue_summary="Inverter fault",
            message_html="<div><p>Evidence pack</p></div>",
        )

        client.create_ticket(payload)

        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "https://example.com")
        self.assertEqual(
            kwargs["json"],
            {
                "type": DEFAULT_TICKET_TYPE,
                "firstName": "Jane",
                "lastName": "Doe",
                "email": "jane@example.com",
                "phone": "+1-555-0100",
                "message": "<div><p>Evidence pack</p></div>",
            },
        )

    @patch("app.adapters.ticket_api_client.httpx.post")
    def test_create_ticket_parses_wrapped_real_api_response(self, mock_post):
        response = Mock()
        response.json.return_value = {
            "message": "Success",
            "data": {
                "id": 115,
                "status": 2,
            },
        }
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        client = TicketApiClient(
            base_url="https://api.datamarshall.au:8443/public/freshdesk",
            api_key=None,
            timeout_seconds=10,
        )
        payload = TicketPayload(
            customer_info=CustomerInfo(firstName="Jane"),
            issue_summary="Fault code",
            message_html="<div>payload</div>",
        )

        ticket = client.create_ticket(payload)

        args, _ = mock_post.call_args
        self.assertEqual(args[0], "https://api.datamarshall.au:8443/public/freshdesk")
        self.assertEqual(ticket.ticket_id, "115")
        self.assertEqual(ticket.status, "created")
        self.assertEqual(ticket.message, "Success")

    @patch("app.adapters.ticket_api_client.httpx.post")
    def test_create_ticket_prefers_string_status_when_upstream_provides_one(self, mock_post):
        response = Mock()
        response.json.return_value = {
            "status": "created",
            "message": "Success",
            "data": {
                "id": 116,
                "status": 2,
            },
        }
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        client = TicketApiClient(
            base_url="https://api.datamarshall.au:8443/public/freshdesk",
            api_key=None,
            timeout_seconds=10,
        )
        payload = TicketPayload(
            customer_info=CustomerInfo(firstName="Jane"),
            issue_summary="Fault code",
            message_html="<div>payload</div>",
        )

        ticket = client.create_ticket(payload)

        self.assertEqual(ticket.ticket_id, "116")
        self.assertEqual(ticket.status, "created")
        self.assertEqual(ticket.message, "Success")

    @patch("app.adapters.ticket_api_client.httpx.post")
    def test_create_ticket_uses_configured_url_verbatim(self, mock_post):
        response = Mock()
        response.json.return_value = {
            "ticket_id": "T-101",
            "status": "created",
            "message": "created",
        }
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        client = TicketApiClient(
            base_url="https://api.datamarshall.au:8443/public/website/contactForm",
            api_key=None,
            timeout_seconds=10,
        )
        payload = TicketPayload(
            customer_info=CustomerInfo(firstName="Jane"),
            issue_summary="Fault code",
            message_html="<div>payload</div>",
        )

        client.create_ticket(payload)

        args, _ = mock_post.call_args
        self.assertEqual(args[0], "https://api.datamarshall.au:8443/public/website/contactForm")


if __name__ == "__main__":
    unittest.main()
