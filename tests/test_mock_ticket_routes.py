import unittest

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover - optional test dependency in this environment
    FastAPI = None
    TestClient = None

    router = None
    InMemoryMockTicketStore = None
else:
    from app.api.mock_ticket_routes import router
    from app.services.mock_ticket_store import InMemoryMockTicketStore


class StubTicketAdapter:
    def __init__(self, base_url: str | None):
        self.base_url = base_url

    @property
    def contact_form_url(self) -> str:
        return self.base_url or ""


class StubTicketService:
    def __init__(self, base_url: str | None):
        self.adapter = StubTicketAdapter(base_url=base_url)


@unittest.skipIf(FastAPI is None or TestClient is None, "fastapi is not installed in this environment")
class MockTicketRoutesTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.state.mock_ticket_store = InMemoryMockTicketStore()
        app.state.ticket_service = StubTicketService("http://127.0.0.1:8000/public/website/contactForm")
        app.include_router(router)
        self.client = TestClient(app)

    def test_contact_form_submission_is_stored_in_memory(self):
        response = self.client.post(
            "/public/website/contactForm",
            json={
                "type": "Sales - Marshall",
                "firstName": "Jane",
                "lastName": "Doe",
                "email": "jane@example.com",
                "phone": "+61-400-000-000",
                "message": "<div><p>Demo HTML</p></div>",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "mock_created")

        stored = self.client.get("/ticket/mock/submissions")
        self.assertEqual(stored.status_code, 200)
        self.assertEqual(len(stored.json()), 1)
        self.assertEqual(stored.json()[0]["firstName"], "Jane")
        self.assertEqual(stored.json()[0]["message"], "<div><p>Demo HTML</p></div>")

    def test_status_endpoint_reports_current_target(self):
        response = self.client.get("/ticket/mock/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["ticket_api_base_url"], "http://127.0.0.1:8000/public/website/contactForm")
        self.assertEqual(payload["resolved_target_url"], "http://127.0.0.1:8000/public/website/contactForm")
        self.assertEqual(payload["mock_endpoint"], "/public/website/contactForm")
        self.assertTrue(payload["using_local_mock_endpoint"])


if __name__ == "__main__":
    unittest.main()
