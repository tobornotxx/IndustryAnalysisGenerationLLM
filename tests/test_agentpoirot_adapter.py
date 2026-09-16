import tempfile
import unittest
from pathlib import Path

from run_on_benchmark.agentpoirot_adapter import make_deepseek_chat, normalize_insights


class _Client:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.request = None
        outer = self
        class Completions:
            def create(self, **request):
                outer.request = request
                message = type("Message", (), {"content": "ok"})()
                return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()
        self.chat = type("Chat", (), {"completions": Completions()})()


class AgentPoirotAdapterTests(unittest.TestCase):
    def test_chat_uses_deepseek_endpoint_and_canonical_model(self):
        clients = []
        def factory(**kwargs):
            clients.append(_Client(**kwargs))
            return clients[-1]
        chat = make_deepseek_chat(api_key="test-key", client_factory=factory)
        self.assertEqual(chat("hello"), "ok")
        self.assertEqual(clients[0].kwargs["base_url"], "https://api.deepseek.com")
        self.assertEqual(clients[0].request["model"], "deepseek-flash")

    def test_image_is_encoded_in_the_same_request(self):
        clients = []
        factory = lambda **kwargs: clients.append(_Client(**kwargs)) or clients[-1]
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "plot.png"
            image.write_bytes(b"fake-png")
            make_deepseek_chat(api_key="x", client_factory=factory)("inspect", str(image))
        content = clients[0].request["messages"][0]["content"]
        self.assertEqual(len(content), 2)
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_missing_upstream_plot_falls_back_to_text(self):
        clients = []
        factory = lambda **kwargs: clients.append(_Client(**kwargs)) or clients[-1]
        chat = make_deepseek_chat(api_key="x", client_factory=factory)
        self.assertEqual(chat("inspect", "missing-plot.jpg"), "ok")
        self.assertEqual(clients[0].request["messages"][0]["content"], "inspect")

    def test_normalize_official_output(self):
        result = normalize_insights([
            {"header": "Trend", "question": "Why?", "insight": "Sales increased."},
            "  plain insight  ", {},
        ])
        self.assertEqual(result, ["Trend Why? Sales increased.", "plain insight"])


if __name__ == "__main__":
    unittest.main()
