"""
Tests para use_cases/chat.py — construcción de ChatRequest.
Ejecutar: python3 -m unittest tests.test_use_cases_chat -v
"""

import unittest

from langfuse_ollama.domain.entities import ChatMessage
from langfuse_ollama.use_cases.chat import make_chat_request


class TestMakeChatRequest(unittest.TestCase):

    def _request(self, **overrides):
        kwargs = dict(
            history=[ChatMessage(role="user", content="hi")],
            system_prompt="Be helpful.",
            model="llama3.1",
            session_id="s1",
            user_id="u1",
        )
        kwargs.update(overrides)
        return make_chat_request(**kwargs)

    def test_system_prompt_goes_first(self):
        request = self._request()
        self.assertEqual(request.messages[0], ChatMessage(role="system", content="Be helpful."))
        self.assertEqual(request.messages[1], ChatMessage(role="user", content="hi"))

    def test_default_tags_include_model(self):
        request = self._request(model="mistral")
        self.assertEqual(request.tags, ["ollama", "mistral"])

    def test_explicit_tags_kept(self):
        request = self._request(tags=["t1", "t2"])
        self.assertEqual(request.tags, ["t1", "t2"])

    def test_params_carried(self):
        request = self._request(temperature=1.5, max_tokens=4096, trace_name="tr")
        self.assertEqual(request.temperature, 1.5)
        self.assertEqual(request.max_tokens, 4096)
        self.assertEqual(request.trace_name, "tr")
        self.assertEqual(request.session_id, "s1")
        self.assertEqual(request.user_id, "u1")


if __name__ == "__main__":
    unittest.main()
