"""
Tests para use_cases/batch.py — lógica de aplicación pura, sin Streamlit ni red.
Ejecutar: python3 -m unittest tests.test_use_cases_batch -v
"""

import unittest

from langfuse_ollama.domain.entities import BatchDefaults
from langfuse_ollama.use_cases import batch as br


DEFAULTS = BatchDefaults(
    model="llama3.1",
    system="You are a helpful assistant.",
    user_id="u1",
    trace_name="t1",
    tags=["a", "b"],
    temperature=0.7,
    max_tokens=2048,
)


class FakeGateway:
    """ChatGateway de prueba — registra las requests recibidas."""

    def __init__(self, response="ok", chunks=None, error=None):
        self._response = response
        self._chunks = chunks if chunks is not None else ["ok"]
        self._error = error
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if self._error:
            raise self._error
        return self._response

    def stream(self, request):
        self.requests.append(request)
        if self._error:
            raise self._error
        return iter(self._chunks)


class TestParseJsonl(unittest.TestCase):

    def test_valid_lines(self):
        text = '{"prompt": "one"}\n{"prompt": "two", "model": "mistral"}'
        entries, errors = br.parse_jsonl(text)
        self.assertEqual(errors, [])
        self.assertEqual(entries, [(1, {"prompt": "one"}),
                                   (2, {"prompt": "two", "model": "mistral"})])

    def test_skips_blank_lines_keeping_line_numbers(self):
        text = '\n{"prompt": "one"}\n\n{"prompt": "two"}\n'
        entries, _ = br.parse_jsonl(text)
        self.assertEqual([n for n, _ in entries], [2, 4])

    def test_invalid_json_reported(self):
        entries, errors = br.parse_jsonl('not json\n{"prompt": "ok"}')
        self.assertEqual(len(entries), 1)
        self.assertEqual(errors[0][0], 1)
        self.assertIn("invalid JSON", errors[0][1])

    def test_missing_prompt_reported(self):
        entries, errors = br.parse_jsonl('{"model": "mistral"}')
        self.assertEqual(entries, [])
        self.assertEqual(errors, [(1, "missing 'prompt' key")])

    def test_non_object_line_reported(self):
        entries, errors = br.parse_jsonl('["prompt", "x"]')
        self.assertEqual(entries, [])
        self.assertEqual(len(errors), 1)

    def test_empty_text(self):
        self.assertEqual(br.parse_jsonl(""), ([], []))


class TestNormalizeTags(unittest.TestCase):

    def test_list_passthrough(self):
        self.assertEqual(br.normalize_tags(["x", "y"], ["f"]), ["x", "y"])

    def test_string_split(self):
        self.assertEqual(br.normalize_tags("a, b,c", ["f"]), ["a", "b", "c"])

    def test_drops_empty_items(self):
        self.assertEqual(br.normalize_tags("a,,b,", ["f"]), ["a", "b"])
        self.assertEqual(br.normalize_tags(["a", "", "  "], ["f"]), ["a"])

    def test_fallback_on_other_types(self):
        self.assertEqual(br.normalize_tags(None, ["f"]), ["f"])
        self.assertEqual(br.normalize_tags(42, ["f"]), ["f"])


class TestResolveParams(unittest.TestCase):

    def test_no_overrides_returns_defaults(self):
        p = br.resolve_params({"prompt": "x"}, DEFAULTS)
        self.assertEqual(p, DEFAULTS)

    def test_overrides_applied(self):
        p = br.resolve_params(
            {"prompt": "x", "model": "mistral", "temperature": 0.1, "tags": "t1,t2"},
            DEFAULTS,
        )
        self.assertEqual(p.model, "mistral")
        self.assertEqual(p.temperature, 0.1)
        self.assertEqual(p.tags, ["t1", "t2"])
        self.assertEqual(p.max_tokens, DEFAULTS.max_tokens)

    def test_defaults_not_mutated(self):
        br.resolve_params({"prompt": "x", "model": "other"}, DEFAULTS)
        self.assertEqual(DEFAULTS.model, "llama3.1")


class TestRunEntry(unittest.TestCase):

    def test_success_result(self):
        gateway = FakeGateway(response="the response")

        result = br.run_entry(gateway, 3, {"prompt": "hello"}, DEFAULTS)

        self.assertEqual(result["line"], 3)
        self.assertEqual(result["prompt"], "hello")
        self.assertEqual(result["response"], "the response")
        self.assertIsNone(result["error"])

        request = gateway.requests[0]
        self.assertEqual(request.messages[0].role, "system")
        self.assertEqual(request.messages[0].content, DEFAULTS.system)
        self.assertEqual(request.messages[1].role, "user")
        self.assertEqual(request.messages[1].content, "hello")
        self.assertEqual(request.session_id, result["session_id"])

    def test_entry_overrides_reach_gateway(self):
        gateway = FakeGateway()
        br.run_entry(gateway, 1,
                     {"prompt": "p", "model": "mistral", "system": "Be brief."},
                     DEFAULTS)
        request = gateway.requests[0]
        self.assertEqual(request.model, "mistral")
        self.assertEqual(request.messages[0].content, "Be brief.")

    def test_streaming_with_on_chunk(self):
        gateway = FakeGateway(chunks=["Hel", "lo"])
        seen = []
        result = br.run_entry(gateway, 1, {"prompt": "p"}, DEFAULTS,
                              on_chunk=seen.append)
        self.assertEqual(seen, ["Hel", "lo"])
        self.assertEqual(result["response"], "Hello")
        self.assertIsNone(result["error"])

    def test_error_captured(self):
        gateway = FakeGateway(error=RuntimeError("boom"))
        result = br.run_entry(gateway, 1, {"prompt": "p"}, DEFAULTS)
        self.assertIsNone(result["response"])
        self.assertEqual(result["error"], "boom")

    def test_each_run_gets_fresh_session(self):
        gateway = FakeGateway()
        r1 = br.run_entry(gateway, 1, {"prompt": "p"}, DEFAULTS)
        r2 = br.run_entry(gateway, 1, {"prompt": "p"}, DEFAULTS)
        self.assertNotEqual(r1["session_id"], r2["session_id"])


if __name__ == "__main__":
    unittest.main()
