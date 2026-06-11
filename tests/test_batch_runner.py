"""
Tests para batch_runner.py — lógica de dominio pura, sin Streamlit ni red.
Ejecutar: python3 -m unittest tests.test_batch_runner -v
"""

import unittest

from langfuse_ollama.core import batch_runner as br


DEFAULTS = br.BatchDefaults(
    model="llama3.1",
    system="You are a helpful assistant.",
    user_id="u1",
    trace_name="t1",
    tags=["a", "b"],
    temperature=0.7,
    max_tokens=2048,
)


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
        captured = {}

        def fake_chat(**kwargs):
            captured.update(kwargs)
            return "the response"

        result = br.run_entry(3, {"prompt": "hello"}, DEFAULTS, chat_fn=fake_chat)

        self.assertEqual(result["line"], 3)
        self.assertEqual(result["prompt"], "hello")
        self.assertEqual(result["response"], "the response")
        self.assertIsNone(result["error"])
        self.assertEqual(captured["messages"][0],
                         {"role": "system", "content": DEFAULTS.system})
        self.assertEqual(captured["messages"][1],
                         {"role": "user", "content": "hello"})
        self.assertEqual(captured["session_id"], result["session_id"])

    def test_entry_overrides_reach_chat(self):
        captured = {}

        def fake_chat(**kwargs):
            captured.update(kwargs)
            return "ok"

        br.run_entry(1, {"prompt": "p", "model": "mistral", "system": "Be brief."},
                     DEFAULTS, chat_fn=fake_chat)
        self.assertEqual(captured["model"], "mistral")
        self.assertEqual(captured["messages"][0]["content"], "Be brief.")

    def test_error_captured(self):
        def fake_chat(**kwargs):
            raise RuntimeError("boom")

        result = br.run_entry(1, {"prompt": "p"}, DEFAULTS, chat_fn=fake_chat)
        self.assertIsNone(result["response"])
        self.assertEqual(result["error"], "boom")

    def test_each_run_gets_fresh_session(self):
        def fake_chat(**kwargs):
            return "ok"

        r1 = br.run_entry(1, {"prompt": "p"}, DEFAULTS, chat_fn=fake_chat)
        r2 = br.run_entry(1, {"prompt": "p"}, DEFAULTS, chat_fn=fake_chat)
        self.assertNotEqual(r1["session_id"], r2["session_id"])


if __name__ == "__main__":
    unittest.main()
