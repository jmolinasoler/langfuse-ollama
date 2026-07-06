"""
Tests para trace_cli.py — unittest stdlib.
Ejecutar: python3 -m unittest tests.test_trace_cli -v

Invocan main() real con sys.argv parcheado y un FakeGateway en lugar del
adapter: testean el parser y el wiring de verdad, sin Ollama ni Langfuse.
"""

import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch
import importlib


class FakeGateway:
    """ChatGateway de prueba — registra las requests recibidas."""

    def __init__(self, response="ok", chunks=None):
        self._response = response
        self._chunks = chunks if chunks is not None else ["Hello", " ", "World"]
        self.requests = []
        self.stream_calls = 0
        self.complete_calls = 0

    def complete(self, request):
        self.requests.append(request)
        self.complete_calls += 1
        return self._response

    def stream(self, request):
        self.requests.append(request)
        self.stream_calls += 1
        return iter(self._chunks)


def _run_cli(argv, gateway=None):
    """Ejecuta trace_cli.main() con el adapter mockeado. Retorna (stdout, gateway)."""
    gateway = gateway or FakeGateway()
    with patch("langfuse_ollama.adapters.langfuse_ollama.get_gateway", return_value=gateway), \
         patch("langfuse_ollama.adapters.langfuse_ollama.flush"), \
         patch("langfuse_ollama.use_cases.batch.new_session_id", return_value="fake-session-id"), \
         patch("sys.argv", ["trace_cli.py"] + argv):
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            import trace_cli
            trace_cli = importlib.reload(trace_cli)
            trace_cli.main()
    return captured.getvalue(), gateway


class TestCliSingleStream(unittest.TestCase):

    def test_stream_mode_uses_gateway_stream(self):
        from langfuse_ollama import config
        _, gateway = _run_cli(["--prompt", "Test prompt"])
        self.assertEqual(gateway.stream_calls, 1)
        self.assertEqual(gateway.complete_calls, 0)
        request = gateway.requests[0]
        self.assertEqual(request.model, config.DEFAULT_MODEL)
        self.assertEqual(request.session_id, "fake-session-id")
        self.assertEqual(len(request.messages), 2)  # system + user

    def test_stream_output_contains_chunks(self):
        out, _ = _run_cli(["--prompt", "Test"], gateway=FakeGateway(chunks=["Alpha", "Beta"]))
        self.assertIn("Alpha", out)
        self.assertIn("Beta", out)

    def test_session_id_printed(self):
        out, _ = _run_cli(["--prompt", "Test"])
        self.assertIn("fake-session-id", out)


class TestCliSingleComplete(unittest.TestCase):

    def test_no_stream_uses_gateway_complete(self):
        _, gateway = _run_cli(["--prompt", "Test", "--no-stream"])
        self.assertEqual(gateway.complete_calls, 1)
        self.assertEqual(gateway.stream_calls, 0)

    def test_no_stream_output_contains_response(self):
        out, _ = _run_cli(["--prompt", "Test", "--no-stream"],
                          gateway=FakeGateway(response="Full response text"))
        self.assertIn("Full response text", out)


class TestCliArgWiring(unittest.TestCase):
    """Los argumentos CLI llegan hasta la ChatRequest."""

    def _request(self, argv):
        _, gateway = _run_cli(argv + ["--no-stream"])
        return gateway.requests[0]

    def test_custom_model(self):
        self.assertEqual(self._request(["--prompt", "Hi", "--model", "mistral"]).model, "mistral")

    def test_custom_temperature(self):
        self.assertAlmostEqual(self._request(["--prompt", "Hi", "--temperature", "1.2"]).temperature, 1.2)

    def test_custom_max_tokens(self):
        self.assertEqual(self._request(["--prompt", "Hi", "--max-tokens", "4096"]).max_tokens, 4096)

    def test_tags_parsed_to_list(self):
        self.assertEqual(self._request(["--prompt", "Hi", "--tags", "mica,compliance,test"]).tags,
                         ["mica", "compliance", "test"])

    def test_custom_user_id(self):
        self.assertEqual(self._request(["--prompt", "Hi", "--user-id", "julio"]).user_id, "julio")

    def test_custom_trace_name(self):
        self.assertEqual(self._request(["--prompt", "Hi", "--trace-name", "audit"]).trace_name, "audit")

    def test_messages_include_system_and_user(self):
        request = self._request(["--prompt", "Explain MiCA", "--system", "Be technical"])
        self.assertEqual(request.messages[0].role, "system")
        self.assertEqual(request.messages[0].content, "Be technical")
        self.assertEqual(request.messages[1].role, "user")
        self.assertEqual(request.messages[1].content, "Explain MiCA")


class TestCliBatch(unittest.TestCase):

    def _write_batch(self, lines):
        fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
        fh.write("\n".join(lines))
        fh.close()
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def test_batch_runs_all_entries(self):
        path = self._write_batch(['{"prompt": "one"}', '{"prompt": "two"}'])
        _, gateway = _run_cli(["--batch-file", path, "--no-stream"])
        self.assertEqual(gateway.complete_calls, 2)

    def test_batch_entry_overrides_model(self):
        path = self._write_batch(['{"prompt": "x", "model": "mistral"}'])
        _, gateway = _run_cli(["--batch-file", path, "--no-stream"])
        self.assertEqual(gateway.requests[0].model, "mistral")

    def test_batch_output_file_includes_errors_and_results(self):
        path = self._write_batch(['not json', '{"model": "x"}', '{"prompt": "ok"}'])
        out_path = path + ".out"
        self.addCleanup(lambda: os.path.exists(out_path) and os.unlink(out_path))

        _run_cli(["--batch-file", path, "--no-stream", "--output", out_path])

        with open(out_path, encoding="utf-8") as f:
            records = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(records), 3)
        self.assertIn("invalid JSON", records[0]["error"])
        self.assertIn("missing 'prompt' key", records[1]["error"])
        self.assertEqual(records[2]["response"], "ok")
        self.assertEqual(records[2]["line"], 3)


if __name__ == "__main__":
    unittest.main()
