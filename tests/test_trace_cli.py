"""
Tests para trace_cli.py — unittest stdlib.
Ejecutar: python3 -m unittest tests.test_trace_cli -v

Invocan main() real con sys.argv parcheado y ollama_client mockeado:
testean el parser y el wiring de verdad, sin Ollama ni Langfuse activos.
"""

import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import importlib


def _run_cli(argv, stream_chunks=None, complete_response="ok"):
    """Ejecuta trace_cli.main() con mocks. Retorna (stdout, mocks)."""
    mocks = {}
    with patch("langfuse_ollama.core.ollama_client.get_chat_client", return_value=MagicMock()) as m_client, \
         patch("langfuse_ollama.core.ollama_client.chat_stream") as m_stream, \
         patch("langfuse_ollama.core.ollama_client.chat_complete") as m_complete, \
         patch("langfuse_ollama.core.ollama_client.flush") as m_flush, \
         patch("langfuse_ollama.core.ollama_client.new_session_id", return_value="fake-session-id"), \
         patch("sys.argv", ["trace_cli.py"] + argv):
        m_stream.side_effect = lambda **kw: iter(stream_chunks or ["Hello", " ", "World"])
        m_complete.return_value = complete_response
        mocks.update(client=m_client, stream=m_stream, complete=m_complete, flush=m_flush)

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            import trace_cli
            trace_cli = importlib.reload(trace_cli)
            trace_cli.main()
    return captured.getvalue(), mocks


class TestCliSingleStream(unittest.TestCase):

    def test_stream_mode_calls_chat_stream(self):
        from langfuse_ollama import config
        _, mocks = _run_cli(["--prompt", "Test prompt"])
        mocks["stream"].assert_called_once()
        kwargs = mocks["stream"].call_args.kwargs
        self.assertEqual(kwargs["model"], config.DEFAULT_MODEL)
        self.assertEqual(kwargs["session_id"], "fake-session-id")
        self.assertEqual(len(kwargs["messages"]), 2)  # system + user

    def test_stream_output_contains_chunks(self):
        out, _ = _run_cli(["--prompt", "Test"], stream_chunks=["Alpha", "Beta"])
        self.assertIn("Alpha", out)
        self.assertIn("Beta", out)

    def test_session_id_printed(self):
        out, _ = _run_cli(["--prompt", "Test"])
        self.assertIn("fake-session-id", out)


class TestCliSingleComplete(unittest.TestCase):

    def test_no_stream_calls_chat_complete(self):
        _, mocks = _run_cli(["--prompt", "Test", "--no-stream"])
        mocks["complete"].assert_called_once()
        mocks["stream"].assert_not_called()

    def test_no_stream_output_contains_response(self):
        out, _ = _run_cli(["--prompt", "Test", "--no-stream"],
                          complete_response="Full response text")
        self.assertIn("Full response text", out)


class TestCliArgWiring(unittest.TestCase):
    """Los argumentos CLI llegan hasta la llamada al modelo."""

    def _kwargs(self, argv):
        _, mocks = _run_cli(argv + ["--no-stream"])
        return mocks["complete"].call_args.kwargs

    def test_custom_model(self):
        self.assertEqual(self._kwargs(["--prompt", "Hi", "--model", "mistral"])["model"], "mistral")

    def test_custom_temperature(self):
        self.assertAlmostEqual(self._kwargs(["--prompt", "Hi", "--temperature", "1.2"])["temperature"], 1.2)

    def test_custom_max_tokens(self):
        self.assertEqual(self._kwargs(["--prompt", "Hi", "--max-tokens", "4096"])["max_tokens"], 4096)

    def test_tags_parsed_to_list(self):
        self.assertEqual(self._kwargs(["--prompt", "Hi", "--tags", "mica,compliance,test"])["tags"],
                         ["mica", "compliance", "test"])

    def test_custom_user_id(self):
        self.assertEqual(self._kwargs(["--prompt", "Hi", "--user-id", "julio"])["user_id"], "julio")

    def test_custom_trace_name(self):
        self.assertEqual(self._kwargs(["--prompt", "Hi", "--trace-name", "audit"])["trace_name"], "audit")

    def test_messages_include_system_and_user(self):
        messages = self._kwargs(["--prompt", "Explain MiCA", "--system", "Be technical"])["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": "Be technical"})
        self.assertEqual(messages[1], {"role": "user", "content": "Explain MiCA"})


class TestCliBatch(unittest.TestCase):

    def _write_batch(self, lines):
        fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
        fh.write("\n".join(lines))
        fh.close()
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def test_batch_runs_all_entries(self):
        path = self._write_batch(['{"prompt": "one"}', '{"prompt": "two"}'])
        _, mocks = _run_cli(["--batch-file", path, "--no-stream"])
        self.assertEqual(mocks["complete"].call_count, 2)

    def test_batch_entry_overrides_model(self):
        path = self._write_batch(['{"prompt": "x", "model": "mistral"}'])
        _, mocks = _run_cli(["--batch-file", path, "--no-stream"])
        self.assertEqual(mocks["complete"].call_args.kwargs["model"], "mistral")

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
