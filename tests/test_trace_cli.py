"""
Tests para trace_cli.py — unittest stdlib.
Ejecutar: python3 -m unittest tests.test_trace_cli -v

Verifica parsing de argumentos e invocación correcta de ollama_client.
"""

import unittest
from unittest.mock import patch, MagicMock, call
import importlib
import io
import sys


class TestCliArgParsing(unittest.TestCase):
    """Verifica que argparse parsea correctamente los argumentos del CLI."""

    def _parse(self, args_list):
        """Helper: importa trace_cli y ejecuta el parser con args dados."""
        import trace_cli
        trace_cli = importlib.reload(trace_cli)
        import argparse
        parser = argparse.ArgumentParser(description="Langfuse × Ollama CLI Tracer")
        parser.add_argument("--model",       default="llama3.1")
        parser.add_argument("--prompt",      required=True)
        parser.add_argument("--system",      default="You are a helpful assistant.")
        parser.add_argument("--user-id",     default="cli-user")
        parser.add_argument("--trace-name",  default="ollama-cli")
        parser.add_argument("--tags",        default="cli,ollama")
        parser.add_argument("--temperature", type=float, default=0.7)
        parser.add_argument("--max-tokens",  type=int, default=2048)
        parser.add_argument("--stream",      action="store_true", default=True)
        parser.add_argument("--no-stream",   action="store_false", dest="stream")
        return parser.parse_args(args_list)

    def test_minimal_args(self):
        args = self._parse(["--prompt", "Hello"])
        self.assertEqual(args.prompt, "Hello")
        self.assertEqual(args.model, "llama3.1")
        self.assertTrue(args.stream)

    def test_custom_model(self):
        args = self._parse(["--prompt", "Hi", "--model", "mistral"])
        self.assertEqual(args.model, "mistral")

    def test_no_stream_flag(self):
        args = self._parse(["--prompt", "Hi", "--no-stream"])
        self.assertFalse(args.stream)

    def test_custom_temperature(self):
        args = self._parse(["--prompt", "Hi", "--temperature", "1.2"])
        self.assertAlmostEqual(args.temperature, 1.2)

    def test_custom_max_tokens(self):
        args = self._parse(["--prompt", "Hi", "--max-tokens", "4096"])
        self.assertEqual(args.max_tokens, 4096)

    def test_custom_tags(self):
        args = self._parse(["--prompt", "Hi", "--tags", "mica,compliance,test"])
        self.assertEqual(args.tags, "mica,compliance,test")

    def test_custom_user_id(self):
        args = self._parse(["--prompt", "Hi", "--user-id", "julio"])
        self.assertEqual(args.user_id, "julio")

    def test_custom_trace_name(self):
        args = self._parse(["--prompt", "Hi", "--trace-name", "audit-trace"])
        self.assertEqual(args.trace_name, "audit-trace")

    def test_custom_system_prompt(self):
        args = self._parse(["--prompt", "Hi", "--system", "Be concise."])
        self.assertEqual(args.system, "Be concise.")


class TestCliStreamExecution(unittest.TestCase):
    """Verifica que main() invoca chat_stream cuando streaming está activo."""

    @patch("ollama_client.chat_stream")
    @patch("ollama_client.new_session_id", return_value="fake-session-id")
    def test_stream_mode_calls_chat_stream(self, mock_sid, mock_stream):
        mock_stream.return_value = iter(["Hello", " ", "World"])

        with patch("sys.argv", ["trace_cli.py", "--prompt", "Test prompt"]):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                import trace_cli
                trace_cli = importlib.reload(trace_cli)
                trace_cli.main()

        mock_stream.assert_called_once()
        call_kwargs = mock_stream.call_args
        self.assertEqual(call_kwargs.kwargs["model"], "llama3.1")
        self.assertEqual(call_kwargs.kwargs["session_id"], "fake-session-id")
        self.assertEqual(len(call_kwargs.kwargs["messages"]), 2)  # system + user

    @patch("ollama_client.chat_stream")
    @patch("ollama_client.new_session_id", return_value="fake-session-id")
    def test_stream_output_contains_chunks(self, mock_sid, mock_stream):
        mock_stream.return_value = iter(["Alpha", "Beta"])

        with patch("sys.argv", ["trace_cli.py", "--prompt", "Test"]):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                import trace_cli
                trace_cli = importlib.reload(trace_cli)
                trace_cli.main()

        output = captured.getvalue()
        self.assertIn("Alpha", output)
        self.assertIn("Beta", output)


class TestCliCompleteExecution(unittest.TestCase):
    """Verifica que main() invoca chat_complete cuando --no-stream."""

    @patch("ollama_client.chat_complete")
    @patch("ollama_client.new_session_id", return_value="fake-session-id")
    def test_no_stream_calls_chat_complete(self, mock_sid, mock_complete):
        mock_complete.return_value = "Complete response"

        with patch("sys.argv", ["trace_cli.py", "--prompt", "Test", "--no-stream"]):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                import trace_cli
                trace_cli = importlib.reload(trace_cli)
                trace_cli.main()

        mock_complete.assert_called_once()

    @patch("ollama_client.chat_complete")
    @patch("ollama_client.new_session_id", return_value="fake-session-id")
    def test_no_stream_output_contains_response(self, mock_sid, mock_complete):
        mock_complete.return_value = "Full response text"

        with patch("sys.argv", ["trace_cli.py", "--prompt", "Test", "--no-stream"]):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                import trace_cli
                trace_cli = importlib.reload(trace_cli)
                trace_cli.main()

        output = captured.getvalue()
        self.assertIn("Full response text", output)


class TestCliMessageConstruction(unittest.TestCase):
    """Verifica que los mensajes se construyen correctamente."""

    @patch("ollama_client.chat_complete")
    @patch("ollama_client.new_session_id", return_value="s1")
    def test_messages_include_system_and_user(self, mock_sid, mock_complete):
        mock_complete.return_value = "ok"

        with patch("sys.argv", [
            "trace_cli.py",
            "--prompt", "Explain MiCA",
            "--system", "Be technical",
            "--no-stream",
        ]):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                import trace_cli
                trace_cli = importlib.reload(trace_cli)
                trace_cli.main()

        messages = mock_complete.call_args.kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "Be technical")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "Explain MiCA")

    @patch("ollama_client.chat_complete")
    @patch("ollama_client.new_session_id", return_value="s1")
    def test_tags_parsed_correctly(self, mock_sid, mock_complete):
        mock_complete.return_value = "ok"

        with patch("sys.argv", [
            "trace_cli.py",
            "--prompt", "Hi",
            "--tags", "mica,compliance,test",
            "--no-stream",
        ]):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                import trace_cli
                trace_cli = importlib.reload(trace_cli)
                trace_cli.main()

        tags = mock_complete.call_args.kwargs["tags"]
        self.assertEqual(tags, ["mica", "compliance", "test"])


if __name__ == "__main__":
    unittest.main()
