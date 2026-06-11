"""
Tests para ollama_client.py — unittest stdlib + DI con Fakes.
Ejecutar: python3 -m unittest tests.test_ollama_client -v

Todos los tests corren SIN Ollama ni Langfuse activos.
"""

import unittest
from unittest.mock import patch, MagicMock
import uuid
import importlib


class TestNewSessionId(unittest.TestCase):
    """new_session_id() debe retornar UUIDs válidos y únicos."""

    def setUp(self):
        import ollama_client as oc
        self.oc = oc

    def test_returns_valid_uuid(self):
        sid = self.oc.new_session_id()
        parsed = uuid.UUID(sid)
        self.assertEqual(str(parsed), sid)

    def test_returns_unique_ids(self):
        ids = {self.oc.new_session_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)

    def test_returns_string_type(self):
        self.assertIsInstance(self.oc.new_session_id(), str)


class TestPing(unittest.TestCase):
    """ping() distingue Ollama disponible de no disponible."""

    def setUp(self):
        import ollama_client as oc
        self.oc = importlib.reload(oc)

    @patch("ollama_client.httpx.get")
    def test_true_on_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        self.assertTrue(self.oc.ping())

    @patch("ollama_client.httpx.get")
    def test_false_on_connection_error(self, mock_get):
        mock_get.side_effect = ConnectionError("refused")
        self.assertFalse(self.oc.ping())

    @patch("ollama_client.httpx.get")
    def test_false_on_http_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("500")
        mock_get.return_value = mock_response
        self.assertFalse(self.oc.ping())

    @patch("ollama_client.httpx.get")
    def test_uses_given_base_url(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        self.oc.ping("http://remote:11434")
        self.assertIn("http://remote:11434/api/version", mock_get.call_args.args)


class TestListModels(unittest.TestCase):
    """list_models() con httpx mockeado. Sin fallback que enmascare fallos."""

    def setUp(self):
        import ollama_client as oc
        self.oc = importlib.reload(oc)

    @patch("ollama_client.httpx.get")
    def test_returns_model_names_on_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.1"},
                {"name": "mistral"},
                {"name": "codellama"},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        models = self.oc.list_models()
        self.assertEqual(models, ["llama3.1", "mistral", "codellama"])

    @patch("ollama_client.httpx.get")
    def test_returns_empty_on_connection_error(self, mock_get):
        mock_get.side_effect = ConnectionError("refused")
        self.assertEqual(self.oc.list_models(), [])

    @patch("ollama_client.httpx.get")
    def test_returns_empty_on_http_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("500")
        mock_get.return_value = mock_response
        self.assertEqual(self.oc.list_models(), [])

    @patch("ollama_client.httpx.get")
    def test_handles_empty_models_list(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        self.assertEqual(self.oc.list_models(), [])

    @patch("ollama_client.httpx.get")
    def test_handles_missing_models_key(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        self.assertEqual(self.oc.list_models(), [])


class TestGetChatClient(unittest.TestCase):
    """get_chat_client() cachea por configuración."""

    def setUp(self):
        import ollama_client as oc
        self.oc = importlib.reload(oc)

    def test_same_config_returns_same_client(self):
        with patch.object(self.oc, "_build_client", side_effect=lambda *a: object()) as mb:
            c1 = self.oc.get_chat_client("http://x:11434", "pk", "sk", "http://lf")
            c2 = self.oc.get_chat_client("http://x:11434", "pk", "sk", "http://lf")
        self.assertIs(c1, c2)
        self.assertEqual(mb.call_count, 1)

    def test_different_config_returns_different_client(self):
        with patch.object(self.oc, "_build_client", side_effect=lambda *a: object()):
            c1 = self.oc.get_chat_client("http://x:11434", "pk-a", "sk", "http://lf")
            c2 = self.oc.get_chat_client("http://x:11434", "pk-b", "sk", "http://lf")
        self.assertIsNot(c1, c2)


class TestTraceRoot(unittest.TestCase):
    """_trace_root() fija los atributos de trace vía span raíz (patrón v3)."""

    def setUp(self):
        import ollama_client as oc
        self.oc = importlib.reload(oc)

    def test_noop_without_public_key(self):
        with patch("langfuse.get_client") as mock_gc:
            with self.oc._trace_root("t", "s1", "u1", ["a"], lf_public_key=None):
                pass
        mock_gc.assert_not_called()

    def test_sets_trace_attributes(self):
        mock_span = MagicMock()
        mock_lf = MagicMock()
        mock_lf.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_lf.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("langfuse.get_client", return_value=mock_lf) as mock_gc:
            with self.oc._trace_root("my-trace", "sess-1", "user-1", ["t1"], lf_public_key="pk-x"):
                pass

        mock_gc.assert_called_once_with(public_key="pk-x")
        mock_lf.start_as_current_span.assert_called_once_with(name="my-trace")
        mock_span.update_trace.assert_called_once_with(
            name="my-trace", session_id="sess-1", user_id="user-1", tags=["t1"],
        )


class FakeChoice:
    """Fake para simular response.choices[0]."""
    def __init__(self, content):
        self.message = MagicMock(content=content)


class FakeResponse:
    """Fake para simular respuesta de chat.completions.create()."""
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeStreamChunk:
    """Fake para simular chunks de streaming."""
    def __init__(self, content):
        self.choices = [MagicMock(delta=MagicMock(content=content))]


class FakeClient:
    """
    Fake del OpenAI client Langfuse-wrapped — registra llamadas para assertions.
    Igual que el wrapper real, acepta los kwargs Langfuse (name, session_id,
    user_id, tags, langfuse_public_key) sin reenviarlos a OpenAI.
    """
    def __init__(self, response_content="test response", stream_chunks=None):
        self._response_content = response_content
        self._stream_chunks = stream_chunks if stream_chunks is not None else ["chunk1", "chunk2", "chunk3"]
        self.last_call_kwargs = None
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = self._create

    def _create(self, **kwargs):
        self.last_call_kwargs = kwargs
        if kwargs.get("stream"):
            return iter([FakeStreamChunk(c) for c in self._stream_chunks])
        return FakeResponse(self._response_content)


class TestChatComplete(unittest.TestCase):
    """chat_complete() con client inyectado (DI)."""

    def setUp(self):
        import ollama_client as oc
        self.oc = importlib.reload(oc)

    def test_returns_response_content(self):
        fake = FakeClient(response_content="Hello from Ollama")
        result = self.oc.chat_complete(
            messages=[{"role": "user", "content": "hi"}],
            model="llama3.1",
            session_id="test-session",
            user_id="test-user",
            client=fake,
        )
        self.assertEqual(result, "Hello from Ollama")

    def test_passes_correct_model(self):
        fake = FakeClient()
        self.oc.chat_complete(
            messages=[{"role": "user", "content": "hi"}],
            model="mistral",
            session_id="s1",
            user_id="u1",
            client=fake,
        )
        self.assertEqual(fake.last_call_kwargs["model"], "mistral")

    def test_passes_temperature_and_max_tokens(self):
        fake = FakeClient()
        self.oc.chat_complete(
            messages=[{"role": "user", "content": "hi"}],
            model="llama3.1",
            session_id="s1",
            user_id="u1",
            temperature=1.5,
            max_tokens=4096,
            client=fake,
        )
        self.assertEqual(fake.last_call_kwargs["temperature"], 1.5)
        self.assertEqual(fake.last_call_kwargs["max_tokens"], 4096)

    def test_trace_attrs_not_leaked_to_create(self):
        """session_id/user_id/tags van por _trace_root, NO como kwargs de create()
        — el wrapper v3 no los extrae y romperían la llamada OpenAI real."""
        fake = FakeClient()
        self.oc.chat_complete(
            messages=[{"role": "user", "content": "hi"}],
            model="llama3.1",
            session_id="sess-123",
            user_id="user-456",
            trace_name="custom-trace",
            tags=["tag1", "tag2"],
            client=fake,
        )
        self.assertEqual(fake.last_call_kwargs["name"], "custom-trace")
        self.assertNotIn("session_id", fake.last_call_kwargs)
        self.assertNotIn("user_id", fake.last_call_kwargs)
        self.assertNotIn("tags", fake.last_call_kwargs)

    def test_lf_public_key_routes_call(self):
        fake = FakeClient()
        with patch("langfuse.get_client") as mock_gc:
            mock_gc.return_value.start_as_current_span.return_value.__enter__ = MagicMock()
            mock_gc.return_value.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            self.oc.chat_complete(
                messages=[{"role": "user", "content": "hi"}],
                model="llama3.1",
                session_id="s1",
                user_id="u1",
                client=fake,
                lf_public_key="pk-route",
            )
        self.assertEqual(fake.last_call_kwargs["langfuse_public_key"], "pk-route")

    def test_no_lf_public_key_omits_kwarg(self):
        fake = FakeClient()
        self.oc.chat_complete(
            messages=[{"role": "user", "content": "hi"}],
            model="llama3.1",
            session_id="s1",
            user_id="u1",
            client=fake,
        )
        self.assertNotIn("langfuse_public_key", fake.last_call_kwargs)

    def test_does_not_set_stream(self):
        fake = FakeClient()
        self.oc.chat_complete(
            messages=[{"role": "user", "content": "hi"}],
            model="llama3.1",
            session_id="s1",
            user_id="u1",
            client=fake,
        )
        self.assertNotIn("stream", fake.last_call_kwargs)


class TestChatStream(unittest.TestCase):
    """chat_stream() con client inyectado (DI)."""

    def setUp(self):
        import ollama_client as oc
        self.oc = importlib.reload(oc)

    def test_yields_all_chunks(self):
        fake = FakeClient(stream_chunks=["Hello", " ", "World"])
        chunks = list(self.oc.chat_stream(
            messages=[{"role": "user", "content": "hi"}],
            model="llama3.1",
            session_id="s1",
            user_id="u1",
            client=fake,
        ))
        self.assertEqual(chunks, ["Hello", " ", "World"])

    def test_skips_none_deltas(self):
        fake = FakeClient()
        chunks_raw = [FakeStreamChunk(c) for c in ["A", None, "B"]]
        fake.chat.completions.create = lambda **kw: iter(chunks_raw)

        chunks = list(self.oc.chat_stream(
            messages=[{"role": "user", "content": "hi"}],
            model="llama3.1",
            session_id="s1",
            user_id="u1",
            client=fake,
        ))
        self.assertEqual(chunks, ["A", "B"])

    def test_sets_stream_true(self):
        fake = FakeClient()
        list(self.oc.chat_stream(
            messages=[{"role": "user", "content": "hi"}],
            model="llama3.1",
            session_id="s1",
            user_id="u1",
            client=fake,
        ))
        self.assertTrue(fake.last_call_kwargs["stream"])

    def test_trace_attrs_not_leaked_in_stream(self):
        fake = FakeClient()
        list(self.oc.chat_stream(
            messages=[{"role": "user", "content": "hi"}],
            model="llama3.1",
            session_id="sess-stream",
            user_id="user-stream",
            trace_name="stream-trace",
            tags=["s1", "s2"],
            client=fake,
        ))
        self.assertEqual(fake.last_call_kwargs["name"], "stream-trace")
        self.assertNotIn("session_id", fake.last_call_kwargs)
        self.assertNotIn("tags", fake.last_call_kwargs)

    def test_empty_stream(self):
        fake = FakeClient(stream_chunks=[])
        chunks = list(self.oc.chat_stream(
            messages=[{"role": "user", "content": "hi"}],
            model="llama3.1",
            session_id="s1",
            user_id="u1",
            client=fake,
        ))
        self.assertEqual(chunks, [])


if __name__ == "__main__":
    unittest.main()
