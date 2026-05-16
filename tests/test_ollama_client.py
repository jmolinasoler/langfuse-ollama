"""
Tests para ollama_client.py — unittest stdlib + DI con Fakes.
Ejecutar: python3 -m unittest tests.test_ollama_client -v

Todos los tests corren SIN Ollama ni Langfuse activos.
"""

import unittest
from unittest.mock import patch, MagicMock
import uuid
import os
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


class TestInitLangfuseEnv(unittest.TestCase):
    """init_langfuse_env() debe configurar las variables de entorno correctamente."""

    def setUp(self):
        self._saved = {}
        for key in ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL", "OPENAI_API_KEY"]:
            self._saved[key] = os.environ.pop(key, None)

    def tearDown(self):
        for key, val in self._saved.items():
            if val is not None:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)

    def test_sets_env_vars(self):
        import config
        config = importlib.reload(config)
        config.LANGFUSE_PUBLIC_KEY = "pk-test"
        config.LANGFUSE_SECRET_KEY = "sk-test"
        config.LANGFUSE_BASE_URL = "http://test:3000"

        import ollama_client as oc
        oc = importlib.reload(oc)
        oc.init_langfuse_env()

        self.assertEqual(os.environ["LANGFUSE_PUBLIC_KEY"], "pk-test")
        self.assertEqual(os.environ["LANGFUSE_SECRET_KEY"], "sk-test")
        self.assertEqual(os.environ["LANGFUSE_BASE_URL"], "http://test:3000")
        self.assertEqual(os.environ["OPENAI_API_KEY"], "ollama")


class TestListModels(unittest.TestCase):
    """list_models() con httpx mockeado."""

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
    def test_returns_fallback_on_connection_error(self, mock_get):
        mock_get.side_effect = ConnectionError("refused")

        models = self.oc.list_models()
        self.assertEqual(len(models), 1)  # fallback al DEFAULT_MODEL

    @patch("ollama_client.httpx.get")
    def test_returns_fallback_on_http_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("500")
        mock_get.return_value = mock_response

        models = self.oc.list_models()
        self.assertEqual(len(models), 1)

    @patch("ollama_client.httpx.get")
    def test_handles_empty_models_list(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        models = self.oc.list_models()
        self.assertEqual(models, [])

    @patch("ollama_client.httpx.get")
    def test_handles_missing_models_key(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        models = self.oc.list_models()
        self.assertEqual(models, [])


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
    """Fake OpenAI client inyectable — registra llamadas para assertions."""
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

    def test_langfuse_metadata_not_leaked_to_openai(self):
        """Con DI, los kwargs de Langfuse NO se pasan al create() de OpenAI."""
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
        # El path DI envía solo kwargs puros de OpenAI
        self.assertNotIn("name", fake.last_call_kwargs)
        self.assertNotIn("session_id", fake.last_call_kwargs)
        self.assertNotIn("user_id", fake.last_call_kwargs)
        self.assertNotIn("tags", fake.last_call_kwargs)

    def test_accepts_langfuse_params_without_error(self):
        """Verifica que los parámetros de Langfuse se aceptan sin error."""
        fake = FakeClient()
        # No debe lanzar excepción
        result = self.oc.chat_complete(
            messages=[{"role": "user", "content": "hi"}],
            model="codellama",
            session_id="s1",
            user_id="u1",
            tags=["tag1"],
            client=fake,
        )
        self.assertEqual(result, "test response")

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
        fake = FakeClient(stream_chunks=["A", None, "B"])
        # Override el fake para que maneje None correctamente
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
        # Consumir el generador para que se ejecute la llamada
        list(self.oc.chat_stream(
            messages=[{"role": "user", "content": "hi"}],
            model="llama3.1",
            session_id="s1",
            user_id="u1",
            client=fake,
        ))
        self.assertTrue(fake.last_call_kwargs["stream"])

    def test_langfuse_metadata_not_leaked_in_stream(self):
        """Con DI, los kwargs de Langfuse NO se pasan al create() en streaming."""
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
        self.assertNotIn("name", fake.last_call_kwargs)
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


class TestMakeClient(unittest.TestCase):
    """_make_client() con DI."""

    def setUp(self):
        import ollama_client as oc
        self.oc = importlib.reload(oc)

    def test_returns_injected_client(self):
        fake = object()
        result = self.oc._make_client(client=fake)
        self.assertIs(result, fake)

    def test_returns_injected_none_creates_new(self):
        # Sin inyección, debería intentar crear un OpenAI real.
        # Mockeamos _get_openai_cls para evitar dependencia real.
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        with patch.object(self.oc, "_get_openai_cls", return_value=mock_cls):
            result = self.oc._make_client()
            self.assertIs(result, mock_instance)


if __name__ == "__main__":
    unittest.main()
