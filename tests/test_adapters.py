"""
Tests para la capa de adapters — unittest stdlib + DI con Fakes.
Ejecutar: python3 -m unittest tests.test_adapters -v

Todos los tests corren SIN Ollama ni Langfuse activos.
"""

import unittest
from unittest.mock import patch, MagicMock

from langfuse_ollama.adapters.ollama_api import OllamaApi
from langfuse_ollama.adapters import langfuse_ollama as gw
from langfuse_ollama.domain.entities import ChatMessage, ChatRequest


def make_request(**overrides):
    kwargs = dict(
        messages=[ChatMessage(role="user", content="hi")],
        model="llama3.1",
        session_id="s1",
        user_id="u1",
        trace_name="t1",
        tags=["a"],
        temperature=0.7,
        max_tokens=2048,
    )
    kwargs.update(overrides)
    return ChatRequest(**kwargs)


class TestOllamaApiPing(unittest.TestCase):
    """ping() distingue Ollama disponible de no disponible."""

    def setUp(self):
        self.api = OllamaApi("http://remote:11434")

    @patch("langfuse_ollama.adapters.ollama_api.httpx.get")
    def test_true_on_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        self.assertTrue(self.api.ping())
        self.assertIn("http://remote:11434/api/version", mock_get.call_args.args)

    @patch("langfuse_ollama.adapters.ollama_api.httpx.get")
    def test_false_on_connection_error(self, mock_get):
        mock_get.side_effect = ConnectionError("refused")
        self.assertFalse(self.api.ping())

    @patch("langfuse_ollama.adapters.ollama_api.httpx.get")
    def test_false_on_http_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("500")
        mock_get.return_value = mock_response
        self.assertFalse(self.api.ping())


class TestOllamaApiListModels(unittest.TestCase):
    """list_models() con httpx mockeado. Sin fallback que enmascare fallos."""

    def setUp(self):
        self.api = OllamaApi("http://localhost:11434")

    @patch("langfuse_ollama.adapters.ollama_api.httpx.get")
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
        self.assertEqual(self.api.list_models(), ["llama3.1", "mistral", "codellama"])

    @patch("langfuse_ollama.adapters.ollama_api.httpx.get")
    def test_returns_empty_on_connection_error(self, mock_get):
        mock_get.side_effect = ConnectionError("refused")
        self.assertEqual(self.api.list_models(), [])

    @patch("langfuse_ollama.adapters.ollama_api.httpx.get")
    def test_returns_empty_on_http_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("500")
        mock_get.return_value = mock_response
        self.assertEqual(self.api.list_models(), [])

    @patch("langfuse_ollama.adapters.ollama_api.httpx.get")
    def test_handles_empty_models_list(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        self.assertEqual(self.api.list_models(), [])

    @patch("langfuse_ollama.adapters.ollama_api.httpx.get")
    def test_handles_missing_models_key(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        self.assertEqual(self.api.list_models(), [])


class TestGetGateway(unittest.TestCase):
    """get_gateway() cachea por configuración."""

    def setUp(self):
        gw._gateways.clear()

    def test_same_config_returns_same_gateway(self):
        with patch.object(gw, "LangfuseOllamaGateway", side_effect=lambda *a: object()) as mb:
            g1 = gw.get_gateway("http://x:11434", "pk", "sk", "http://lf")
            g2 = gw.get_gateway("http://x:11434", "pk", "sk", "http://lf")
        self.assertIs(g1, g2)
        self.assertEqual(mb.call_count, 1)

    def test_different_config_returns_different_gateway(self):
        with patch.object(gw, "LangfuseOllamaGateway", side_effect=lambda *a: object()):
            g1 = gw.get_gateway("http://x:11434", "pk-a", "sk", "http://lf")
            g2 = gw.get_gateway("http://x:11434", "pk-b", "sk", "http://lf")
        self.assertIsNot(g1, g2)


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
    Igual que el wrapper real, acepta los kwargs Langfuse (name,
    langfuse_public_key) sin reenviarlos a OpenAI.
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


def make_gateway(fake_client, lf_public_key=""):
    return gw.LangfuseOllamaGateway(
        ollama_url="http://localhost:11434",
        lf_public_key=lf_public_key,
        client=fake_client,
    )


class TestGatewayComplete(unittest.TestCase):
    """complete() con client inyectado (DI)."""

    def test_returns_response_content(self):
        fake = FakeClient(response_content="Hello from Ollama")
        result = make_gateway(fake).complete(make_request())
        self.assertEqual(result, "Hello from Ollama")

    def test_passes_model_and_params(self):
        fake = FakeClient()
        make_gateway(fake).complete(make_request(model="mistral", temperature=1.5, max_tokens=4096))
        self.assertEqual(fake.last_call_kwargs["model"], "mistral")
        self.assertEqual(fake.last_call_kwargs["temperature"], 1.5)
        self.assertEqual(fake.last_call_kwargs["max_tokens"], 4096)

    def test_messages_serialized_as_dicts(self):
        fake = FakeClient()
        make_gateway(fake).complete(make_request(
            messages=[ChatMessage(role="system", content="Be brief."),
                      ChatMessage(role="user", content="hi")],
        ))
        self.assertEqual(fake.last_call_kwargs["messages"],
                         [{"role": "system", "content": "Be brief."},
                          {"role": "user", "content": "hi"}])

    def test_trace_attrs_not_leaked_to_create(self):
        """session_id/user_id/tags van por _trace_root, NO como kwargs de create()
        — el wrapper v3 no los extrae y romperían la llamada OpenAI real."""
        fake = FakeClient()
        make_gateway(fake).complete(make_request(trace_name="custom-trace"))
        self.assertEqual(fake.last_call_kwargs["name"], "custom-trace")
        self.assertNotIn("session_id", fake.last_call_kwargs)
        self.assertNotIn("user_id", fake.last_call_kwargs)
        self.assertNotIn("tags", fake.last_call_kwargs)

    def test_lf_public_key_routes_call(self):
        fake = FakeClient()
        with patch("langfuse.get_client") as mock_gc:
            mock_gc.return_value.start_as_current_span.return_value.__enter__ = MagicMock()
            mock_gc.return_value.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            make_gateway(fake, lf_public_key="pk-route").complete(make_request())
        self.assertEqual(fake.last_call_kwargs["langfuse_public_key"], "pk-route")

    def test_no_lf_public_key_omits_kwarg(self):
        fake = FakeClient()
        make_gateway(fake).complete(make_request())
        self.assertNotIn("langfuse_public_key", fake.last_call_kwargs)

    def test_does_not_set_stream(self):
        fake = FakeClient()
        make_gateway(fake).complete(make_request())
        self.assertNotIn("stream", fake.last_call_kwargs)

    def test_empty_choices_returns_empty_string(self):
        """Algún backend puede devolver choices=[] — no debe romper con IndexError."""
        fake = FakeClient()
        empty = MagicMock(choices=[])
        fake.chat.completions.create = lambda **kw: empty
        self.assertEqual(make_gateway(fake).complete(make_request()), "")

    def test_none_content_returns_empty_string(self):
        fake = FakeClient(response_content=None)
        self.assertEqual(make_gateway(fake).complete(make_request()), "")


class TestGatewayStream(unittest.TestCase):
    """stream() con client inyectado (DI)."""

    def test_yields_all_chunks(self):
        fake = FakeClient(stream_chunks=["Hello", " ", "World"])
        chunks = list(make_gateway(fake).stream(make_request()))
        self.assertEqual(chunks, ["Hello", " ", "World"])

    def test_skips_none_deltas(self):
        fake = FakeClient()
        chunks_raw = [FakeStreamChunk(c) for c in ["A", None, "B"]]
        fake.chat.completions.create = lambda **kw: iter(chunks_raw)
        chunks = list(make_gateway(fake).stream(make_request()))
        self.assertEqual(chunks, ["A", "B"])

    def test_sets_stream_true(self):
        fake = FakeClient()
        list(make_gateway(fake).stream(make_request()))
        self.assertTrue(fake.last_call_kwargs["stream"])

    def test_trace_attrs_not_leaked_in_stream(self):
        fake = FakeClient()
        list(make_gateway(fake).stream(make_request(trace_name="stream-trace")))
        self.assertEqual(fake.last_call_kwargs["name"], "stream-trace")
        self.assertNotIn("session_id", fake.last_call_kwargs)
        self.assertNotIn("tags", fake.last_call_kwargs)

    def test_empty_stream(self):
        fake = FakeClient(stream_chunks=[])
        chunks = list(make_gateway(fake).stream(make_request()))
        self.assertEqual(chunks, [])

    def test_skips_chunks_without_choices(self):
        """El chunk final con usage (choices=[]) no debe romper con IndexError."""
        fake = FakeClient()
        chunks_raw = [FakeStreamChunk("A"), MagicMock(choices=[]), FakeStreamChunk("B")]
        fake.chat.completions.create = lambda **kw: iter(chunks_raw)
        chunks = list(make_gateway(fake).stream(make_request()))
        self.assertEqual(chunks, ["A", "B"])


class TestTraceRoot(unittest.TestCase):
    """_trace_root() fija los atributos de trace vía span raíz (patrón v3)."""

    def test_noop_without_public_key(self):
        gateway = make_gateway(FakeClient(), lf_public_key="")
        with patch("langfuse.get_client") as mock_gc:
            with gateway._trace_root(make_request()):
                pass
        mock_gc.assert_not_called()

    def test_sets_trace_attributes(self):
        mock_span = MagicMock()
        mock_lf = MagicMock()
        mock_lf.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_lf.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        gateway = make_gateway(FakeClient(), lf_public_key="pk-x")
        request = make_request(trace_name="my-trace", session_id="sess-1",
                               user_id="user-1", tags=["t1"])
        with patch("langfuse.get_client", return_value=mock_lf) as mock_gc:
            with gateway._trace_root(request):
                pass

        mock_gc.assert_called_once_with(public_key="pk-x")
        mock_lf.start_as_current_span.assert_called_once_with(name="my-trace")
        mock_span.update_trace.assert_called_once_with(
            name="my-trace", session_id="sess-1", user_id="user-1", tags=["t1"],
        )


if __name__ == "__main__":
    unittest.main()
