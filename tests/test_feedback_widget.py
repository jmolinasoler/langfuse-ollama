"""
Tests para feedback_widget._boot_snippet (función pura).
Ejecutar: python3 -m unittest tests.test_feedback_widget -v

No testeamos render() porque depende del runtime de Streamlit
(st.button, st.session_state, components.html). Esa pieza la verifica
el usuario manualmente cargando la app — no la mockeamos.
"""

import unittest
from langfuse_ollama.ui import feedback_widget


class TestBootSnippet(unittest.TestCase):
    """Verifica que el snippet contiene los valores correctos y la forma esperada."""

    def setUp(self):
        self.snippet = feedback_widget._boot_snippet()

    def test_contains_app_id(self):
        self.assertIn(feedback_widget.FEATUREBASE_APP_ID, self.snippet)

    def test_contains_pinned_sdk_url(self):
        self.assertIn(feedback_widget.SDK_URL, self.snippet)
        self.assertIn(f"featurebase-js@{feedback_widget.SDK_VERSION}", self.snippet)

    def test_uses_esm_sh_cdn(self):
        # Camino B: ESM CDN, no el script legacy de do.featurebase.app.
        self.assertIn("https://esm.sh/", self.snippet)
        self.assertNotIn("do.featurebase.app", self.snippet)

    def test_injects_into_parent_document(self):
        # Crítico: el SDK tiene que montarse en el doc padre, no en el
        # iframe height=0 de components.html.
        self.assertIn("window.parent", self.snippet)
        self.assertIn("parentDoc.head.appendChild", self.snippet)

    def test_idempotency_guard(self):
        # Streamlit re-corre el script en cada rerun; el flag evita boots
        # duplicados.
        self.assertIn("__featurebaseBooted", self.snippet)

    def test_uses_module_script(self):
        # featurebase-js es ESM nativo → tiene que cargarse como
        # type="module".
        self.assertIn("'module'", self.snippet)

    def test_calls_default_export_with_appid(self):
        # Forma exacta del boot anónimo: Featurebase({ appId: '...' }).
        # Sin JWT, sin identify (Fase 1 anónima).
        self.assertIn("Featurebase({", self.snippet)
        self.assertIn("appId:", self.snippet)
        self.assertNotIn("featurebaseJwt", self.snippet)


class TestAppIdConstant(unittest.TestCase):
    """Sanity check sobre el appId hardcodeado."""

    def test_appid_is_24_hex_chars(self):
        # Featurebase appId = MongoDB ObjectId → 24 chars hex.
        self.assertEqual(len(feedback_widget.FEATUREBASE_APP_ID), 24)
        int(feedback_widget.FEATUREBASE_APP_ID, 16)  # no debe lanzar


if __name__ == "__main__":
    unittest.main()
