"""
Tests para config.py — unittest stdlib.
Ejecutar: python3 -m unittest tests.test_config -v
"""

import unittest
from unittest.mock import patch
import os
import importlib


class TestConfigDefaults(unittest.TestCase):
    """Verifica valores por defecto cuando no hay variables de entorno."""

    def setUp(self):
        # Guardar estado original del env
        self._original_env = {}
        self._keys = [
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL",
            "OLLAMA_BASE_URL", "DEFAULT_MODEL", "DEFAULT_SYSTEM_PROMPT", "DEFAULT_USER_ID",
        ]
        for key in self._keys:
            self._original_env[key] = os.environ.pop(key, None)

    def tearDown(self):
        # Restaurar env original
        for key, val in self._original_env.items():
            if val is not None:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)

    def _reload_config(self):
        from langfuse_ollama import config
        with patch("dotenv.load_dotenv"):
            return importlib.reload(config)

    def test_default_langfuse_public_key_empty(self):
        cfg = self._reload_config()
        self.assertEqual(cfg.LANGFUSE_PUBLIC_KEY, "")

    def test_default_langfuse_secret_key_empty(self):
        cfg = self._reload_config()
        self.assertEqual(cfg.LANGFUSE_SECRET_KEY, "")

    def test_default_langfuse_base_url(self):
        cfg = self._reload_config()
        self.assertEqual(cfg.LANGFUSE_BASE_URL, "https://cloud.langfuse.com")

    def test_default_ollama_base_url(self):
        cfg = self._reload_config()
        self.assertEqual(cfg.OLLAMA_BASE_URL, "http://localhost:11434")

    def test_default_model(self):
        cfg = self._reload_config()
        self.assertEqual(cfg.DEFAULT_MODEL, "llama3.1")

    def test_default_system_prompt(self):
        cfg = self._reload_config()
        self.assertEqual(cfg.DEFAULT_SYSTEM_PROMPT, "You are a helpful assistant.")

    def test_default_user_id(self):
        cfg = self._reload_config()
        self.assertEqual(cfg.DEFAULT_USER_ID, "user")


class TestConfigFromEnv(unittest.TestCase):
    """Verifica que las variables se cargan correctamente desde el entorno."""

    def setUp(self):
        self._original_env = {}
        self._test_values = {
            "LANGFUSE_PUBLIC_KEY": "pk-test-123",
            "LANGFUSE_SECRET_KEY": "sk-test-456",
            "LANGFUSE_BASE_URL": "http://custom:3000",
            "OLLAMA_BASE_URL": "http://remote:11434",
            "DEFAULT_MODEL": "mistral",
            "DEFAULT_SYSTEM_PROMPT": "Custom prompt",
            "DEFAULT_USER_ID": "test-user",
        }
        for key, val in self._test_values.items():
            self._original_env[key] = os.environ.get(key)
            os.environ[key] = val

    def tearDown(self):
        for key, val in self._original_env.items():
            if val is not None:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)

    def _reload_config(self):
        from langfuse_ollama import config
        with patch("dotenv.load_dotenv"):
            return importlib.reload(config)

    def test_langfuse_public_key_from_env(self):
        cfg = self._reload_config()
        self.assertEqual(cfg.LANGFUSE_PUBLIC_KEY, "pk-test-123")

    def test_langfuse_secret_key_from_env(self):
        cfg = self._reload_config()
        self.assertEqual(cfg.LANGFUSE_SECRET_KEY, "sk-test-456")

    def test_ollama_base_url_from_env(self):
        cfg = self._reload_config()
        self.assertEqual(cfg.OLLAMA_BASE_URL, "http://remote:11434")

    def test_default_model_from_env(self):
        cfg = self._reload_config()
        self.assertEqual(cfg.DEFAULT_MODEL, "mistral")


class TestLangfuseConfigured(unittest.TestCase):
    """Verifica langfuse_configured() con y sin override dinámico."""

    def setUp(self):
        self._original_pk = os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
        self._original_sk = os.environ.pop("LANGFUSE_SECRET_KEY", None)

    def tearDown(self):
        if self._original_pk is not None:
            os.environ["LANGFUSE_PUBLIC_KEY"] = self._original_pk
        if self._original_sk is not None:
            os.environ["LANGFUSE_SECRET_KEY"] = self._original_sk

    def _reload_config(self):
        from langfuse_ollama import config
        with patch("dotenv.load_dotenv"):
            return importlib.reload(config)

    def test_not_configured_when_empty(self):
        cfg = self._reload_config()
        self.assertFalse(cfg.langfuse_configured())

    def test_configured_with_override(self):
        cfg = self._reload_config()
        self.assertTrue(cfg.langfuse_configured(public_key="pk-x", secret_key="sk-x"))

    def test_not_configured_partial_override(self):
        cfg = self._reload_config()
        self.assertFalse(cfg.langfuse_configured(public_key="pk-x", secret_key=""))

    def test_configured_from_env(self):
        os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-env"
        os.environ["LANGFUSE_SECRET_KEY"] = "sk-env"
        cfg = self._reload_config()
        self.assertTrue(cfg.langfuse_configured())


if __name__ == "__main__":
    unittest.main()
