"""
Tests para el dominio — entidades puras, sin dependencias.
Ejecutar: python3 -m unittest tests.test_domain -v
"""

import unittest
import uuid

from langfuse_ollama.domain.entities import new_session_id


class TestNewSessionId(unittest.TestCase):
    """new_session_id() debe retornar UUIDs válidos y únicos."""

    def test_returns_valid_uuid(self):
        sid = new_session_id()
        parsed = uuid.UUID(sid)
        self.assertEqual(str(parsed), sid)

    def test_returns_unique_ids(self):
        ids = {new_session_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)

    def test_returns_string_type(self):
        self.assertIsInstance(new_session_id(), str)


if __name__ == "__main__":
    unittest.main()
