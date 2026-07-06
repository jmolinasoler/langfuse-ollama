"""
Adapter del puerto ModelCatalog: API nativa de Ollama vía httpx.
"""

import logging
from typing import List

import httpx

logger = logging.getLogger(__name__)


class OllamaApi:
    """Implementación de ModelCatalog contra la API REST de Ollama."""

    def __init__(self, base_url: str):
        self._base_url = base_url

    def ping(self) -> bool:
        """Comprueba si Ollama responde. No confundir con list_models()."""
        try:
            r = httpx.get(f"{self._base_url}/api/version", timeout=3)
            r.raise_for_status()
            return True
        except Exception as e:
            logger.warning("Ollama unreachable at %s: %s", self._base_url, e)
            return False

    def list_models(self) -> List[str]:
        """Lista modelos disponibles en Ollama. Retorna [] si no responde."""
        try:
            r = httpx.get(f"{self._base_url}/api/tags", timeout=5)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception as e:
            logger.warning("Failed to list Ollama models: %s", e)
            return []
