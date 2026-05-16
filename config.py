import os
from dotenv import load_dotenv

load_dotenv()

def get_required(key: str) -> str:
    """Obtiene variable de entorno obligatoria. Lanza EnvironmentError si no existe."""
    val = os.getenv(key, "").strip()
    if not val:
        raise EnvironmentError(f"Missing required env var: {key}")
    return val

LANGFUSE_PUBLIC_KEY  = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY  = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL    = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
OLLAMA_BASE_URL      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL        = os.getenv("DEFAULT_MODEL", "llama3.1")
DEFAULT_SYSTEM_PROMPT = os.getenv("DEFAULT_SYSTEM_PROMPT", "You are a helpful assistant.")
DEFAULT_USER_ID      = os.getenv("DEFAULT_USER_ID", "user")

def langfuse_configured(public_key=None, secret_key=None) -> bool:
    """Verifica si Langfuse está configurado. Acepta override dinámico desde la UI."""
    pk = public_key if public_key is not None else LANGFUSE_PUBLIC_KEY
    sk = secret_key if secret_key is not None else LANGFUSE_SECRET_KEY
    return bool(pk and sk)
