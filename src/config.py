import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_PATH = BASE_DIR / "vectorstore" / "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"


def require_google_api_key() -> str:
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        raise ValueError(
            "A variável GOOGLE_API_KEY não foi encontrada. "
            "Crie um arquivo .env baseado no .env.example."
        )

    return google_api_key
