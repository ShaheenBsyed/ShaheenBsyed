import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(os.getenv("BASE_DIR", Path(__file__).resolve().parent.parent))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", BASE_DIR / "indexes"))


class EmbeddingConfig(BaseModel):
    model_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    device: str = os.getenv("DEVICE", "cpu")
    batch_size: int = int(os.getenv("EMBED_BATCH", "64"))


class GenerationConfig(BaseModel):
    # Using OpenAI-compatible API or local model via transformers
    provider: str = os.getenv("GEN_PROVIDER", "transformers")
    model_name: str = os.getenv("GEN_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    max_tokens: int = int(os.getenv("GEN_MAX_TOKENS", "512"))
    temperature: float = float(os.getenv("GEN_TEMPERATURE", "0.2"))
    top_p: float = float(os.getenv("GEN_TOP_P", "0.95"))


class AppConfig(BaseModel):
    embedding: EmbeddingConfig = EmbeddingConfig()
    generation: GenerationConfig = GenerationConfig()
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    index_dir: Path = INDEX_DIR


settings = AppConfig()
