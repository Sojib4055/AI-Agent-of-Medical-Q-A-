from pathlib import Path
import os

from dotenv import load_dotenv

# === Base paths ===
# BASE_DIR points to your project root: multi_tool_med_agent_full/
BASE_DIR = Path(__file__).resolve().parents[2]

# Load .env from project root
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)

# === API KEYS ===
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")

# Environment label (dev / prod / test)
APP_ENV: str = os.getenv("APP_ENV", "dev")

# === MODEL NAMES (Groq) ===
# Stronger model for routing & explanations
ROUTER_MODEL: str = os.getenv("ROUTER_MODEL", "llama-3.3-70b-versatile")
# Smaller / cheaper model for SQL agents
SQL_AGENT_MODEL: str = os.getenv("SQL_AGENT_MODEL", "llama-3.3-70b-versatile")

# === DATA PATHS ===
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_DIR = DATA_DIR / "db"

HEART_DB_PATH = DB_DIR / "heart_disease.db"
CANCER_DB_PATH = DB_DIR / "cancer.db"
DIABETES_DB_PATH = DB_DIR / "diabetes.db"


def validate_api_keys() -> None:
    """
    Optional helper: call this at startup if you want to ensure keys are set.
    Currently checks GROQ_API_KEY; you can also check TAVILY_API_KEY if required.
    """
    if GROQ_API_KEY is None:
        raise RuntimeError("GROQ_API_KEY is not set in the environment (.env).")
