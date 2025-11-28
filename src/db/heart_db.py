from pathlib import Path

from langchain_community.utilities import SQLDatabase

from src.config import HEART_DB_PATH


def get_heart_sql_database() -> SQLDatabase:
    """
    Return a LangChain SQLDatabase instance for the heart_disease.db.
    This will be used by the HeartDiseaseDBTool.
    """
    db_path: Path = HEART_DB_PATH

    if not db_path.exists():
        raise FileNotFoundError(
            f"Heart disease database not found at {db_path}. "
            "Did you run `python -m src.data_prep.csv_to_sqlite`?"
        )

    uri = f"sqlite:///{db_path}"
    return SQLDatabase.from_uri(uri)
