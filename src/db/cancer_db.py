from pathlib import Path

from langchain_community.utilities import SQLDatabase

from src.config import CANCER_DB_PATH


def get_cancer_sql_database() -> SQLDatabase:
    """
    Return a LangChain SQLDatabase instance for the cancer.db.
    This will be used by the CancerDBTool.
    """
    db_path: Path = CANCER_DB_PATH

    if not db_path.exists():
        raise FileNotFoundError(
            f"Cancer database not found at {db_path}. "
            "Did you run `python -m src.data_prep.csv_to_sqlite`?"
        )

    uri = f"sqlite:///{db_path}"
    return SQLDatabase.from_uri(uri)
