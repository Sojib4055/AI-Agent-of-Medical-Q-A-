import sqlite3
from pathlib import Path
from typing import Dict

from src.config import HEART_DB_PATH, CANCER_DB_PATH, DIABETES_DB_PATH


def get_table_schema(db_path: Path, table_name: str) -> str:
    """
    Return a text description of the columns in a SQLite table using PRAGMA.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(f"PRAGMA table_info({table_name});")
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        return f"No schema found for table '{table_name}' in {db_path}"

    lines = [f"Schema for table '{table_name}':"]
    for cid, name, col_type, notnull, default_value, pk in rows:
        lines.append(
            f"  - {name} ({col_type})"
            + (" NOT NULL" if notnull else "")
            + (" PRIMARY KEY" if pk else "")
        )

    return "\n".join(lines)


def all_schemas() -> Dict[str, str]:
    """
    Return a dict with schema summaries for all three datasets.
    Useful for prompting the model.
    """
    return {
        "heart_disease": get_table_schema(HEART_DB_PATH, "heart_disease"),
        "cancer_data": get_table_schema(CANCER_DB_PATH, "cancer_data"),
        "diabetes_data": get_table_schema(DIABETES_DB_PATH, "diabetes_data"),
    }


if __name__ == "__main__":
    schemas = all_schemas()
    for name, text in schemas.items():
        print("===" * 10)
        print(f"Dataset: {name}")
        print(text)
