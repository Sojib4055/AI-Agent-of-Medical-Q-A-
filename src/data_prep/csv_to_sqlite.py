import sqlite3
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR, DB_DIR, HEART_DB_PATH, CANCER_DB_PATH, DIABETES_DB_PATH


def resolve_csv_path(filename_options: list[str], dataset_label: str) -> Path:
    """
    Return the first existing path in RAW_DIR from a list of candidate names.
    Raises a helpful error if none are found.
    """
    for filename in filename_options:
        candidate = RAW_DIR / filename
        if candidate.exists():
            return candidate
    checked = ", ".join(str(RAW_DIR / name) for name in filename_options)
    raise FileNotFoundError(
        f"{dataset_label} CSV not found. Checked locations: {checked}"
    )


def load_heart_csv() -> pd.DataFrame:
    """
    Load the heart disease CSV file from data/raw.
    You can add cleaning steps here if needed.
    """
    csv_path = resolve_csv_path(
        ["heart_disease.csv", "heart.csv"], dataset_label="Heart disease"
    )
    df = pd.read_csv(csv_path)

    # Example light cleaning (optional)
    # df = df.dropna()  # or handle NaNs more carefully

    return df


def load_cancer_csv() -> pd.DataFrame:
    """
    Load the cancer CSV file from data/raw.
    """
    csv_path = resolve_csv_path(
        ["cancer.csv", "The_Cancer_data_1500_V2.csv"], dataset_label="Cancer"
    )
    df = pd.read_csv(csv_path)
    return df


def load_diabetes_csv() -> pd.DataFrame:
    """
    Load the diabetes CSV file from data/raw.
    """
    csv_path = resolve_csv_path(
        ["diabetes.csv", "diabetes_clean.csv"], dataset_label="Diabetes"
    )
    df = pd.read_csv(csv_path)
    return df


def df_to_sqlite(df: pd.DataFrame, db_path: Path, table_name: str) -> None:
    """
    Write a pandas DataFrame to a SQLite database with a given table name.
    If the table exists, it will be replaced.
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"[OK] Wrote table '{table_name}' to {db_path}")
    finally:
        conn.close()


def build_heart_db() -> None:
    df = load_heart_csv()
    # You can rename columns or enforce dtypes here if needed
    df_to_sqlite(df, HEART_DB_PATH, table_name="heart_disease")


def build_cancer_db() -> None:
    df = load_cancer_csv()
    df_to_sqlite(df, CANCER_DB_PATH, table_name="cancer_data")


def build_diabetes_db() -> None:
    df = load_diabetes_csv()
    df_to_sqlite(df, DIABETES_DB_PATH, table_name="diabetes_data")


def build_all_dbs() -> None:
    """
    Build all three SQLite databases from the raw CSVs.
    Run this once after downloading CSV files from Kaggle.
    """
    print("=== Building Heart Disease DB ===")
    build_heart_db()
    print("=== Building Cancer DB ===")
    build_cancer_db()
    print("=== Building Diabetes DB ===")
    build_diabetes_db()
    print("=== All databases built successfully ===")


if __name__ == "__main__":
    build_all_dbs()
