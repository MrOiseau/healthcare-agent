"""
Functions for loading and preprocessing the healthcare dataset.
Adds robust cleaning steps (case normalisation, numeric / date coercion,
duplicate removal) and builds a `semantic_summary` column for RAG retrieval.

Used by the agent, evaluation scripts, and Streamlit app.
"""

import os
from typing import Optional

import pandas as pd


def load_and_preprocess_data(file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load the healthcare CSV, clean it, and return a ready-to-embed DataFrame.

    Args:
        file_path: Path to the CSV dataset.
        sample_size: If set, randomly sample this many rows (deterministic).

    Raises:
        FileNotFoundError: If `file_path` does not exist.

    Returns:
        Cleaned pandas DataFrame with an extra `semantic_summary` column.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    df = pd.read_csv(file_path)

    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42, ignore_index=True)

    # Column standardization ─ remove ALL non‑alnum chars, then snake‑case
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^0-9a-z]+", "_", regex=True)
        .str.strip("_")
    )
    if "room_number" in df.columns:
        df.rename(columns={"room_number": "room_number_str"}, inplace=True)

    # Text normalization
    text_cols = [
        "name",
        "doctor",
        "hospital",
        "insurance_provider",
        "medical_condition",
        "medication",
        "test_results",
        "admission_type",
        "gender",
        "blood_type",
    ]

    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()  # trim whitespace
        df[f"{col}_lc"] = df[col].str.lower()  # canonical copy

        # Pretty display versions where it matters
        if col in {"name", "doctor", "hospital"}:
            df[col] = df[col].str.title()

    # Numeric & Date correciton
    # Clean monetary column before numeric coercion
    if "billing_amount" in df.columns:
        df["billing_amount"] = (
            df["billing_amount"]
            .astype(str)
            .str.replace(r"[^\d.\-]", "", regex=True)  # strip $ , ** etc.
            .pipe(pd.to_numeric, errors="coerce")
        )

    date_cols = ["date_of_admission", "discharge_date"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)

    # Null‑handling for key text fields
    for col in ["medical_condition", "medication", "test_results"]:
        df[col] = df[col].fillna("Not specified")

    # Duplicate removal
    before = len(df)
    df = df.drop_duplicates(ignore_index=True)
    after = len(df)

    # Quick sanity check (comment out for prod jobs)
    if sample_size is None:
        print(f"[Preprocess] Dropped {before - after} exact duplicate rows.")
        print(df[["billing_amount"]].describe())

    # Semantic summary for embeddings
    def _row_to_summary(row: pd.Series) -> str:
        """Create a natural‑language blurb for semantic search."""
        bill = (
            f"${row['billing_amount']:.2f}"
            if pd.notnull(row["billing_amount"])
            else "an unspecified amount"
        )
        return (
            f"Patient {row['name']} (Age: {row['age']}, Gender: {row['gender']}) "
            f"was admitted on {row['date_of_admission'].date() if pd.notnull(row['date_of_admission']) else 'an unknown date'} "
            f"for {row['medical_condition']}. They were treated by Dr. {row['doctor']} at {row['hospital']}. "
            f"Blood type is {row['blood_type']}. Admission was {row['admission_type']}. "
            f"Prescribed medication was {row['medication']} and test results were {row['test_results']}. "
            f"The total bill was {bill}. "
            f"Discharged on {row['discharge_date'].date() if pd.notnull(row['discharge_date']) else 'an unknown date'}."
        )

    df["semantic_summary"] = df.apply(_row_to_summary, axis=1)

    return df
