import pandas as pd
import os
from typing import Optional


def load_and_preprocess_data(file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Loads the healthcare dataset, optionally samples it, cleans it, and creates
    a synthetic 'semantic_summary' column for embedding.

    Args:
        file_path: The path to the CSV dataset.
        sample_size: The number of rows to sample. If None, the entire dataset is used.

    Returns:
        The preprocessed DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    df = pd.read_csv(file_path)

    if sample_size:
        if sample_size > len(df):
            df = df.sample(n=len(df), random_state=42)
        else:
            df = df.sample(n=sample_size, random_state=42)

    # Standardize column names for easier access in pandas agent
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df.rename(columns={'room_number': 'room_number_str'}, inplace=True)

    # Fill NaNs in key text fields to ensure the semantic summary is always complete
    for col in ['medical_condition', 'medication', 'test_results']:
        df[col] = df[col].fillna('Not specified')

    # Core of the hybrid approach: 
    # create a natural language summary of each structured row to enable semantic search
    df['semantic_summary'] = df.apply(
        lambda row: f"Patient {row['name']} (Age: {row['age']}, Gender: {row['gender']}) was admitted on {row['date_of_admission']} "
                    f"for {row['medical_condition']}. They were treated by Dr. {row['doctor']} at {row['hospital']}. "
                    f"Blood type is {row['blood_type']}. Admission was {row['admission_type']}. "
                    f"Prescribed medication was {row['medication']} and test results were {row['test_results']}. "
                    f"The total bill was ${row['billing_amount']:.2f}. "
                    f"Discharged on {row['discharge_date']}.",
        axis=1
    )
    
    return df
