import importlib
import pandas as pd
import numpy as np

def load_sample_data(difficulty="easy"):
    """
    Dynamically loads a task based on difficulty level.
    Supported: 'easy', 'medium', 'hard'
    """
    try:
        module = importlib.import_module(f"env.tasks.{difficulty}")
        return module.get_task()
    except (ImportError, AttributeError):
        # Fallback to easy if not found
        from env.tasks import easy
        return easy.get_task()

def get_ground_truth(df: pd.DataFrame):
    """
    Generates a cleaned version of the data for internal reward calculation.
    In a real system, this would be the original data before noise injection.
    """
    clean_df = df.copy()
    clean_df = clean_df.drop_duplicates()
    
    # 1. Fill nulls with mean or 'Unknown'
    for col in clean_df.columns:
        if clean_df[col].isnull().sum() > 0:
            if clean_df[col].dtype == np.number:
                clean_df[col] = clean_df[col].fillna(clean_df[col].mean())
            else:
                clean_df[col] = clean_df[col].fillna("Unknown")
    
    # 2. Normalize capitalization
    str_cols = clean_df.select_dtypes(include=['object']).columns
    for col in str_cols:
        clean_df[col] = clean_df[col].astype(str).str.capitalize()
                
    return clean_df