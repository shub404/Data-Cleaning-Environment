import pandas as pd
import numpy as np

def calculate_reward(current_data: pd.DataFrame, ground_truth: pd.DataFrame):
    """
    Calculates a multi-dimensional Data Quality Score ranging from 0 to 1.0.
    """
    if current_data.empty:
        return 0.01
        
    dup_count = current_data.duplicated().sum()
    uniqueness = max(0, 1 - (dup_count / len(current_data)))

    null_count = current_data.isnull().sum().sum()
    total_cells = current_data.size
    completeness = 1 - (null_count / total_cells)

    str_cols = current_data.select_dtypes(include=['object']).columns
    format_hits = 0
    total_samples = 0
    
    for col in str_cols:
        is_consistent = current_data[col].astype(str).str[0].str.isupper().all()
        format_hits += 1 if is_consistent else 0
        total_samples += 1
            
    consistency = (format_hits / total_samples) if total_samples > 0 else 1.0

    original_size = len(ground_truth)
    current_size = len(current_data)
    integrity = min(1.0, current_size / original_size)

    final_score = (
        (uniqueness * 0.20) + 
        (completeness * 0.40) + 
        (consistency * 0.30) + 
        (integrity * 0.10)
    )

    # FINAL FIXES (Robust Scoring for Validator)
    score = float(final_score)

    # Handle NaN
    if np.isnan(score):
        score = 0.5

    # STRICT clamp BEFORE rounding
    score = max(0.01, min(0.99, score))

    # Avoid exact boundary after rounding
    score = round(score, 4)

    if score <= 0.0:
        score = 0.01
    elif score >= 1.0:
        score = 0.99

    return score