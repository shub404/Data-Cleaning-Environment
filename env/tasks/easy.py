import pandas as pd
import numpy as np
from faker import Faker

def get_task():
    """Easy Task: Only missing values in a simple user dataset."""
    fake = Faker()
    data = []
    for i in range(15):
        data.append({
            "user_id": i + 100,
            "name": fake.name(),
            "age": float(fake.random_int(18, 65)) if i % 4 != 0 else np.nan
        })
    return pd.DataFrame(data)
