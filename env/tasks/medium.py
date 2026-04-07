import pandas as pd
import numpy as np
import random
from faker import Faker

def get_task():
    """Medium Task: Duplicates and Case Inconsistencies."""
    fake = Faker()
    data = []
    for _ in range(20):
        name = fake.name()
        if random.random() < 0.2:
            name = name.lower()
        data.append({
            "order_id": fake.uuid4()[:8],
            "customer": name,
            "status": random.choice(["Delivered", "In-Transit", "Pending"])
        })
    
    df = pd.DataFrame(data)
    # Inject 2 duplicates
    dupes = df.sample(n=2).copy()
    return pd.concat([df, dupes], ignore_index=True)
