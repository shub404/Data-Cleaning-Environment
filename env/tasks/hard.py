import pandas as pd
import numpy as np
import random
from faker import Faker

def get_task():
    """Hard Task: Mixed formats, Outliers, and Missing values across multiple columns."""
    fake = Faker()
    data = []
    for _ in range(25):
        data.append({
            "transaction_id": fake.uuid4()[:8],
            "date": fake.date_this_year().strftime("%Y-%m-%d"),
            "amount": round(random.uniform(10.0, 1000.0), 2),
            "category": random.choice(["Food", "Rent", "Salary", "Entertainment"]),
            "notes": fake.sentence(nb_words=3)
        })
    
    df = pd.DataFrame(data)
    
    # 1. Nulls in amount
    df.loc[df.sample(frac=0.1).index, "amount"] = np.nan
    
    # 2. Case inconsistencies
    df.loc[df.sample(frac=0.2).index, "category"] = df["category"].str.lower()
    
    # 3. Outlier amount (very high)
    df.at[0, "amount"] = 99999.99
    
    # 4. Mixed date formats (bonus hard!)
    for i in range(3):
        idx = random.randint(0, len(df)-1)
        df.at[idx, "date"] = fake.date_this_year().strftime("%d/%m/%y")
        
    return df
