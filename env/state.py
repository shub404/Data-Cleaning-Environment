import pandas as pd


def load_sample_data():
    data = [
        {"name": "john", "age": "25", "salary": None},
        {"name": "John", "age": "25", "salary": 50000},
        {"name": "mary", "age": "", "salary": 60000},
    ]
    return pd.DataFrame(data)