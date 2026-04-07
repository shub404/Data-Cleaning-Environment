def calculate_reward(data):
    score = 0.0

    # No duplicates
    if data.duplicated().sum() == 0:
        score += 0.3

    # No missing values
    if data.isnull().sum().sum() == 0:
        score += 0.3

    # Proper formatting (names capitalized)
    if data["name"].str[0].str.isupper().all():
        score += 0.2

    return score