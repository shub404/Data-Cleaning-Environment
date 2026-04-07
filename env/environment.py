from env.models import Observation, Action
from env.state import load_sample_data
from env.reward import calculate_reward
import numpy as np


class DataCleaningEnv:

    def __init__(self):
        self.data = None
        self.done = False

    def reset(self):
        self.data = load_sample_data()
        self.done = False
        return self._get_observation()

    def step(self, action: Action):
        reward = 0.0

        if action.action_type == "remove_duplicates":
            before = len(self.data)
            self.data = self.data.drop_duplicates()
            after = len(self.data)
            reward = 0.2 if after < before else -0.05

        elif action.action_type == "fill_missing":
            self.data = self.data.fillna({
                "name": "Unknown",
                "age": "0",
                "salary": 0
            })
            reward = 0.2

        elif action.action_type == "normalize_text":
            self.data["name"] = self.data["name"].astype(str).str.capitalize()
            reward = 0.1

        elif action.action_type == "done":
            self.done = True
            reward = calculate_reward(self.data)

        else:
            reward = -0.1

        return {
            "observation": self._get_observation(),
            "reward": reward,
            "done": self.done,
            "info": {}
        }

    def _get_observation(self):
        preview_df = self.data.head(5).copy()

        # Fix NaN for JSON
        preview_df = preview_df.replace({np.nan: ""})

        preview = preview_df.to_dict(orient="records")

        return Observation(
            data_preview=preview,
            issues=self._detect_issues()
        )

    def _detect_issues(self):
        issues = []

        if self.data.duplicated().sum() > 0:
            issues.append("duplicates")

        if self.data.isnull().sum().sum() > 0:
            issues.append("missing_values")

        if not self.data["name"].astype(str).str[0].str.isupper().all():
            issues.append("formatting")

        return issues