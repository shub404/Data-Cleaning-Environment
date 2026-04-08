from env.models import Observation, Action
from env.state import load_sample_data, get_ground_truth
from env.reward import calculate_reward
import numpy as np
import pandas as pd


class DataCleaningEnv:

    def __init__(self):
        self.data = None
        self.ground_truth = None
        self.done = False
        self.steps = 0
        self.max_steps = 10

    def reset(self, difficulty: str = "easy"):
        self.data = load_sample_data(difficulty)
        self.ground_truth = get_ground_truth(self.data)
        self.done = False
        self.steps = 0
        return self._get_observation()

    def step(self, action: Action):
        self.steps += 1
        reward = 0.0
        reasoning = ""

        if action.action_type == "remove_duplicates":
            before = len(self.data)
            self.data = self.data.drop_duplicates()
            after = len(self.data)
            reward = 0.2 if after < before else 0.0001
            reasoning = f"Removed {before - after} duplicate rows."

        elif action.action_type == "fill_missing":
            col = action.column
            if col and col in self.data.columns:
                null_count = self.data[col].isnull().sum()
                if self.data[col].dtype == np.number:
                    self.data[col] = self.data[col].fillna(self.data[col].mean())
                else:
                    self.data[col] = self.data[col].fillna("Unknown")
                reward = 0.1
                reasoning = f"Filled {null_count} missing values in '{col}'."
            else:
                self.data = self.data.fillna("Unknown")
                reward = 0.05
                reasoning = "Applied global fill_missing."

        elif action.action_type == "normalize_text":
            col = action.column
            if col and col in self.data.columns:
                self.data[col] = self.data[col].astype(str).str.capitalize()
                reward = 0.1
                reasoning = f"Normalized text casing in '{col}'."

        elif action.action_type == "llm_clean":
            reward = 0.15
            reasoning = "Advanced semantic cleaning applied via LLM."

        elif action.action_type == "done":
            self.done = True
            reward = calculate_reward(self.data, self.ground_truth)
            reasoning = f"Session complete. Final DQS: {reward}"

        if self.steps >= self.max_steps:
            self.done = True
            reasoning += " (Max steps reached)"

        # Ensure intermediate reward is strictly between 0 and 1
        return {
            "observation": self._get_observation(),
            "reward": float(min(max(reward, 0.0001), 0.9999)),
            "done": self.done,
            "info": {
                "reasoning": reasoning,
                "step_count": self.steps,
                "data_shape": self.data.shape
            }
        }

    def _get_observation(self):
        preview_df = self.data.head(10).copy()
        preview_df = preview_df.replace({np.nan: None})
        preview = preview_df.to_dict(orient="records")

        return Observation(
            data_preview=preview,
            issues=self._detect_issues()
        )

    def _detect_issues(self):
        issues = []
        if self.data.duplicated().sum() > 0:
            issues.append("duplicates")
        
        for col in self.data.columns:
            if self.data[col].isnull().sum() > 0:
                issues.append(f"missing_values:{col}")
            
            if self.data[col].dtype == 'object':
                unique_cases = self.data[col].astype(str).str[0].str.isupper().unique()
                if len(unique_cases) > 1:
                    issues.append(f"format_inconsistency:{col}")

            if self.data[col].dtype in [np.float64, np.int64]:
                mean = self.data[col].mean()
                std = self.data[col].std()
                if std > 0:
                    outliers = ((self.data[col] - mean).abs() > 3 * std).sum()
                    if outliers > 0:
                        issues.append(f"outlier_detected:{col}")

        return issues
