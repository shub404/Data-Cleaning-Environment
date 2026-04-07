from pydantic import BaseModel
from typing import List, Optional


class Observation(BaseModel):
    data_preview: List[dict]
    issues: List[str]


class Action(BaseModel):
    action_type: str
    column: Optional[str] = None


class Reward(BaseModel):
    value: float