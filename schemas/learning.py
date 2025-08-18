# app/schemas/learning.py
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import date


class LearningTask(BaseModel):
    taskTitle: str
    resourceUrl: Optional[str] = None
    weekNo: Optional[int] = None


class LearningGoal(BaseModel):
    title: str
    metric: str
    targetValue: Optional[str] = None
    priority: int
    category: str
    tasks: List[LearningTask] = []


class LearningFeignReq(BaseModel):
    sessionUid: str
    resume: Dict[str, Any]
    improvements: List[str] = []


class LearningFeignRes(BaseModel):
    sessionUid: str
    planDate: date
    seed: str
    modelName: str
    promptVersion: str
    goals: List[LearningGoal]
