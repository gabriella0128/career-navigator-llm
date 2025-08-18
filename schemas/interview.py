from pydantic import BaseModel, Field, field_serializer
from typing import Any, Dict, List
from datetime import date, datetime


def to_camel(s: str) -> str:
    parts = s.split('_')
    return parts[0] + ''.join(p.capitalize() for p in parts[1:])

class InterviewQuestionItemRes(BaseModel):
    id: str
    category: str
    difficulty: int
    question: str
    expectedPoints: List[str] = Field(default_factory=list)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    position: int
    qHash: str


class InterviewQuestionFeignRes(BaseModel):
    sessionUid: str
    seed: str
    modelName: str
    promptVersion: str
    practiceDate: date
    questions: List[InterviewQuestionItemRes]


class InterviewQuestionFeignReq(BaseModel):
    sessionUid: str
    resume: Dict[str, Any]
    recent_questions: List[str] = Field(default_factory=list, alias="recentQuestions")

    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }


class InterviewAnswerFeignReq(BaseModel):
    sessionUid: str
    question: str
    answer: str


class InterviewAnswerFeignRes(BaseModel):
    sessionUid: str
    seed: str
    modelName: str
    promptVersion: str
    evaluatedAt: datetime
    @field_serializer('evaluatedAt')
    def ser_dt(self, v: datetime, _info):
        # tz 정보 제거 + 마이크로초까지
        return v.replace(tzinfo=None).isoformat(timespec='microseconds')
    scoreOverall: int
    scores: str
    feedback: str
    strength: str
    improvements: str
