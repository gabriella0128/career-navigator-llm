import hashlib
import json
from datetime import datetime
from typing import Dict, Any, List
from zoneinfo import ZoneInfo

from openai import OpenAI

from schemas.common import CommonResponse
from schemas.learning import LearningFeignReq, LearningFeignRes, LearningGoal, LearningTask
from core.config import settings

from services.seed import _resume_hash, _generate_seed
from utils.templater import render_template

KST = ZoneInfo("Asia/Seoul")
client = OpenAI(api_key=settings.openai_api_key)


def _pick_keywords(resume: Dict[str, Any]) -> Dict[str, List[str]]:
    skills, companies = [], []
    for s in (resume.get("skills") or []):
        if isinstance(s, dict) and s.get("skillName"):
            skills.append(s["skillName"])
        elif isinstance(s, str):
            skills.append(s)
    for e in (resume.get("experiences") or []):
        if isinstance(e, dict) and e.get("companyName"):
            companies.append(e["companyName"])
    return {"skills": skills[:8], "companies": companies[:8]}


def build_learning_plan(
        req: LearningFeignReq
) -> CommonResponse[LearningFeignRes]:
    if not isinstance(req.resume, dict):
        raise ValueError("Resume must be a dict")

    seed = _generate_seed(req.sessionUid, req.resume)
    today = datetime.now(KST).date()
    hints = _pick_keywords(req.resume)

    user_msg = render_template(
        "learning_plan_prompt.j2",
        seed=seed,
        improvements=req.improvements,
        resume=req.resume
    )

    system = "당신은 시니어 코칭 어시스턴트입니다. 한국어로만 답하고, 오직 JSON만 출력하세요."
    resp = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
    )
    data = json.loads(resp.choices[0].message.content or "{}")

    goals: List[LearningGoal] = []
    for g in data.get("goals", []):
        tasks = [LearningTask(**t) for t in g.get("tasks", [])]
        # weekNo 누락시 자동 배치
        if tasks:
            w = 1
            for t in tasks:
                if t.weekNo is None:
                    t.weekNo = w
                    w = 1 if w >= 4 else w + 1

        goals.append(LearningGoal(
            title=g.get("title", ""),
            metric=g.get("metric", "정량 지표를 명시"),
            targetValue=g.get("targetValue"),
            dueDate=None,
            priority=int(g.get("priority", 3)),
            category=g.get("category", "process"),
            tasks=tasks
        ))

    return CommonResponse[LearningFeignRes](
        success=True,
        code="Success",
        message="질문 생성 성공",
        data=LearningFeignRes(
            sessionUid=req.sessionUid,
            planDate=today,
            seed=seed,
            modelName=settings.llm_model,
            promptVersion=settings.prompt_version,
            goals=goals
        )
    )
