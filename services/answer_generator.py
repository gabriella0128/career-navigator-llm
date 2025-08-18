import hashlib
import hmac
import json
from datetime import datetime
from typing import Tuple, Dict, List, Any
from zoneinfo import ZoneInfo

from openai import OpenAI

from core.config import settings
from schemas.common import CommonResponse
from schemas.interview import InterviewAnswerFeignReq, InterviewAnswerFeignRes
from utils.templater import render_template

KST = ZoneInfo("Asia/Seoul")
client = OpenAI(api_key=settings.openai_api_key)


def _fallback_structure(q: str, a: str) -> Tuple[int, Dict[str, int], str, List[str], List[str]]:
    scores = {"structure": 3, "specificity": 3, "impact": 3, "clarity": 3}
    strengths = ["질문의 의도를 파악하고 답변을 시도했습니다."]
    improvements = [
        "구체적 수치/근거를 제시하세요.",
        "의사결정 기준과 대안 비교를 명확히 하세요.",
        "성과/결과를 지표로 요약하세요."
    ]
    feedback = (
        "답변의 방향은 적절하나, 사례와 정량 지표가 부족합니다. "
        "핵심 선택의 근거와 대안 비교를 포함해 구조화(상황-문제-행동-결과)해 주세요."
    )
    overall = round(sum(scores.values()) / len(scores))
    return overall, scores, feedback, strengths, improvements


def _hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def _seed(req: InterviewAnswerFeignReq) -> str:
    base = f"{req.sessionUid or 'no-session'}:{_hash(req.question)}:{_hash(req.answer)}:{settings.prompt_version}:{settings.llm_model}"
    return hmac.new(settings.app_secret.encode("utf-8"), base.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def _coerce_int(v: Any, default: int = 3) -> int:
    try:
        i = int(v)
        return max(1, min(5, i))
    except Exception:
        return default


def generate_daily_question_answer(
        req: InterviewAnswerFeignReq,
        overgenerate: int = 24,
) -> CommonResponse[InterviewAnswerFeignRes]:
    seed = _seed(req)
    now = datetime.now(KST)

    user_payload = render_template(
        "daily_question_answer_prompt.j2",
        seed=seed,
        question=req.question,
        answer=req.answer,
        prompt_version=settings.prompt_version,
        model_name=settings.llm_model
    )
    system = (
        "너는 시니어 인터뷰 코치다. 한국어로만 답하고, 오직 JSON만 출력하라. "
        "형식을 지키지 못할 경우 평가가 무효 처리된다."
    )

    # 2) OpenAI 호출 (JSON 강제)
    try:
        resp = client.chat.completions.create(
            model=settings.llm_model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_payload},
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
    except Exception:
        data = None

    if not isinstance(data, dict):
        # LLM 실패 → 기본값
        overall, scores_map, feedback, strengths, improvements = _fallback_structure()
    else:
        scores_map = data.get("scores") or {}
        if not isinstance(scores_map, dict):
            scores_map = {}
        for k in ("structure", "specificity", "impact", "clarity"):
            scores_map[k] = _coerce_int(scores_map.get(k, 3), default=3)

        overall = data.get("score_overall")
        if overall is None:
            # 없으면 서브스코어 평균
            overall = round(sum(scores_map.values()) / max(1, len(scores_map)))
        overall = _coerce_int(overall, default=3)

        feedback = str(data.get("feedback") or "")

        strengths = data.get("strength") or data.get("strengths") or []
        if isinstance(strengths, str):
            strengths = [s.strip() for s in strengths.split("\n") if s.strip()]

        improvements = data.get("improvements") or []
        if isinstance(improvements, str):
            improvements = [s.strip() for s in improvements.split("\n") if s.strip()]

        # 3) DTO 구성 (JSON 필드는 문자열로 직렬화)
    res = InterviewAnswerFeignRes(
        sessionUid=req.sessionUid or "",
        seed=seed,
        modelName=settings.llm_model,
        promptVersion=settings.prompt_version,
        evaluatedAt=now,
        scoreOverall=int(overall),
        scores=json.dumps(scores_map, ensure_ascii=False),
        feedback=feedback,
        strength=json.dumps(strengths, ensure_ascii=False),
        improvements=json.dumps(improvements, ensure_ascii=False),
    )
    return CommonResponse[InterviewAnswerFeignRes](
        success=True,
        code="Success",
        message="답변 생성 성공",
        data=res
    )
