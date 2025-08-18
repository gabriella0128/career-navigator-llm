# app/services/question_generator.py
from __future__ import annotations

import hashlib
import json, uuid, random
from collections.abc import Set, Collection
from datetime import datetime
from typing import Any, Dict, List
from zoneinfo import ZoneInfo
from openai import OpenAI
import numpy as np

from core.config import settings
from schemas.common import CommonResponse
from schemas.interview import (
    InterviewQuestionItemRes, InterviewQuestionFeignRes, InterviewQuestionFeignReq,
)
from services.seed import _generate_seed
from utils.jsonutils import normalize_json
from utils.templater import render_template

KST = ZoneInfo("Asia/Seoul")
PLAN = [("tech", 2), ("project", 2), ("behavior", 1)]
NEAR_DUP_THRESHOLD = 0.88
RECENT_FOR_PROMPT_MAX_CHAR = 50

client = OpenAI(api_key=settings.openai_api_key)


# =========================
# 해시/정규화/유사도
# =========================

def _normalize_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _qhash_text_only(question: str) -> str:
    base = _normalize_text(question)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _token_bigrams(s: str) -> set[str]:
    toks = _normalize_text(s).split()
    return {" ".join(toks[i:i + 2]) for i in range(max(0, len(toks) - 1))}


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _embed_batch(texts: list[str]) -> np.ndarray:
    if not texts: return np.zeros((0, 1536), dtype=np.float32)
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype=np.float32)


def _cosine_max(v: np.ndarray, M: np.ndarray) -> float:
    if M.size == 0: return 0.0
    v = v / (np.linalg.norm(v) + 1e-9)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    sims = M @ v
    return float(sims.max(initial=0.0))


def _near_dup_lexical(q: str, recent_texts: list[str]) -> bool:
    a3 = {_normalize_text(q)[i:i + 3] for i in range(max(0, len(_normalize_text(q)) - 2))}
    ab = _token_bigrams(q)
    for prev in recent_texts:
        b3 = {_normalize_text(prev)[i:i + 3] for i in range(max(0, len(_normalize_text(prev)) - 2))}
        bb = _token_bigrams(prev)
        if _jaccard(a3, b3) >= 0.80:  # ↓ 0.88 → 0.80로 완화
            return True
        if _jaccard(ab, bb) >= 0.60:
            return True
    return False


def _jaccard_3gram(a: str, b: str) -> float:
    def shingles(s: str, k=3) -> Set[str]:
        s = _normalize_text(s)
        return {s[i:i + k] for i in range(max(0, len(s) - k + 1))}

    _a, _b = shingles(a), shingles(b)
    if not _a or not _b:
        return 0.0
    return len(_a & _b) / len(_a | _b)


def _is_near_duplicate(q: str, recent_texts: list[str], recent_vecs: np.ndarray | None = None) -> bool:
    # 1) 어휘 기반
    if _near_dup_lexical(q, recent_texts):
        return True
    # 2) 의미 기반
    if recent_vecs is not None and recent_vecs.size:
        qv = _embed_batch([q])[0]
        if _cosine_max(qv, recent_vecs) >= 0.86:
            return True
    return False


def _mmr_select(items: list[dict], k: int, embed_cache: dict[str, np.ndarray] | None = None) -> list[dict]:
    texts = [(it.get("question") or "").strip() for it in items]
    vecs = _embed_batch(texts)  # 캐시 있으면 교체
    picked_idx: list[int] = []
    lambda_div = 0.5  # 0=순수 다양성, 1=순수 중심성

    # 중심성: 평균 벡터와의 유사도
    centroid = vecs.mean(axis=0, keepdims=True)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

    scores_center = (vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)) @ centroid.T  # (n,1)

    while len(picked_idx) < min(k, len(items)):
        best, best_score = -1, -1e9
        for i in range(len(items)):
            if i in picked_idx:
                continue
            # 다양성: 현재까지 선택된 것들과의 최대 유사도
            if picked_idx:
                sims = []
                vi = vecs[i] / (np.linalg.norm(vecs[i]) + 1e-9)
                for j in picked_idx:
                    vj = vecs[j] / (np.linalg.norm(vecs[j]) + 1e-9)
                    sims.append(float(vi @ vj))
                max_sim = max(sims)
            else:
                max_sim = 0.0
            score = lambda_div * float(scores_center[i]) - (1 - lambda_div) * max_sim
            if score > best_score:
                best, best_score = i, score
        picked_idx.append(best)
    return [items[i] for i in picked_idx]


# =========================
# seed/resume 키워드
# =========================

def _resume_hash(resume: Dict[str, Any]) -> str:
    return hashlib.sha256(normalize_json(resume).encode("utf-8")).hexdigest()


def _derive_seed(seed: str, tag: str) -> str:
    raw = (seed + tag).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


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


# =========================
# OpenAI 후보 생성
# =========================

def _call_openai_candidates(
        seed_hex: str,
        resume: Dict[str, Any],
        plan: List[Dict[str, Any]],
        hints: Dict[str, List[str]],
        recent_questions_for_prompt: List[str],
        n_overgenerate: int = 24
) -> List[Dict[str, Any]]:
    client = OpenAI(api_key=settings.openai_api_key)

    user_msg = render_template(
        "daily_questions_prompt.j2",
        seed=seed_hex,
        plan=plan,
        hints=hints,
        resume=resume,
        category="tech",
        uuid8=uuid.uuid4().hex[:8],
    )

    system_msg = (
        "당신은 시니어 개발 면접관입니다. 한국어로만 답하고, 오직 JSON만 출력하세요. "
        "사용자의 이력서/힌트를 근거로 실전형 질문 다수를 생성하되, 제공된 최근 질문들과 동일/유사 의미를 피하세요."
    )

    resp = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    )

    text = resp.choices[0].message.content or "{}"
    obj = json.loads(text)
    questions = obj.get("questions", [])
    if not isinstance(questions, list):
        questions = []
    return questions


FALLBACK_TEMPLATES = {
    "tech": [
        "최근 {skill} 사용 시 가장 어려웠던 이슈와 해결 전략은?",
        "{skill} 기반 설계에서 성능과 안정성 트레이드오프를 어떻게 결정했나요?"
    ],
    "project": [
        "{company}에서 수행한 프로젝트의 병목과 개선 전/후 수치를 설명해 주세요.",
        "{company} 프로젝트에서 요구 변경/리스크를 어떻게 관리했나요?"
    ],
    "behavior": [
        "팀 내 갈등을 해결했던 사례를 STAR 구조로 설명해 주세요.",
        "실패 경험과 학습/재발 방지 대책을 말씀해 주세요."
    ],
    "system": [
        "대규모 트래픽을 처리하는 캐시/큐/샤딩 설계에서 병목과 완화책을 설명해 주세요.",
        "가용성 목표(예: 99.9%) 달성을 위한 장애 대응/복구 전략은?"
    ],
}


def _safe_load_fallback_bank(hints: Dict[str, List[str]]) -> Dict[str, List[str]]:
    try:
        txt = render_template("daily_questions_fallback.j2", hints=hints)
        bank = json.loads(txt)
        for k in ("tech", "project", "behavior", "system"):
            if k not in bank or not isinstance(bank[k], list) or not bank[k]:
                bank[k] = FALLBACK_TEMPLATES.get(k, [])
        return bank
    except Exception:
        return FALLBACK_TEMPLATES


def _format_question_from_template(tpl: str, rnd: random.Random, hints: Dict[str, List[str]]) -> tuple[
    str, Dict[str, Any]]:
    text = tpl
    evidence = {"from": "behavior"}  # 기본값

    if "{skill}" in text:
        skill = rnd.choice(hints.get("skills") or ["핵심 기술"])
        text = text.replace("{skill}", str(skill))
        evidence = {"from": "skills", "skills": [skill]}

    if "{company}" in text:
        company = rnd.choice(hints.get("companies") or ["최근 프로젝트"])
        text = text.replace("{company}", str(company))
        # skill evidence가 이미 있다면 company만 추가, 없으면 experience로 대체
        if evidence.get("from") == "skills":
            evidence["company"] = company
        else:
            evidence = {"from": "experience", "company": company}

    return text, evidence


def _fallback_candidates(seed_hex: str, hints: Dict[str, List[str]], need: int) -> List[Dict[str, Any]]:
    rnd = random.Random(int(seed_hex, 16))
    bank = _safe_load_fallback_bank(hints)
    out: List[Dict[str, Any]] = []
    order: List[str] = []
    for category, cnt in PLAN:
        order.extend([category] * cnt)
    rnd.shuffle(order)

    while len(out) < need:
        if not order:
            order = [c for c, _ in PLAN]
            rnd.shuffle(order)
        cat = order.pop(0)
        tmpl = bank.get(cat) or FALLBACK_TEMPLATES.get(cat, [])
        if not tmpl:
            cat = "tech"
            templates = bank.get("tech") or FALLBACK_TEMPLATES["tech"]

        tpl = rnd.choice(tmpl)  # ← 여기서 문자열 리스트에서 고름
        q_text, evidence = _format_question_from_template(tpl, rnd, hints)

        item = {
            "id": f"q_{cat}_{uuid.uuid4().hex[:8]}",
            "category": cat,
            "difficulty": rnd.choice([1, 2, 3]) if cat != "behavior" else rnd.choice([1, 2]),
            "question": q_text,
            "expected_points": ["정량 지표", "대안 비교", "의사결정 근거"],
            "evidence": evidence,
        }
        out.append(item)

    return out


# =========================
# 필터 & 선별
# =========================

def _filter_and_pick(
        seed_hex: str,
        candidates: list[dict],
        recent_qhashes: Collection[str],
        recent_texts: list[str],
        k: int = 5,
        recent_vecs: np.ndarray | None = None,
) -> list[dict]:
    rnd = random.Random(int(seed_hex, 16))
    rnd.shuffle(candidates)

    picked: list[dict] = []
    seen_hashes: set[str] = set()

    for item in candidates:
        q = (item.get("question") or "").strip()
        if not q:
            continue
        h = _qhash_text_only(q)

        if h in recent_qhashes:
            continue
        if h in seen_hashes:
            continue
        if _is_near_duplicate(q, recent_texts, recent_vecs):
            continue

        item["qHash"] = h
        picked.append(item)
        seen_hashes.add(h)
        if len(picked) == k:
            break
    return picked


_ALLOWED_CATEGORIES = {"tech", "project", "behavior", "system"}


def _to_item_res(raw: Dict[str, Any], pos: int) -> InterviewQuestionItemRes:
    cat = str(raw.get("category", "tech")).lower()
    if cat not in _ALLOWED_CATEGORIES:
        cat = "tech"

    q_text = (raw.get("question") or "").strip() or "최근 수행한 업무에서 가장 큰 기술적 난관과 해결 과정을 설명해 주세요."

    try:
        diff = max(1, min(3, int(raw.get("difficulty", 2))))
    except Exception:
        diff = 2

    expected = raw.get("expected_points") or raw.get("expectedPoints") or []
    if not isinstance(expected, list):
        expected = [str(expected)]

    ev = raw.get("evidence") or {}
    if not isinstance(ev, dict):
        ev = {}

    qid = raw.get("id") or f"q_{cat}_{uuid.uuid4().hex[:8]}"
    qhash = raw.get("qHash") or _qhash_text_only(q_text)

    return InterviewQuestionItemRes(
        id=qid,
        category=cat,
        difficulty=diff,
        question=q_text,
        expectedPoints=[str(x) for x in expected][:6],
        evidence=ev,
        position=pos,
        qHash=qhash,
    )


def generate_daily_questions(
        req: InterviewQuestionFeignReq,
        overgenerate: int = 24,
) -> CommonResponse[InterviewQuestionFeignRes]:
    if not isinstance(req.resume, dict):
        raise ValueError("Resume must be a dict")

    recent_texts: List[str] = req.recent_questions or []
    recent_qhashes: Set[str] = {_qhash_text_only(t) for t in recent_texts}

    recent_vecs = _embed_batch(recent_texts) if recent_texts else None

    practice_date = datetime.now(KST).date()
    seed = _generate_seed(req.sessionUid, req.resume)
    hints = _pick_keywords(req.resume)
    plan = [{"category": c, "count": n} for c, n in PLAN]

    raw_candidates = _call_openai_candidates(
        seed_hex=seed,
        resume=req.resume,
        plan=plan,
        hints=hints,
        recent_questions_for_prompt=recent_texts,
        n_overgenerate=overgenerate,
    )

    selected = _filter_and_pick(
        seed, raw_candidates, recent_qhashes, recent_texts, k=20,
        recent_vecs=recent_vecs
    )

    if len(selected) < 5:
        need = 5 - len(selected)
        seed2 = _derive_seed(seed, ":alt")
        more = _fallback_candidates(seed2, hints, need * 3)
        more_sel = _filter_and_pick(seed2, more, recent_qhashes, recent_texts, k=need)
        selected.extend(more_sel)

    final5 = _mmr_select(selected, k=5)

    items: List[InterviewQuestionItemRes] = []
    for pos, r in enumerate(final5, start=1):
        items.append(_to_item_res(r, pos))

    return CommonResponse[InterviewQuestionFeignRes](
        success=True,
        code="Success",
        message="질문 생성 성공",
        data=InterviewQuestionFeignRes(
            sessionUid=req.sessionUid,
            seed=seed,
            modelName=settings.llm_model,
            promptVersion=settings.prompt_version,
            practiceDate=practice_date,
            questions=items
        )
    )
