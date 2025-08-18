from __future__ import annotations
from zoneinfo import ZoneInfo
import hmac, hashlib
from datetime import datetime
from typing import Any, Dict, List

from core.config import settings
from utils.jsonutils import normalize_json

KST = ZoneInfo("Asia/Seoul")


def _resume_hash(resume: Dict[str, Any]) -> str:
    return hashlib.sha256(normalize_json(resume).encode("utf-8")).hexdigest()


def _generate_seed(session_uid: str, resume: Dict[str, Any]) -> str:
    d = datetime.now(KST).date().isoformat()
    source = f"{session_uid}:{d}:{_resume_hash(resume)}:{settings.prompt_version}:{settings.llm_model}:0"
    sig = hmac.new(settings.app_secret.encode(), source.encode(), hashlib.sha256).hexdigest()
    return sig[:32]  #
