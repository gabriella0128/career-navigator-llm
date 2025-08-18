import hashlib


def _norm_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def calc_qhash(question: str, evidence: dict) -> str:
    base = _norm_text(question) + "|" + _norm_text(str(evidence.get("from", "")))
    return hashlib.sha1(base.encode("utf-8")).hexdigest()
