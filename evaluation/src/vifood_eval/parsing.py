from __future__ import annotations

import json
import re
from typing import Any

ANSWER_RE = re.compile(r"(?im)^\s*(?:final\s+)?answer\s*[:：]\s*(?:\*\*)?\s*([ABCD])\b")
LOCALIZED_ANSWER_RE = re.compile(
    r"(?im)^\s*(?:đáp\s*án|dap\s*an|chọn|chon)\s*[:：]?\s*(?:\*\*)?\s*([ABCD])\b"
)
FINAL_LETTER_LINE_RE = re.compile(r"(?i)^\s*(?:\*\*)?\s*([ABCD])\s*(?:\*\*)?\s*(?:[.)].*)?$")
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_answer_letter(text: str) -> tuple[str | None, str]:
    matches = ANSWER_RE.findall(text or "")
    if matches:
        return matches[-1].upper(), "ok"

    localized_matches = LOCALIZED_ANSWER_RE.findall(text or "")
    if localized_matches:
        return localized_matches[-1].upper(), "ok_localized"

    stripped = (text or "").strip().upper()
    if stripped in {"A", "B", "C", "D"}:
        return stripped, "ok_single_letter"

    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if lines:
        final_line_match = FINAL_LETTER_LINE_RE.match(lines[-1])
        if final_line_match:
            return final_line_match.group(1).upper(), "ok_final_line"

    fallback = re.findall(r"\b([ABCD])\b", stripped)
    if len(fallback) == 1:
        return fallback[0], "ok_single_letter_fallback"

    return None, "unparsed"


def parse_classifier_response(text: str, qtypes: set[str]) -> tuple[dict[str, Any], str]:
    payload = _extract_json(text)
    if not isinstance(payload, dict):
        return {"qtype": None, "food_items": []}, "unparsed"

    qtype = str(payload.get("qtype") or "").strip()
    if qtype not in qtypes:
        qtype = None

    food_items = payload.get("food_items") or []
    if not isinstance(food_items, list):
        food_items = []

    clean_items = [" ".join(str(item).split()) for item in food_items]
    clean_items = [item for item in clean_items if item]
    return {"qtype": qtype, "food_items": clean_items}, "ok" if qtype else "invalid_qtype"


def _extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = JSON_OBJECT_RE.search(text or "")
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
