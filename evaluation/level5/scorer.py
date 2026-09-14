"""Level 5 answer scoring: rule-based first, LLM judge fallback for tricky equivalence."""
from __future__ import annotations

from typing import Optional

import re

from evaluation.common.judge_cache import JudgeCache, make_key
from evaluation.common.judge_client import JudgeClient
from evaluation.common.math_scoring import is_equal as _rule_is_equal
from evaluation.level5.answer_extractor import normalize_answer
from evaluation.level5.choice import (
    option_answer_candidates,
    parse_choice_label,
    parse_choice_options,
)
from evaluation.level5.llm_extractor import llm_extract

_INVALID_PRED_RE = re.compile(r"^</?(?:image\d*|think|aux\d*|img\d*)>$", re.IGNORECASE)


_LLM_SYS = (
    "You are a strict mathematical equivalence judge. "
    "Determine whether the model prediction and the ground-truth answer are "
    "mathematically equivalent. Symbolic expressions are equivalent if they "
    "denote the same value/function for all admissible variable assignments. "
    "Units (degrees, °, etc.) should be ignored when both sides are angles. "
    "Numerical answers are equivalent if they match to ~0.5% tolerance."
)

_LLM_USER = (
    "Ground-truth: {gt}\n"
    "Prediction:   {pred}\n\n"
    "Respond ONLY with a JSON object: "
    "{{\"equivalent\": true/false, \"reason\": \"<short>\"}}"
)


def judge_equivalent(
    *,
    pred: str,
    gt: str,
    text_judge: JudgeClient,
    cache: JudgeCache,
    question: str = "",
    choice_label: str | None = None,
    choice_text: str = "",
) -> dict:
    """LLM equivalence call with cache.

    Single-item failures (network blip, malformed JSON, empty content) MUST
    NOT take down the whole evaluation — they're reported as `equivalent=False`
    with the error in the reason. Successful results ARE cached; failures are
    NOT cached so the next run can retry.
    """
    if choice_label:
        user = (
            f"Question and options:\n{question}\n\n"
            f"Ground-truth option label: {choice_label}\n"
            f"Ground-truth option text: {choice_text}\n"
            f"Prediction: {pred}\n\n"
            "Determine whether the prediction selects, states, or is mathematically "
            "equivalent to the ground-truth option. Respond ONLY with a JSON object: "
            '{"equivalent": true/false, "reason": "<short>"}'
        )
    else:
        user = _LLM_USER.format(pred=pred, gt=gt)
    key = make_key(model=text_judge.cfg.model, system=_LLM_SYS, user=user)
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        result = text_judge.chat_json(_LLM_SYS, user, max_tokens=1024)
    except Exception as e:
        return {
            "equivalent": False,
            "reason": f"judge call failed: {str(e)[:300]}",
        }
    # Normalize.
    out = {
        "equivalent": bool(result.get("equivalent", False)),
        "reason": str(result.get("reason", ""))[:500],
    }
    cache.put(key, out)
    return out


def score(
    *,
    pred_raw: str,
    gt: str,
    text_judge: Optional[JudgeClient] = None,
    cache: Optional[JudgeCache] = None,
    question: str = "",
    response_text: str = "",
) -> dict:
    """Return {pred_raw, pred_norm, gt_norm, correct, method, reason}.

    Extraction escalation:
      - start with the rule-extracted pred_raw
      - if it is empty, ask text_judge to extract a short answer from
        response_text (cached) and use that instead
      - preserve non-empty textual/proof/multi-part answers for equivalence
        judging instead of reducing them to one scalar or LaTeX span
    """
    pred_norm = normalize_answer(pred_raw)
    gt_norm = normalize_answer(gt)
    extraction_method = "rule"
    parsed_gt_label = parse_choice_label(gt)
    choice_options = parse_choice_options(question) if parsed_gt_label else {}
    # A bare A-D answer is treated as a choice label only when the question
    # actually contains a complete option table.  This avoids misclassifying a
    # legitimate free-response answer such as the variable/name ``A``.
    choice_gt_label = parsed_gt_label if len(choice_options) == 4 else None
    choice_gt_text = choice_options.get(choice_gt_label, "") if choice_gt_label else ""

    # 0. Treat bare markup tags as invalid (prevents LLM judge hallucination).
    if _INVALID_PRED_RE.match(pred_norm):
        pred_norm = ""

    # 0b. LLM extractor fallback when rule extraction looks bad.
    # Cache is optional — only the judge client is required.
    if text_judge is not None and response_text:
        if not pred_norm:
            llm_pred = llm_extract(
                question=question,
                response=response_text,
                text_judge=text_judge,
                cache=cache,
            )
            if llm_pred and not _INVALID_PRED_RE.match(normalize_answer(llm_pred)):
                pred_raw = llm_pred
                pred_norm = normalize_answer(llm_pred)
                extraction_method = "llm_extract"

    # 1. Trivially empty prediction.
    if not pred_norm:
        return {
            "pred_raw": pred_raw,
            "pred_norm": pred_norm,
            "gt_norm": gt_norm,
            "correct": False,
            "method": "empty",
            "extraction_method": extraction_method,
            "reason": "empty prediction",
        }

    # 2. Multiple-choice scoring.  The benchmark may store only a label while
    # the model emits the corresponding option value/expression.
    if choice_gt_label:
        predicted_label = parse_choice_label(pred_raw, prediction=True)
        if predicted_label is not None:
            return {
                "pred_raw": pred_raw,
                "pred_norm": pred_norm,
                "gt_norm": gt_norm,
                "correct": predicted_label == choice_gt_label,
                "method": "choice_label",
                "extraction_method": extraction_method,
                "reason": (
                    "choice label matched" if predicted_label == choice_gt_label
                    else f"predicted option {predicted_label}, expected {choice_gt_label}"
                ),
                "choice_gt_label": choice_gt_label,
                "choice_gt_text": choice_gt_text,
                "choice_pred_label": predicted_label,
            }

        matched_options: list[str] = []
        for label, option_text in choice_options.items():
            if any(
                _rule_is_equal(pred_norm, normalize_answer(candidate))
                for candidate in option_answer_candidates(option_text)
            ):
                matched_options.append(label)
        if len(matched_options) == 1:
            matched_label = matched_options[0]
            return {
                "pred_raw": pred_raw,
                "pred_norm": pred_norm,
                "gt_norm": gt_norm,
                "correct": matched_label == choice_gt_label,
                "method": "choice_option_rule",
                "extraction_method": extraction_method,
                "reason": (
                    f"prediction matches option {matched_label} content"
                    if matched_label == choice_gt_label
                    else f"prediction matches option {matched_label}, expected {choice_gt_label}"
                ),
                "choice_gt_label": choice_gt_label,
                "choice_gt_text": choice_gt_text,
                "choice_pred_label": matched_label,
            }

    # 3. Rule-based equality (string + latex2sympy numeric).
    if _rule_is_equal(pred_norm, gt_norm):
        return {
            "pred_raw": pred_raw,
            "pred_norm": pred_norm,
            "gt_norm": gt_norm,
            "correct": True,
            "method": "rule",
            "extraction_method": extraction_method,
            "reason": "rule matched",
        }

    # 4. LLM equivalence fallback (cached).
    if text_judge is not None and cache is not None:
        verdict = judge_equivalent(
            pred=pred_norm,
            gt=gt_norm,
            text_judge=text_judge,
            cache=cache,
            question=question,
            choice_label=choice_gt_label,
            choice_text=choice_gt_text,
        )
        return {
            "pred_raw": pred_raw,
            "pred_norm": pred_norm,
            "gt_norm": gt_norm,
            "correct": bool(verdict["equivalent"]),
            "method": "llm",
            "extraction_method": extraction_method,
            "reason": verdict["reason"],
        }

    return {
        "pred_raw": pred_raw,
        "pred_norm": pred_norm,
        "gt_norm": gt_norm,
        "correct": False,
        "method": "rule",
        "extraction_method": extraction_method,
        "reason": "rule mismatch, llm fallback disabled",
    }
