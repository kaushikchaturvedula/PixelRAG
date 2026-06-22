#!/usr/bin/env python3
"""QA grader for the PixelRAG reader — scores a predicted answer against gold. No manual labeling.

Gold answers already exist (gold.jsonl has `answer` + `reference_list` for all 25 iNat queries), so
grading is automatic. Two methods, both reusing the proven Phase-1 eval logic (replicated, not
imported, to avoid pulling in the heavy `eval` package):

- **judge** (default): LLM-as-judge, byte-faithful to the paper grader — `judge_worldvqa_prompt.txt`
  + `parse_label` (Correct/Incorrect/Unattempted), ground truth = `"Any of: " + reference_list`
  (encyclopedic_vqa convention), judge model default **gpt-4.1-2025-04-14** (matches
  eval/lib/grader.py so QA accuracy stays comparable to the paper's other benchmarks — this is
  independent of the reader model), temp 0, seed 42. The judge call is injectable (`judge_fn`) so
  tests grade offline with a mock judge — zero API spend.
- **exact**: free, no API — `is_exact_match` normalization (verbatim from eval/lib/grader.py).

`grade(...)` returns one verdict dict; `aggregate(...)` rolls verdicts into accuracy = #correct / N.
"""

from __future__ import annotations

import os
import re
import string
from pathlib import Path

_ASSET = (
    Path(__file__).resolve().parent.parent
    / "eval"
    / "repro_assets"
    / "judge_worldvqa_prompt.txt"
)
JUDGE_PROMPT = _ASSET.read_text()

DEFAULT_JUDGE_MODEL = "gpt-4.1-2025-04-14"  # matches eval/lib/grader.py (the paper grader)
JUDGE_SYSTEM_MESSAGE = "You are a helpful assistant."  # matches eval/lib/grader.py sampler
JUDGE_MAX_TOKENS = 1000
JUDGE_SEED = 42

VERDICTS = ("correct", "incorrect", "unattempted")


# --- reused verbatim from eval/lib/grader.py -------------------------------------------------
def strip_think(text: str) -> str:
    if text is None:
        return ""
    if "<think>" in text and "</think>" in text:
        return text.split("</think>")[-1].strip()
    elif "think>" in text:
        return text.split("think>")[-1].strip()
    return text


def parse_label(judge_text: str) -> str:
    m = re.search(r"Label:\s*(Correct|Incorrect|Unattempted)", judge_text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    tl = (judge_text or "").lower()
    if "incorrect" in tl:
        return "incorrect"
    if "unattempted" in tl:
        return "unattempted"
    if "correct" in tl:
        return "correct"
    return "incorrect"


def _normalize_text(s: str) -> str:
    s = re.sub(
        r"\b(a|an|the)\b",
        " ",
        s.lower().translate(str.maketrans("", "", string.punctuation)),
    )
    return " ".join(s.split())


def is_exact_match(prediction: str, golds) -> bool:
    prediction = (prediction or "").replace("Exact Answer: ", "").strip()
    pred_norm = _normalize_text(prediction)
    return any(_normalize_text(str(g)) == pred_norm for g in golds)


# --- ground truth + judge --------------------------------------------------------------------
def build_ground_truth(reference_list, answer: str | None = None) -> str:
    """encyclopedic_vqa convention: ANY reference matching == correct."""
    refs = [r for r in (reference_list or []) if r]
    if refs:
        return "Any of: " + " | ".join(refs)
    return answer or ""


def judge_openai(question: str, prediction: str, ground_truth: str, model: str) -> tuple[str, dict]:
    """Default judge: OpenAI chat.completions (text-only). Returns (verdict_label, usage_dict)."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL")
    )
    prompt = JUDGE_PROMPT.format(
        question=question, model_answer=prediction, ground_truth_answer=ground_truth
    )
    r = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=JUDGE_MAX_TOKENS,
        seed=JUDGE_SEED,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
    )
    u = r.usage
    usage = (
        {
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        }
        if u
        else {}
    )
    return parse_label(r.choices[0].message.content or ""), usage


def grade(
    question: str,
    prediction: str,
    reference_list,
    answer: str | None = None,
    method: str = "judge",
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_fn=None,
) -> dict:
    """Grade one prediction. Returns {verdict, method, ground_truth, prediction}.

    method='judge' uses judge_fn (or judge_openai) — the only path that calls an API.
    method='exact' is offline. <think> tags are stripped before grading (paper-faithful).
    """
    pred = strip_think(prediction)
    gt = build_ground_truth(reference_list, answer)
    judge_usage: dict = {}
    if method == "exact":
        golds = [r for r in (reference_list or []) if r] or ([answer] if answer else [])
        verdict = "correct" if is_exact_match(pred, golds) else "incorrect"
    elif method == "judge":
        fn = judge_fn or judge_openai
        res = fn(question, pred, gt, judge_model)
        # judge_fn may return a bare label (e.g. a mock) or (label, usage_dict) like judge_openai
        if isinstance(res, tuple):
            verdict, judge_usage = res[0], (res[1] or {})
        else:
            verdict = res
    else:
        raise ValueError(f"unknown grade method: {method!r} (choose 'judge' or 'exact')")
    return {"verdict": verdict, "method": method, "ground_truth": gt,
            "prediction": pred, "judge_usage": judge_usage}


def aggregate(verdicts: list[str]) -> dict:
    """accuracy = #correct / N (errors excluded from N, reported separately)."""
    valid = [v for v in verdicts if v in VERDICTS]
    errors = len(verdicts) - len(valid)
    n = len(valid)
    c = valid.count("correct")
    return {
        "n": n,
        "correct": c,
        "incorrect": valid.count("incorrect"),
        "unattempted": valid.count("unattempted"),
        "errors": errors,
        "accuracy": round(c / n, 4) if n else 0.0,
    }
