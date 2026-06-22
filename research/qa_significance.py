#!/usr/bin/env python3
"""Exact McNemar paired significance for the 4-cell iNat QA sweep. Stdlib only (json, math).

Each query is a paired binary trial: outcome = 1 iff verdict == "correct" (incorrect AND
unattempted both → 0). For a contrast between two arms we count the discordant pairs
  b = #(arm_a correct, arm_b wrong)   c = #(arm_a wrong, arm_b correct)
and compute the EXACT two-sided McNemar p-value from the binomial:

    p = min(1, 2 * sum_{i=0}^{min(b,c)} C(b+c, i) * 0.5^(b+c)),   p = 1.0 if b+c == 0.

NO chi-square, NO normal/continuity-corrected approximation — n is tiny (25 queries), so only the
exact binomial test is valid. Arms are aligned by qid and the qid sets MUST be identical across all
four files (the four cells are scored on the same 25 paired queries); a mismatch prints the diff and
aborts.

    python research/qa_significance.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

FILES = {
    "fixed_flat": "qa_fixed_flat.json",
    "fixed_hier": "qa_fixed_hier.json",
    "ca_flat": "qa_ca_flat.json",
    "ca_hier": "qa_ca_hier.json",
}

# (id, label, arm_a, arm_b) — delta is acc(arm_b) - acc(arm_a), so a positive delta = b improves on a.
CONTRASTS = [
    ("C1", "chunking (flat): fixed_flat -> ca_flat", "fixed_flat", "ca_flat"),
    ("C2", "expansion (content_aware): ca_flat -> ca_hier", "ca_flat", "ca_hier"),
    ("C3", "best vs baseline: fixed_flat -> ca_hier", "fixed_flat", "ca_hier"),
    ("C4", "expansion-on-fixed: fixed_flat -> fixed_hier", "fixed_flat", "fixed_hier"),
]


def load_outcomes(path: Path) -> dict[str, int]:
    """qid -> 1 if verdict 'correct' else 0 (incorrect and unattempted both score 0)."""
    d = json.loads(path.read_text())
    return {r["qid"]: (1 if r.get("verdict") == "correct" else 0) for r in d.get("per_query", [])}


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p from the binomial on the b+c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def _selftest() -> None:
    # exact two-sided sign-test reference values (== scipy binomtest(k, n, 0.5, 'two-sided'))
    assert abs(mcnemar_exact_p(5, 0) - 0.0625) < 1e-12
    assert abs(mcnemar_exact_p(4, 0) - 0.125) < 1e-12
    assert abs(mcnemar_exact_p(8, 0) - 0.0078125) < 1e-12
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(3, 3) == 1.0  # perfectly symmetric -> capped at 1.0


def main() -> int:
    _selftest()

    arms: dict[str, dict[str, int]] = {}
    for name, fn in FILES.items():
        p = RESULTS / fn
        if not p.exists():
            print(f"ABORT: missing {p}")
            return 1
        arms[name] = load_outcomes(p)

    # the four cells must be the SAME 25 paired queries
    ref_name = "fixed_flat"
    ref = set(arms[ref_name])
    mismatch = False
    for name in FILES:
        s = set(arms[name])
        if s != ref:
            mismatch = True
            print(f"ABORT: qid set mismatch {name} vs {ref_name}")
            print(f"  only in {name}:       {sorted(s - ref)}")
            print(f"  only in {ref_name}: {sorted(ref - s)}")
    if mismatch:
        return 1

    qids = sorted(ref)
    n = len(qids)

    def acc(name: str) -> float:
        return sum(arms[name][q] for q in qids) / n

    rows = []
    for cid, label, a, b in CONTRASTS:
        oa, ob = arms[a], arms[b]
        n_b = sum(1 for q in qids if oa[q] == 1 and ob[q] == 0)  # a correct, b wrong
        n_c = sum(1 for q in qids if oa[q] == 0 and ob[q] == 1)  # a wrong, b correct
        rows.append({
            "id": cid,
            "contrast": label,
            "arm_a": a,
            "arm_b": b,
            "n": n,
            "acc_a": round(acc(a), 4),
            "acc_b": round(acc(b), 4),
            "delta": round(acc(b) - acc(a), 4),
            "b_a_correct_b_wrong": n_b,
            "c_a_wrong_b_correct": n_c,
            "discordant": n_b + n_c,
            "p_exact_mcnemar": round(mcnemar_exact_p(n_b, n_c), 5),
        })

    out = {
        "test": "exact two-sided McNemar (binomial); no approximation",
        "n_queries": n,
        "binary_rule": "1 if verdict=='correct' else 0 (incorrect and unattempted both 0)",
        "sources": {name: FILES[name] for name in FILES},
        "accuracies": {name: round(acc(name), 4) for name in FILES},
        "contrasts": rows,
    }
    out_path = RESULTS / "qa_significance.json"
    out_path.write_text(json.dumps(out, indent=2))

    # ---- table ----
    print(f"\nExact two-sided McNemar (binomial), n={n} paired queries  [b=a✓/b✗  c=a✗/b✓]")
    print("accuracies: " + "  ".join(f"{k}={v}" for k, v in out["accuracies"].items()))
    hdr = f"{'id':<3} {'contrast':<44} {'acc_a':>6} {'acc_b':>6} {'delta':>7} {'b':>3} {'c':>3} {'p_exact':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['id']:<3} {r['contrast']:<44} {r['acc_a']:>6.4f} {r['acc_b']:>6.4f} "
              f"{r['delta']:>+7.4f} {r['b_a_correct_b_wrong']:>3} {r['c_a_wrong_b_correct']:>3} "
              f"{r['p_exact_mcnemar']:>8.4f}")
    print(f"\n[qa-sig] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
