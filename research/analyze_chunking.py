#!/usr/bin/env python3
"""Zero-compute forensics on existing result JSONs: why did content-aware (CA) chunking help on
iNat QA (+0.24) but not on NQ / NQ-Tables? Reads results/*.json only — no rendering / embedding /
API / new corpus. Prints the numbers behind research/CHUNKING_ANALYSIS.md.

    python research/analyze_chunking.py
"""

from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"


def L(name: str) -> dict:
    return json.loads((R / f"{name}.json").read_text())


# benchmark -> (fixed_retrieval, ca_retrieval, [fixed_flat, fixed_hier, ca_flat, ca_hier] qa cells)
BENCH = {
    "iNat": ("baseline_clean", "content_aware",
             ["qa_fixed_flat", "qa_fixed_hier", "qa_ca_flat", "qa_ca_hier"]),
    "NQ-Tables": ("nqt_fixed_retrieval", "nqt_ca_retrieval",
                  ["qa_nqt_fixed_flat", "qa_nqt_fixed_hier", "qa_nqt_ca_flat", "qa_nqt_ca_hier"]),
    "NQ": ("nq_fixed_retrieval", "nq_ca_retrieval",
           ["qa_nq_fixed_flat", "qa_nq_fixed_hier", "qa_nq_ca_flat", "qa_nq_ca_hier"]),
}


def _cost(name: str) -> dict:
    c = L(name)["cost"]
    tp, ch = c["tiles_per_page"], c["chunk_height_px_dist"]
    return {"total": c["total_chunks"], "pages": c["pages"], "tpp_mean": tp["mean"],
            "tpp_std": tp["std"], "tpp_min": tp["min"], "tpp_max": tp["max"],
            "h_mean": ch["mean"], "h_std": ch["std"], "h_min": ch["min"]}


def q1_chunker_effect():
    print("\n========== Q1: DID THE CHUNKER DO ANYTHING? (fixed vs CA) ==========")
    for b, (fx, ca, _) in BENCH.items():
        f, c = _cost(fx), _cost(ca)
        d_total = c["total"] - f["total"]
        print(f"\n[{b}]  pages={f['pages']}")
        print(f"  total_chunks   fixed={f['total']:5d}   ca={c['total']:5d}   Δ={d_total:+d} ({100*d_total/f['total']:+.1f}%)")
        print(f"  tiles/page     fixed={f['tpp_mean']:.2f}±{f['tpp_std']:.2f} [{f['tpp_min']:.0f},{f['tpp_max']:.0f}]"
              f"   ca={c['tpp_mean']:.2f}±{c['tpp_std']:.2f} [{c['tpp_min']:.0f},{c['tpp_max']:.0f}]"
              f"   Δmean={c['tpp_mean']-f['tpp_mean']:+.2f}")
        print(f"  chunk-height   fixed={f['h_mean']:.1f}±{f['h_std']:.1f} (min {f['h_min']:.0f})"
              f"   ca={c['h_mean']:.1f}±{c['h_std']:.1f} (min {c['h_min']:.0f})"
              f"   Δmean={c['h_mean']-f['h_mean']:+.1f}  Δstd={c['h_std']-f['h_std']:+.1f}")
        noop = (d_total == 0 and abs(c["h_mean"] - f["h_mean"]) < 1.0 and abs(c["tpp_mean"] - f["tpp_mean"]) < 0.01)
        print(f"  -> CA {'== fixed (NO-OP: identical chunk count + height)' if noop else 'DIFFERS materially from fixed'}")


def q3_modality():
    print("\n========== Q3: QUERY-MODALITY CONFOUND ==========")
    for b, (fx, ca, _) in BENCH.items():
        qm = L(fx)["query_modality"]
        print(f"  {b:10s}: image_queries={qm['n_image_queries']:3d}  text_queries(fallback)={qm['n_text_fallback']:3d}  (n={qm['n_total']})")


def _acc(cell: str) -> float:
    return L(cell)["accuracy"]


def q4_expansion():
    print("\n========== Q4: EXPANSION (flat -> hier) per arm x benchmark ==========")
    print(f"  {'bench':10s} {'arm':6s} {'flat':>7s} {'hier':>7s} {'Δ(hier-flat)':>13s}")
    for b, (_, _, cells) in BENCH.items():
        ff, fh, cf, ch = cells
        for arm, flat, hier in (("fixed", ff, fh), ("ca", cf, ch)):
            a_flat, a_hier = _acc(flat), _acc(hier)
            print(f"  {b:10s} {arm:6s} {a_flat:7.4f} {a_hier:7.4f} {a_hier-a_flat:+13.4f}")
    print("\n  fixed-vs-CA at FLAT (same retrieval depth; tests whether CA changed anything):")
    for b, (_, _, cells) in BENCH.items():
        ff, _, cf, _ = cells
        print(f"  {b:10s} fixed_flat={_acc(ff):.4f}  ca_flat={_acc(cf):.4f}  Δ={_acc(cf)-_acc(ff):+.4f}")


def q2_ceiling():
    print("\n========== Q2: READER-CEILING vs RETRIEVAL-CEILING on misses (best CA cell) ==========")
    # best CA cell = higher-accuracy of {ca_flat, ca_hier}; classify misses by whether the gold
    # page (gold_article_id) was among the tiles the reader actually saw (qa row 'tiles').
    pick = {  # (best_ca_qa_cell, ca_retrieval_json, modality_rels_key)
        "iNat": (None, "content_aware", "rels_image"),
        "NQ-Tables": (None, "nqt_ca_retrieval", "rels_text"),
        "NQ": (None, "nq_ca_retrieval", "rels_text"),
    }
    for b, (_, ca_ret, cells) in BENCH.items():
        _, _, cf, ch = cells
        best = cf if _acc(cf) >= _acc(ch) else ch
        rels_key = pick[b][2]
        ret = L(ca_ret)
        gold_aid = {q["qid"]: q.get("gold_article_id") for q in ret["per_query"]}
        # was gold anywhere in the stored flat top-k hits? (retrieval recall cross-check)
        gold_in_topk = {q["qid"]: (1 in (q.get(rels_key) or [])) for q in ret["per_query"]}
        qa = L(best)
        reader_ceiling = retrieval_ceiling = 0
        retr_ceiling_but_retrievable = 0  # gold not in reader tiles, but WAS in stored top-k
        misses = 0
        for row in qa["per_query"]:
            if row["verdict"] == "correct":
                continue
            misses += 1
            qid = row["qid"]
            reader_aids = {t[0] for t in row.get("tiles", [])}
            if gold_aid.get(qid) in reader_aids:
                reader_ceiling += 1            # (a) reader saw a gold tile, still wrong
            else:
                retrieval_ceiling += 1         # (b) reader never saw a gold tile
                if gold_in_topk.get(qid):
                    retr_ceiling_but_retrievable += 1
        acc = _acc(best)
        ratio = f"{reader_ceiling}:{retrieval_ceiling}"
        print(f"\n[{b}] best CA cell = {best} (acc={acc:.4f}, {misses} misses)")
        print(f"  (a) reader ceiling  (gold tile WAS read, answered wrong) = {reader_ceiling}")
        print(f"  (b) retrieval ceiling (gold tile NOT in reader's tiles)  = {retrieval_ceiling}"
              f"   [of which {retr_ceiling_but_retrievable} had gold in the stored top-k but cut before the reader]")
        print(f"  (a):(b) = {ratio}  -> {'reader-bound' if reader_ceiling>retrieval_ceiling else 'retrieval-bound'}")


def q5_modality_flip():
    """Disambiguating experiment: iNat QA with the SAME corpus + CA tiles, flipping only the query
    modality (image vs text). Separates 'CA chunking helps' (A) from 'image queries help' (B)."""
    print("\n========== iNat: IMAGE-query vs TEXT-query CA (disambiguating experiment) ==========")
    cells = [  # (label, qa_cell, retrieval_json_for_gold_article_id)
        ("image fixed-flat", "qa_fixed_flat", "baseline_clean"),
        ("image fixed-hier", "qa_fixed_hier", "baseline_clean"),
        ("image ca-flat", "qa_ca_flat", "content_aware"),
        ("image ca-hier", "qa_ca_hier", "content_aware"),
        ("text  fixed-flat", "qa_inat_text_fixed_flat", "baseline_clean"),
        ("text  fixed-hier", "qa_inat_text_fixed_hier", "baseline_clean"),
        ("text  ca-flat", "qa_inat_text_ca_flat", "content_aware"),
        ("text  ca-hier", "qa_inat_text_ca_hier", "content_aware"),
    ]
    gmaps: dict[str, dict] = {}

    def gmap(ret: str) -> dict:
        if ret not in gmaps:
            gmaps[ret] = {q["qid"]: q.get("gold_article_id") for q in L(ret)["per_query"]}
        return gmaps[ret]

    print(f"  {'cell':18s} {'acc':>6s} {'reader-saw-gold':>16s}")
    acc: dict[str, float] = {}
    for label, qa, ret in cells:
        d = L(qa)
        pq, gm = d["per_query"], gmap(ret)
        seen = sum(1 for r in pq if gm.get(r["qid"]) in {t[0] for t in r.get("tiles", [])})
        acc[" ".join(label.split())] = d["accuracy"]  # collapse alignment whitespace
        print(f"  {label:18s} {d['accuracy']:6.3f} {seen:>4d}/{len(pq)} ({100*seen/len(pq):4.0f}%)")
    print("\n  CA effect (ca − fixed), per modality x retrieval:")
    for mod in ("image", "text"):
        for r in ("flat", "hier"):
            ca, fx = acc[f"{mod} ca-{r}"], acc[f"{mod} fixed-{r}"]
            print(f"    {mod:5s} {r:4s}:  ca {ca:.3f} − fixed {fx:.3f} = {ca - fx:+.3f}")


if __name__ == "__main__":
    q1_chunker_effect()
    q2_ceiling()
    q3_modality()
    q4_expansion()
    q5_modality_flip()
    print()
