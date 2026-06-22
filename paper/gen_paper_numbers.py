#!/usr/bin/env python3
"""Generate EVERY numeric figure for the paper from committed results/*.json.

Data discipline (same as research/gen_phase3_experiments.py): no number in paper/main.tex is
hand-typed — they all flow through here. Reuses research/analyze_chunking.py primitives
(L, _cost, _acc, BENCH) so the analysis and the paper cannot disagree. stdlib + matplotlib only.

Outputs:
  paper/numbers.tex          \\newcommand macros for every inline number
  paper/tables/*.tex         booktabs tabular blocks (\\input-ed inside table floats)
  paper/figures/*.{pdf,png}  matplotlib figures

Run:  .venv-mac/bin/python paper/gen_paper_numbers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
from analyze_chunking import L, _cost, _acc, BENCH  # noqa: E402  reuse, do not reimplement

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
FIGS = PAPER / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# ───────────────────────── helpers ─────────────────────────
MACROS: dict[str, str] = {}


def macro(name: str, value) -> None:
    assert name.isalpha(), f"LaTeX macro name must be letters-only: {name!r}"
    assert name not in MACROS, f"duplicate macro {name}"
    MACROS[name] = str(value)


def f2(x: float) -> str:
    return f"{x:.2f}"


def signed2(x: float) -> str:
    return f"{x:+.2f}"


def pct(x: float, dec: int = 0) -> str:
    return f"{x:.{dec}f}"


def gold_aid(ret_json: str) -> dict:
    return {q["qid"]: q.get("gold_article_id") for q in L(ret_json)["per_query"]}


def saw_gold(qa_cell: str, gmap: dict) -> tuple[int, int]:
    """(#queries where the gold page reached the reader, total) — gold_article_id in the read tiles."""
    pq = L(qa_cell)["per_query"]
    seen = sum(1 for r in pq if gmap.get(r["qid"]) in {t[0] for t in r.get("tiles", [])})
    return seen, len(pq)


def ceiling(qa_cell: str, ret_json: str) -> tuple[int, int]:
    """Misses split into (reader-ceiling: gold tile read but wrong, retrieval-ceiling: gold not read)."""
    gmap = gold_aid(ret_json)
    rc = tc = 0
    for r in L(qa_cell)["per_query"]:
        if r["verdict"] == "correct":
            continue
        if gmap.get(r["qid"]) in {t[0] for t in r.get("tiles", [])}:
            rc += 1
        else:
            tc += 1
    return rc, tc


# ───────────────────────── corpus metadata (all from results/ provenance) ─────────────────────────
# gold = provenance.n_queries (one query per gold page); total = cost.pages; distractors = total - gold.
CORPORA = {  # key -> (label, fixed_retrieval_json, ca_retrieval_json, modality)
    "inat": ("iNaturalist", "baseline_clean", "content_aware", "image"),
    "nqt": ("NQ-Tables", "nqt_fixed_retrieval", "nqt_ca_retrieval", "text"),
    "nq": ("NQ", "nq_fixed_retrieval", "nq_ca_retrieval", "text"),
}
PFX = {"inat": "Inat", "nqt": "Nqt", "nq": "Nq"}

for key, (label, fx_ret, _ca_ret, _mod) in CORPORA.items():
    d = L(fx_ret)
    gold = d["provenance"]["n_queries"]
    pages = d["cost"]["pages"]
    macro(PFX[key] + "Gold", gold)
    macro(PFX[key] + "Distr", pages - gold)
    macro(PFX[key] + "Pages", pages)
    macro(PFX[key] + "N", gold)

# models (strings; escape underscores for LaTeX text)
_reader = L("qa_ca_hier")["reader"]["model"]
_judge = L("qa_ca_hier")["judge_model"]
_embed = L("baseline_clean")["provenance"]["model"]
macro("ReaderModel", _reader)
macro("JudgeModel", _judge)
macro("EmbedModel", _embed.replace("_", r"\_"))

# ───────────────────────── Q1: layout-gating (tile counts) ─────────────────────────
for key, (label, fx_ret, ca_ret, _mod) in CORPORA.items():
    f, c = _cost(fx_ret), _cost(ca_ret)
    p = PFX[key]
    macro(p + "ChunksFixed", f["total"])
    macro(p + "ChunksCa", c["total"])
    macro(p + "TppFixed", f2(f["tpp_mean"]))
    macro(p + "TppCa", f2(c["tpp_mean"]))
    macro(p + "HStdFixed", f"{f['h_std']:.0f}")
    macro(p + "HStdCa", f"{c['h_std']:.0f}")
    if f["total"]:
        macro(p + "ChunksDeltaPct", f"{100*(c['total']-f['total'])/f['total']:.1f}")

# ───────────────────────── boundary violation (probe set) ─────────────────────────
bv = L("boundary_violation_probe")
bvf, bvc = bv["fixed"], bv["content_aware"]
macro("BvFixedBlock", f"{100*bvf['block_split_rate']:.2f}")
macro("BvCaBlock", f"{100*bvc['block_split_rate']:.2f}")
macro("BvFixedRow", f"{100*bvf['row_split_rate']:.2f}")
macro("BvCaRow", f"{100*bvc['row_split_rate']:.2f}")
macro("BvBlockRatio", f"{bvf['block_split_rate']/bvc['block_split_rate']:.1f}")
macro("BvRowRatio", f"{bvf['row_split_rate']/bvc['row_split_rate']:.1f}")
macro("BvFixedBlockN", bvf["blocks_split"])
macro("BvCaBlockN", bvc["blocks_split"])
macro("BvFixedRowN", bvf["rows_split"])
macro("BvCaRowN", bvc["rows_split"])

# ───────────────────────── QA accuracies (16 cells) ─────────────────────────
QA = {  # macro-prefix -> qa cell json
    "InatFixedFlat": "qa_fixed_flat", "InatFixedHier": "qa_fixed_hier",
    "InatCaFlat": "qa_ca_flat", "InatCaHier": "qa_ca_hier",
    "InatTextFixedFlat": "qa_inat_text_fixed_flat", "InatTextFixedHier": "qa_inat_text_fixed_hier",
    "InatTextCaFlat": "qa_inat_text_ca_flat", "InatTextCaHier": "qa_inat_text_ca_hier",
    "NqtFixedFlat": "qa_nqt_fixed_flat", "NqtFixedHier": "qa_nqt_fixed_hier",
    "NqtCaFlat": "qa_nqt_ca_flat", "NqtCaHier": "qa_nqt_ca_hier",
    "NqFixedFlat": "qa_nq_fixed_flat", "NqFixedHier": "qa_nq_fixed_hier",
    "NqCaFlat": "qa_nq_ca_flat", "NqCaHier": "qa_nq_ca_hier",
}
ACC = {k: _acc(v) for k, v in QA.items()}
for k, v in ACC.items():
    macro(k + "Acc", f2(v))

# ───────────────────────── Q5: modality flip (iNat image vs text) ─────────────────────────
# CA effect = ca - fixed, per modality x retrieval depth
ca_img_flat = ACC["InatCaFlat"] - ACC["InatFixedFlat"]
ca_img_hier = ACC["InatCaHier"] - ACC["InatFixedHier"]
ca_txt_flat = ACC["InatTextCaFlat"] - ACC["InatTextFixedFlat"]
ca_txt_hier = ACC["InatTextCaHier"] - ACC["InatTextFixedHier"]
macro("CaImgFlatDelta", signed2(ca_img_flat))
macro("CaImgHierDelta", signed2(ca_img_hier))
macro("CaTxtFlatDelta", signed2(ca_txt_flat))
macro("CaTxtHierDelta", signed2(ca_txt_hier))

# reader-saw-gold (flat cells; image vs text, fixed vs ca)
gimg = gold_aid("content_aware")
gimg_fx = gold_aid("baseline_clean")
img_fx_seen, _ = saw_gold("qa_fixed_flat", gimg_fx)
img_ca_seen, n25 = saw_gold("qa_ca_flat", gimg)
txt_fx_seen, _ = saw_gold("qa_inat_text_fixed_flat", gimg_fx)
txt_ca_seen, _ = saw_gold("qa_inat_text_ca_flat", gimg)
macro("ImgFixedSawGold", f"{100*img_fx_seen/n25:.0f}")
macro("ImgCaSawGold", f"{100*img_ca_seen/n25:.0f}")
macro("TxtFixedSawGold", f"{100*txt_fx_seen/n25:.0f}")
macro("TxtCaSawGold", f"{100*txt_ca_seen/n25:.0f}")

# ───────────────────────── Q2: reader vs retrieval ceiling (best CA cell) ─────────────────────────
BEST_CA = {  # bench -> (best ca qa cell, ca retrieval json) — best = higher-accuracy ca cell
    "inat": ("qa_ca_hier" if ACC["InatCaHier"] >= ACC["InatCaFlat"] else "qa_ca_flat", "content_aware"),
    "nqt": ("qa_nqt_ca_hier" if ACC["NqtCaHier"] >= ACC["NqtCaFlat"] else "qa_nqt_ca_flat", "nqt_ca_retrieval"),
    "nq": ("qa_nq_ca_hier" if ACC["NqCaHier"] >= ACC["NqCaFlat"] else "qa_nq_ca_flat", "nq_ca_retrieval"),
}
ceil_rows = {}
reader_pcts = []
for key, (cell, ret) in BEST_CA.items():
    rc, tc = ceiling(cell, ret)
    ceil_rows[key] = (cell, rc, tc)
    macro(PFX[key] + "ReaderCeil", rc)
    macro(PFX[key] + "RetrCeil", tc)
    if rc + tc:
        reader_pcts.append(100 * rc / (rc + tc))
macro("ReaderCeilPctMin", f"{min(reader_pcts):.0f}")
macro("ReaderCeilPctMax", f"{max(reader_pcts):.0f}")

# ───────────────────────── McNemar (iNat image headline) ─────────────────────────
con = {c["id"]: c for c in L("qa_significance")["contrasts"]}
c3 = con["C3"]
macro("McNemarDelta", signed2(c3["delta"]))
macro("McNemarP", f"{c3['p_exact_mcnemar']:.3f}")
macro("McNemarDiscordant", c3["c_a_wrong_b_correct"] + c3["b_a_correct_b_wrong"])
macro("McNemarFavor", c3["c_a_wrong_b_correct"])  # all discordant favor the full method (b=0)
macro("McNemarBaselineAcc", f2(c3["acc_a"]))
macro("McNemarBestAcc", f2(c3["acc_b"]))
macro("McNemarN", L("qa_significance")["n_queries"])

# ───────────────────────── retrieval recall@1 (key inline numbers) ─────────────────────────
def recall1(ret_json: str, mod: str) -> float:
    return L(ret_json)["retrieval"][mod]["recall@1"]


macro("InatImgRecallFixed", f2(recall1("baseline_clean", "image")))
macro("InatImgRecallCa", f2(recall1("content_aware", "image")))
macro("InatTxtRecallFixed", f2(recall1("baseline_clean", "text")))
macro("InatTxtRecallCa", f2(recall1("content_aware", "text")))
macro("NqtTxtRecallFixed", f2(recall1("nqt_fixed_retrieval", "text")))
macro("NqtTxtRecallCa", f2(recall1("nqt_ca_retrieval", "text")))
macro("NqTxtRecallFixed", f2(recall1("nq_fixed_retrieval", "text")))
macro("NqTxtRecallCa", f2(recall1("nq_ca_retrieval", "text")))

# ───────────────────────── write numbers.tex ─────────────────────────
lines = ["% AUTO-GENERATED by paper/gen_paper_numbers.py — DO NOT EDIT. Every number sourced from results/*.json.",
         ""]
for k in sorted(MACROS):
    lines.append(f"\\newcommand{{\\{k}}}{{{MACROS[k]}}}")
(PAPER / "numbers.tex").write_text("\n".join(lines) + "\n")


# ───────────────────────── tables ─────────────────────────
def write_table(name: str, body: str) -> None:
    (TABLES / name).write_text(
        "% AUTO-GENERATED by paper/gen_paper_numbers.py — DO NOT EDIT.\n" + body.rstrip() + "\n"
    )


# main QA table: 3 corpora x 4 cells
qa_main = [r"\begin{tabular}{llcc}", r"\toprule",
           r"Corpus (query) & Chunker & Flat & Hier-expand \\", r"\midrule"]
rows = [
    ("iNaturalist (image)", "fixed", "InatFixedFlat", "InatFixedHier", False),
    ("", "content-aware", "InatCaFlat", "InatCaHier", True),
    ("NQ-Tables (text)", "fixed", "NqtFixedFlat", "NqtFixedHier", False),
    ("", "content-aware", "NqtCaFlat", "NqtCaHier", False),
    ("NQ (text)", "fixed", "NqFixedFlat", "NqFixedHier", False),
    ("", "content-aware", "NqCaFlat", "NqCaHier", False),
]
for i, (corp, chunk, kf, kh, boldhier) in enumerate(rows):
    if corp and i:
        qa_main.append(r"\midrule")
    flat = f2(ACC[kf])
    hier = (r"\textbf{" + f2(ACC[kh]) + "}") if boldhier else f2(ACC[kh])
    qa_main.append(f"{corp} & {chunk} & {flat} & {hier} \\\\")
qa_main += [r"\bottomrule", r"\end{tabular}"]
write_table("tab_qa_main.tex", "\n".join(qa_main))

# modality-flip table (iNat image vs text): acc + CA delta + reader-saw-gold
mf = [r"\begin{tabular}{llccc}", r"\toprule",
      r"Modality & Retrieval & Fixed & Content-aware & CA $\Delta$ \\", r"\midrule"]
mf_rows = [
    ("Image", "flat", "InatFixedFlat", "InatCaFlat", ca_img_flat),
    ("", "hier-expand", "InatFixedHier", "InatCaHier", ca_img_hier),
    ("Text", "flat", "InatTextFixedFlat", "InatTextCaFlat", ca_txt_flat),
    ("", "hier-expand", "InatTextFixedHier", "InatTextCaHier", ca_txt_hier),
]
for i, (mod, retr, kf, kc, dlt) in enumerate(mf_rows):
    if mod == "Text":
        mf.append(r"\midrule")
    mf.append(f"{mod} & {retr} & {f2(ACC[kf])} & {f2(ACC[kc])} & {signed2(dlt)} \\\\")
mf += [r"\bottomrule", r"\end{tabular}"]
write_table("tab_modality_flip.tex", "\n".join(mf))

# layout-gating table: tiles/page, chunks, height std, fixed vs CA per corpus
lay = [r"\begin{tabular}{lrrrr}", r"\toprule",
       r"Corpus & Tiles/pg (fixed$\to$CA) & Chunks (fixed$\to$CA) & Height std (fixed$\to$CA) & Engaged? \\",
       r"\midrule"]
for key, (label, fx_ret, ca_ret, _mod) in CORPORA.items():
    f, c = _cost(fx_ret), _cost(ca_ret)
    engaged = "yes" if f["total"] != c["total"] else r"\textbf{no-op}"
    lay.append(
        f"{label} & {f2(f['tpp_mean'])}$\\to${f2(c['tpp_mean'])} & "
        f"{f['total']}$\\to${c['total']} & {f['h_std']:.0f}$\\to${c['h_std']:.0f} & {engaged} \\\\"
    )
lay += [r"\bottomrule", r"\end{tabular}"]
write_table("tab_layout.tex", "\n".join(lay))

# retrieval table: recall@1 / recall@5 / nDCG@10 per corpus per arm (relevant modality)
def rmet(ret_json: str, mod: str, m: str):
    return L(ret_json)["retrieval"][mod][m]


ret = [r"\begin{tabular}{llccc}", r"\toprule",
       r"Corpus (modality) & Chunker & R@1 & R@5 & nDCG@10 \\", r"\midrule"]
for key, (label, fx_ret, ca_ret, mod) in CORPORA.items():
    for arm, rj in (("fixed", fx_ret), ("content-aware", ca_ret)):
        ret.append(
            f"{label if arm=='fixed' else ''} ({mod}) & {arm} & "
            f"{f2(rmet(rj, mod, 'recall@1'))} & {f2(rmet(rj, mod, 'recall@5'))} & "
            f"{f2(rmet(rj, mod, 'ndcg@10'))} \\\\"
        )
    if key != "nq":
        ret.append(r"\midrule")
ret += [r"\bottomrule", r"\end{tabular}"]
write_table("tab_retrieval.tex", "\n".join(ret))

# McNemar table (the 4 contrasts)
mc = [r"\begin{tabular}{llccc}", r"\toprule",
      r"Contrast & Arms & $\Delta$acc & Discordant (b:c) & $p$ (exact) \\", r"\midrule"]
labels = {"C1": "Chunking (flat)", "C2": "Expansion (CA)", "C3": "Best vs.\\ baseline",
          "C4": "Expansion (fixed)"}
for cid in ("C1", "C2", "C3", "C4"):
    cc = con[cid]
    arms = f"{cc['arm_a']}$\\to${cc['arm_b']}".replace("_", r"\_")
    star = r"\,$^\star$" if cc["p_exact_mcnemar"] < 0.05 else ""
    mc.append(
        f"{labels[cid]} & {arms} & {signed2(cc['delta'])} & "
        f"{cc['b_a_correct_b_wrong']}:{cc['c_a_wrong_b_correct']} & "
        f"{cc['p_exact_mcnemar']:.3f}{star} \\\\"
    )
mc += [r"\bottomrule", r"\end{tabular}"]
write_table("tab_mcnemar.tex", "\n".join(mc))

# boundary-violation table
bvt = [r"\begin{tabular}{lccc}", r"\toprule",
       r"Split type & Fixed & Content-aware & Reduction \\", r"\midrule",
       f"Block split & {MACROS['BvFixedBlock']}\\% & {MACROS['BvCaBlock']}\\% & "
       f"{MACROS['BvBlockRatio']}$\\times$ \\\\",
       f"Table-row split & {MACROS['BvFixedRow']}\\% & {MACROS['BvCaRow']}\\% & "
       f"{MACROS['BvRowRatio']}$\\times$ \\\\",
       r"\bottomrule", r"\end{tabular}"]
write_table("tab_boundary.tex", "\n".join(bvt))

# ───────────────────────── figures ─────────────────────────
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C_FIX, C_CA = "#9aa0a6", "#1a73e8"
C_IMG, C_TXT = "#1a73e8", "#e8710a"


def savefig(fig, stem: str) -> None:
    fig.savefig(FIGS / f"{stem}.pdf")
    fig.savefig(FIGS / f"{stem}.png", dpi=150)
    plt.close(fig)


# Fig 1: layout-gating — tiles/page by corpus x chunker (with std error bars)
fig, ax = plt.subplots(figsize=(5.2, 3.2))
labels1 = [CORPORA[k][0] for k in CORPORA]
fx_m = [_cost(CORPORA[k][1])["tpp_mean"] for k in CORPORA]
ca_m = [_cost(CORPORA[k][2])["tpp_mean"] for k in CORPORA]
fx_s = [_cost(CORPORA[k][1])["tpp_std"] for k in CORPORA]
ca_s = [_cost(CORPORA[k][2])["tpp_std"] for k in CORPORA]
x = np.arange(len(labels1))
w = 0.36
ax.bar(x - w / 2, fx_m, w, yerr=fx_s, capsize=3, label="fixed", color=C_FIX)
ax.bar(x + w / 2, ca_m, w, yerr=ca_s, capsize=3, label="content-aware", color=C_CA)
ax.set_xticks(x)
ax.set_xticklabels(labels1)
ax.set_ylabel("tiles / page")
ax.set_title("Layout-gating: content-aware chunking only engages on heterogeneous layout")
ax.legend(frameon=False)
ax.annotate("identical\n(no-op)", xy=(1, ca_m[1]), xytext=(1.05, ca_m[1] + 2.5),
            ha="center", fontsize=9, color="dimgray",
            arrowprops=dict(arrowstyle="->", color="dimgray"))
savefig(fig, "fig1_layout_gating")

# Fig 2: modality flip (CENTERPIECE) — iNat CA delta, image vs text x flat/hier
fig, ax = plt.subplots(figsize=(5.2, 3.4))
groups = ["flat", "hier-expand"]
img_d = [ca_img_flat, ca_img_hier]
txt_d = [ca_txt_flat, ca_txt_hier]
x = np.arange(len(groups))
ax.axhline(0, color="black", lw=0.8)
b1 = ax.bar(x - w / 2, img_d, w, label="image queries", color=C_IMG)
b2 = ax.bar(x + w / 2, txt_d, w, label="text queries", color=C_TXT)
ax.bar_label(b1, labels=[signed2(v) for v in img_d], padding=3, fontsize=9)
ax.bar_label(b2, labels=[signed2(v) for v in txt_d], padding=3, fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel("content-aware QA effect (CA $-$ fixed)")
ax.set_title("Same iNat corpus & tiles: the CA benefit is image-query-specific")
ax.legend(frameon=False, loc="upper right")
ax.margins(y=0.22)
savefig(fig, "fig2_modality_flip")

# Fig 3: reader-ceiling vs retrieval-ceiling per benchmark (stacked, fraction of misses)
fig, ax = plt.subplots(figsize=(5.2, 3.2))
labels3 = [CORPORA[k][0] for k in BEST_CA]
rc = np.array([ceil_rows[k][1] for k in BEST_CA], float)
tc = np.array([ceil_rows[k][2] for k in BEST_CA], float)
tot = rc + tc
x = np.arange(len(labels3))
ax.bar(x, rc / tot, 0.55, label="reader ceiling (gold read, misread)", color=C_CA)
ax.bar(x, tc / tot, 0.55, bottom=rc / tot, label="retrieval ceiling (gold not read)", color=C_FIX)
for i in range(len(labels3)):
    ax.text(i, 0.5 * rc[i] / tot[i], f"{int(rc[i])}", ha="center", va="center", color="white", fontsize=10)
    if tc[i]:
        ax.text(i, rc[i] / tot[i] + 0.5 * tc[i] / tot[i], f"{int(tc[i])}", ha="center", va="center", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(labels3)
ax.set_ylabel("fraction of QA misses")
ax.set_ylim(0, 1)
ax.set_title("Most errors are reading failures on correctly-retrieved content")
ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=1)
savefig(fig, "fig3_reader_ceiling")

# Fig 4 (optional): boundary-violation rate fixed vs CA on the probe set
fig, ax = plt.subplots(figsize=(4.6, 3.0))
cats = ["block split", "table-row split"]
fxr = [100 * bvf["block_split_rate"], 100 * bvf["row_split_rate"]]
car = [100 * bvc["block_split_rate"], 100 * bvc["row_split_rate"]]
x = np.arange(len(cats))
ax.bar(x - w / 2, fxr, w, label="fixed", color=C_FIX)
ax.bar(x + w / 2, car, w, label="content-aware", color=C_CA)
ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.set_ylabel("region-split rate (\\%)" if False else "region-split rate (%)")
ax.set_title("Mechanism: CA respects content boundaries (probe set)")
ax.legend(frameon=False)
savefig(fig, "fig4_boundary_violation")

# ───────────────────────── summary ─────────────────────────
print(f"wrote {len(MACROS)} macros -> paper/numbers.tex")
print(f"wrote tables: {sorted(p.name for p in TABLES.glob('*.tex'))}")
print(f"wrote figures: {sorted(p.stem for p in FIGS.glob('*.pdf'))}")
print("\nKey numbers (cross-check):")
print(f"  layout no-op: NQT chunks {MACROS['NqtChunksFixed']}->{MACROS['NqtChunksCa']}, "
      f"iNat {MACROS['InatChunksFixed']}->{MACROS['InatChunksCa']} (+{MACROS['InatChunksDeltaPct']}%)")
print(f"  modality flip CA delta: image {MACROS['CaImgFlatDelta']}/{MACROS['CaImgHierDelta']}, "
      f"text {MACROS['CaTxtFlatDelta']}/{MACROS['CaTxtHierDelta']}")
print(f"  reader-saw-gold: image {MACROS['ImgFixedSawGold']}/{MACROS['ImgCaSawGold']}%, "
      f"text {MACROS['TxtFixedSawGold']}/{MACROS['TxtCaSawGold']}%")
print(f"  reader:retr ceiling: iNat {MACROS['InatReaderCeil']}:{MACROS['InatRetrCeil']}, "
      f"NQT {MACROS['NqtReaderCeil']}:{MACROS['NqtRetrCeil']}, NQ {MACROS['NqReaderCeil']}:{MACROS['NqRetrCeil']}")
print(f"  McNemar: {MACROS['McNemarDelta']} p={MACROS['McNemarP']} discordant={MACROS['McNemarDiscordant']} "
      f"favor-full={MACROS['McNemarFavor']}")
print(f"  boundary: block {MACROS['BvFixedBlock']}->{MACROS['BvCaBlock']}% ({MACROS['BvBlockRatio']}x), "
      f"row {MACROS['BvFixedRow']}->{MACROS['BvCaRow']}% ({MACROS['BvRowRatio']}x)")
