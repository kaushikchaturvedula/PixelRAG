#!/usr/bin/env python3
"""Phase-3 substrate: a page→section→tile tree for hierarchical retrieve-then-expand.

Built entirely from metadata the content-aware chunker already emits (chunks.json): `heading_path`
→ section nodes, `reading_order` → ordered leaves + neighbor edges, `(tile_index, chunk_index)` →
leaf keys. No re-embed, no render, CPU-only.

`expand()` is the retrieve-then-expand operator: given the flat FAISS top-k leaves, add context
(same-section siblings and/or reading-order neighbors), dedup, keep the seed hits first, cap the
total — the expanded tile set is what the reader sees. The flat retrieval metrics are unchanged.

Fixed arm: fixed chunks have no `heading_path` (and no `reading_order`); we derive the reading order
from the `(tile_index, chunk_index)` sequence and build a **neighbors-only** tree (no sections), so
the fixed arm can still get the reading-order expansion it supports — section-structure expansion is
the content_aware-specific advantage.
"""

from __future__ import annotations

import json
from pathlib import Path

LEAD = "(lead)"  # section bucket for chunks before the first heading (empty heading_path)


def _article_id(dir_name: str) -> int | None:
    try:
        return int(dir_name.split(".")[0])
    except (ValueError, IndexError):
        return None


def build_tree(tiles_dir: str) -> dict:
    """Build {article_id: {order, sections, leaf_to_section, pos, has_sections}} from chunks.json.

    leaf key = (tile_index, chunk_index). `order` is the reading-order leaf list; `pos` maps a leaf
    to its index in `order` (for neighbor lookup); `sections` maps heading_path → leaves (in order).
    """
    tree: dict[int, dict] = {}
    for cj in sorted(Path(tiles_dir).glob("*.png.tiles/chunks.json")):
        aid = _article_id(cj.parent.name)
        if aid is None:
            continue
        chunks = json.loads(cj.read_text()).get("chunks", [])
        if not chunks:
            continue
        # reading order: explicit reading_order if present (content_aware), else (tile_index, chunk_index)
        if all("reading_order" in c for c in chunks):
            chunks = sorted(chunks, key=lambda c: c["reading_order"])
        else:
            chunks = sorted(chunks, key=lambda c: (c["tile_index"], c["chunk_index"]))
        order = [(c["tile_index"], c["chunk_index"]) for c in chunks]
        leaf_to_section: dict[tuple, str] = {}
        sections: dict[str, list] = {}
        has_sections = any((c.get("heading_path") or "").strip() for c in chunks)
        for c in chunks:
            key = (c["tile_index"], c["chunk_index"])
            sec = (c.get("heading_path") or "").strip() or LEAD
            leaf_to_section[key] = sec
            sections.setdefault(sec, []).append(key)
        tree[aid] = {
            "order": order,
            "pos": {k: i for i, k in enumerate(order)},
            "sections": sections,
            "leaf_to_section": leaf_to_section,
            "has_sections": has_sections,
        }
    return tree


def expand(hits: list[tuple], tree: dict, mode: str = "section+neighbors",
           neighbors: int = 1, cap: int = 8) -> list[tuple]:
    """Expand flat hits with context. hits/return are (article_id, tile_index, chunk_index).

    mode: 'neighbors' (reading-order ±neighbors only — works for fixed and content_aware) or
    'section+neighbors' (also add same-section siblings — content_aware). Seed hits are kept first
    and never dropped; added context is appended in a stable order; total is capped at `cap`.
    """
    out: list[tuple] = []
    seen: set[tuple] = set()

    def add(item):
        if item not in seen:
            seen.add(item)
            out.append(item)

    # 1) seeds first (never dropped, retrieval order preserved)
    for h in hits:
        add(h)
    # 2) context per seed, in seed order
    for aid, ti, ci in hits:
        art = tree.get(aid)
        if not art:
            continue
        key = (ti, ci)
        if mode == "section+neighbors" and art["has_sections"]:
            sec = art["leaf_to_section"].get(key)
            for sib in art["sections"].get(sec, []):
                add((aid, sib[0], sib[1]))
        p = art["pos"].get(key)
        if p is not None:
            for d in range(1, neighbors + 1):
                for j in (p - d, p + d):
                    if 0 <= j < len(art["order"]):
                        nb = art["order"][j]
                        add((aid, nb[0], nb[1]))
    # 3) cap — but never below the seed count (seeds are first in `out`)
    if cap is not None and cap > 0 and len(out) > cap:
        out = out[:max(cap, len(hits))]
    return out


def validate(tree: dict) -> list[str]:
    """Structural checks; returns a list of issues (empty == well-formed)."""
    issues = []
    for aid, art in tree.items():
        order, sections, l2s = art["order"], art["sections"], art["leaf_to_section"]
        if len(order) != len(set(order)):
            issues.append(f"article {aid}: duplicate leaf in order")
        if set(order) != set(l2s):
            issues.append(f"article {aid}: order/leaf_to_section leaf sets differ")
        sec_leaves = [k for ks in sections.values() for k in ks]
        if sorted(sec_leaves) != sorted(order):
            issues.append(f"article {aid}: sections don't cover exactly the leaves")
        for sec, ks in sections.items():
            if not ks:
                issues.append(f"article {aid}: empty section {sec!r}")
    return issues


def stats(tree: dict) -> dict:
    n_art = len(tree)
    leaves = sum(len(a["order"]) for a in tree.values())
    secs = sum(len(a["sections"]) for a in tree.values())
    lead = sum(len(a["sections"].get(LEAD, [])) for a in tree.values())
    with_sec = sum(1 for a in tree.values() if a["has_sections"])
    return {
        "articles": n_art, "leaves": leaves,
        "sections_total": secs, "sections_per_article": round(secs / n_art, 2) if n_art else 0,
        "articles_with_sections": with_sec, "lead_only_leaves": lead,
    }
