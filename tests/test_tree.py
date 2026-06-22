#!/usr/bin/env python3
"""Unit tests for the hier-expand tree (research/tree.py). Stdlib only.

Run: PYTHONPATH=research python tests/test_tree.py   (or via pytest)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))
import tree as T  # noqa: E402


def _mk(tmp: Path, aid: int, chunks: list[dict]):
    d = tmp / f"{aid}.png.tiles"
    d.mkdir(parents=True)
    (d / "chunks.json").write_text(json.dumps({"chunks": chunks}))


def _ca_chunks():
    # content_aware-style: heading_path + reading_order
    return [
        {"tile_index": 0, "chunk_index": 0, "heading_path": "", "reading_order": 0},
        {"tile_index": 0, "chunk_index": 1, "heading_path": "Sec A", "reading_order": 1},
        {"tile_index": 0, "chunk_index": 2, "heading_path": "Sec A", "reading_order": 2},
        {"tile_index": 0, "chunk_index": 3, "heading_path": "Sec B", "reading_order": 3},
        {"tile_index": 1, "chunk_index": 0, "heading_path": "Sec B", "reading_order": 4},
    ]


def _fixed_chunks():
    # fixed-style: no heading_path / reading_order → reading order from (tile_index, chunk_index)
    return [{"tile_index": 0, "chunk_index": i} for i in range(5)]


def test_build_and_validate():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _mk(tmp, 0, _ca_chunks())
        _mk(tmp, 1, _fixed_chunks())
        tree = T.build_tree(str(tmp))
        assert T.validate(tree) == [], T.validate(tree)
        # content_aware article 0: sections (lead) + Sec A + Sec B; has_sections True
        a0 = tree[0]
        assert a0["has_sections"] and set(a0["sections"]) == {T.LEAD, "Sec A", "Sec B"}
        assert a0["sections"]["Sec A"] == [(0, 1), (0, 2)]
        assert len(a0["order"]) == 5
        # fixed article 1: no sections, all under (lead), order by (tile,chunk)
        a1 = tree[1]
        assert not a1["has_sections"] and set(a1["sections"]) == {T.LEAD}
        assert a1["order"] == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        print("  build + validate OK (content_aware sections; fixed neighbors-only)")


def test_expand_section_mode():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); _mk(tmp, 0, _ca_chunks())
        tree = T.build_tree(str(tmp))
        # seed = the first chunk of Sec A; section+neighbors should pull Sec A sibling + ro-neighbors
        # seed = first chunk of Sec A (reading-order pos 1). section sibling = (0,0,2);
        # ±1 reading-order neighbors = pos 0 (0,0,0) and pos 2 (0,0,2). (0,0,3) is distance-2 → excluded.
        exp = T.expand([(0, 0, 1)], tree, mode="section+neighbors", neighbors=1, cap=8)
        assert exp[0] == (0, 0, 1), "seed must come first"
        assert (0, 0, 2) in exp, "same-section sibling missing"
        assert (0, 0, 0) in exp, "reading-order neighbor (pos-1) missing"
        assert (0, 0, 3) not in exp, "distance-2 leaf should not appear at neighbors=1"
        # with neighbors=2 the distance-2 leaf (0,0,3) appears
        exp2 = T.expand([(0, 0, 1)], tree, mode="section+neighbors", neighbors=2, cap=8)
        assert (0, 0, 3) in exp2, "neighbors=2 should reach pos+2"
        assert len(exp) == len(set(exp)), "dedup failed"
        print(f"  section+neighbors expand OK -> {exp}")


def test_expand_neighbors_only_fixed():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); _mk(tmp, 1, _fixed_chunks())
        tree = T.build_tree(str(tmp))
        exp = T.expand([(1, 0, 2)], tree, mode="neighbors", neighbors=1, cap=8)
        assert exp == [(1, 0, 2), (1, 0, 1), (1, 0, 3)] or set(exp) == {(1, 0, 2), (1, 0, 1), (1, 0, 3)}
        assert exp[0] == (1, 0, 2)
        print(f"  neighbors-only expand (fixed) OK -> {exp}")


def test_expand_cap_keeps_seeds():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); _mk(tmp, 0, _ca_chunks())
        tree = T.build_tree(str(tmp))
        seeds = [(0, 0, 1), (0, 0, 3), (0, 1, 0)]
        exp = T.expand(seeds, tree, mode="section+neighbors", neighbors=2, cap=2)
        # cap < expansion, but never below seed count, and seeds preserved & first
        assert exp[:len(seeds)] == seeds, "seeds must be preserved and first even under tight cap"
        assert len(exp) == len(seeds), f"cap should clamp to seed count, got {len(exp)}"
        print(f"  cap keeps seeds OK -> {exp}")


if __name__ == "__main__":
    test_build_and_validate()
    test_expand_section_mode()
    test_expand_neighbors_only_fixed()
    test_expand_cap_keeps_seeds()
    print("ALL tree tests PASSED")
