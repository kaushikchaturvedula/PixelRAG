#!/usr/bin/env python3
"""Build + validate the page→section→tile tree from a chunked tiles dir; dump tree.json + report.

    python research/build_tree.py --tiles-dir research/mini_corpus/output_ca/tiles
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tree as tree_mod  # research/ on sys.path, or run from research/


def _serialize(tree: dict) -> dict:
    out = {}
    for aid, a in tree.items():
        out[str(aid)] = {
            "order": [list(k) for k in a["order"]],
            "sections": {sec: [list(k) for k in ks] for sec, ks in a["sections"].items()},
            "leaf_to_section": {f"{k[0]}_{k[1]}": v for k, v in a["leaf_to_section"].items()},
            "has_sections": a["has_sections"],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build + validate the hier-expand tree.")
    ap.add_argument("--tiles-dir", required=True)
    ap.add_argument("--out", default=None, help="Write tree.json (default: <tiles-dir>/../tree.json).")
    args = ap.parse_args()

    t = tree_mod.build_tree(args.tiles_dir)
    issues = tree_mod.validate(t)
    s = tree_mod.stats(t)
    print("tree stats:", json.dumps(s, indent=2))
    if issues:
        print(f"VALIDATION FAILED ({len(issues)} issues):")
        for i in issues[:20]:
            print("  -", i)
        return 1
    print("validation: OK")
    out = Path(args.out) if args.out else Path(args.tiles_dir).parent / "tree.json"
    out.write_text(json.dumps(_serialize(t)))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
