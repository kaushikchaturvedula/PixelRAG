#!/usr/bin/env python3
"""px-reconciliation sanity check on REAL rendered pages (not synthetic).

Confirms the DOM region coordinate space lines up with the rendered tile pixels before we trust the
content-aware cuts. For each checked page it reports devicePixelRatio and three alignment facts, and
FAILS LOUDLY if they drift:
  * dpr ~= 1 (renderer uses deviceScaleFactor=1, so CSS px == tile px),
  * DOM page_height (regions.json) ~= rendered page_height (tiles.json) — navigate-only layout matched
    the layout the tiles were rendered at,
  * max region bottom (dpr-scaled) fits within the total rendered tile pixel height.

Stdlib + Pillow. Usage:
    PYTHONPATH=embed/src python research/px_sanity.py --tiles-dir research/mini_corpus/output/tiles --n 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def _check(art: Path, tol_frac: float) -> dict:
    regions = json.loads((art / "regions.json").read_text())
    tiles = json.loads((art / "tiles.json").read_text())
    dpr = float(regions.get("device_pixel_ratio", 1) or 1)
    dom_ph = float(regions.get("page_height", 0))
    render_ph = float(tiles.get("page_height", 0))
    total_tile_px = 0
    for t in tiles.get("tiles", []):
        tp = art / t
        if tp.exists():
            total_tile_px += Image.open(tp).size[1]
    rs = regions.get("regions", [])
    max_bottom = max(((r["y"] + r["height"]) * dpr for r in rs), default=0.0)
    n_regions = len(rs)
    n_tables = sum(1 for r in rs if r.get("kind") == "table")

    ph_ok = render_ph > 0 and abs(dom_ph - render_ph) <= tol_frac * render_ph
    fit_ok = max_bottom <= total_tile_px + 0.02 * max(total_tile_px, 1)
    dpr_ok = abs(dpr - 1.0) < 0.01
    return {
        "page": art.name, "dpr": dpr, "dpr_ok": dpr_ok,
        "dom_page_height": dom_ph, "render_page_height": render_ph, "page_height_match": ph_ok,
        "max_region_bottom_px": round(max_bottom, 1), "total_tile_px": total_tile_px, "fits_tile": fit_ok,
        "n_regions": n_regions, "n_tables": n_tables,
        "PASS": bool(dpr_ok and ph_ok and fit_ok),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="px-reconciliation sanity check on real pages.")
    ap.add_argument("--tiles-dir", required=True)
    ap.add_argument("--n", type=int, default=2, help="How many pages with regions.json to check.")
    ap.add_argument("--tolerance", type=float, default=0.05, help="Allowed page_height drift fraction.")
    args = ap.parse_args()

    arts = [a for a in sorted(Path(args.tiles_dir).glob("*.png.tiles")) if (a / "regions.json").exists()]
    if not arts:
        print("no pages with regions.json found")
        return 1
    ok = True
    for art in arts[: args.n]:
        r = _check(art, args.tolerance)
        ok = ok and r["PASS"]
        print(f"  {r['page']}: dpr={r['dpr']} (ok={r['dpr_ok']})  "
              f"page_height dom={r['dom_page_height']:.0f} render={r['render_page_height']:.0f} "
              f"match={r['page_height_match']}  region_bottom={r['max_region_bottom_px']:.0f} "
              f"<= tile_px={r['total_tile_px']} fit={r['fits_tile']}  "
              f"regions={r['n_regions']} tables={r['n_tables']}  -> {'PASS' if r['PASS'] else 'FAIL'}")
    print(f"\npx-sanity: {'ALL PASS' if ok else 'FAIL — coordinates drift, do not trust cuts'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
