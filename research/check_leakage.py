#!/usr/bin/env python3
"""Leakage check: confirm iNaturalist query photos are NOT (near-)duplicates of any image rendered
on their own gold Wikipedia page. Source independence is the whole point of the iNaturalist switch.

For each gold question's query image, perceptual-hash it (aHash + dHash, 64-bit) and compare against
every image on the gold Wikipedia article (fetched via the MediaWiki API). A small Hamming distance
to ANY article image means the query photo likely also appears on the page (leakage). We report the
closest match per page; a clean run has all minimum distances comfortably above the threshold.

Needs only Pillow + stdlib. Network: ~1 API call + a handful of thumbnail downloads per gold page.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

UA = "PixelRAG-research/0.1 (https://github.com/StarTrail-org/PixelRAG; leakage check) urllib"
WIKI_API = "https://en.wikipedia.org/w/api.php"


def _ahash(img: Image.Image) -> int:
    px = list(img.convert("L").resize((8, 8), Image.BILINEAR).getdata())
    avg = sum(px) / 64
    bits = 0
    for i, v in enumerate(px):
        if v >= avg:
            bits |= 1 << i
    return bits


def _dhash(img: Image.Image) -> int:
    px = list(img.convert("L").resize((9, 8), Image.BILINEAR).getdata())
    bits = 0
    k = 0
    for r in range(8):
        for c in range(8):
            if px[r * 9 + c] < px[r * 9 + c + 1]:
                bits |= 1 << k
            k += 1
    return bits


def _ham(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _hashes(path_or_bytes) -> tuple[int, int] | None:
    try:
        import io
        img = Image.open(path_or_bytes if isinstance(path_or_bytes, (str, Path)) else io.BytesIO(path_or_bytes))
        img.load()
        return _ahash(img), _dhash(img)
    except Exception:
        return None


def _wiki_image_thumbs(title: str, limit: int, width: int, timeout: float) -> list[str]:
    params = {
        "action": "query", "format": "json", "titles": title,
        "generator": "images", "gimlimit": str(limit),
        "prop": "imageinfo", "iiprop": "url", "iiurlwidth": str(width),
    }
    url = f"{WIKI_API}?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        data = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except Exception:
        return []
    out = []
    for p in (data.get("query", {}).get("pages", {}) or {}).values():
        for ii in p.get("imageinfo", []) or []:
            t = ii.get("thumburl") or ii.get("url")
            if t and not t.lower().endswith(".svg"):
                out.append(t)
    return out


def _download_bytes(url: str, timeout: float) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        return urllib.request.urlopen(req, timeout=timeout).read()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="iNat query-image vs gold-page leakage check.")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--query-images-dir", default=None, help="Default: <gold-dir>/query_images.")
    ap.add_argument("--threshold", type=int, default=10, help="Hamming bits ≤ this ⇒ likely same image.")
    ap.add_argument("--max-images-per-page", type=int, default=15)
    ap.add_argument("--thumb-width", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=12.0)
    args = ap.parse_args()

    gold_path = Path(args.gold).resolve()
    qdir = Path(args.query_images_dir).resolve() if args.query_images_dir else gold_path.parent / "query_images"
    gold = [json.loads(l) for l in gold_path.read_text().splitlines() if l.strip()]

    results = []
    leaks = 0
    checked = 0
    for q in gold:
        qid = q["id"]
        page = q["gold_url"].split("/wiki/")[-1]
        title = urllib.parse.unquote(page)
        qimg = qdir / f"{qid}.jpg"
        if not (qimg.exists() and qimg.stat().st_size > 0):
            results.append({"qid": qid, "page": page, "status": "no_query_image"})
            continue
        qh = _hashes(qimg)
        if qh is None:
            results.append({"qid": qid, "page": page, "status": "unhashable_query"})
            continue

        thumbs = _wiki_image_thumbs(title, args.max_images_per_page, args.thumb_width, args.timeout)
        best = 64
        n_img = 0
        for turl in thumbs:
            raw = _download_bytes(turl, args.timeout)
            if not raw:
                continue
            ch = _hashes(raw)
            if ch is None:
                continue
            n_img += 1
            best = min(best, _ham(qh[0], ch[0]), _ham(qh[1], ch[1]))
            time.sleep(0.05)
        checked += 1
        leak = best <= args.threshold
        leaks += int(leak)
        results.append({"qid": qid, "page": page, "n_page_images": n_img,
                        "min_hamming": best, "leak": leak})
        print(f"  {page:42} page_imgs={n_img:2} min_hamming={best:2} {'⚠ LEAK' if leak else 'ok'}")
        time.sleep(0.1)

    out = {"threshold": args.threshold, "n_checked": checked, "n_leaks": leaks, "results": results}
    (qdir / "leakage.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[leakage] checked {checked} pages; potential leaks (min_hamming ≤ {args.threshold}): {leaks}")
    print(f"[leakage] {'PASS — query photos are independent of gold-page images' if leaks == 0 else 'REVIEW the flagged pages'}  → {qdir/'leakage.json'}")
    return 1 if leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
