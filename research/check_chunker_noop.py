#!/usr/bin/env python3
"""HARD REGRESSION GATE: `pixelrag chunk --chunker fixed` must be byte-identical to unflagged.

Runs the chunker twice over the SAME tiles — once with no flag (legacy path) and once with
`--chunker fixed` — into two scratch dirs, then asserts every produced chunk_*.png is
byte-for-byte identical and chunks.json is structurally identical. Exits non-zero on any diff.

`--chunker fixed` is a no-op by construction (the flag is only read by an assert; the
output-producing functions are unchanged), but this gate proves it empirically and guards
against regressions when the content-aware path is added in Phase 2.

Usage:
    # On the real rendered mini-corpus (preferred — run this BEFORE run_eval.py):
    python research/check_chunker_noop.py --tiles-dir research/mini_corpus/output/tiles
    # Env-light synthetic smoke test (needs only Pillow):
    python research/check_chunker_noop.py --synthetic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _make_synthetic(dst: Path) -> None:
    """Fabricate one *.png.tiles dir with two tall tiles + tiles.json (needs Pillow)."""
    from PIL import Image  # local import so the file imports without Pillow

    art = dst / "0.png.tiles"
    art.mkdir(parents=True, exist_ok=True)
    names = []
    for i, h in enumerate((3000, 1500)):  # both exceed CHUNK_HEIGHT=1024 → real splitting
        img = Image.new("RGB", (875, h))
        # deterministic, non-uniform content so crops are meaningful
        for y in range(0, h, 4):
            for x in range(0, 875, 64):
                img.putpixel((x, y), ((x + y) % 256, (y * 2) % 256, (x * 3) % 256))
        name = f"tile_{i:04d}.jpg"
        img.save(art / name, "JPEG", quality=85)
        names.append(name)
    (art / "tiles.json").write_text(json.dumps(
        {"url": "synthetic://t", "page_height": 4500, "viewport_width": 875, "tiles": names}
    ))


def _run_chunk(shard_dir: Path, extra: list[str]) -> None:
    subprocess.run(
        [sys.executable, "-m", "pixelrag_embed.chunk",
         "--shard-dir", str(shard_dir), "--workers", "1", *extra],
        check=True,
    )


def _collect_chunks(shard_dir: Path) -> dict[str, str]:
    """Map chunk filename -> sha256, across every *.png.tiles dir in shard_dir."""
    out: dict[str, str] = {}
    for art in sorted(shard_dir.glob("*.png.tiles")):
        for c in sorted(art.glob("chunk_*")):
            out[f"{art.name}/{c.name}"] = _sha(c)
    return out


def _collect_manifests(shard_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for art in sorted(shard_dir.glob("*.png.tiles")):
        cj = art / "chunks.json"
        if cj.exists():
            out[art.name] = json.loads(cj.read_text())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Byte-identical regression gate for --chunker fixed.")
    ap.add_argument("--tiles-dir", default=None, help="Existing tiles dir with *.png.tiles/.")
    ap.add_argument("--synthetic", action="store_true", help="Fabricate synthetic tiles (needs Pillow).")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="chunker_noop_"))
    try:
        src = work / "src"
        src.mkdir()
        if args.tiles_dir and not args.synthetic:
            for art in Path(args.tiles_dir).glob("*.png.tiles"):
                dst = src / art.name
                dst.mkdir(parents=True, exist_ok=True)
                for f in art.iterdir():
                    if f.name.startswith("tile_") or f.name == "tiles.json":
                        shutil.copy2(f, dst / f.name)
            if not any(src.glob("*.png.tiles")):
                print(f"FAIL: no *.png.tiles with tiles in {args.tiles_dir}")
                return 1
        else:
            _make_synthetic(src)

        # two independent copies → chunk each a different way
        a, b = work / "unflagged", work / "flagged"
        shutil.copytree(src, a)
        shutil.copytree(src, b)
        _run_chunk(a, [])
        _run_chunk(b, ["--chunker", "fixed"])

        ca, cb = _collect_chunks(a), _collect_chunks(b)
        ma, mb = _collect_manifests(a), _collect_manifests(b)

        ok = True
        if set(ca) != set(cb):
            ok = False
            print(f"FAIL: chunk file sets differ: only-unflagged={set(ca)-set(cb)} only-flagged={set(cb)-set(ca)}")
        diffs = [name for name in ca if name in cb and ca[name] != cb[name]]
        if diffs:
            ok = False
            print(f"FAIL: {len(diffs)} chunk(s) differ in bytes: {diffs[:5]}")
        if ma != mb:
            ok = False
            print("FAIL: chunks.json manifests differ")

        n_chunks = len(ca)
        if ok:
            print(f"PASS: --chunker fixed is byte-identical to unflagged "
                  f"({n_chunks} chunks across {len(ma)} page(s) compared).")
            return 0
        return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
