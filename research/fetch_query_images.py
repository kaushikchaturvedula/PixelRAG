#!/usr/bin/env python3
"""Resolve EncyclopedicVQA query images for the gold set (for IMAGE-query retrieval).

EVQA gives each question a query photo via ``dataset_image_ids``. For the *landmarks* subset
(what build_mini_corpus.py selects) those are Google-Landmarks-v2 (GLDv2) ids. This script
mirrors eval/lib/retrieval.py's landmark resolution with stdlib only:

  1. (optional) look for a locally-staged GLDv2 image at {data}/{split}/{a}/{b}/{c}/{id}.jpg;
  2. else read GLDv2 ``train.csv`` (id -> Flickr URL; ~525MB) and download the photo.

Per question it tries each candidate id in order and keeps the first that downloads a valid
image (>= --min-bytes). Many GLDv2 Flickr URLs are dead (404) — that's expected; questions with
no fetchable image are recorded as "missing" and run_eval.py falls back to a text query for them.

Outputs (under --out-dir, default <gold-dir>/query_images):
  {qid}.jpg            one query photo per resolved question
  manifest.json        {summary:{...}, items:{qid:{status, img_id, source_url}}}

Disclosed download: GLDv2 ``train.csv`` is ~525MB (s3.amazonaws.com/google-landmark/metadata).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

GLDV2_TRAIN_CSV_URL = "https://s3.amazonaws.com/google-landmark/metadata/train.csv"
# iNaturalist EVQA query images: EVQA's dataset_image_ids are iNat2021 COMPETITION image ids
# (val.json maps id -> file_name), NOT iNaturalist open-data photo_ids. There is NO per-image URL
# for the competition set, so correct images require the iNat2021 val image tarball (val.tar.gz,
# ~8.93GB) extracted under --inat-data-dir. (Using inaturalist-open-data S3 with these ids returns
# WRONG, unrelated photos — a different id space — so it is intentionally NOT used.)
INAT2021_VAL_JSON_TARGZ = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz"
# GLDv2 landmark URLs resolve to upload.wikimedia.org, which 429-throttles generic bot UAs and
# requires a descriptive User-Agent with contact info (Wikimedia UA policy). Be polite.
USER_AGENT = "PixelRAG-research/0.1 (https://github.com/StarTrail-org/PixelRAG; research eval) urllib"

_inat_map_cache: dict[int, str] | None = None


def _load_inat2021_map(inat_dir: Path) -> dict[int, str]:
    """iNat2021 competition image_id -> file_name (val split). Auto-downloads val.json (~9.8MB)."""
    global _inat_map_cache
    if _inat_map_cache is not None:
        return _inat_map_cache
    import tarfile
    inat_dir.mkdir(parents=True, exist_ok=True)
    val_json = inat_dir / "val.json"
    if not val_json.exists():
        tar = inat_dir / "val.json.tar.gz"
        if not tar.exists():
            print(f"[fetch] downloading iNat2021 val.json mapping (~9.8MB) from {INAT2021_VAL_JSON_TARGZ}", flush=True)
            urllib.request.urlretrieve(INAT2021_VAL_JSON_TARGZ, tar)
        with tarfile.open(tar, "r:gz") as tf:
            tf.extractall(inat_dir)
    data = json.loads(val_json.read_text())
    _inat_map_cache = {img["id"]: img["file_name"] for img in data["images"]}
    return _inat_map_cache


INAT_API_TAXA_URL = "https://api.inaturalist.org/v1/taxa"


def _inat_sci_name(file_name: str) -> str | None:
    """'Genus species' from an iNat2021 file_name dir: val/<id>_<k>_<p>_<c>_<o>_<f>_<genus>_<species>/..."""
    try:
        toks = file_name.split("/")[1].split("_")
        return " ".join(toks[-2:]) if len(toks) >= 2 else None
    except Exception:
        return None


def _inat_api_default_photo(sci_name: str, timeout: float) -> tuple[str | None, str | None]:
    """iNaturalist API → a representative (default) photo URL for a species. (url, matched_name)."""
    import urllib.parse
    q = urllib.parse.urlencode({"q": sci_name, "rank": "species", "per_page": 1})
    try:
        req = urllib.request.Request(f"{INAT_API_TAXA_URL}?{q}", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        r = (data.get("results") or [{}])[0]
        dp = r.get("default_photo") or {}
        return (dp.get("medium_url") or dp.get("url"), r.get("name"))
    except Exception:
        return (None, None)


def _load_gold(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _local_gldv2_path(data_dir: Path, img_id: str) -> Path | None:
    """GLDv2 on-disk layout: {split}/{a}/{b}/{c}/{id}.jpg (a,b,c = first 3 id chars)."""
    if len(img_id) < 3:
        return None
    sub = f"{img_id[0]}/{img_id[1]}/{img_id[2]}/{img_id}.jpg"
    for split in ("train", "index", "test"):
        p = data_dir / split / sub
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _build_url_map(train_csv: Path, needed: set[str]) -> dict[str, str]:
    """Single low-memory pass over train.csv collecting URLs only for the ids we need."""
    csv.field_size_limit(10_000_000)
    out: dict[str, str] = {}
    with open(train_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            i = (row.get("id") or "").strip()
            if i in needed:
                out[i] = (row.get("url") or "").strip()
                if len(out) == len(needed):
                    break
    return out


def _download(url: str, dst: Path, timeout: float, min_bytes: int, retries: int = 3) -> bool:
    """Download with polite 429/503 backoff (honors Retry-After). Returns True on a valid image."""
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if len(data) >= min_bytes:
                dst.write_bytes(data)
                return True
            return False  # too small (placeholder / error body)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < retries:
                ra = e.headers.get("Retry-After")
                wait = float(ra) if (ra and str(ra).isdigit()) else delay
                time.sleep(min(wait, 30.0))
                delay *= 2
                continue
            return False
        except Exception:
            return False
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch EVQA (landmarks) query images for the gold set.")
    ap.add_argument("--gold", required=True, help="gold.jsonl from build_mini_corpus.py.")
    ap.add_argument("--out-dir", default=None, help="Default: <gold-dir>/query_images.")
    ap.add_argument("--landmark-train-csv", default=None,
                    help="Path to GLDv2 train.csv (id->url). Required unless --landmark-data-dir resolves all.")
    ap.add_argument("--landmark-data-dir", default=None,
                    help="Optional locally-staged GLDv2 image tree ({split}/{a}/{b}/{c}/{id}.jpg).")
    ap.add_argument("--inat-data-dir", default=None,
                    help="Dir with iNat2021 val.json + (for --inat-source val-tar) extracted val "
                         "images. Default: <gold-dir>/cache/inat2021. val.json (~9.8MB) auto-downloads.")
    ap.add_argument("--inat-source", default="api", choices=["api", "val-tar"],
                    help="iNaturalist query-image source. 'api' (default): a representative photo per "
                         "species via the iNaturalist API (leak-free, no big download, NOT EVQA-exact). "
                         "'val-tar': the exact EVQA photo from the extracted iNat2021 val.tar.gz (8.93GB).")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--min-bytes", type=int, default=1000)
    args = ap.parse_args()

    gold_path = Path(args.gold).resolve()
    gold = _load_gold(gold_path)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else gold_path.parent / "query_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.landmark_data_dir).resolve() if args.landmark_data_dir else None
    inat_dir = (Path(args.inat_data_dir).resolve() if args.inat_data_dir
                else gold_path.parent / "cache" / "inat2021")

    # Which ids might we need a URL for? (landmark questions not already on local disk)
    needed: set[str] = set()
    for q in gold:
        if (q.get("dataset_name") or "").lower() != "landmarks":
            continue
        for img_id in q.get("dataset_image_ids", []):
            if not (data_dir and _local_gldv2_path(data_dir, img_id)):
                needed.add(img_id)
    url_map: dict[str, str] = {}
    if needed:
        if not args.landmark_train_csv or not Path(args.landmark_train_csv).exists():
            print(f"[fetch] WARNING: {len(needed)} ids need URLs but --landmark-train-csv is missing; "
                  f"download it (~525MB): {GLDV2_TRAIN_CSV_URL}")
        else:
            print(f"[fetch] scanning {args.landmark_train_csv} for {len(needed)} ids ...", flush=True)
            url_map = _build_url_map(Path(args.landmark_train_csv), needed)
            print(f"[fetch] resolved {len(url_map)}/{len(needed)} ids to URLs", flush=True)

    items: dict[str, dict] = {}
    n_ok = n_missing = n_unsupported = 0
    for q in gold:
        qid = q["id"]
        ds = (q.get("dataset_name") or "").lower()
        img_ids = q.get("dataset_image_ids", [])
        dst = out_dir / f"{qid}.jpg"
        if dst.exists() and dst.stat().st_size >= args.min_bytes:
            items[qid] = {"status": "ok", "img_id": "cached", "source_url": None}
            n_ok += 1
            continue

        resolved = None
        if ds == "inaturalist":
            # EVQA ids are iNat2021 competition ids (val.json: id -> file_name). NO open-data S3
            # (wrong id space → wrong photos). Two sources, selectable via --inat-source.
            id_map = _load_inat2021_map(inat_dir)
            sci = None
            mapped = False
            for img_id in img_ids:
                if img_id.isdigit() and int(img_id) in id_map:
                    mapped = True
                    sci = sci or _inat_sci_name(id_map[int(img_id)])
            sci = sci or (q.get("gold_title") or "").strip() or None

            if args.inat_source == "api":
                # Representative photo for the SPECIES via the iNat API (leak-free, not EVQA-exact).
                purl, matched = _inat_api_default_photo(sci, args.timeout) if sci else (None, None)
                time.sleep(0.4)  # polite to api.inaturalist.org
                if purl and _download(purl, dst, args.timeout, args.min_bytes):
                    resolved = {"status": "ok", "img_id": "inat_api", "source_url": purl,
                                "species_query": sci, "matched_taxon": matched,
                                "note": "iNat API representative photo (same species, NOT EVQA-exact)"}
                else:
                    items[qid] = {"status": "missing", "reason": "inat_api_no_photo",
                                  "species_query": sci, "tried": img_ids}
                    n_missing += 1
                    continue
            else:  # val-tar: the exact EVQA photo from the extracted competition val images
                for img_id in img_ids:
                    if not img_id.isdigit():
                        continue
                    fn = id_map.get(int(img_id))
                    if not fn:
                        continue
                    src = inat_dir / fn
                    if src.exists() and src.stat().st_size > 0:
                        dst.write_bytes(src.read_bytes())
                        resolved = {"status": "ok", "img_id": img_id, "source_url": f"file://{src}"}
                        break
                if not resolved:
                    items[qid] = {"status": "missing",
                                  "reason": ("image_absent_needs_inat2021_val_tar_8.93GB" if mapped
                                             else "id_not_in_inat2021_val_json"),
                                  "tried": img_ids}
                    n_missing += 1
                    continue
        elif ds == "landmarks":
            for img_id in img_ids:
                local = _local_gldv2_path(data_dir, img_id) if data_dir else None
                if local:
                    dst.write_bytes(local.read_bytes())
                    resolved = {"status": "ok", "img_id": img_id, "source_url": f"file://{local}"}
                    break
                url = url_map.get(img_id)
                if url and _download(url, dst, args.timeout, args.min_bytes):
                    resolved = {"status": "ok", "img_id": img_id, "source_url": url}
                    break
                time.sleep(0.4)  # polite spacing between Wikimedia (Commons) requests
        else:
            items[qid] = {"status": "unsupported", "note": f"dataset_name={ds!r}: no URL resolver"}
            n_unsupported += 1
            continue

        if resolved:
            items[qid] = resolved
            n_ok += 1
        else:
            items[qid] = {"status": "missing", "img_id": None, "source_url": None, "tried": img_ids}
            n_missing += 1

    # "potential" = resolvable in principle (id maps in val.json) but image bytes not yet on disk —
    # i.e. coverage we'd get once the iNat2021 val.tar.gz (8.93GB) is extracted under --inat-data-dir.
    n_potential = n_ok + sum(
        1 for v in items.values()
        if v.get("status") == "missing" and str(v.get("reason", "")).startswith("image_absent")
    )
    manifest = {
        "summary": {"n_total": len(gold), "n_ok": n_ok, "n_missing": n_missing,
                    "n_unsupported": n_unsupported, "n_potential_with_val_images": n_potential},
        "gldv2_train_csv": args.landmark_train_csv,
        "inat_data_dir": str(inat_dir),
        "items": items,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n[fetch] query images: {n_ok}/{len(gold)} on disk, {n_missing} missing, "
          f"{n_unsupported} unsupported → {out_dir}")
    if n_potential > n_ok:
        print(f"[fetch] potential coverage WITH iNat2021 val images: {n_potential}/{len(gold)} "
              f"(ids map in val.json; need val.tar.gz ~8.93GB extracted under {inat_dir}).")
    print(f"[fetch] {n_missing} questions would fall back to TEXT queries in run_eval.py.")


if __name__ == "__main__":
    main()
