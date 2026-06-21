"""Parse pixelrag.yaml with parameter forwarding."""

import os
from pathlib import Path

import yaml

from .sources import SOURCES

DEFAULT_CONFIG = {
    "ingest": {"backend": "cdp", "quality": 85, "tile_height": 8192},
    # Chunking strategy for stage 2. "fixed" = equal-height 1024px strips (baseline);
    # "content_aware" = cut at DOM block boundaries. region_source: dom (regions.json) | vision (stub).
    "chunk": {"chunker": "fixed", "region_source": "dom"},
    "embed": {"model": "Qwen/Qwen3-VL-Embedding-2B", "device": "cuda"},
    "output": "./index",
}


def load_config(path=None):
    if path is None:
        for c in [Path("pixelrag.yaml"), Path("pixelrag.yml")]:
            if c.exists():
                path = str(c)
                break
    if path and os.path.exists(path):
        with open(path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
    return {**DEFAULT_CONFIG, **config}


def make_source(config):
    source_config = dict(config.get("source", {}))
    source_type = source_config.pop("type", "local")
    # Expand ~ in any string values that look like paths
    for k, v in source_config.items():
        if isinstance(v, str) and ("/" in v or "~" in v):
            source_config[k] = str(Path(v).expanduser())
    return SOURCES[source_type](**source_config)
