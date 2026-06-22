#!/usr/bin/env python3
"""Swappable hosted VLM reader — pixel-native QA over retrieved chunk tiles.

`read(question, image_paths, provider, model) -> (answer, usage)` sends the retrieved chunk
**PNGs** (never parsed text — PixelRAG stays pixel-native) plus the question to a hosted VLM and
returns its answer. Providers behind `--reader {openai,claude,gemini,qwen,mock}`, default
**openai/gpt-4o**. API keys come **only from the environment** (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`/`GOOGLE_API_KEY`, `QWEN_API_KEY`+`QWEN_BASE_URL`) — never
hardcoded. The `mock` provider needs no key and never reads the image files, so the whole
qa_eval pipeline can be validated offline with zero API spend.

openai+qwen go through the `openai` SDK (chat.completions vision, data-URL images); claude through
the `anthropic` SDK (base64 image blocks); gemini through `google-genai` (optional). SDKs are
imported lazily so a missing optional dep never blocks an unused provider (or the mock path).

Cost control (the big lever — a full-page tile at OpenAI's default `detail:"high"` is ~tens of
thousands of vision tokens):
- `detail` ∈ {low, high, auto}, **default low** — passed into the OpenAI/qwen `image_url` block.
  `low` caps each image at ~85 tokens. Anthropic/Gemini have no per-image detail param, so `detail`
  is ignored there; use `image_maxdim` to bound their cost.
- `image_maxdim` (default off) — PIL-downscale each tile to this long-side max BEFORE base64. For
  when `low` blurs dense tables but full resolution is overkill; applies to every provider.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

# Pixel-native: answer ONLY from the screenshots. Mirrors eval/lib/llm.py SYSTEM_PROMPT_EVIDENCE_QA,
# tuned for retrieved image tiles. Kept provider-agnostic (sent as `system` for claude, as a system
# message for openai/qwen, prepended for gemini).
SYSTEM_PROMPT = (
    "You are a research assistant who answers questions from provided screenshot tiles of a "
    "Wikipedia page. Read the answer directly off the images. Answer ONLY from the screenshots; "
    "do not use outside knowledge. Give a short, direct answer (a few words). If the screenshots "
    "do not contain the answer, say you cannot find it."
)

DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "qwen": "qwen-vl-max",
    "claude": "claude-opus-4-8",
    "gemini": "gemini-2.0-flash",
}

PROVIDERS = ("openai", "qwen", "claude", "gemini", "mock")


def _b64(path: str, maxdim: int | None = None) -> str:
    """base64-encode a PNG, optionally PIL-downscaled so its long side <= maxdim (re-encoded PNG).

    Resizes (and re-encodes) only when the image is larger than maxdim; otherwise the original
    bytes are returned untouched. PIL is imported lazily so it's only required when maxdim is set.
    """
    if maxdim:
        import io

        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
            if max(w, h) > maxdim:
                scale = maxdim / max(w, h)
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))))
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                return base64.standard_b64encode(buf.getvalue()).decode()
    return base64.standard_b64encode(Path(path).read_bytes()).decode()


def read(
    question: str,
    image_paths: list[str],
    provider: str = "openai",
    model: str | None = None,
    max_tokens: int = 512,
    system: str = SYSTEM_PROMPT,
    detail: str = "low",
    image_maxdim: int | None = None,
) -> tuple[str, dict]:
    """Answer `question` from the chunk PNGs at `image_paths`. Returns (answer, usage_dict).

    `detail` (low|high|auto) is the OpenAI/qwen per-image token control (default low, ~85 tok/image;
    ignored by claude/gemini — no equivalent). `image_maxdim` PIL-downscales every tile's long side
    before encoding (default off; the cost lever for claude/gemini). usage_dict has prompt_tokens /
    completion_tokens / total_tokens (0s for mock). Raises ValueError on an unknown provider; lets
    SDK/IO errors propagate (caller decides retry).
    """
    provider = (provider or "openai").lower()
    if provider == "mock":
        return _read_mock(question, image_paths)
    model = model or DEFAULT_MODELS.get(provider)
    if provider in ("openai", "qwen"):
        return _read_openai(question, image_paths, provider, model, max_tokens, system,
                            detail, image_maxdim)
    if provider == "claude":
        return _read_claude(question, image_paths, model, max_tokens, system, image_maxdim)
    if provider == "gemini":
        return _read_gemini(question, image_paths, model, max_tokens, system, image_maxdim)
    raise ValueError(f"unknown reader provider: {provider!r} (choose from {PROVIDERS})")


def _read_mock(question: str, image_paths: list[str]) -> tuple[str, dict]:
    """Offline stand-in: deterministic, never touches the API or the image files.

    Returns a canned answer that encodes how many tiles were routed to the reader, so a qa_eval
    --reader mock dry-run confirms the flat/hier-expand tile selection without any spend.
    """
    answer = f"[mock answer | {len(image_paths)} tiles | q={question[:40]!r}]"
    return answer, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _read_openai(question, image_paths, provider, model, max_tokens, system,
                 detail="low", image_maxdim=None):
    from openai import OpenAI

    if provider == "qwen":
        base_url = os.environ.get("QWEN_BASE_URL")
        if not base_url:
            raise RuntimeError("QWEN_BASE_URL must be set in the environment for --reader qwen")
        client = OpenAI(api_key=os.environ.get("QWEN_API_KEY", "EMPTY"), base_url=base_url)
    else:
        client = OpenAI()  # OPENAI_API_KEY (+ optional OPENAI_BASE_URL) from env
    content = [{"type": "text", "text": question}]
    for p in image_paths:
        # `detail` is the per-image token cap: low ~85 tok, high = full multi-tile cost.
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{_b64(p, image_maxdim)}",
                    "detail": detail,
                },
            }
        )
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
    )
    u = resp.usage
    usage = (
        {
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        }
        if u
        else {}
    )
    return (resp.choices[0].message.content or ""), usage


def _read_claude(question, image_paths, model, max_tokens, system, image_maxdim=None):
    # Anthropic vision = base64 image blocks (per the claude-api skill). No temperature on Opus 4.8
    # (rejected); thinking omitted — a VQA reader wants a direct answer, not a reasoning trace.
    # No per-image `detail` param exists; bound cost with image_maxdim (downscale) instead.
    import anthropic

    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY from env
    content = [{"type": "text", "text": question}]
    for p in image_paths:
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png",
                           "data": _b64(p, image_maxdim)},
            }
        )
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    usage = {
        "prompt_tokens": resp.usage.input_tokens,
        "completion_tokens": resp.usage.output_tokens,
        "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
    }
    return text, usage


def _read_gemini(question, image_paths, model, max_tokens, system, image_maxdim=None):
    import base64 as _b

    from google import genai
    from google.genai.types import Blob, GenerateContentConfig, Content, Part

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY must be set for --reader gemini")
    client = genai.Client(api_key=api_key)
    parts = [Part(text=f"{system}\n\n{question}")]
    for p in image_paths:
        # no per-image detail param; image_maxdim downscales to bound cost (reuse _b64's resize)
        data = _b.b64decode(_b64(p, image_maxdim))
        parts.append(Part(inline_data=Blob(mime_type="image/png", data=data)))
    resp = client.models.generate_content(
        model=model,
        contents=[Content(role="user", parts=parts)],
        config=GenerateContentConfig(temperature=0, max_output_tokens=max_tokens),
    )
    text = resp.text if getattr(resp, "text", None) else ""
    usage = {}
    um = getattr(resp, "usage_metadata", None)
    if um:
        usage = {
            "prompt_tokens": getattr(um, "prompt_token_count", 0),
            "completion_tokens": getattr(um, "candidates_token_count", 0),
            "total_tokens": getattr(um, "total_token_count", 0),
        }
    return text, usage
