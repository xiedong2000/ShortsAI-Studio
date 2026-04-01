from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI


def _fallback_metadata(transcript: str) -> dict[str, Any]:
    t = transcript.strip()
    snippet = t.replace("\n", " ")[:120].strip()
    if len(snippet) < len(t):
        snippet = snippet.rsplit(" ", 1)[0] + "…"
    title = snippet[:70] if snippet else "My Short"
    desc = (t[:4000] + ("…" if len(t) > 4000 else "")) if t else ""
    return {
        "title": title or "My Short",
        "description": desc or "Created with ShortsAI-Studio.",
        "tags": ["shorts", "youtubeshorts", "verticalvideo"],
        "attribution": "",
        "notes": "Set OPENAI_API_KEY for GPT-generated title, description, and tags.",
    }


def _parse_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        raise ValueError("No JSON object in model output")
    return json.loads(m.group())


def generate_metadata(transcript: str, *, api_key: str | None) -> dict[str, Any]:
    if not api_key:
        return _fallback_metadata(transcript)

    client = OpenAI(api_key=api_key)
    prompt = (
        "You help with YouTube Shorts. Given the spoken transcript, return ONE JSON object only, "
        'with keys: title (string, under 70 chars, catchy), description (string, 1-3 short paragraphs, '
        "SEO-friendly, no hashtag spam), tags (array of 5-10 short strings, no # prefix), "
        'attribution (string, empty unless user provided music credit text). '
        "Do not invent facts not implied by the transcript.\n\nTranscript:\n"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Reply with valid JSON only. No markdown fences."},
            {"role": "user", "content": prompt + transcript[:12000]},
        ],
        temperature=0.6,
    )
    content = (resp.choices[0].message.content or "").strip()
    try:
        data = _parse_json_object(content)
    except (json.JSONDecodeError, ValueError):
        return _fallback_metadata(transcript)

    title = str(data.get("title", "")).strip() or "My Short"
    description = str(data.get("description", "")).strip()
    tags = data.get("tags")
    if not isinstance(tags, list):
        tags = []
    tags = [str(x).strip().lstrip("#") for x in tags if str(x).strip()][:15]
    attribution = str(data.get("attribution", "") or "").strip()

    return {
        "title": title[:100],
        "description": description[:5000],
        "tags": tags or ["shorts", "youtubeshorts"],
        "attribution": attribution,
        "notes": "",
    }


def generate_overlay_text_from_vision_segments(
    segment_frames: list[list[Path]],
    *,
    duration_sec: float,
    api_key: str | None,
    spoken_context: str = "",
) -> list[str]:
    """
    One caption per segment: images for each group come only from that part of the timeline,
    so each line matches the scene shown while that caption is on screen.
    """
    if not api_key or not segment_frames:
        return []

    n = len(segment_frames)
    client = OpenAI(api_key=api_key)
    hint = spoken_context.strip()[:500]

    header = (
        f"This is a {duration_sec:.2f}s vertical (9:16) short. Images are grouped in time order. "
        f"Each group is labeled with its time range in the clip. "
        f"Reply with a JSON array only (no markdown): exactly {n} strings. "
        f"String i (1-based: first string for part 1, etc.) must describe ONLY what is visible in that part's images—"
        f"that moment of the video—not other parts. Max 48 characters per string. "
        f"Literal, concrete (who/what/where/action/lighting). No generic motivational filler. "
        f"No hashtags. No nested quotes inside strings."
    )
    if hint:
        header += f" Optional hint (may be wrong—prefer the grouped images): {hint}"

    user_content: list[dict[str, Any]] = [{"type": "text", "text": header}]

    for i, paths in enumerate(segment_frames):
        t_a = duration_sec * i / n
        t_b = duration_sec * (i + 1) / n
        user_content.append(
            {
                "type": "text",
                "text": f"— Part {i + 1} of {n}: ~{t_a:.2f}s to ~{t_b:.2f}s —",
            }
        )
        if not paths:
            user_content.append(
                {
                    "type": "text",
                    "text": "(No frame for this part—output a very short neutral line like 'Scene continues' under 48 chars.)",
                }
            )
            continue
        for p in paths[:2]:
            try:
                raw = p.read_bytes()
            except OSError:
                continue
            b64 = base64.standard_b64encode(raw).decode("ascii")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                }
            )

    image_parts = sum(1 for x in user_content if x.get("type") == "image_url")
    if image_parts == 0:
        return []

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"Reply with valid JSON only: a JSON array of exactly {n} strings.",
                },
                {"role": "user", "content": user_content},
            ],
            temperature=0.35,
            max_tokens=450,
        )
        content = (resp.choices[0].message.content or "").strip()
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.I)
        content = re.sub(r"\s*```\s*$", "", content.strip())
        m = re.search(r"\[[\s\S]*\]", content)
        if not m:
            return []
        arr = json.loads(m.group())
        if not isinstance(arr, list):
            return []
        out = [str(x).strip() for x in arr if str(x).strip()]
        # Model sometimes returns one combined line; pad so each timed segment has text.
        while len(out) < n:
            out.append("Scene continues")
        return out[:n]
    except Exception:
        return []


def generate_overlay_text(scene_description: str, *, api_key: str | None) -> list[str]:
    """Generate 1-3 concise on-screen caption lines from text only (fallback when vision is unavailable)."""
    scene_description = scene_description.strip()
    if not scene_description:
        return []

    # Fallback simple split if OpenAI is not configured
    if not api_key:
        words = scene_description.replace("_", " ").replace("-", " ").split()
        first = " ".join(words[:8])
        return [first[:70]] if first else []

    client = OpenAI(api_key=api_key)
    prompt = (
        "You create short on-screen captions for a vertical video. "
        "Given ONLY the text below (no images), return a JSON array of 1-3 lines (<= 50 chars each). "
        "Each line must reflect concrete details from that text—no generic slogans or clichés unrelated to the description."
        f"\n\nText: {scene_description}"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Reply with valid JSON only. No markdown fences."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        content = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\[.*\]", content, re.S)
        if not m:
            m = re.search(r"\{.*\}", content, re.S)
        if m:
            arr = json.loads(m.group())
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()][:3]
    except Exception:
        pass

    # Ultimate fallback simple text
    words = scene_description.split()
    return [" ".join(words[:8])][:1]
