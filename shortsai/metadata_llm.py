from __future__ import annotations

import json
import re
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
