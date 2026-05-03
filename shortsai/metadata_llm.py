from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from . import ffmpeg_util

# Double quotes + guillemets: remove everywhere (model often wraps lines in "..." or „…").
_OVERLAY_QUOTE_GLOBAL = frozenset(
    "\""
    "\u201c\u201d"  # “ ”
    "\u201e\u201f"  # „ ‟
    "\uff02"  # ＂ fullwidth
    "\u00ab\u00bb"  # « »
    "\u2039\u203a"  # ‹ ›
)
# Singles: strip only from line ends so internal apostrophes (It's) stay intact.
_OVERLAY_QUOTE_EDGE_SINGLE = frozenset("'\u2018\u2019\u201a\u201b")


def strip_overlay_quotes(s: str) -> str:
    """Remove wrapping and decorative quotes from a single overlay line."""
    s = (s or "").strip()
    s = "".join(c for c in s if c not in _OVERLAY_QUOTE_GLOBAL)
    s = s.strip()
    while s and s[0] in _OVERLAY_QUOTE_EDGE_SINGLE:
        s = s[1:].strip()
    while s and s[-1] in _OVERLAY_QUOTE_EDGE_SINGLE:
        s = s[:-1].strip()
    return s


def sanitize_overlay_lines(lines: list[str]) -> list[str]:
    """Strip quotes from each line and drop empties."""
    return [x for x in (strip_overlay_quotes(t) for t in lines) if x]


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


def _transcript_insufficient_for_topic(transcript: str) -> bool:
    """True when input is empty, filename-only, or too short—LLMs tend to invent generic Shorts how-to."""
    t = transcript.strip()
    if not t:
        return True
    words = [w for w in re.split(r"\s+", t) if w]
    if len(words) < 6 and len(t) < 50:
        return True
    return False


def _metadata_dict_from_parsed(data: dict[str, Any], *, notes: str) -> dict[str, Any]:
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
        "notes": notes,
    }


def generate_metadata_from_video_frames(
    video: Path,
    work_dir: Path,
    *,
    hint: str,
    api_key: str,
) -> dict[str, Any] | None:
    """
    Title/description/tags from actual video frames when speech transcript is missing or useless.
    """
    prefix = "meta_vision"
    try:
        paths = ffmpeg_util.extract_jpeg_frames_evenly(
            video, work_dir, max_frames=4, file_prefix=prefix
        )
        if not paths:
            return None
        client = OpenAI(api_key=api_key)
        hint_s = (hint or "").strip()[:400]
        header = (
            "You write YouTube Shorts metadata (not on-screen captions). "
            "These JPEGs are evenly spaced frames from ONE vertical Short. "
            "Return ONE JSON object only with keys: title (string, under 70 chars), "
            "description (string, 1-3 short paragraphs, no hashtag spam), "
            "tags (array of 5-10 short strings, no # prefix), "
            "attribution (string, empty unless a visible on-screen credit implies music attribution). "
            "Describe what is actually shown or strongly implied (subject, setting, activity, mood). "
            "Do NOT write generic tutorials about uploading to YouTube, Shorts SEO, 'maximize reach', "
            "or 'content strategy' unless the frames literally show that topic. "
            "If the clip is ambiguous, stay concrete about what you see rather than inventing a lesson."
        )
        if hint_s:
            header += f" Optional extra hint (may be filename only—prefer the images): {hint_s}"

        user_content: list[dict[str, Any]] = [{"type": "text", "text": header}]
        for p in paths:
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
        if sum(1 for x in user_content if x.get("type") == "image_url") == 0:
            return None

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Reply with valid JSON only. No markdown fences.",
                },
                {"role": "user", "content": user_content},
            ],
            temperature=0.35,
            max_tokens=700,
        )
        content = (resp.choices[0].message.content or "").strip()
        try:
            data = _parse_json_object(content)
        except (json.JSONDecodeError, ValueError):
            return None
        return _metadata_dict_from_parsed(
            data, notes="Generated from video frames (speech transcript was missing or too short)."
        )
    except Exception:
        return None
    finally:
        for fp in work_dir.glob(f"{prefix}_*.jpg"):
            fp.unlink(missing_ok=True)


def generate_metadata(
    transcript: str,
    *,
    api_key: str | None,
    video_path: Path | None = None,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    if not api_key:
        out = _fallback_metadata(transcript)
        out["metadata_source"] = "fallback_no_api_key"
        return out

    if _transcript_insufficient_for_topic(transcript):
        if video_path and work_dir:
            vision_meta = generate_metadata_from_video_frames(
                video_path, work_dir, hint=transcript, api_key=api_key
            )
            if vision_meta is not None:
                vision_meta["metadata_source"] = "vision_llm"
                return vision_meta
        out = _fallback_metadata(transcript)
        out["metadata_source"] = "fallback_transcript_only"
        return out

    client = OpenAI(api_key=api_key)
    prompt = (
        "You write metadata for ONE YouTube Short. The block below is the spoken transcript (or a tiny "
        "filename fallback if there was almost no speech). "
        "Return ONE JSON object only, with keys: title (string, under 70 chars, catchy), "
        "description (string, 1-3 short paragraphs, SEO-friendly, no hashtag spam), "
        "tags (array of 5-10 short strings, no # prefix), "
        "attribution (string, empty unless the transcript mentions music to credit). "
        "Every topical claim MUST come from the transcript. "
        "If the transcript is very short or looks like a filename, describe only what those words could "
        "honestly mean—do NOT substitute a generic article about uploading Shorts, 'mastering' the platform, "
        "reach, engagement, or content strategy. "
        "Never write meta-tutorials about YouTube or Shorts unless the speaker is actually discussing that.\n\n"
        "Transcript:\n"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Reply with valid JSON only. No markdown fences."},
            {"role": "user", "content": prompt + transcript[:12000]},
        ],
        temperature=0.45,
    )
    content = (resp.choices[0].message.content or "").strip()
    try:
        data = _parse_json_object(content)
    except (json.JSONDecodeError, ValueError):
        out = _fallback_metadata(transcript)
        out["metadata_source"] = "fallback_parse_error"
        return out

    out = _metadata_dict_from_parsed(data, notes="")
    out["metadata_source"] = "transcript_llm"
    return out


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
        f"This is a {duration_sec:.2f}s vertical (9:16) YouTube Short. Images are grouped in time order; "
        f"each group covers one segment of the timeline. "
        f"Reply with a JSON array only (no markdown): exactly {n} strings. "
        f"String i must be the on-screen caption shown WHILE that part plays. "
        f"Goal: lines that help **get views and watch time**—hooks, curiosity, tension, relatability, "
        f"or a clear payoff tease—NOT dry picture captions like 'a person in a room'. "
        f"Each line must still be **grounded in what is actually visible** in that part's images (and the optional hint); "
        f"do not invent facts, stunts, or outcomes that contradict the frames. "
        f"Vary style across segments (e.g. question, bold claim tied to the scene, 'POV:', 'Here's why…', "
        f"relatable joke, countdown vibe) where it fits; avoid repeating the same opening word every line. "
        f"Max 48 characters per string. No hashtags, no emojis. Write plain words only—never surround a line "
        f"with \" or ' or typographic quotes; no quotation marks anywhere in the caption. "
        f"No nested double quotes inside strings."
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
                    "text": (
                        "(No frame for this part—output one very short retention-friendly line under 48 chars, "
                        "e.g. teasing what comes next, still plausible for a Short.)"
                    ),
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
                    "content": (
                        f"You write YouTube Shorts on-screen captions that boost engagement. "
                        f"Reply with valid JSON only: a JSON array of exactly {n} strings."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            temperature=0.55,
            max_tokens=500,
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
        out = [strip_overlay_quotes(str(x)) for x in arr]
        out = [x for x in out if x]
        # Model sometimes returns one combined line; pad so each timed segment has text.
        while len(out) < n:
            out.append("Keep watching")
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
        first = strip_overlay_quotes(" ".join(words[:8]))
        return [first[:70]] if first else []

    client = OpenAI(api_key=api_key)
    prompt = (
        "You write on-screen captions for a vertical YouTube Short. "
        "Given ONLY the text below (no images), return a JSON array of 1-3 lines (<= 50 chars each). "
        "Lines should help attract and retain viewers: hooks, curiosity, relatability, or a clear tease—"
        "not bland scene summaries. Stay truthful to the text; do not invent events or claims the text does not support. "
        "No hashtags, no emojis, no quotation marks in the lines. Vary tone across lines when you output more than one."
        f"\n\nText: {scene_description}"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Reply with valid JSON only. No markdown fences. Captions optimized for Shorts engagement.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.75,
        )
        content = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\[.*\]", content, re.S)
        if not m:
            m = re.search(r"\{.*\}", content, re.S)
        if m:
            arr = json.loads(m.group())
            if isinstance(arr, list):
                lines = [strip_overlay_quotes(str(x)) for x in arr]
                return [x for x in lines if x][:3]
    except Exception:
        pass

    # Ultimate fallback simple text
    words = scene_description.split()
    u = strip_overlay_quotes(" ".join(words[:8]))
    return [u] if u else []
