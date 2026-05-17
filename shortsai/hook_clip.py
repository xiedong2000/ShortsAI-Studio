"""Pick a short 'hook' window from source video and prepend it as a cold open (FFmpeg)."""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from . import ffmpeg_util

DEFAULT_HOOK_SEC = 5.0
MIN_SOURCE_SEC_FOR_HOOK = 10.0
MAX_CANDIDATES = 8


@dataclass(frozen=True)
class HookSelection:
    """Window in the **source** video timeline (seconds)."""

    start: float
    end: float
    score: float
    method: str  # vision | fallback | override
    reason: str = ""


def hook_duration_from_env() -> float:
    raw = (os.environ.get("SHORTSAI_AI_HOOK_SEC") or "").strip()
    if not raw:
        return DEFAULT_HOOK_SEC
    try:
        return max(3.0, min(8.0, float(raw)))
    except ValueError:
        return DEFAULT_HOOK_SEC


def candidate_hook_windows(
    duration_sec: float,
    hook_len: float,
    *,
    max_candidates: int = MAX_CANDIDATES,
) -> list[tuple[float, float]]:
    """Non-overlapping-ish candidate (start, end) windows inside the source clip."""
    if duration_sec <= hook_len + 0.5:
        return [(0.0, min(duration_sec, hook_len))]
    last_start = max(0.0, duration_sec - hook_len)
    n = min(max_candidates, max(3, int(last_start / max(hook_len, 1.0)) + 1))
    if n <= 1:
        return [(0.0, hook_len)]
    out: list[tuple[float, float]] = []
    for i in range(n):
        start = last_start * i / (n - 1) if n > 1 else 0.0
        start = max(0.0, min(start, last_start))
        out.append((start, start + hook_len))
    return out


def _extract_frame(video: Path, work_dir: Path, t_sec: float, name: str) -> Path | None:
    out = work_dir / name
    try:
        ffmpeg_util.run_ffmpeg(
            [
                "-y",
                "-ss",
                f"{max(0.0, t_sec):.3f}",
                "-i",
                str(video.resolve()),
                "-frames:v",
                "1",
                "-q:v",
                "3",
                out.name,
            ],
            cwd=work_dir,
        )
    except ffmpeg_util.FFmpegError:
        return None
    if out.is_file() and out.stat().st_size > 0:
        return out
    return None


def _fallback_pick_window(
    duration_sec: float,
    hook_len: float,
    *,
    transcript_peaks: list[float] | None = None,
) -> HookSelection:
    """Prefer mid-clip; optional boost near speech-heavy timestamps."""
    candidates = candidate_hook_windows(duration_sec, hook_len)
    if not candidates:
        return HookSelection(0.0, min(hook_len, duration_sec), 5.0, "fallback", "start of clip")

    if transcript_peaks:
        best = candidates[0]
        best_d = float("inf")
        for start, end in candidates:
            mid = (start + end) / 2
            d = min(abs(mid - p) for p in transcript_peaks)
            if d < best_d:
                best_d = d
                best = (start, end)
        return HookSelection(
            best[0],
            best[1],
            6.0,
            "fallback",
            "near speech activity",
        )

    # Default: ~35% in — often past pure logo intros on longer clips
    target = duration_sec * 0.35
    best = min(candidates, key=lambda w: abs((w[0] + w[1]) / 2 - target))
    return HookSelection(
        best[0],
        best[1],
        5.0,
        "fallback",
        "mid-clip heuristic",
    )


def select_hook_window(
    video: Path,
    work_dir: Path,
    *,
    duration_sec: float,
    hook_len: float,
    api_key: str | None,
    transcript_peaks: list[float] | None = None,
    log: Callable[[str], None] | None = None,
) -> HookSelection | None:
    """
    Return the best hook window in source time, or None if the clip is too short.
    Uses vision when ``api_key`` is set; otherwise a simple heuristic.
    """
    emit = log or (lambda _m: None)
    if duration_sec < MIN_SOURCE_SEC_FOR_HOOK:
        emit(
            f"AI hook skipped: clip is {duration_sec:.1f}s "
            f"(need at least {MIN_SOURCE_SEC_FOR_HOOK:.0f}s)."
        )
        return None

    candidates = candidate_hook_windows(duration_sec, hook_len)
    if not api_key:
        sel = _fallback_pick_window(duration_sec, hook_len, transcript_peaks=transcript_peaks)
        emit(f"AI hook (heuristic): {sel.start:.2f}s–{sel.end:.2f}s — {sel.reason}")
        return sel

    frames_dir = work_dir / "hook_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"You are picking a {hook_len:.1f}s cold-open hook for a YouTube Short from a "
                f"{duration_sec:.1f}s source clip. Each image is the **middle frame** of one "
                f"candidate window (labeled with its index). Rate how strong each would be as an "
                f"attention-grabbing opening (faces, action, surprise, emotional peak, clear visual "
                f"story)—not a boring static title card or empty scene.\n\n"
                f"Reply with JSON only: {{\"scores\": [numbers], \"best_index\": int, \"reason\": \"short\"}} "
                f"where scores has exactly {len(candidates)} numbers 1–10 (one per candidate in order), "
                f"best_index is 0-based, reason is one short phrase."
            ),
        }
    ]

    for i, (start, end) in enumerate(candidates):
        mid = (start + end) / 2
        fp = _extract_frame(video, frames_dir, mid, f"hook_cand_{i:02d}.jpg")
        user_content.append(
            {
                "type": "text",
                "text": f"Candidate {i}: source {start:.2f}s – {end:.2f}s",
            }
        )
        if fp:
            b64 = base64.standard_b64encode(fp.read_bytes()).decode("ascii")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                }
            )
        else:
            user_content.append({"type": "text", "text": "(frame missing)"})

    if not any(x.get("type") == "image_url" for x in user_content):
        sel = _fallback_pick_window(duration_sec, hook_len, transcript_peaks=transcript_peaks)
        emit(f"AI hook: vision frames failed; using heuristic ({sel.reason}).")
        return sel

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Reply with valid JSON only. No markdown fences.",
                },
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```\s*$", "", raw.strip())
        data = json.loads(raw)
        scores = data.get("scores")
        best_i = int(data.get("best_index", 0))
        reason = str(data.get("reason", "")).strip() or "vision pick"
        if isinstance(scores, list) and scores:
            best_i = max(0, min(best_i, len(candidates) - 1))
            try:
                score_val = float(scores[best_i])
            except (TypeError, ValueError):
                score_val = 7.0
        else:
            best_i = max(0, min(best_i, len(candidates) - 1))
            score_val = 7.0
        start, end = candidates[best_i]
        emit(
            f"AI hook (vision): candidate {best_i} {start:.2f}s–{end:.2f}s "
            f"(score {score_val:.1f}) — {reason}"
        )
        return HookSelection(start, end, score_val, "vision", reason)
    except Exception as e:
        sel = _fallback_pick_window(duration_sec, hook_len, transcript_peaks=transcript_peaks)
        emit(f"AI hook: vision failed ({str(e)[:120]}); using heuristic ({sel.reason}).")
        return sel


def apply_hook_cold_open(
    video_in: Path,
    video_out: Path,
    *,
    hook: HookSelection,
    hook_len: float,
    max_output_sec: float,
    cwd: Path,
) -> float:
    """
    Prepend ``hook`` segment, then play from 0 for ``main_len`` seconds.
    Returns output duration (seconds).
    """
    dur = ffmpeg_util.probe_duration_seconds(video_in)
    main_len = max(0.5, min(dur, max_output_sec - hook_len))
    hook_start = max(0.0, min(hook.start, max(0.0, dur - hook_len)))
    hook_end = min(dur, hook_start + hook_len)
    has_audio = ffmpeg_util.has_audio_stream(video_in)

    src_name = video_in.name
    if video_in.parent.resolve() != cwd.resolve():
        import shutil

        shutil.copy2(video_in, cwd / src_name)

    v_part = (
        f"[0:v]trim=start={hook_start}:end={hook_end},setpts=PTS-STARTPTS[vhook];"
        f"[0:v]trim=start=0:end={main_len},setpts=PTS-STARTPTS[vmain];"
        f"[vhook][vmain]concat=n=2:v=1:a=0[vout]"
    )
    if has_audio:
        filter_complex = (
            v_part
            + f";[0:a]atrim=start={hook_start}:end={hook_end},asetpts=PTS-STARTPTS[ahook];"
            f"[0:a]atrim=start=0:end={main_len},asetpts=PTS-STARTPTS[amain];"
            f"[ahook][amain]concat=n=2:v=0:a=1[aout]"
        )
        maps = ["-map", "[vout]", "-map", "[aout]"]
    else:
        filter_complex = v_part
        maps = ["-map", "[vout]"]

    cmd = [
        "-y",
        "-i",
        src_name,
        "-filter_complex",
        filter_complex,
        *maps,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
    ]
    if has_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    else:
        cmd.append("-an")
    cmd.append(str(video_out.name))

    ffmpeg_util.run_ffmpeg(cmd, cwd=cwd)
    return ffmpeg_util.probe_duration_seconds(video_out)


def hook_selection_to_meta(hook: HookSelection, *, hook_len: float, output_duration: float) -> dict[str, Any]:
    return {
        "applied": True,
        "hook_sec": round(hook_len, 2),
        "source_start": round(hook.start, 2),
        "source_end": round(hook.end, 2),
        "score": round(hook.score, 2),
        "method": hook.method,
        "reason": hook.reason,
        "output_duration_sec": round(output_duration, 2),
    }


def hook_selection_from_meta(raw: Any) -> HookSelection | None:
    if not isinstance(raw, dict) or not raw.get("applied"):
        return None
    try:
        return HookSelection(
            start=float(raw["source_start"]),
            end=float(raw["source_end"]),
            score=float(raw.get("score", 0)),
            method=str(raw.get("method", "override")),
            reason=str(raw.get("reason", "")),
        )
    except (KeyError, TypeError, ValueError):
        return None
