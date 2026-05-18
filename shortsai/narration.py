"""Scene-timed AI narration: one line per video segment, placed on the timeline."""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from . import ffmpeg_util
from .transcribe import WordSpan

DEFAULT_VOICE = "alloy"
DEFAULT_ORIGINAL_DUCK = 0.14
DEFAULT_NARRATION_VOL = 1.45
MAX_NARRATION_VOL = 3.0


@dataclass(frozen=True)
class NarrationCue:
    start_sec: float
    end_sec: float
    text: str


def narration_voice_from_env() -> str:
    v = (os.environ.get("SHORTSAI_NARRATION_VOICE") or DEFAULT_VOICE).strip().lower()
    allowed = frozenset({"alloy", "echo", "fable", "onyx", "nova", "shimmer"})
    return v if v in allowed else DEFAULT_VOICE


def original_duck_volume_from_env() -> float:
    raw = (os.environ.get("SHORTSAI_NARRATION_ORIG_VOL") or "").strip()
    if not raw:
        return DEFAULT_ORIGINAL_DUCK
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return DEFAULT_ORIGINAL_DUCK


def narration_volume_from_env() -> float:
    raw = (os.environ.get("SHORTSAI_NARRATION_VOL") or "").strip()
    if not raw:
        return DEFAULT_NARRATION_VOL
    try:
        return max(0.25, min(MAX_NARRATION_VOL, float(raw)))
    except ValueError:
        return DEFAULT_NARRATION_VOL


def clamp_narration_volume(vol: float) -> float:
    return max(0.25, min(MAX_NARRATION_VOL, float(vol)))


def narration_segment_count(duration_sec: float) -> int:
    raw = (os.environ.get("SHORTSAI_NARRATION_SEGMENTS") or "").strip()
    if raw:
        try:
            return max(2, min(6, int(raw)))
        except ValueError:
            pass
    return max(2, min(6, int(duration_sec / 9.0) + 1))


def segment_windows(duration_sec: float, n: int) -> list[tuple[float, float]]:
    if n < 1:
        return [(0.0, duration_sec)]
    return [
        (duration_sec * i / n, duration_sec * (i + 1) / n) for i in range(n)
    ]


def cue_speech_start_sec(
    cue: NarrationCue,
    tts_dur: float,
    *,
    align_finale: bool,
) -> float:
    """When to start TTS within a segment window (finale lines end near segment end)."""
    span = max(0.08, cue.end_sec - cue.start_sec)
    tts_dur = max(0.05, tts_dur)
    if align_finale:
        start = cue.end_sec - tts_dur - 0.15
        return max(cue.start_sec, min(start, cue.end_sec - 0.2))
    inset = min(span * 0.12, max(0.0, span - tts_dur - 0.15))
    return cue.start_sec + inset


def _transcript_insufficient_for_topic(transcript: str) -> bool:
    t = transcript.strip()
    if not t:
        return True
    words = [w for w in re.split(r"\s+", t) if w]
    if len(words) < 6 and len(t) < 50:
        return True
    return False


_GENERIC_NARRATION_RE = re.compile(
    r"|".join(
        [
            r"upload(?:ing)?\s+content",
            r"few\s+clicks\s+away",
            r"platforms?\s+(?:designed\s+)?for\s+sharing",
            r"content\s+strategy",
            r"maximize\s+(?:your\s+)?reach",
            r"how\s+easy\s+it\s+is\s+to\s+upload",
            r"whether\s+it['\u2019]?s\s+a\s+video,\s*photo",
            r"youtube\s+shorts?\s+(?:tips|tutorial|guide)",
            r"mastering\s+(?:the\s+)?platform",
        ]
    ),
    re.I,
)


def _clean_script_line(raw: str) -> str:
    script = (raw or "").strip().replace("\n", " ")
    if len(script) >= 2 and script[0] == script[-1] and script[0] in '"\'':
        script = script[1:-1].strip()
    return script


def looks_like_generic_platform_narration(script: str) -> bool:
    if not script.strip():
        return True
    return bool(_GENERIC_NARRATION_RE.search(script))


def _words_in_window(words: list[WordSpan], t0: float, t1: float) -> str:
    picked: list[str] = []
    for w in words:
        mid = (w.start + w.end) / 2
        if t0 <= mid < t1 and w.text.strip():
            picked.append(w.text.strip())
    return " ".join(picked).strip()


def generate_scene_narration_lines(
    video: Path,
    work_dir: Path,
    *,
    duration_sec: float,
    api_key: str,
    transcript: str = "",
    words: list[WordSpan] | None = None,
    visual_hint: str = "",
    log: Callable[[str], None] | None = None,
) -> list[NarrationCue]:
    """One narration line per chronological segment, aligned to video time."""
    emit = log or (lambda _m: None)
    n = narration_segment_count(duration_sec)
    windows = segment_windows(duration_sec, n)
    prefix = "narr_scene"
    frame_groups = ffmpeg_util.extract_jpeg_frames_per_segment(
        video, work_dir, n_segments=n, file_prefix=prefix
    )

    max_words_per = max(6, min(22, int(duration_sec / n * 2.0)))
    hint_s = (visual_hint or transcript or "").strip()[:400]

    header = (
        f"This {duration_sec:.1f}s vertical Short is split into {n} chronological segments. "
        f"Reply with JSON only: a JSON array of exactly {n} strings.\n"
        f"String i is the narrator line spoken DURING segment i only (see time ranges and images). "
        f"Each line: at most {max_words_per} words, English, conversational.\n"
        f"Describe only what is visible or spoken in that segment—do not summarize the whole video in line 0.\n"
        f"The last array string MUST describe the finale/payoff (what happens at the end of the clip), not only the start of that segment.\n"
        f"Do NOT write generic tutorials about uploading content, YouTube tips, or 'platforms for sharing'.\n"
        f"If a segment has nothing to say, use a single dash \"-\"."
    )
    if hint_s:
        header += f"\nOptional hint (may be filename only—prefer segment images): {hint_s}"

    user_content: list[dict[str, Any]] = [{"type": "text", "text": header}]
    word_list = words or []

    for i, (t0, t1) in enumerate(windows):
        seg_words = _words_in_window(word_list, t0, t1)
        seg_note = (
            f"— Segment {i + 1} of {n}: {t0:.2f}s to {t1:.2f}s —"
        )
        if seg_words:
            seg_note += f"\nSpoken in this segment (may help): {seg_words[:500]}"
        if i == n - 1:
            seg_note += (
                "\nFINAL segment — include the climax, twist, or most exciting beat "
                "(often near the end of this time range). Extra stills may show the ending."
            )
        user_content.append({"type": "text", "text": seg_note})
        paths = frame_groups[i] if i < len(frame_groups) else []
        if not paths:
            user_content.append({"type": "text", "text": "(No frame—use \"-\" if nothing to narrate.)"})
            continue
        for p in paths[:3]:
            try:
                b64 = base64.standard_b64encode(p.read_bytes()).decode("ascii")
            except OSError:
                continue
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                }
            )

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Reply with valid JSON only: array of strings. "
                        "Each string is one timed narrator line for that segment only."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            temperature=0.28,
            max_tokens=500,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```\s*$", "", raw.strip())
        m = re.search(r"\[[\s\S]*\]", raw)
        if not m:
            raise ValueError("No JSON array in scene narration response")
        arr = json.loads(m.group())
        if not isinstance(arr, list):
            raise ValueError("Scene narration response is not a list")

        cues: list[NarrationCue] = []
        for i, (t0, t1) in enumerate(windows):
            line = "-"
            if i < len(arr):
                line = _clean_script_line(str(arr[i]))
            if line and line != "-" and looks_like_generic_platform_narration(line):
                line = "-"
                emit(f"Segment {i + 1}: dropped generic line.")
            if line and line != "-":
                cues.append(NarrationCue(t0, t1, line))
                emit(f"Narration seg {i + 1} ({t0:.1f}s–{t1:.1f}s): {line[:80]}")

        if not cues:
            raise ValueError("No usable scene narration lines after filtering")
        return cues
    finally:
        for fp in work_dir.glob(f"{prefix}_*.jpg"):
            fp.unlink(missing_ok=True)


def resolve_scene_narration_cues(
    *,
    transcript: str,
    words: list[WordSpan] | None,
    duration_sec: float,
    api_key: str,
    video_path: Path,
    work_dir: Path,
    visual_hint: str = "",
    log: Callable[[str], None] | None = None,
) -> tuple[list[NarrationCue], str]:
    """Build timed narration cues from per-segment vision (+ speech snippets)."""
    emit = log or (lambda _m: None)
    emit(f"Writing scene-timed narration ({narration_segment_count(duration_sec)} segments)…")
    cues = generate_scene_narration_lines(
        video_path,
        work_dir,
        duration_sec=duration_sec,
        api_key=api_key,
        transcript=transcript,
        words=words,
        visual_hint=visual_hint,
        log=log,
    )
    return cues, "scene_timed"


def synthesize_narration(
    script: str,
    out_mp3: Path,
    *,
    api_key: str,
    voice: str | None = None,
    log: Callable[[str], None] | None = None,
) -> Path:
    emit = log or (lambda _m: None)
    text = script.strip()
    if not text:
        raise ValueError("Narration script is empty")

    voice_id = (voice or narration_voice_from_env()).strip().lower()
    client = OpenAI(api_key=api_key)
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice_id,  # type: ignore[arg-type]
        input=text[:4096],
    )
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    response.stream_to_file(str(out_mp3))
    if not out_mp3.is_file() or out_mp3.stat().st_size < 100:
        raise RuntimeError("TTS output missing or too small")
    dur = ffmpeg_util.probe_duration_seconds(out_mp3)
    emit(f"TTS line ({dur:.1f}s): {text[:60]}{'…' if len(text) > 60 else ''}")
    return out_mp3


def build_timed_narration_track(
    cues: list[NarrationCue],
    out_mp3: Path,
    *,
    video_duration: float,
    work_dir: Path,
    api_key: str,
    voice: str | None = None,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Place one TTS clip at each cue's start time; output spans full video duration."""
    emit = log or (lambda _m: None)
    if not cues:
        raise ValueError("No narration cues")

    emit(f"Building timed narration across {video_duration:.1f}s ({len(cues)} lines)…")
    staged: list[tuple[float, str]] = []
    last_cue_end = cues[-1].end_sec if cues else 0.0
    for i, cue in enumerate(cues):
        seg_mp3 = work_dir / f"narr_line_{i:02d}.mp3"
        synthesize_narration(
            cue.text,
            seg_mp3,
            api_key=api_key,
            voice=voice,
            log=None,
        )
        tts_dur = ffmpeg_util.probe_duration_seconds(seg_mp3)
        align_finale = cue.end_sec >= last_cue_end - 0.05
        start_at = cue_speech_start_sec(cue, tts_dur, align_finale=align_finale)
        staged.append((start_at, seg_mp3.name))

    if len(staged) == 1:
        start, name = staged[0]
        ms = int(max(0.0, start) * 1000)
        pad = max(0.0, video_duration)
        filter_complex = (
            f"[0:a]adelay={ms}|{ms},asetpts=PTS-STARTPTS,apad=pad_dur={pad:.3f}:whole_dur={pad:.3f}[aout]"
        )
        ffmpeg_util.run_ffmpeg(
            [
                "-y",
                "-i",
                name,
                "-filter_complex",
                filter_complex,
                "-map",
                "[aout]",
                "-t",
                f"{pad:.3f}",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "192k",
                str(out_mp3.name),
            ],
            cwd=work_dir,
        )
        return out_mp3

    inputs: list[str] = ["-y"]
    filters: list[str] = []
    labels: list[str] = []
    for i, (start, fname) in enumerate(staged):
        inputs.extend(["-i", fname])
        ms = int(max(0.0, start) * 1000)
        filters.append(
            f"[{i}:a]adelay={ms}|{ms},asetpts=PTS-STARTPTS,volume=1.0[na{i}]"
        )
        labels.append(f"[na{i}]")

    pad = max(0.1, video_duration)
    filters.append(
        f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:dropout_transition=0:"
        f"normalize=0,apad=pad_dur={pad:.3f}:whole_dur={pad:.3f}[aout]"
    )
    ffmpeg_util.run_ffmpeg(
        inputs
        + [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[aout]",
            "-t",
            f"{pad:.3f}",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "192k",
            str(out_mp3.name),
        ],
        cwd=work_dir,
    )
    emit(f"Timed narration track ready ({pad:.1f}s).")
    return out_mp3


def _stage_input(path: Path, cwd: Path) -> str:
    if path.parent.resolve() != cwd.resolve():
        shutil.copy2(path, cwd / path.name)
    return path.name


def mix_video_audio_bed(
    video_in: Path,
    video_out: Path,
    *,
    cwd: Path,
    narration: Path | None = None,
    music: Path | None = None,
    music_volume: float = 0.18,
    original_volume: float = 1.0,
    narration_volume: float | None = None,
    narration_timed: bool = False,
) -> None:
    """Replace video audio with ducked original + narration (+ optional music)."""
    v_name = _stage_input(video_in, cwd)
    vdur = ffmpeg_util.probe_duration_seconds(cwd / v_name)
    has_orig = ffmpeg_util.has_audio_stream(cwd / v_name)

    narr_name: str | None = None
    music_name: str | None = None
    if narration is not None and narration.is_file():
        narr_name = _stage_input(narration, cwd)
    if music is not None and music.is_file():
        music_name = _stage_input(music, cwd)

    vol_orig = max(0.0, min(1.0, original_volume))
    vol_narr = clamp_narration_volume(
        narration_volume if narration_volume is not None else narration_volume_from_env()
    )
    vol_music = max(0.0, min(1.0, music_volume))
    if narr_name:
        vol_orig = min(vol_orig, original_duck_volume_from_env())

    inputs: list[str] = ["-y", "-i", v_name]
    filters: list[str] = []
    mix: list[str] = []
    in_idx = 1

    if has_orig:
        filters.append(f"[0:a]volume={vol_orig}[aorig]")
        mix.append("[aorig]")

    if narr_name:
        inputs.extend(["-i", narr_name])
        if narration_timed:
            filters.append(f"[{in_idx}:a]volume={vol_narr}[anarr]")
        else:
            trim_end = max(0.5, vdur - 0.05)
            filters.append(
                f"[{in_idx}:a]atrim=0:{trim_end:.3f},asetpts=PTS-STARTPTS,volume={vol_narr}[anarr]"
            )
        mix.append("[anarr]")
        in_idx += 1

    if music_name:
        inputs.extend(["-i", music_name])
        filters.append(f"[{in_idx}:a]volume={vol_music}[amus]")
        mix.append("[amus]")
        in_idx += 1

    if not mix:
        args = inputs + ["-map", "0:v:0"]
        if has_orig:
            args.extend(["-map", "0:a:0", "-c:a", "aac", "-b:a", "192k"])
        else:
            args.append("-an")
        ffmpeg_util.run_ffmpeg(
            args + ["-c:v", "copy", "-shortest", str(video_out.name)],
            cwd=cwd,
        )
        return

    if len(mix) > 1:
        filters.append(
            f"{''.join(mix)}amix=inputs={len(mix)}:duration=first:dropout_transition=2:"
            f"normalize=0[aout]"
        )
        audio_map = "[aout]"
    else:
        audio_map = mix[0]

    filter_complex = ";".join(filters)
    ffmpeg_util.run_ffmpeg(
        inputs
        + [
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            audio_map,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(video_out.name),
        ],
        cwd=cwd,
    )


def narration_to_meta(
    *,
    applied: bool,
    script: str = "",
    voice: str = "",
    source: str = "",
    segment_lines: list[dict[str, Any]] | None = None,
    volume: float | None = None,
    error: str = "",
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "applied": applied,
        "script": script,
        "voice": voice,
        "source": source,
        "segment_lines": segment_lines or [],
        "error": error[:300] if error else "",
    }
    if volume is not None:
        out["volume"] = round(clamp_narration_volume(volume), 2)
    return out


def narration_cues_to_srt(cues: list[NarrationCue], *, max_chars: int = 36) -> str:
    """SubRip text for burned-in captions matching narration segment times."""
    from shortsai.srt_build import segment_timed_srt

    return segment_timed_srt(
        [(c.start_sec, c.end_sec, c.text) for c in cues],
        max_chars=max_chars,
    )


def cues_to_meta_segments(cues: list[NarrationCue]) -> list[dict[str, Any]]:
    return [
        {
            "start": round(c.start_sec, 2),
            "end": round(c.end_sec, 2),
            "text": c.text,
        }
        for c in cues
    ]
