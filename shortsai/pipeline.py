from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Literal, cast

from faster_whisper import WhisperModel

from shortsai import ffmpeg_util
from shortsai.metadata_llm import (
    _transcript_insufficient_for_topic,
    generate_metadata,
    generate_onscreen_subtitle_srt_from_segments,
    generate_overlay_text,
    generate_overlay_text_from_vision_segments,
    sanitize_overlay_lines,
    strip_overlay_quotes,
)
from shortsai.hook_clip import (
    apply_hook_cold_open,
    hook_duration_from_env,
    hook_selection_from_meta,
    hook_selection_to_meta,
    select_hook_window,
)
from shortsai.narration import (
    NarrationCue,
    build_timed_narration_track,
    cues_to_meta_segments,
    mix_video_audio_bed,
    narration_cues_to_srt,
    narration_cues_from_meta,
    narration_to_meta,
    narration_voice_from_env,
    narration_voice_from_meta,
    narration_volume_from_env,
    resolve_scene_narration_cues,
)
from shortsai.srt_build import words_to_srt
from shortsai.transcribe import TranscriptResult, WordSpan, transcribe

VerticalFitMode = Literal["letterbox", "crop", "blur_fill"]
OverlayPosition = Literal["upper", "middle", "lower"]

DEFAULT_MAX_DURATION_SEC = 120.0
HARD_MAX_DURATION_SEC = 120.0  # upper cap (2 minutes)
# Back-compat alias for imports expecting MAX_DURATION_SEC
MAX_DURATION_SEC = DEFAULT_MAX_DURATION_SEC
SHORT_WIDTH = 1080
SHORT_HEIGHT = 1920

# Vertical centers (fraction of frame height) for on-screen scene text.
# Picked to stay clear of the YouTube Shorts UI: top app bar (~0–10%) and
# bottom action rail / caption block (~80–100%).
def max_duration_sec_from_env() -> float:
    """Max input/output clip length (seconds). Default 120; set SHORTSAI_MAX_DURATION_SEC up to 120."""
    raw = (os.environ.get("SHORTSAI_MAX_DURATION_SEC") or "").strip()
    if not raw:
        return DEFAULT_MAX_DURATION_SEC
    try:
        return max(10.0, min(HARD_MAX_DURATION_SEC, float(raw)))
    except ValueError:
        return DEFAULT_MAX_DURATION_SEC


def clamp_max_duration_sec(sec: float) -> float:
    return max(10.0, min(HARD_MAX_DURATION_SEC, float(sec)))


_OVERLAY_Y_FRACTIONS: dict[str, float] = {
    "upper": 0.25,
    "middle": 0.50,
    "lower": 0.75,
}


def _coerce_overlay_position(raw: str | OverlayPosition | None) -> OverlayPosition:
    """Env / CLI / UI: upper | middle | lower (also top/bottom and *_center). Default: middle."""
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        return "middle"
    if raw in ("upper", "middle", "lower"):
        return cast(OverlayPosition, raw)
    x = str(raw).strip().lower().replace("-", "_")
    if x in ("upper", "top", "upper_center", "upper_third"):
        return "upper"
    if x in ("lower", "bottom", "lower_center", "lower_third"):
        return "lower"
    if x in ("middle", "center", "centre", "mid"):
        return "middle"
    return "middle"


def coerce_overlay_position(raw: str | OverlayPosition | None) -> OverlayPosition:
    """Public alias for :func:`_coerce_overlay_position` (UI / JSON sidecars)."""
    return _coerce_overlay_position(raw)


def normalize_overlay_line_positions(
    count: int,
    positions: list[OverlayPosition] | None,
    default: OverlayPosition,
) -> list[OverlayPosition]:
    """Build a length-``count`` list; missing entries use ``default``."""
    if count <= 0:
        return []
    out: list[OverlayPosition] = []
    for i in range(count):
        if positions and i < len(positions):
            out.append(_coerce_overlay_position(positions[i]))
        else:
            out.append(default)
    return out


def default_overlay_line_times(count: int, duration_sec: float) -> list[tuple[float, float]]:
    """Equal timeline windows for ``count`` scene lines (same as auto burn-in)."""
    if count <= 0:
        return []
    return [_overlay_text_segment_window(i, count, duration_sec) for i in range(count)]


def normalize_overlay_line_times(
    count: int,
    times: list[tuple[float, float]] | None,
    duration_sec: float,
) -> list[tuple[float, float]]:
    """Build a length-``count`` list of (start_sec, end_sec); clamp to [0, duration]."""
    duration_sec = max(0.1, float(duration_sec))
    if count <= 0:
        return []
    defaults = default_overlay_line_times(count, duration_sec)
    min_gap = 0.12
    out: list[tuple[float, float]] = []
    for i in range(count):
        t0, t1 = defaults[i]
        if times and i < len(times):
            try:
                t0 = float(times[i][0])
                t1 = float(times[i][1])
            except (TypeError, ValueError, IndexError):
                pass
        t0 = max(0.0, min(t0, duration_sec))
        t1 = max(0.0, min(t1, duration_sec))
        if t1 <= t0:
            t1 = min(duration_sec, t0 + min_gap)
        if t1 - t0 < min_gap:
            t1 = min(duration_sec, t0 + min_gap)
        out.append((t0, t1))
    return out


def overlay_times_to_meta(times: list[tuple[float, float]]) -> list[dict[str, float]]:
    return [{"start": round(a, 2), "end": round(b, 2)} for a, b in times]


def overlay_times_from_meta(
    raw: Any,
    *,
    count: int,
    duration_sec: float,
) -> list[tuple[float, float]] | None:
    """Parse ``on_screen_overlay_times`` from metadata; returns None if unusable."""
    if not isinstance(raw, list) or not raw:
        return None
    parsed: list[tuple[float, float]] = []
    for item in raw[:count]:
        try:
            if isinstance(item, dict):
                parsed.append((float(item["start"]), float(item["end"])))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                parsed.append((float(item[0]), float(item[1])))
            else:
                return None
        except (KeyError, TypeError, ValueError):
            return None
    if len(parsed) != count:
        return None
    return normalize_overlay_line_times(count, parsed, duration_sec)


def _overlay_y_expression(position: OverlayPosition) -> str:
    """drawtext y= expression that vertically centers the line at the chosen band."""
    if position == "middle":
        # Equivalent to h*0.5-text_h/2; keep the original form for clarity.
        return "(h-text_h)/2"
    frac = _OVERLAY_Y_FRACTIONS.get(position, 0.5)
    return f"h*{frac:g}-text_h/2"

# Copied into work_dir so drawtext can use fontfile=name.ttf (no Windows drive colons in -vf).
_OVERLAY_FONT_LOCAL_NAME = "shortsai_overlay.ttf"

def _scene_overlay_font_size_from_env() -> int:
    """drawtext fontsize for red timed AI scene lines (SHORTSAI_SCENE_OVERLAY_FONT_SIZE; not speech subtitles)."""
    raw = (os.environ.get("SHORTSAI_SCENE_OVERLAY_FONT_SIZE") or "64").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 64
    return max(36, min(100, n))


def resolve_scene_overlay_font_size(override: int | None = None) -> int:
    """Scene line drawtext size; ``override`` wins, else env (default 64, clamped 36–100)."""
    if override is not None:
        try:
            return max(36, min(100, int(override)))
        except (TypeError, ValueError):
            pass
    return _scene_overlay_font_size_from_env()


def _scene_overlay_drawtext_style(font_size: int | None = None) -> str:
    fs = resolve_scene_overlay_font_size(font_size)
    bw = max(2, min(4, fs // 18))
    sh = max(3, min(6, fs // 12))
    return (
        f"fontcolor=0xFF0000:fontsize={fs}:"
        f"borderw={bw}:bordercolor=black@0.85:"
        f"shadowcolor=black@0.75:shadowx={sh}:shadowy={sh}"
    )


def _split_long_tokens(words: list[str], max_chars: int) -> list[str]:
    """Break tokens longer than max_chars so word-wrap can place them."""
    out: list[str] = []
    for w in words:
        if len(w) <= max_chars:
            out.append(w)
        else:
            for i in range(0, len(w), max_chars):
                out.append(w[i : i + max_chars])
    return out


def _wrap_scene_overlay_line(
    text: str,
    *,
    video_width: int,
    fontsize: int,
    max_lines: int = 2,
    margin_px: int = 96,
) -> str:
    """
    Word-wrap for FFmpeg drawtext so large red scene labels fit 9:16 width (center crop safe).
    Returns up to ``max_lines`` lines separated by newlines (UTF-8 textfile for drawtext).
    """
    text = text.strip()
    if not text:
        return text
    usable = max(120, int(video_width) - margin_px)
    # Conservative glyph width so Latin/CJK mix rarely clips horizontally.
    approx_cw = max(float(fontsize) * 0.5, 10.0)
    max_chars = max(6, int(usable / approx_cw))

    words = _split_long_tokens(text.split(), max_chars)
    if not words:
        return text

    wrapped_rows: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur and cur_len + add > max_chars:
            wrapped_rows.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += add
    if cur:
        wrapped_rows.append(" ".join(cur))

    if len(wrapped_rows) <= max_lines:
        return "\n".join(wrapped_rows)

    head = "\n".join(wrapped_rows[: max_lines - 1])
    tail = " ".join(wrapped_rows[max_lines - 1 :]).strip()
    if len(tail) > max_chars:
        tail = tail[: max_chars - 1].rstrip() + "\u2026"
    return f"{head}\n{tail}"


def _drawtext_enable_between(t_start: float, t_end: float) -> str:
    """drawtext enable=between(...); commas must be \\, or FFmpeg splits the -vf chain."""
    return f"enable=between(t\\,{t_start}\\,{t_end})"


def _overlay_text_segment_window(
    segment_index: int, num_segments: int, duration_sec: float
) -> tuple[float, float]:
    """
    Split [0, duration] into num_segments sequential windows for timed captions.
    Slightly shortens non-final segments so inclusive between() does not double-draw at boundaries.
    """
    if num_segments <= 0:
        return 0.0, duration_sec
    if num_segments == 1:
        return 0.0, duration_sec
    lo = duration_sec * segment_index / num_segments
    hi = duration_sec * (segment_index + 1) / num_segments
    if segment_index == num_segments - 1:
        return lo, duration_sec
    # Shrink end so the next segment's start does not overlap both enables for the same t.
    hi = max(lo + 0.12, hi - 0.03)
    return lo, min(hi, duration_sec)


def _resolve_overlay_font_source() -> Path | None:
    env = (os.environ.get("SHORTSAI_DRAWTEXT_FONT") or "").strip()
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    if sys.platform == "win32":
        windir = os.environ.get("WINDIR", "C:\\Windows")
        fonts = Path(windir) / "Fonts"
        candidates.extend(
            [
                fonts / "arialbd.ttf",
                fonts / "calibrib.ttf",
                fonts / "segoeuib.ttf",
                fonts / "arial.ttf",
            ]
        )
    elif sys.platform == "darwin":
        supp = Path("/System/Library/Fonts/Supplemental")
        candidates.extend(
            [
                supp / "Arial Bold.ttf",
                supp / "Arial.ttf",
                Path("/Library/Fonts/Arial Bold.ttf"),
                Path("/Library/Fonts/Arial.ttf"),
            ]
        )
    else:
        candidates.extend(
            [
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
                Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
            ]
        )
    for p in candidates:
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    return None


def _prepare_overlay_font_in_cwd(cwd: Path) -> str:
    """Copy a TTF into cwd; return ':fontfile=shortsai_overlay.ttf' or ''."""
    src = _resolve_overlay_font_source()
    if src is None:
        return ""
    dest = cwd / _OVERLAY_FONT_LOCAL_NAME
    try:
        shutil.copy2(src, dest)
    except OSError:
        return ""
    return f":fontfile={_OVERLAY_FONT_LOCAL_NAME}"


def _coerce_vertical_fit(raw: str | None) -> VerticalFitMode:
    """Env / CLI: letterbox | crop | blur_fill (also blur-fill, blurfill). Default: crop."""
    if not raw or not str(raw).strip():
        return "crop"
    x = str(raw).strip().lower().replace("-", "_")
    if x in ("blur_fill", "blurfill"):
        return "blur_fill"
    if x == "crop":
        return "crop"
    if x == "letterbox":
        return "letterbox"
    return "crop"


def _caption_font_size_from_env() -> int:
    """ASS FontSize for burned-in speech subtitles only (SHORTSAI_CAPTION_FONT_SIZE, default 10)."""
    raw = (os.environ.get("SHORTSAI_CAPTION_FONT_SIZE") or "10").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 10
    return max(1, min(20, n))


def resolve_caption_font_size(override: int | None = None) -> int:
    """Speech caption ASS FontSize; ``override`` wins, else env (default 10, clamped 1–20)."""
    if override is not None:
        try:
            return max(1, min(20, int(override)))
        except (TypeError, ValueError):
            pass
    return _caption_font_size_from_env()


def _speech_subtitles_vf(srt_basename: str, *, font_size: int | None = None) -> str:
    """SRT burn-in with UTF-8 and configurable font size (SHORTSAI_CAPTION_FONT_SIZE, default 10)."""
    fs = resolve_caption_font_size(font_size)
    style = f"FontSize={fs}\\,Outline=2\\,Shadow=0.5"
    return f"subtitles={srt_basename}:charenc=UTF-8:force_style={style}"


def _is_probable_srt(s: str) -> bool:
    t = s.strip()
    return len(t) >= 12 and "-->" in t


def _srt_plain_for_metadata(srt: str) -> str:
    """Join non-timing lines from SRT for title/description context after user edits."""
    parts: list[str] = []
    for line in srt.splitlines():
        s = line.strip()
        if not s or s.isdigit() or "-->" in s:
            continue
        parts.append(s)
    return " ".join(parts).strip()


def _whisper_cache_dict(tr: TranscriptResult) -> dict[str, Any]:
    return {
        "language": tr.language,
        "text": tr.text,
        "words": [{"start": w.start, "end": w.end, "text": w.text} for w in tr.words],
    }


def _transcript_from_whisper_cache(cache: dict[str, Any]) -> TranscriptResult:
    words_raw = cache.get("words")
    if not isinstance(words_raw, list):
        words_raw = []
    words: list[WordSpan] = []
    for w in words_raw:
        if not isinstance(w, dict):
            continue
        try:
            words.append(
                WordSpan(
                    start=float(w["start"]),
                    end=float(w["end"]),
                    text=str(w.get("text", "")).strip(),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    words = [w for w in words if w.text]
    text = str(cache.get("text") or "").strip()
    if not text and words:
        text = " ".join(w.text for w in words)
    lang = str(cache.get("language") or "en")
    return TranscriptResult(language=lang, text=text, words=words)


def _extract_audio(video: Path, wav_out: Path) -> None:
    ffmpeg_util.run_ffmpeg(
        [
            "-y",
            "-i",
            str(video),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(wav_out),
        ]
    )


def _scale_and_subs(
    video_in: Path,
    video_out: Path,
    *,
    cwd: Path,
    srt_name: str | None,
    vertical_fit: VerticalFitMode = "crop",
    caption_font_size: int | None = None,
) -> None:
    w, h = SHORT_WIDTH, SHORT_HEIGHT
    use_subs = False
    if srt_name:
        srt_path = cwd / srt_name
        use_subs = srt_path.exists() and srt_path.stat().st_size > 0

    sub_clause = (
        f",{_speech_subtitles_vf(srt_name, font_size=caption_font_size)}" if use_subs else ""
    )

    if vertical_fit == "letterbox":
        inner = (
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2{sub_clause}"
        )
        filter_complex = f"[0:v]{inner}[vout]"
    elif vertical_fit == "crop":
        inner = (
            f"scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h}:(iw-ow)/2:(ih-oh)/2{sub_clause}"
        )
        filter_complex = f"[0:v]{inner}[vout]"
    else:
        # Blurred full-frame background + letterboxed foreground (boxblur is in default lavfi builds).
        bg = (
            f"scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h}:(iw-ow)/2:(ih-oh)/2,boxblur=25:5"
        )
        fg = (
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"
        )
        if use_subs:
            filter_complex = (
                f"[0:v]split=2[vb_in][fg_in];"
                f"[vb_in]{bg}[bg];"
                f"[fg_in]{fg}[fg];"
                f"[bg][fg]overlay=0:0[vpre];"
                f"[vpre]{_speech_subtitles_vf(srt_name, font_size=caption_font_size)}[vout]"
            )
        else:
            filter_complex = (
                f"[0:v]split=2[vb_in][fg_in];"
                f"[vb_in]{bg}[bg];"
                f"[fg_in]{fg}[fg];"
                f"[bg][fg]overlay=0:0[vout]"
            )

    # Only video through the filter graph; audio is mapped unchanged from input 0.
    # Using -vf + -map 0:a together can yield silent MP4s on some Windows/ffmpeg builds.

    src = cwd / video_in.name
    has_audio = ffmpeg_util.has_audio_stream(src)
    cmd: list[str] = [
        "-y",
        "-i",
        str(video_in.name),
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
    ]
    if has_audio:
        cmd.extend(
            [
                "-map",
                "0:a:0",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
            ]
        )
    cmd.append(str(video_out.name))

    ffmpeg_util.run_ffmpeg(cmd, cwd=cwd)


def _add_text_overlay(
    video_in: Path,
    video_out: Path,
    *,
    overlay_texts: list[str],
    cwd: Path,
    progress_emit: Callable[[str], None],
    overlay_position: OverlayPosition = "middle",
    overlay_positions: list[OverlayPosition] | None = None,
    overlay_times: list[tuple[float, float]] | None = None,
    scene_font_size: int | None = None,
) -> None:
    """Burn scene-based on-screen lines only (title/description/tags stay in metadata, not here).

    ``overlay_position`` is the default vertical band when ``overlay_positions`` is shorter than
    the number of lines. ``overlay_positions[i]`` sets the band for timed line ``i`` (upper /
    middle / lower). ``overlay_times[i]`` is ``(start_sec, end_sec)`` on the video timeline.
    """
    lines = [strip_overlay_quotes(x.strip()) for x in overlay_texts[:4] if x.strip()]
    lines = [x for x in lines if x]
    if not lines:
        raise ValueError("overlay_texts must contain at least one non-empty line")

    pos_per_line = normalize_overlay_line_positions(len(lines), overlay_positions, overlay_position)

    duration_sec = ffmpeg_util.probe_duration_seconds(video_in)
    times_per_line = normalize_overlay_line_times(len(lines), overlay_times, duration_sec)
    # drawtext 't' follows stream PTS; align segment windows to real timeline.
    pts0 = ffmpeg_util.probe_video_start_time_seconds(video_in)
    font_clause = _prepare_overlay_font_in_cwd(cwd)

    def write_drawtext_textfile(line_idx: int, row_idx: int, raw: str, *, max_len: int) -> str:
        """Write overlay string to a cwd-local file; avoids filter parse breaks on : , ' etc."""
        text = strip_overlay_quotes((raw or "").strip()).replace("\n", " ")
        text = text.replace("%", "%%")  # drawtext expands strftime sequences
        if len(text) > max_len:
            text = text[: max_len - 1].rstrip() + "\u2026"
        fname = f"overlay_t_{line_idx:02d}_{row_idx}.txt"
        with open(cwd / fname, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        return fname

    fs_overlay = resolve_scene_overlay_font_size(scene_font_size)
    style = _scene_overlay_drawtext_style(fs_overlay)
    overlay_filters: list[str] = []
    n_lines = len(lines)
    for idx, text_content in enumerate(lines):
        base_y = _overlay_y_expression(pos_per_line[idx])
        wrapped = _wrap_scene_overlay_line(
            text_content,
            video_width=SHORT_WIDTH,
            fontsize=fs_overlay,
            max_lines=2,
        )
        t0, t1 = times_per_line[idx]
        enable = _drawtext_enable_between(pts0 + t0, pts0 + t1)
        parts = [p.strip() for p in wrapped.split("\n") if p.strip()]
        if not parts:
            continue
        # textfile= with cwd-local names (no drive colons); safe for :, commas, quotes in AI text.
        row_gap = max(int(fs_overlay * 0.92), 14)
        if len(parts) >= 2:
            tf_a = write_drawtext_textfile(idx, 0, parts[0], max_len=500)
            tf_b = write_drawtext_textfile(idx, 1, parts[1], max_len=500)
            overlay_filters.append(
                f"drawtext=textfile={tf_a}{font_clause}:{style}:"
                f"x=(w-text_w)/2:y={base_y}-{row_gap}:{enable}"
            )
            overlay_filters.append(
                f"drawtext=textfile={tf_b}{font_clause}:{style}:"
                f"x=(w-text_w)/2:y={base_y}+{max(row_gap // 2, 6)}:{enable}"
            )
        else:
            tf = write_drawtext_textfile(idx, 0, parts[0], max_len=900)
            overlay_filters.append(
                f"drawtext=textfile={tf}{font_clause}:{style}:"
                f"x=(w-text_w)/2:y={base_y}:{enable}"
            )

    # Input is already 9:16 from _scale_and_subs (music step copies video); only burn drawtext.
    vf_inner = ",".join(overlay_filters)
    filter_complex = f"[0:v]{vf_inner}[vout]"

    progress_emit(f"FFmpeg overlay filter_complex: {filter_complex}")
    src = cwd / video_in.name
    has_audio = ffmpeg_util.has_audio_stream(src)
    cmd: list[str] = [
        "-y",
        "-i",
        str(video_in.name),
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
    ]
    if has_audio:
        cmd.extend(["-map", "0:a:0", "-c:a", "copy"])  # keep speech / mixed audio as-is
    cmd.append(str(video_out.name))

    ffmpeg_util.run_ffmpeg(cmd, cwd=cwd)


def _mix_music(
    video_in: Path,
    music: Path,
    video_out: Path,
    *,
    music_volume: float,
    cwd: Path,
) -> None:
    vol = max(0.0, min(1.0, music_volume))
    v_src = cwd / video_in.name
    has_speech = ffmpeg_util.has_audio_stream(v_src)
    # amix requires two inputs; silent video still gets music under speech-style ducking path.
    if has_speech:
        filter_complex = (
            f"[1:a]volume={vol}[m];[0:a][m]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )
    else:
        filter_complex = f"[1:a]volume={vol}[aout]"

    ffmpeg_util.run_ffmpeg(
        [
            "-y",
            "-i",
            str(video_in.name),
            "-i",
            str(music.name),
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
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


def process_upload(
    input_video: Path,
    *,
    work_dir: Path,
    whisper_model: str = "base",
    whisper: WhisperModel | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
    openai_api_key: str | None = None,
    music_path: Path | None = None,
    music_volume: float = 0.18,
    manual_overlay_text: str | None = None,
    progress: Callable[[str], None] | None = None,
    whisper_task: Literal["transcribe", "translate"] | None = None,
    whisper_language_hint: str | None = None,
    vertical_fit: VerticalFitMode | None = None,
    caption_srt_override: str | None = None,
    reuse_whisper_cache: dict[str, Any] | None = None,
    vision_onscreen_subtitles: bool = False,
    vision_onscreen_subtitles_english: bool = False,
    overlay_position: OverlayPosition | None = None,
    overlay_positions: list[OverlayPosition] | None = None,
    scene_overlay_lines_override: list[str] | None = None,
    scene_overlay_times_override: list[tuple[float, float]] | None = None,
    ai_hook_cold_open: bool = False,
    ai_hook_meta: dict[str, Any] | None = None,
    ai_narration: bool = False,
    narration_volume: float | None = None,
    ai_narration_meta: dict[str, Any] | None = None,
    caption_font_size: int | None = None,
    scene_overlay_font_size: int | None = None,
    burn_subtitles: bool = True,
) -> tuple[bytes, dict[str, Any]]:
    """
    Full pipeline: validate duration → transcribe → SRT → 9:16 + burn-in subs → optional music
    → metadata JSON (title/desc/tags) → timed scene text overlays on the export (not metadata fields).
    Returns (mp4_bytes, metadata_dict).

    ``vertical_fit``: letterbox / crop / blur_fill; ``None`` uses env ``SHORTSAI_VERTICAL_FIT`` (default crop).

    ``caption_srt_override``: full SRT file contents to burn instead of auto cues from Whisper word timings
    (must look like valid SRT, e.g. contain ``-->``). Use with ``reuse_whisper_cache`` from a prior export
    to skip re-running Whisper on re-export.

    ``reuse_whisper_cache``: dict with keys ``language``, ``text``, ``words`` (list of ``start``/``end``/``text``)
    from metadata ``whisper_cache`` — skips transcription when valid.

    ``vision_onscreen_subtitles``: when speech is missing or very thin, sample the source video and use
    vision to read on-screen text into burned-in SRT (needs ``openai_api_key``).

    ``vision_onscreen_subtitles_english``: if True with vision subtitles, translate visible text to English.

    ``overlay_position``: default vertical band when a line has no entry in ``overlay_positions``
    (and for env ``SHORTSAI_OVERLAY_POSITION`` when ``None``).

    ``overlay_positions``: optional list, one ``upper`` / ``middle`` / ``lower`` per timed scene line
    (after sanitization). Shorter lists are padded with ``overlay_position``.

    ``scene_overlay_lines_override``: if set to a non-empty list of strings, skip vision/text overlay
    generation and burn these lines instead (e.g. re-export with new positions only).

    ``scene_overlay_times_override``: optional ``(start_sec, end_sec)`` per line; same order as
    override lines (or auto-generated lines). Clamped to video duration.

    ``ai_hook_cold_open``: prepend a short AI-picked clip from the source as a cold open (needs
    ``openai_api_key`` for vision; heuristic fallback without). Total length stays ≤ configured max
    (``SHORTSAI_MAX_DURATION_SEC``, default 120s, max 120s).
    (hook + trimmed main). Pass prior ``ai_hook`` metadata via ``ai_hook_meta`` to reuse the same window
    on re-export.

    ``ai_narration``: generate a short English voiceover (GPT script + OpenAI TTS), duck original
    speech, and mix optional background music under the narration. Requires ``openai_api_key``.

    ``ai_narration_meta``: prior export ``ai_narration`` block with ``segment_lines`` — reuses the
    same script/timing on re-export (re-synthesizes TTS) instead of re-planning from vision/GPT.

    ``caption_font_size`` / ``scene_overlay_font_size``: burned-in speech caption ASS size (1–20)
    and red timed scene drawtext size (36–100). ``None`` uses env defaults.

    ``burn_subtitles``: when False, still transcribe and save SRT in metadata but skip burning
    speech/narration captions into the video (red scene overlays are unchanged).

    Speech subtitles are unaffected by scene overlay size; scene lines are unaffected by caption size.
    """
    log = progress or (lambda _m: None)

    resolved_caption_font_size = resolve_caption_font_size(caption_font_size)
    resolved_scene_font_size = resolve_scene_overlay_font_size(scene_overlay_font_size)

    resolved_vertical_fit: VerticalFitMode = (
        vertical_fit if vertical_fit is not None else _coerce_vertical_fit(os.environ.get("SHORTSAI_VERTICAL_FIT"))
    )
    resolved_overlay_position: OverlayPosition = (
        overlay_position
        if overlay_position is not None
        else _coerce_overlay_position(os.environ.get("SHORTSAI_OVERLAY_POSITION"))
    )

    ffmpeg_util.require_ffmpeg()
    max_dur = max_duration_sec_from_env()
    dur = ffmpeg_util.probe_duration_seconds(input_video)
    if dur > max_dur + 0.05:
        raise ValueError(
            f"Video is {dur:.1f}s; max allowed is {max_dur:.0f}s "
            f"(set SHORTSAI_MAX_DURATION_SEC up to {HARD_MAX_DURATION_SEC:.0f})."
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    stem = "src" + input_video.suffix.lower()
    local_in = work_dir / stem
    shutil.copy2(input_video, local_in)

    ai_hook_block: dict[str, Any] = {"applied": False}
    if ai_hook_cold_open:
        hook_len = hook_duration_from_env()
        prior = hook_selection_from_meta(ai_hook_meta) if ai_hook_meta else None
        if prior:
            log(
                f"AI hook: reusing {prior.start:.2f}s–{prior.end:.2f}s from prior export "
                f"({prior.reason or prior.method})."
            )
            hook_sel = prior
        else:
            log(f"AI hook: scanning source for a {hook_len:.1f}s cold open…")
            hook_sel = select_hook_window(
                local_in,
                work_dir,
                duration_sec=dur,
                hook_len=hook_len,
                api_key=openai_api_key,
                log=log,
            )
        if hook_sel:
            hooked_path = work_dir / "with_hook_src.mp4"
            try:
                out_dur = apply_hook_cold_open(
                    local_in,
                    hooked_path,
                    hook=hook_sel,
                    hook_len=hook_len,
                    max_output_sec=max_dur,
                    cwd=work_dir,
                )
                local_in = hooked_path
                dur = out_dur
                ai_hook_block = hook_selection_to_meta(
                    hook_sel, hook_len=hook_len, output_duration=out_dur
                )
                log(f"AI hook applied; timeline is now {dur:.2f}s.")
            except ffmpeg_util.FFmpegError as e:
                log(f"AI hook failed ({str(e)[:200]}); continuing with original clip.")
                ai_hook_block = {"applied": False, "error": str(e)[:300]}

    log("Extracting audio…")
    wav = work_dir / "speech.wav"

    # Check if video has audio stream
    if not ffmpeg_util.has_audio_stream(local_in):
        log("Warning: Video file has no audio stream")
        # Create empty WAV file so transcription doesn't fail
        wav.write_bytes(b"")  # Empty file
    else:
        _extract_audio(local_in, wav)

    # Check if audio file was created and has content
    if wav.exists():
        audio_size = wav.stat().st_size
        log(f"Audio extracted: {audio_size} bytes")
        if audio_size < 1000:  # Very small file might be empty
            log("Warning: Audio file is very small, may be empty")
    else:
        log("Error: Audio file was not created")

    log("Transcribing (first run may download the Whisper model)…")

    visual_fallback = ""
    use_whisper_cache = (
        reuse_whisper_cache is not None
        and isinstance(reuse_whisper_cache, dict)
        and isinstance(reuse_whisper_cache.get("words"), list)
        and len(reuse_whisper_cache["words"]) > 0
    )

    if use_whisper_cache:
        log("Skipping Whisper — reusing cached word timings from a prior run.")
        tr = _transcript_from_whisper_cache(reuse_whisper_cache)
    elif wav.stat().st_size == 0:
        log("Audio file is empty - no transcription possible")
        tr = TranscriptResult(language="en", text="", words=[])
    else:
        if whisper_task is not None:
            resolved_wtask: Literal["transcribe", "translate"] = whisper_task
        else:
            task_raw = (os.environ.get("SHORTSAI_WHISPER_TASK") or "transcribe").strip().lower()
            resolved_wtask = cast(
                Literal["transcribe", "translate"],
                "translate" if task_raw == "translate" else "transcribe",
            )
        if resolved_wtask == "translate":
            log("Whisper task=translate (subtitles and transcript will be English).")
        # None = caller did not pass (e.g. script): use .env. Empty str from UI = auto-detect.
        if whisper_language_hint is None:
            lang_hint = (os.environ.get("SHORTSAI_WHISPER_LANGUAGE") or "").strip().lower() or None
        else:
            lang_hint = whisper_language_hint.strip().lower() or None
        if lang_hint:
            log(f"Whisper language hint: {lang_hint}")
        tr = transcribe(
            wav,
            model=whisper,
            model_name=whisper_model,
            device=device,
            compute_type=compute_type,
            language=lang_hint,
            task=resolved_wtask,
        )

    if not tr.words:
        visual_fallback = input_video.stem.replace("_", " ").replace("-", " ")

    log(f"Transcription result: language={tr.language}, text length={len(tr.text)}, words detected={len(tr.words)}")
    if not tr.words:
        log("No words detected - check if audio has speech or if volume is too low")
        log("Using filename-based fallback as visual context for metadata")
        visual_fallback = input_video.stem.replace("_", " ").replace("-", " ")
    else:
        visual_fallback = ""

    auto_srt = words_to_srt(tr.words)
    ov = (caption_srt_override or "").strip()

    narration_cues: list[NarrationCue] | None = None
    prior_narr = narration_cues_from_meta(ai_narration_meta) if ai_narration_meta else None
    if ai_narration and prior_narr:
        log(f"Reusing AI narration ({len(prior_narr)} lines) from prior export.")
        narration_cues = prior_narr
    elif (
        ai_narration
        and openai_api_key
        and not (ov and _is_probable_srt(ov))
    ):
        try:
            log("Planning scene narration (for voice + captions)…")
            narration_cues, _narr_plan_src = resolve_scene_narration_cues(
                transcript=tr.text,
                words=tr.words,
                duration_sec=dur,
                api_key=openai_api_key,
                video_path=local_in,
                work_dir=work_dir,
                visual_hint=visual_fallback or tr.text,
                log=log,
            )
        except Exception as e:
            log(f"Scene narration planning failed ({str(e)[:160]}); captions stay on speech.")
            narration_cues = None

    burned_subtitle_source = "none"
    if ov and _is_probable_srt(ov):
        srt_text = ov
        burned_subtitle_source = "override"
        log("Using user-provided SRT for burned-in speech captions.")
    elif narration_cues:
        srt_text = narration_cues_to_srt(narration_cues)
        burned_subtitle_source = "ai_narration"
        log("Burned-in captions will show AI narration lines (timed per scene).")
    elif ov:
        log("Caption SRT override ignored (not valid SRT); using auto-generated cues from speech.")
        srt_text = auto_srt
        burned_subtitle_source = "whisper" if auto_srt.strip() else "none"
    else:
        srt_text = auto_srt
        burned_subtitle_source = "whisper" if auto_srt.strip() else "none"

    thin_for_vision = not tr.words or _transcript_insufficient_for_topic(tr.text.strip())
    use_onscreen = (
        vision_onscreen_subtitles
        and openai_api_key
        and thin_for_vision
        and not (ov and _is_probable_srt(ov))
        and not narration_cues
    )
    if use_onscreen:
        log("Building burned-in captions from on-screen text (vision)…")
        n_sub = 6
        prefix = "sub_vision"
        try:
            groups = ffmpeg_util.extract_jpeg_frames_per_segment(
                local_in,
                work_dir,
                n_segments=n_sub,
                file_prefix=prefix,
            )
            if any(groups):
                vision_srt = generate_onscreen_subtitle_srt_from_segments(
                    groups,
                    duration_sec=dur,
                    api_key=openai_api_key,
                    translate_to_english=vision_onscreen_subtitles_english,
                )
                if vision_srt.strip():
                    srt_text = vision_srt
                    burned_subtitle_source = (
                        "vision_onscreen_en" if vision_onscreen_subtitles_english else "vision_onscreen"
                    )
                    log("Vision-based on-screen subtitle SRT applied.")
                else:
                    log("Vision on-screen subtitles returned empty; keeping speech-based SRT if any.")
        except ffmpeg_util.FFmpegError as e:
            log(f"Frame extraction for on-screen subtitles failed ({str(e)[:200]}).")
        finally:
            for fp in work_dir.glob(f"{prefix}_*.jpg"):
                fp.unlink(missing_ok=True)

    srt_path = work_dir / "captions.srt"
    srt_name: str | None = None
    if srt_text.strip():
        srt_path.write_text(srt_text, encoding="utf-8")
        srt_name = srt_path.name
        log(f"SRT file created with {len(srt_text.splitlines())//4} subtitle cues")
        # Verify file was written
        if srt_path.exists() and srt_path.stat().st_size > 0:
            log(f"SRT file exists and has {srt_path.stat().st_size} bytes")
        else:
            log("Warning: SRT file was created but appears empty")
            srt_name = None
    else:
        log("No SRT content generated (no words detected)")

    srt_burn_name = srt_name if burn_subtitles else None
    if srt_name and not burn_subtitles:
        log("Skipping burned-in speech captions (disabled); SRT still saved in metadata.")
    log(
        "Rendering vertical 9:16 video"
        + (f" ({resolved_vertical_fit})" if resolved_vertical_fit != "crop" else "")
        + (" with captions…" if srt_burn_name else "…")
    )
    scaled = work_dir / "scaled_subs.mp4"
    _scale_and_subs(
        local_in,
        scaled,
        cwd=work_dir,
        srt_name=srt_burn_name,
        vertical_fit=resolved_vertical_fit,
        caption_font_size=resolved_caption_font_size,
    )

    log("Generating metadata…")
    meta_hint = _srt_plain_for_metadata(srt_text) if srt_text.strip() else ""
    metadata_input = (meta_hint or tr.text).strip() or visual_fallback
    caption_plain = metadata_input
    if _transcript_insufficient_for_topic(metadata_input) and openai_api_key:
        log("Transcript short or empty—using video frames for title/description when possible…")
    meta = generate_metadata(
        metadata_input,
        api_key=openai_api_key,
        video_path=scaled,
        work_dir=work_dir,
    )
    meta["transcript"] = tr.text
    meta["subtitles_burned"] = bool(burn_subtitles and srt_name)
    meta["burned_subtitle_source"] = burned_subtitle_source if burn_subtitles else "none"
    if not burn_subtitles and burned_subtitle_source != "none":
        meta["subtitle_srt_source"] = burned_subtitle_source
    meta["visual_description"] = visual_fallback
    meta["language"] = tr.language
    meta["ai_hook"] = ai_hook_block
    meta["vertical_fit"] = resolved_vertical_fit
    meta["caption_font_size"] = resolved_caption_font_size
    meta["scene_overlay_font_size"] = resolved_scene_font_size

    narration_block: dict[str, Any] = narration_to_meta(applied=False)
    narration_path: Path | None = None
    if ai_narration:
        if not openai_api_key:
            log("AI narration skipped: set OPENAI_API_KEY in .env.")
            narration_block = narration_to_meta(
                applied=False, error="OPENAI_API_KEY not set"
            )
        elif not narration_cues:
            log("AI narration skipped: no scene lines were planned.")
            narration_block = narration_to_meta(
                applied=False, error="scene narration planning failed"
            )
        else:
            try:
                scaled_dur = ffmpeg_util.probe_duration_seconds(scaled)
                voice = narration_voice_from_meta(ai_narration_meta)
                cues = narration_cues
                narr_mp3 = work_dir / "narration_timed.mp3"
                build_timed_narration_track(
                    cues,
                    narr_mp3,
                    video_duration=scaled_dur,
                    work_dir=work_dir,
                    api_key=openai_api_key,
                    voice=voice,
                    log=log,
                )
                narration_path = narr_mp3
                script_display = " | ".join(c.text for c in cues)
                narr_vol_meta = (
                    narration_volume
                    if narration_volume is not None
                    else narration_volume_from_env()
                )
                narration_block = narration_to_meta(
                    applied=True,
                    script=script_display,
                    voice=voice,
                    source="scene_timed",
                    segment_lines=cues_to_meta_segments(cues),
                    volume=narr_vol_meta,
                )
            except Exception as e:
                log(f"AI narration failed ({str(e)[:200]}); export without voiceover.")
                narration_block = narration_to_meta(applied=False, error=str(e))

    meta["ai_narration"] = narration_block

    music_local: Path | None = None
    if music_path is not None and music_path.is_file():
        music_local = work_dir / ("music" + music_path.suffix.lower())
        shutil.copy2(music_path, music_local)

    final_video = scaled
    if narration_path is not None or music_local is not None:
        narr_vol = (
            narration_volume
            if narration_volume is not None
            else narration_volume_from_env()
        )
        if narration_path:
            log(
                f"Mixing AI narration (original speech ducked, narration gain {narr_vol:.2f})…"
            )
        if music_local:
            log(f"Mixing music: {music_local.name} at volume {music_volume}")
        out_mix = work_dir / "final.mp4"
        mix_video_audio_bed(
            scaled,
            out_mix,
            cwd=work_dir,
            narration=narration_path,
            music=music_local,
            music_volume=music_volume,
            original_volume=1.0,
            narration_volume=narr_vol,
            narration_timed=bool(narration_path),
        )
        final_video = out_mix
        log("Audio mix complete")

    dur = ffmpeg_util.probe_duration_seconds(final_video)
    meta["duration_seconds"] = round(dur, 2)

    # On-screen lines: always scene-based (vision or text). Title/description/tags stay in metadata.json only.
    overlay_texts: list[str] = []
    meta["overlay_text_source"] = None

    override_raw = (
        [x for x in (scene_overlay_lines_override or []) if x is not None and str(x).strip()]
    )
    if override_raw:
        overlay_texts = sanitize_overlay_lines([str(x).strip() for x in override_raw])
        meta["overlay_text_source"] = "override"
    elif manual_overlay_text and manual_overlay_text.strip():
        overlay_texts = [strip_overlay_quotes(manual_overlay_text.strip())]
        meta["overlay_text_source"] = "manual"
    else:
        spoken = " ".join(x for x in [caption_plain, visual_fallback] if x).strip()
        if openai_api_key:
            log("Analyzing video by segment for on-screen text (vision)…")
            vision_groups: list[list[Path]] = []
            try:
                n_vision_segments = 3
                vision_groups = ffmpeg_util.extract_jpeg_frames_per_segment(
                    final_video,
                    work_dir,
                    n_segments=n_vision_segments,
                    file_prefix="overlay_vision",
                )
                flat_paths = [p for g in vision_groups for p in g]
                if flat_paths:
                    overlay_dur = ffmpeg_util.probe_duration_seconds(final_video)
                    overlay_texts = generate_overlay_text_from_vision_segments(
                        vision_groups,
                        duration_sec=overlay_dur,
                        api_key=openai_api_key,
                        spoken_context=spoken,
                    )
                    if overlay_texts:
                        meta["overlay_text_source"] = "vision"
                    else:
                        log("Vision analysis returned no caption lines; using text fallback.")
            except ffmpeg_util.FFmpegError as e:
                log(f"Frame extraction for vision failed ({str(e)[:200]}); using text fallback.")
            finally:
                for fp in work_dir.glob("overlay_vision_*.jpg"):
                    fp.unlink(missing_ok=True)

        if not overlay_texts:
            visual_source = (
                meta.get("description", "") or meta.get("title", "") or visual_fallback or ""
            )
            text_for_lines = "\n".join(
                x for x in [visual_source, caption_plain[:4000]] if x
            ).strip()
            if not text_for_lines:
                text_for_lines = visual_fallback or "Short clip"
            overlay_texts = generate_overlay_text(text_for_lines, api_key=openai_api_key)
            if not overlay_texts:
                raw_fallback = " ".join(
                    x.strip()
                    for x in [meta.get("title", ""), meta.get("description", ""), visual_fallback, caption_plain]
                    if x
                )
                if raw_fallback:
                    overlay_texts = generate_overlay_text(raw_fallback, api_key=openai_api_key)
                if not overlay_texts:
                    overlay_texts = [text_for_lines[:50] if text_for_lines else "Short clip"]
            if meta.get("overlay_text_source") is None:
                meta["overlay_text_source"] = "text_fallback"

    overlay_texts = sanitize_overlay_lines(overlay_texts)

    per_line_pos = normalize_overlay_line_positions(
        len(overlay_texts), overlay_positions, resolved_overlay_position
    )
    overlay_dur = ffmpeg_util.probe_duration_seconds(final_video)
    per_line_times = normalize_overlay_line_times(
        len(overlay_texts),
        scene_overlay_times_override,
        overlay_dur,
    )
    meta["on_screen_overlay_lines"] = list(overlay_texts)
    meta["on_screen_overlay_positions"] = list(per_line_pos)
    meta["on_screen_overlay_times"] = overlay_times_to_meta(per_line_times)
    meta["scene_overlay_applied"] = False
    meta["overlay_position"] = resolved_overlay_position

    log(
        f"Applying scene text overlays… overlay_texts={overlay_texts} "
        f"positions={per_line_pos} times={per_line_times}"
    )
    try:
        overlay_video = work_dir / "with_overlays.mp4"
        _add_text_overlay(
            final_video,
            overlay_video,
            overlay_texts=overlay_texts,
            cwd=work_dir,
            progress_emit=log,
            overlay_position=resolved_overlay_position,
            overlay_positions=per_line_pos,
            overlay_times=per_line_times,
            scene_font_size=resolved_scene_font_size,
        )
        final_video = overlay_video
        meta["scene_overlay_applied"] = True
        log(f"Scene text overlays applied; lines={len(overlay_texts)}")
    except Exception as e:
        log(f"Warning: Scene text overlay failed ({str(e)[:300]}), continuing without overlays")
        meta["scene_overlay_error"] = str(e)[:500]

    meta["speech_srt"] = srt_text.strip()
    meta["whisper_cache"] = _whisper_cache_dict(tr)

    mp4_bytes = final_video.read_bytes()
    return mp4_bytes, meta


def metadata_to_json_bytes(meta: dict[str, Any]) -> bytes:
    # whisper_cache is large and only for in-app re-export; keep sidecar JSON smaller.
    slim = {k: v for k, v in meta.items() if k != "whisper_cache"}
    return json.dumps(slim, ensure_ascii=False, indent=2).encode("utf-8")
