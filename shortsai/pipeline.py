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
    generate_overlay_text,
    generate_overlay_text_from_vision_segments,
    sanitize_overlay_lines,
    strip_overlay_quotes,
)
from shortsai.srt_build import words_to_srt
from shortsai.transcribe import TranscriptResult, WordSpan, transcribe

VerticalFitMode = Literal["letterbox", "crop", "blur_fill"]

MAX_DURATION_SEC = 60.0
SHORT_WIDTH = 1080
SHORT_HEIGHT = 1920

# Copied into work_dir so drawtext can use fontfile=name.ttf (no Windows drive colons in -vf).
_OVERLAY_FONT_LOCAL_NAME = "shortsai_overlay.ttf"

# Scene overlay (drawtext): bold face via TTF, red fill, shadow + thin outline for contrast.
_OVERLAY_DRAWTEXT_STYLE = (
    "fontcolor=0xFF0000:fontsize=44:"
    "borderw=2:bordercolor=black@0.85:"
    "shadowcolor=black@0.75:shadowx=4:shadowy=4"
)


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
    """ASS FontSize for burned-in speech subtitles (libass default can look large on 1080×1920)."""
    raw = (os.environ.get("SHORTSAI_CAPTION_FONT_SIZE") or "14").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 14
    return max(10, min(44, n))


def _speech_subtitles_vf(srt_basename: str) -> str:
    """SRT burn-in with UTF-8 and configurable font size (SHORTSAI_CAPTION_FONT_SIZE, default 14)."""
    fs = _caption_font_size_from_env()
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
) -> None:
    w, h = SHORT_WIDTH, SHORT_HEIGHT
    use_subs = False
    if srt_name:
        srt_path = cwd / srt_name
        use_subs = srt_path.exists() and srt_path.stat().st_size > 0

    sub_clause = f",{_speech_subtitles_vf(srt_name)}" if use_subs else ""

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
                f"[vpre]{_speech_subtitles_vf(srt_name)}[vout]"
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
) -> None:
    """Burn scene-based on-screen lines only (title/description/tags stay in metadata, not here)."""
    lines = [strip_overlay_quotes(x.strip()) for x in overlay_texts[:4] if x.strip()]
    lines = [x for x in lines if x]
    if not lines:
        raise ValueError("overlay_texts must contain at least one non-empty line")

    duration_sec = ffmpeg_util.probe_duration_seconds(video_in)
    # drawtext 't' follows stream PTS; align segment windows to real timeline.
    pts0 = ffmpeg_util.probe_video_start_time_seconds(video_in)
    font_clause = _prepare_overlay_font_in_cwd(cwd)

    # Helper to escape text for drawtext filter
    def escape_text(text: str) -> str:
        # Commas separate filters in -vf; escape so text like "a, b" stays inside drawtext.
        text = text.replace("\\", "\\\\")
        text = text.replace(",", "\\,")
        text = text.replace("'", "`")
        text = text.replace('"', '\\"')
        text = text.replace("%", "%%")
        text = text.replace(":", "\\:")
        text = text.replace("[", "\\[")
        text = text.replace("]", "\\]")
        return text

    overlay_filters: list[str] = []
    n_lines = len(lines)
    for idx, text_content in enumerate(lines):
        esc_text = escape_text(text_content)
        t0, t1 = _overlay_text_segment_window(idx, n_lines, duration_sec)
        overlay_filters.append(
            f'drawtext=text="{esc_text}"{font_clause}:{_OVERLAY_DRAWTEXT_STYLE}:'
            f"x=(w-text_w)/2:y=(h-text_h)/2:{_drawtext_enable_between(pts0 + t0, pts0 + t1)}"
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
    """
    log = progress or (lambda _m: None)

    resolved_vertical_fit: VerticalFitMode = (
        vertical_fit if vertical_fit is not None else _coerce_vertical_fit(os.environ.get("SHORTSAI_VERTICAL_FIT"))
    )

    ffmpeg_util.require_ffmpeg()
    dur = ffmpeg_util.probe_duration_seconds(input_video)
    if dur > MAX_DURATION_SEC + 0.05:
        raise ValueError(f"Video is {dur:.1f}s; max allowed is {MAX_DURATION_SEC:.0f}s for this version.")

    work_dir.mkdir(parents=True, exist_ok=True)
    stem = "src" + input_video.suffix.lower()
    local_in = work_dir / stem
    shutil.copy2(input_video, local_in)

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
    if ov and _is_probable_srt(ov):
        srt_text = ov
        log("Using user-provided SRT for burned-in speech captions.")
    elif ov:
        log("Caption SRT override ignored (not valid SRT); using auto-generated cues from speech.")
        srt_text = auto_srt
    else:
        srt_text = auto_srt

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

    log(
        "Rendering vertical 9:16 video"
        + (f" ({resolved_vertical_fit})" if resolved_vertical_fit != "crop" else "")
        + (" with captions…" if srt_name else "…")
    )
    scaled = work_dir / "scaled_subs.mp4"
    _scale_and_subs(
        local_in,
        scaled,
        cwd=work_dir,
        srt_name=srt_name,
        vertical_fit=resolved_vertical_fit,
    )

    final_video = scaled
    if music_path is not None and music_path.is_file():
        log(f"Mixing music: {music_path.name} at volume {music_volume}")
        music_local = work_dir / ("music" + music_path.suffix.lower())
        shutil.copy2(music_path, music_local)
        out_mix = work_dir / "final.mp4"
        _mix_music(scaled, music_local, out_mix, music_volume=music_volume, cwd=work_dir)
        final_video = out_mix
        log("Music mixing complete")
    else:
        log("No music selected or music file not found")

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
    meta["visual_description"] = visual_fallback
    meta["language"] = tr.language
    meta["duration_seconds"] = round(dur, 2)
    meta["vertical_fit"] = resolved_vertical_fit

    # On-screen lines: always scene-based (vision or text). Title/description/tags stay in metadata.json only.
    overlay_texts: list[str] = []
    meta["overlay_text_source"] = None
    if manual_overlay_text and manual_overlay_text.strip():
        overlay_texts = [manual_overlay_text.strip()]
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

    meta["on_screen_overlay_lines"] = list(overlay_texts)
    meta["scene_overlay_applied"] = False

    log(f"Applying scene text overlays… overlay_texts={overlay_texts}")
    try:
        overlay_video = work_dir / "with_overlays.mp4"
        _add_text_overlay(
            final_video,
            overlay_video,
            overlay_texts=overlay_texts,
            cwd=work_dir,
            progress_emit=log,
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
