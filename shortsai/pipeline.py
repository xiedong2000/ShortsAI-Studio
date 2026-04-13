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
    generate_metadata,
    generate_overlay_text,
    generate_overlay_text_from_vision_segments,
    sanitize_overlay_lines,
    strip_overlay_quotes,
)
from shortsai.srt_build import words_to_srt
from shortsai.transcribe import transcribe


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
) -> None:
    base = (
        f"scale={SHORT_WIDTH}:{SHORT_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={SHORT_WIDTH}:{SHORT_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
    )

    # Check if SRT file exists and has content before using subtitles filter
    if srt_name:
        srt_path = cwd / srt_name
        if srt_path.exists() and srt_path.stat().st_size > 0:
            # Use just the filename for the subtitles filter (ffmpeg will look in cwd)
            # Avoid Windows path issues by not using absolute paths with backslashes
            vf_inner = f"{base},subtitles={srt_name}"
        else:
            vf_inner = base  # No subtitles if file doesn't exist
    else:
        vf_inner = base

    # Only video through the filter graph; audio is mapped unchanged from input 0.
    # Using -vf + -map 0:a together can yield silent MP4s on some Windows/ffmpeg builds.
    filter_complex = f"[0:v]{vf_inner}[vout]"

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

    # Build filter complex
    vf_parts = [f"scale={SHORT_WIDTH}:{SHORT_HEIGHT}:force_original_aspect_ratio=decrease,pad={SHORT_WIDTH}:{SHORT_HEIGHT}:(ow-iw)/2:(oh-ih)/2"]
    vf_parts.extend(overlay_filters)

    vf_inner = ",".join(vf_parts)
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
) -> tuple[bytes, dict[str, Any]]:
    """
    Full pipeline: validate duration → transcribe → SRT → 9:16 + burn-in subs → optional music
    → metadata JSON (title/desc/tags) → timed scene text overlays on the export (not metadata fields).
    Returns (mp4_bytes, metadata_dict).
    """
    log = progress or (lambda _m: None)

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
    # Check if audio file is empty
    if wav.stat().st_size == 0:
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


    srt_text = words_to_srt(tr.words)
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

    log("Rendering vertical 9:16 video" + (" with captions…" if srt_name else "…"))
    scaled = work_dir / "scaled_subs.mp4"
    _scale_and_subs(local_in, scaled, cwd=work_dir, srt_name=srt_name)

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
    metadata_input = tr.text.strip() or visual_fallback
    meta = generate_metadata(metadata_input, api_key=openai_api_key)
    meta["transcript"] = tr.text
    meta["visual_description"] = visual_fallback
    meta["language"] = tr.language
    meta["duration_seconds"] = round(dur, 2)

    # On-screen lines: always scene-based (vision or text). Title/description/tags stay in metadata.json only.
    overlay_texts: list[str] = []
    meta["overlay_text_source"] = None
    if manual_overlay_text and manual_overlay_text.strip():
        overlay_texts = [manual_overlay_text.strip()]
        meta["overlay_text_source"] = "manual"
    else:
        spoken = " ".join(x for x in [tr.text.strip(), visual_fallback] if x).strip()
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
                x for x in [visual_source, tr.text.strip()[:4000]] if x
            ).strip()
            if not text_for_lines:
                text_for_lines = visual_fallback or "Short clip"
            overlay_texts = generate_overlay_text(text_for_lines, api_key=openai_api_key)
            if not overlay_texts:
                raw_fallback = " ".join(
                    x.strip()
                    for x in [meta.get("title", ""), meta.get("description", ""), visual_fallback, tr.text.strip()]
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

    mp4_bytes = final_video.read_bytes()
    return mp4_bytes, meta


def metadata_to_json_bytes(meta: dict[str, Any]) -> bytes:
    return json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")
