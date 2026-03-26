from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

from faster_whisper import WhisperModel

from shortsai import ffmpeg_util
from shortsai.metadata_llm import generate_metadata
from shortsai.srt_build import words_to_srt
from shortsai.transcribe import transcribe


MAX_DURATION_SEC = 60.0
SHORT_WIDTH = 1080
SHORT_HEIGHT = 1920


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
    vf = f"{base},subtitles={srt_name}" if srt_name else base
    ffmpeg_util.run_ffmpeg(
        [
            "-y",
            "-i",
            str(video_in.name),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(video_out.name),
        ],
        cwd=cwd,
    )


def _mix_music(
    video_in: Path,
    music: Path,
    video_out: Path,
    *,
    music_volume: float,
    cwd: Path,
) -> None:
    vol = max(0.0, min(1.0, music_volume))
    filter_complex = (
        f"[1:a]volume={vol}[m];[0:a][m]amix=inputs=2:duration=first:dropout_transition=2[a]"
    )
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
            "0:v",
            "-map",
            "[a]",
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
    progress: Callable[[str], None] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """
    Full pipeline: validate duration → transcribe → SRT → 9:16 + burn-in subs → optional music → metadata JSON.
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
    _extract_audio(local_in, wav)

    log("Transcribing (first run may download the Whisper model)…")
    tr = transcribe(
        wav,
        model=whisper,
        model_name=whisper_model,
        device=device,
        compute_type=compute_type,
    )

    srt_text = words_to_srt(tr.words)
    srt_path = work_dir / "captions.srt"
    srt_name: str | None = None
    if srt_text.strip():
        srt_path.write_text(srt_text, encoding="utf-8")
        srt_name = srt_path.name

    log("Rendering vertical 9:16 video" + (" with captions…" if srt_name else "…"))
    scaled = work_dir / "scaled_subs.mp4"
    _scale_and_subs(local_in, scaled, cwd=work_dir, srt_name=srt_name)

    final_video = scaled
    if music_path is not None and music_path.is_file():
        log("Mixing YouTube Audio Library track…")
        music_local = work_dir / ("music" + music_path.suffix.lower())
        shutil.copy2(music_path, music_local)
        out_mix = work_dir / "final.mp4"
        _mix_music(scaled, music_local, out_mix, music_volume=music_volume, cwd=work_dir)
        final_video = out_mix

    log("Generating metadata…")
    meta = generate_metadata(tr.text, api_key=openai_api_key)
    meta["transcript"] = tr.text
    meta["language"] = tr.language
    meta["duration_seconds"] = round(dur, 2)

    mp4_bytes = final_video.read_bytes()
    return mp4_bytes, meta


def metadata_to_json_bytes(meta: dict[str, Any]) -> bytes:
    return json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")
