from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def _exe_names() -> tuple[str, str]:
    if sys.platform == "win32":
        return "ffmpeg.exe", "ffprobe.exe"
    return "ffmpeg", "ffprobe"


def ffmpeg_binaries() -> tuple[Path, Path]:
    """
    Resolve ffmpeg and ffprobe executables.

    Order:
    1. FFMPEG_PATH and FFPROBE_PATH (full paths to each binary)
    2. SHORTSAI_FFMPEG_DIR (folder containing both; typical on Windows when PATH is not set)
    3. PATH (shutil.which)
    """
    ff_name, fp_name = _exe_names()

    p_ffmpeg = os.environ.get("FFMPEG_PATH", "").strip()
    p_ffprobe = os.environ.get("FFPROBE_PATH", "").strip()
    if p_ffmpeg and p_ffprobe:
        a, b = Path(p_ffmpeg), Path(p_ffprobe)
        if a.is_file() and b.is_file():
            return a, b
        raise FFmpegError(
            "FFMPEG_PATH / FFPROBE_PATH are set but one or both files were not found. "
            f"FFMPEG_PATH={p_ffmpeg!r} FFPROBE_PATH={p_ffprobe!r}"
        )

    d = os.environ.get("SHORTSAI_FFMPEG_DIR", "").strip()
    if d:
        folder = Path(d).expanduser()
        a, b = folder / ff_name, folder / fp_name
        if a.is_file() and b.is_file():
            return a, b
        raise FFmpegError(
            f"SHORTSAI_FFMPEG_DIR is set to {d!r} but {ff_name} and/or {fp_name} were not found there. "
            "Point it at the folder that contains both executables (often ...\\ffmpeg\\bin after unzipping)."
        )

    w1 = shutil.which("ffmpeg")
    w2 = shutil.which("ffprobe")
    if w1 and w2:
        return Path(w1), Path(w2)

    raise FFmpegError(
        "Could not find ffmpeg and ffprobe.\n\n"
        "• Install them and ensure your **terminal** sees them on PATH, then **restart** the terminal "
        "and Streamlit (Windows: `winget install Gyan.FFmpeg`).\n"
        "• Or set **SHORTSAI_FFMPEG_DIR** in `.env` to the folder that contains `ffmpeg.exe` and "
        "`ffprobe.exe` (for example `C:\\\\ffmpeg\\\\bin`).\n"
        "• Or set **FFMPEG_PATH** and **FFPROBE_PATH** to each executable."
    )


def require_ffmpeg() -> None:
    ffmpeg_binaries()


def probe_duration_seconds(path: Path) -> float:
    _, ffprobe = ffmpeg_binaries()
    r = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        raise FFmpegError(r.stderr or "ffprobe failed")
    try:
        return float(r.stdout.strip())
    except ValueError as e:
        raise FFmpegError("Could not read duration") from e


def probe_video_start_time_seconds(path: Path) -> float:
    """
    First video stream PTS start (seconds). Often 0; some MP4/MOV use a positive offset.
    drawtext enable=between(t,...) uses this timeline—windows must be offset or only the
    first split-second matches.
    """
    _, ffprobe = ffmpeg_binaries()
    r = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=start_time",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        return 0.0
    s = (r.stdout or "").strip()
    if not s or s.lower() in ("n/a", "nan"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def has_audio_stream(path: Path) -> bool:
    """True if the container has at least one audio stream (robust across ffprobe builds)."""
    try:
        if not path.is_file():
            return False
    except OSError:
        return False
    _, ffprobe = ffmpeg_binaries()
    r = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        return False
    try:
        data = json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        return False
    return any(s.get("codec_type") == "audio" for s in data.get("streams", []))


def extract_jpeg_frames_evenly(
    video: Path,
    work_dir: Path,
    *,
    max_frames: int = 6,
    file_prefix: str = "vision_frame",
) -> list[Path]:
    """
    Grab evenly spaced stills from a video (for vision APIs).
    Uses absolute path for input so cwd only needs to hold outputs.
    """
    dur = probe_duration_seconds(video)
    if dur <= 0:
        return []
    # ~one frame every ~2.5s, at least 2 for context (or 1 if very short)
    n = min(max_frames, max(1, min(max_frames, int(dur / 2.5) + 1)))
    if dur < 1.2:
        n = 1
    paths: list[Path] = []
    v_abs = str(video.resolve())
    for i in range(n):
        t = dur * (i + 1) / (n + 1)
        out = work_dir / f"{file_prefix}_{i:02d}.jpg"
        run_ffmpeg(
            [
                "-y",
                "-ss",
                f"{t:.3f}",
                "-i",
                v_abs,
                "-frames:v",
                "1",
                "-q:v",
                "3",
                out.name,
            ],
            cwd=work_dir,
        )
        if out.is_file() and out.stat().st_size > 0:
            paths.append(out)
    return paths


def extract_jpeg_frames_per_segment(
    video: Path,
    work_dir: Path,
    *,
    n_segments: int,
    file_prefix: str = "overlay_seg",
) -> list[list[Path]]:
    """
    For each chronological segment of the video, extract 1–2 JPEGs from inside that segment
    (for vision captions aligned with on-screen timing).
    """
    dur = probe_duration_seconds(video)
    if dur <= 0 or n_segments < 1:
        return []
    v_abs = str(video.resolve())
    groups: list[list[Path]] = []
    for i in range(n_segments):
        t0 = dur * i / n_segments
        t1 = dur * (i + 1) / n_segments
        span = t1 - t0
        # Two samples per segment if long enough; else center of segment
        fracs = (0.35, 0.65) if span >= 1.0 else (0.5,)
        group: list[Path] = []
        for j, frac in enumerate(fracs):
            t = t0 + frac * span
            out = work_dir / f"{file_prefix}_{i:02d}_{j}.jpg"
            run_ffmpeg(
                [
                    "-y",
                    "-ss",
                    f"{t:.3f}",
                    "-i",
                    v_abs,
                    "-frames:v",
                    "1",
                    "-q:v",
                    "3",
                    out.name,
                ],
                cwd=work_dir,
            )
            if out.is_file() and out.stat().st_size > 0:
                group.append(out)
        groups.append(group)
    return groups


def run_ffmpeg(args: list[str], *, cwd: Path | None = None) -> None:
    ffmpeg, _ = ffmpeg_binaries()
    cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "info", *args]
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        check=False,
    )
    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "").strip() or "ffmpeg failed"
        # Include the command that failed for debugging
        cmd_str = " ".join(cmd)
        raise FFmpegError(f"ffmpeg command failed: {cmd_str}\n\n{msg}")
