#!/usr/bin/env python3
"""
End-to-end CLI for the ShortsAI pipeline (same path as Streamlit `app.py`).

Loads `.env` from the project root, runs transcribe → captions → 9:16 export
→ optional music → metadata + scene overlays, then writes MP4 + JSON sidecar.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ShortsAI-Studio export: vertical MP4 + metadata JSON (MVP smoke / CI).",
    )
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input video path (mp4, mov, webm, etc.; max 60s).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output MP4 path. Default: <input_stem>_shorts.mp4 next to the input file.",
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Output metadata JSON path. Default: same path as --output with .json extension.",
    )
    p.add_argument(
        "--music",
        type=Path,
        default=None,
        help="Optional background music file (e.g. from YouTube Audio Library).",
    )
    p.add_argument(
        "--music-volume",
        type=float,
        default=0.18,
        help="Music mix volume 0..1 (default: 0.18).",
    )
    sub = p.add_mutually_exclusive_group()
    sub.add_argument(
        "--translate",
        action="store_true",
        help="Whisper translate task (English subtitles/transcript for any speech).",
    )
    sub.add_argument(
        "--whisper-task",
        choices=("transcribe", "translate"),
        default=None,
        help="Override SHORTSAI_WHISPER_TASK from .env (default: transcribe).",
    )
    p.add_argument(
        "--language-hint",
        default="",
        metavar="CODE",
        help="Optional ISO 639-1 speech hint (e.g. zh, ja). Empty = auto-detect.",
    )
    p.add_argument(
        "--manual-overlay",
        default="",
        help="Optional single line of on-screen text (replaces AI scene lines).",
    )
    p.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Do not delete the temp work directory (prints its path for debugging).",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print errors and the final output paths.",
    )
    return p.parse_args(argv)


def _log(msg: str, *, quiet: bool) -> None:
    if not quiet:
        print(msg, flush=True)


def _default_output_mp4(inp: Path) -> Path:
    stem = inp.stem or "export"
    return inp.resolve().parent / f"{stem}_shorts.mp4"


def _deps_help() -> str:
    return (
        "Missing Python dependencies. Either:\n"
        f"  1) Use the project venv: {ROOT / '.venv' / 'Scripts' / 'python.exe'} shorts_generator.py ...\n"
        "  2) Or from this folder: pip install -r requirements.txt\n"
    )


def main(argv: list[str] | None = None) -> int:
    # Parse first so `python shorts_generator.py --help` works without project deps installed.
    args = _parse_args(argv)

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        print(_deps_help(), file=sys.stderr)
        print("error: no module named 'dotenv' (install package python-dotenv).", file=sys.stderr)
        return 2

    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError:
        print(_deps_help(), file=sys.stderr)
        print("error: no module named 'faster_whisper'.", file=sys.stderr)
        return 2

    from shortsai import ffmpeg_util
    from shortsai.pipeline import (
        MAX_DURATION_SEC,
        metadata_to_json_bytes,
        process_upload,
    )

    load_dotenv(ROOT / ".env")

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        print(f"error: input not found: {inp}", file=sys.stderr)
        return 1

    out_mp4 = (args.output or _default_output_mp4(inp)).expanduser().resolve()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    out_json = (
        args.metadata.expanduser().resolve()
        if args.metadata is not None
        else out_mp4.with_suffix(".json")
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)

    music_path: Path | None = None
    if args.music is not None:
        music_path = args.music.expanduser().resolve()
        if not music_path.is_file():
            print(f"error: music file not found: {music_path}", file=sys.stderr)
            return 1

    if args.translate:
        whisper_task: str | None = "translate"
    else:
        whisper_task = args.whisper_task

    lang_hint = args.language_hint.strip() or None
    manual_overlay = args.manual_overlay.strip() or None
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip() or None

    model_name = os.environ.get("SHORTSAI_WHISPER_MODEL", "base")
    device = os.environ.get("SHORTSAI_WHISPER_DEVICE", "cpu")
    compute_type = os.environ.get("SHORTSAI_WHISPER_COMPUTE", "int8")

    try:
        ffmpeg_util.require_ffmpeg()
    except ffmpeg_util.FFmpegError as e:
        print(str(e), file=sys.stderr)
        return 2

    dur = ffmpeg_util.probe_duration_seconds(inp)
    if dur > MAX_DURATION_SEC + 0.05:
        print(
            f"error: video is {dur:.1f}s; max allowed is {MAX_DURATION_SEC:.0f}s.",
            file=sys.stderr,
        )
        return 1

    work_dir = Path(tempfile.mkdtemp(prefix="shortsai_cli_"))
    _log(f"work dir: {work_dir}", quiet=args.quiet)

    def on_progress(msg: str) -> None:
        _log(msg, quiet=args.quiet)

    try:
        _log("Loading Whisper model…", quiet=args.quiet)
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        _log("Processing…", quiet=args.quiet)
        mp4_bytes, meta = process_upload(
            inp,
            work_dir=work_dir,
            whisper=model,
            whisper_model=model_name,
            device=device,
            compute_type=compute_type,
            openai_api_key=api_key,
            music_path=music_path,
            music_volume=args.music_volume,
            manual_overlay_text=manual_overlay,
            progress=on_progress,
            whisper_task=whisper_task,
            whisper_language_hint=lang_hint,
        )
        if music_path is not None:
            meta["music_file"] = music_path.name
        else:
            meta["music_file"] = None

        out_mp4.write_bytes(mp4_bytes)
        out_json.write_bytes(metadata_to_json_bytes(meta))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ffmpeg_util.FFmpegError as e:
        print(str(e), file=sys.stderr)
        return 2
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    finally:
        if args.keep_work_dir:
            _log(f"kept work dir: {work_dir}", quiet=args.quiet)
        else:
            shutil.rmtree(work_dir, ignore_errors=True)

    print(out_mp4, flush=True)
    print(out_json, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
