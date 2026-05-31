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
_MUSIC_EXTS = frozenset({".mp3", ".wav", ".m4a", ".aac", ".ogg"})


def _music_library_dir() -> Path:
    return ROOT / "assets" / "music"


def _list_music_paths() -> list[Path]:
    d = _music_library_dir()
    if not d.is_dir():
        return []
    return sorted(
        p for p in d.iterdir() if p.is_file() and p.suffix.lower() in _MUSIC_EXTS
    )


def _print_music_list() -> None:
    paths = _list_music_paths()
    print(f"Tracks under {_music_library_dir()}:", flush=True)
    if not paths:
        print("  (none — add .mp3 etc. from YouTube Audio Library)", flush=True)
        return
    for i, p in enumerate(paths):
        print(f"  [{i}] {p.name}", flush=True)
    print('Pass a path, or use --music first, or --music "ExactFileName.mp3"', flush=True)


def _resolve_music_arg(raw: str | None) -> Path | None:
    """Path to an audio file, keyword ``first`` (first sorted library track), or a basename under assets/music/."""
    if raw is None or not str(raw).strip():
        return None
    s = str(raw).strip()
    if s.lower() in ("first", "auto", "@first"):
        paths = _list_music_paths()
        if not paths:
            raise ValueError(
                "No music in assets/music/. Add .mp3 files or pass a full path: --music C:\\path\\track.mp3"
            )
        return paths[0].resolve()
    p = Path(s).expanduser()
    if p.is_file():
        return p.resolve()
    lib = _music_library_dir() / s
    if lib.is_file():
        return lib.resolve()
    raise ValueError(
        f"Music not found: {s!r}. Use --list-music, --music first, or a full path to an audio file."
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ShortsAI-Studio export: vertical MP4 + metadata JSON (MVP smoke / CI).",
    )
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help=(
            "Input video path (mp4, mov, webm, etc.). Max length: env SHORTSAI_MAX_DURATION_SEC "
            "(default 120s, max 120s). Required unless --list-music."
        ),
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
        type=str,
        default=None,
        metavar="PATH_OR_FIRST",
        help='Background music: path to .mp3/.wav, or keyword "first" (first file in assets/music/), '
        'or a filename inside assets/music/ (see --list-music).',
    )
    p.add_argument(
        "--list-music",
        action="store_true",
        help="List tracks in assets/music/ and exit (no -i needed).",
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
        "--overlay-position",
        choices=("upper", "middle", "lower"),
        default=None,
        help="Default vertical band for timed scene lines (env SHORTSAI_OVERLAY_POSITION or middle).",
    )
    p.add_argument(
        "--overlay-positions",
        default=None,
        metavar="CSV",
        help='Comma-separated bands per timed line, e.g. "upper,middle,lower". Pads with --overlay-position.',
    )
    p.add_argument(
        "--vertical-fit",
        choices=("letterbox", "crop", "blur_fill"),
        default=None,
        help="How landscape clips fit 9:16 (default: env SHORTSAI_VERTICAL_FIT or crop).",
    )
    p.add_argument(
        "--ai-narration",
        action="store_true",
        help=(
            "Add English AI voiceover (GPT script + OpenAI TTS); ducks original speech. "
            "Needs OPENAI_API_KEY."
        ),
    )
    p.add_argument(
        "--narration-volume",
        type=float,
        default=None,
        metavar="GAIN",
        help=(
            "AI narration mix gain 0.25–3.0 (default: env SHORTSAI_NARRATION_VOL or 1.45)."
        ),
    )
    p.add_argument(
        "--ai-hook",
        action="store_true",
        help=(
            "Prepend a ~5s AI-picked cold open from the source, then the clip from the start "
            "(total ≤ max duration). Vision needs OPENAI_API_KEY; else uses a heuristic."
        ),
    )
    p.add_argument(
        "--max-duration",
        type=float,
        default=None,
        metavar="SEC",
        help=(
            "Override max clip length in seconds (10–120). Default: SHORTSAI_MAX_DURATION_SEC or 60."
        ),
    )
    p.add_argument(
        "--vision-onscreen-captions",
        action="store_true",
        help="When speech is missing or very thin, burn subtitles from on-screen text (OpenAI vision; needs key).",
    )
    p.add_argument(
        "--vision-onscreen-captions-english",
        action="store_true",
        help="With --vision-onscreen-captions, translate visible text to English in burned-in captions.",
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
    args = p.parse_args(argv)
    if not args.list_music and args.input is None:
        p.error("the following arguments are required: -i/--input")
    return args


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

    if args.list_music:
        _print_music_list()
        return 0

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
        clamp_max_duration_sec,
        coerce_overlay_position,
        max_duration_sec_from_env,
        metadata_to_json_bytes,
        process_upload,
    )

    load_dotenv(ROOT / ".env")
    if args.max_duration is not None:
        os.environ["SHORTSAI_MAX_DURATION_SEC"] = str(
            clamp_max_duration_sec(args.max_duration)
        )

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

    try:
        music_path = _resolve_music_arg(args.music)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.translate:
        whisper_task: str | None = "translate"
    else:
        whisper_task = args.whisper_task

    lang_hint = args.language_hint.strip() or None
    manual_overlay = args.manual_overlay.strip() or None
    overlay_pos_default = (
        coerce_overlay_position(args.overlay_position) if args.overlay_position else None
    )
    csv_ov = (args.overlay_positions or "").strip()
    overlay_pos_list = None
    if csv_ov:
        parts = [p.strip() for p in csv_ov.split(",") if p.strip()]
        if parts:
            overlay_pos_list = [coerce_overlay_position(p) for p in parts]
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip() or None

    model_name = os.environ.get("SHORTSAI_WHISPER_MODEL", "base")
    device = os.environ.get("SHORTSAI_WHISPER_DEVICE", "cpu")
    compute_type = os.environ.get("SHORTSAI_WHISPER_COMPUTE", "int8")

    try:
        ffmpeg_util.require_ffmpeg()
    except ffmpeg_util.FFmpegError as e:
        print(str(e), file=sys.stderr)
        return 2

    max_dur = max_duration_sec_from_env()
    dur = ffmpeg_util.probe_duration_seconds(inp)
    if dur > max_dur + 0.05:
        print(
            f"error: video is {dur:.1f}s; max allowed is {max_dur:.0f}s.",
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
            vertical_fit=args.vertical_fit,
            vision_onscreen_subtitles=bool(args.vision_onscreen_captions),
            vision_onscreen_subtitles_english=bool(args.vision_onscreen_captions_english),
            overlay_position=overlay_pos_default,
            overlay_positions=overlay_pos_list,
            ai_hook_cold_open=bool(args.ai_hook),
            ai_narration=bool(args.ai_narration),
            narration_volume=args.narration_volume,
        )
        if music_path is not None:
            meta["music_file"] = music_path.name
        else:
            meta["music_file"] = None

        if not meta.get("scene_overlay_applied"):
            err = meta.get("scene_overlay_error", "")
            extra = f" ({err})" if err else ""
            print(
                f"warning: timed scene text overlays were not burned in{extra}. "
                "Check progress messages above or metadata scene_overlay_error.",
                file=sys.stderr,
            )

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
