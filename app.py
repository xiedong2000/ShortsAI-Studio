from __future__ import annotations

import os
from typing import Literal
import shutil
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from faster_whisper import WhisperModel

from shortsai import ffmpeg_util
from shortsai.pipeline import (
    MAX_DURATION_SEC,
    metadata_to_json_bytes,
    process_upload,
)

load_dotenv()

ROOT = Path(__file__).resolve().parent
MUSIC_DIR = ROOT / "assets" / "music"
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg"}


@st.cache_resource
def _whisper_model() -> WhisperModel:
    name = os.environ.get("SHORTSAI_WHISPER_MODEL", "base")
    device = os.environ.get("SHORTSAI_WHISPER_DEVICE", "cpu")
    ctype = os.environ.get("SHORTSAI_WHISPER_COMPUTE", "int8")
    return WhisperModel(name, device=device, compute_type=ctype)


def _list_music() -> list[Path]:
    if not MUSIC_DIR.is_dir():
        return []
    return sorted(
        p for p in MUSIC_DIR.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXT
    )


def main() -> None:
    st.set_page_config(page_title="ShortsAI Studio", page_icon="🎬", layout="centered")
    st.title("ShortsAI Studio")
    st.caption(
        "Upload a short clip → burned-in speech captions (if any), 9:16 export, scene text overlays, "
        "and title/description/tags in metadata.json for YouTube (not burned as title/desc/hashtags)."
    )

    try:
        ffmpeg_util.require_ffmpeg()
    except ffmpeg_util.FFmpegError as e:
        st.error(str(e))
        st.stop()

    api_key = os.environ.get("OPENAI_API_KEY") or ""
    if not api_key:
        st.info("Optional: set `OPENAI_API_KEY` in a `.env` file for GPT metadata; otherwise simple fallbacks are used.")

    uploaded = st.file_uploader(
        "Video (max 60s)",
        type=["mp4", "mov", "webm", "mkv", "avi"],
    )

    music_files = _list_music()
    music_choice: Path | None = None
    if music_files:
        options = ["None"] + [p.name for p in music_files]
        sel = st.selectbox(
            "Background music (YouTube Audio Library files in `assets/music/`)",
            options,
        )
        if sel != "None":
            music_choice = next(p for p in music_files if p.name == sel)
    else:
        st.caption("Add tracks under `assets/music/` (from YouTube Studio → Audio library) to enable music mixing.")

    music_vol = st.slider("Music volume (relative to speech)", 0.0, 0.5, 0.18, 0.02)

    manual_overlay_text = st.text_input(
        "Optional: override on-screen scene text (one line; replaces AI lines for this export)",
        value="",
        help="Leave empty to auto-generate timed scene lines (vision + GPT). Title, description, and tags are only in metadata.json.",
    )

    _env_task = (os.environ.get("SHORTSAI_WHISPER_TASK") or "transcribe").strip().lower()
    _sub_default = 1 if _env_task == "translate" else 0
    sub_choice = st.radio(
        "Subtitles & transcript",
        (
            "Same as spoken language",
            "English (translate speech to English)",
        ),
        index=_sub_default,
        help="Translate uses Whisper to output English for burned-in captions and the transcript, even when speech is not English.",
    )
    whisper_task_ui: Literal["transcribe", "translate"] = (
        "translate" if sub_choice.startswith("English") else "transcribe"
    )
    _hint_default = (os.environ.get("SHORTSAI_WHISPER_LANGUAGE") or "").strip()
    speech_lang_hint_ui = st.text_input(
        "Optional speech language hint (ISO 639-1, e.g. zh, ja, ko). Leave empty for auto-detect.",
        value=_hint_default,
        help="Helps Whisper when auto-detection is wrong. Does not change subtitle language unless you use translation above.",
    ).strip()

    if uploaded is None:
        st.stop()

    st.video(uploaded)

    if st.button("Process", type="primary"):
        suffix = Path(uploaded.name).suffix or ".mp4"
        work = Path(tempfile.mkdtemp(prefix="shortsai_"))
        src = work / f"upload{suffix}"
        src.write_bytes(uploaded.getvalue())

        status = st.status("Working…", expanded=True)
        prog_msgs: list[str] = []

        def on_progress(msg: str) -> None:
            prog_msgs.append(msg)
            status.write(msg)

        try:
            dur = ffmpeg_util.probe_duration_seconds(src)
            if dur > MAX_DURATION_SEC + 0.05:
                status.update(label="Too long", state="error")
                st.error(f"This clip is {dur:.1f}s. Max is {MAX_DURATION_SEC:.0f}s for v1.")
                return

            model = _whisper_model()
            mp4_bytes, meta = process_upload(
                src,
                work_dir=work,
                whisper=model,
                whisper_model=os.environ.get("SHORTSAI_WHISPER_MODEL", "base"),
                openai_api_key=api_key.strip() or None,
                music_path=music_choice,
                music_volume=music_vol,
                manual_overlay_text=manual_overlay_text.strip() or None,
                progress=on_progress,
                whisper_task=whisper_task_ui,
                whisper_language_hint=speech_lang_hint_ui,
            )
            if music_choice is not None:
                meta["music_file"] = music_choice.name
            else:
                meta["music_file"] = None

            status.update(label="Done", state="complete")
        except ValueError as e:
            status.update(label="Failed", state="error")
            st.error(str(e))
            return
        except ffmpeg_util.FFmpegError as e:
            status.update(label="Failed", state="error")
            st.error(str(e))
            return
        except Exception as e:
            status.update(label="Failed", state="error")
            st.exception(e)
            return
        finally:
            shutil.rmtree(work, ignore_errors=True)

        st.success("Export ready.")
        st.subheader("Processed preview")
        st.video(mp4_bytes, format="video/mp4")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download MP4",
                data=mp4_bytes,
                file_name="shorts_export.mp4",
                mime="video/mp4",
            )
        with c2:
            st.download_button(
                "Download metadata.json",
                data=metadata_to_json_bytes(meta),
                file_name="metadata.json",
                mime="application/json",
            )

        with st.expander("Preview metadata"):
            st.json({k: v for k, v in meta.items() if k != "transcript"})
        with st.expander("Transcript"):
            st.write(meta.get("transcript", ""))


if __name__ == "__main__":
    main()
