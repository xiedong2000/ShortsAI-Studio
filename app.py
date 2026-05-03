from __future__ import annotations

import io
import os
from typing import Any, cast
import shutil
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from faster_whisper import WhisperModel

from shortsai import ffmpeg_util
from shortsai.pipeline import (
    MAX_DURATION_SEC,
    VerticalFitMode,
    metadata_to_json_bytes,
    process_upload,
)

load_dotenv()

ROOT = Path(__file__).resolve().parent
MUSIC_DIR = ROOT / "assets" / "music"
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg"}

_SS_MP4 = "export_mp4_bytes"
_SS_META = "export_meta_dict"
_SS_HAS_AUDIO = "export_has_audio"
_SCROLL_PENDING = "scroll_pending_step"
_SS_UPLOAD_BYTES = "upload_bytes"
_SS_UPLOAD_NAME = "upload_name"
_SS_MUSIC_VOL = "music_vol"
_SS_MANUAL_OVERLAY = "manual_overlay"
_SS_WHISPER_TASK = "whisper_task_ui"
_SS_SPEECH_HINT = "speech_lang_hint"
_SS_MUSIC_PICK = "music_file_pick"
_SS_VERTICAL_FIT = "vertical_fit_ui"
_SS_WHISPER_CACHE = "whisper_word_cache"
_CURRENT_STEP = "current_step"


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


def _step_anchor(step: int) -> None:
    """Invisible anchor for scroll-into-view; scroll-margin keeps headings clear of the top bar."""
    st.markdown(
        f'<div id="shortsai-step-{step}" style="scroll-margin-top: 5.5rem;"></div>',
        unsafe_allow_html=True,
    )


def _scroll_to_step_after_render(step: int) -> None:
    """Scroll main page to anchor (runs in parent frame after Streamlit renders blocks)."""
    anchor = f"shortsai-step-{step}"
    components.html(
        f"""
        <script>
            const doc = window.parent.document;
            const el = doc.getElementById("{anchor}");
            if (el) {{
                el.scrollIntoView({{ behavior: "smooth", block: "start" }});
            }}
        </script>
        """,
        height=0,
        width=0,
    )


def _step_nav_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Steps (same order as the page)")
        st.markdown(
            """
| # | Section |
|---|---------|
| **1** | Upload |
| **2** | Options |
| **3** | Generate |
| **4** | Results |
"""
        )
        st.divider()
        st.caption(
            "The **filled circle** is the current step. Use **Back / Next** at the bottom of each section, "
            "or tap the circles. **☰** opens this panel on mobile."
        )


def _step_header(n: int, title: str, help_text: str | None = None) -> None:
    st.markdown(f"## Step {n} · {title}")
    if help_text:
        st.caption(help_text)
    st.markdown("")


def _init_session_defaults() -> None:
    if _SS_MP4 not in st.session_state:
        st.session_state[_SS_MP4] = None
    if _SS_META not in st.session_state:
        st.session_state[_SS_META] = None
    if _SS_HAS_AUDIO not in st.session_state:
        st.session_state[_SS_HAS_AUDIO] = True
    if _SS_MUSIC_VOL not in st.session_state:
        st.session_state[_SS_MUSIC_VOL] = 0.18
    if _SS_MANUAL_OVERLAY not in st.session_state:
        st.session_state[_SS_MANUAL_OVERLAY] = ""
    if _SS_WHISPER_TASK not in st.session_state:
        _env = (os.environ.get("SHORTSAI_WHISPER_TASK") or "transcribe").strip().lower()
        st.session_state[_SS_WHISPER_TASK] = "translate" if _env == "translate" else "transcribe"
    if _SS_SPEECH_HINT not in st.session_state:
        st.session_state[_SS_SPEECH_HINT] = (os.environ.get("SHORTSAI_WHISPER_LANGUAGE") or "").strip()
    if _SS_MUSIC_PICK not in st.session_state:
        st.session_state[_SS_MUSIC_PICK] = "None"
    if _SS_VERTICAL_FIT not in st.session_state:
        st.session_state[_SS_VERTICAL_FIT] = _default_vertical_fit_from_env()
    if _SS_WHISPER_CACHE not in st.session_state:
        st.session_state[_SS_WHISPER_CACHE] = None
    if _CURRENT_STEP not in st.session_state:
        st.session_state[_CURRENT_STEP] = 1


def _default_vertical_fit_from_env() -> str:
    raw = (os.environ.get("SHORTSAI_VERTICAL_FIT") or "").strip().lower().replace("-", "_")
    if not raw:
        return "crop"
    if raw in ("crop",):
        return "crop"
    if raw in ("blur_fill", "blurfill"):
        return "blur_fill"
    if raw in ("letterbox",):
        return "letterbox"
    return "crop"


def _step_footer(step: int) -> None:
    """Back / Next between steps (scroll + update current step highlight)."""
    st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
    c_back, _, c_next = st.columns([1, 2, 1])
    with c_back:
        if step > 1:
            if st.button("← Back", key=f"footer_back_{step}", use_container_width=True):
                st.session_state[_CURRENT_STEP] = step - 1
                st.session_state[_SCROLL_PENDING] = step - 1
                st.rerun()
    with c_next:
        if step < 4:
            if st.button("Next →", key=f"footer_next_{step}", type="primary", use_container_width=True):
                st.session_state[_CURRENT_STEP] = step + 1
                st.session_state[_SCROLL_PENDING] = step + 1
                st.rerun()


def _wizard_jump_bar() -> None:
    """Numbered circles: primary style = current step (filled); secondary = other steps (outline)."""
    cur = max(1, min(4, int(st.session_state.get(_CURRENT_STEP, 1))))
    st.session_state[_CURRENT_STEP] = cur

    st.markdown("##### Steps")
    st.caption(
        "The **filled** circle is the current step. Tap any circle to go there, or use **← Back** / **Next →** under each section."
    )
    # Only the first horizontal block on the main canvas should be circular (wizard), not Back/Next rows.
    st.markdown(
        """
<style>
section.main div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="column"] button {
  border-radius: 50% !important;
  width: 2.75rem !important;
  height: 2.75rem !important;
  min-height: 2.75rem !important;
  padding: 0 !important;
  font-weight: 700 !important;
  font-size: 1.05rem !important;
}
/* Softer inactive (secondary) circles — Streamlit uses data-testid on buttons */
section.main div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="column"] button[data-testid="baseButton-secondary"] {
  background: #f3f4f6 !important;
  border: 2px solid #e5e7eb !important;
  color: #6b7280 !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

    w1, w2, w3, w4 = st.columns(4)
    labels = ["Upload", "Options", "Generate", "Results"]
    hints = [
        "Go to upload",
        "Music & captions",
        "Preview & generate",
        "Downloads & metadata",
    ]
    for i, col in enumerate([w1, w2, w3, w4], start=1):
        with col:
            if st.button(
                str(i),
                key=f"wiz_circ_{i}",
                use_container_width=True,
                type="primary" if cur == i else "secondary",
                help=hints[i - 1],
            ):
                st.session_state[_CURRENT_STEP] = i
                st.session_state[_SCROLL_PENDING] = i
                st.rerun()

    l1, l2, l3, l4 = st.columns(4)
    for col, lbl in zip([l1, l2, l3, l4], labels):
        with col:
            st.caption(lbl)

    st.divider()


def main() -> None:
    st.set_page_config(
        page_title="ShortsAI Studio",
        page_icon="🎬",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    _init_session_defaults()

    st.title("ShortsAI Studio")
    st.caption(
        "Turn a clip into a vertical Short: burned-in speech captions, optional music, scene text overlays, "
        "and YouTube-ready title, description, and tags."
    )
    st.caption("Follow the steps below. The **highlighted circle** matches your current step.")

    _step_nav_sidebar()

    try:
        ffmpeg_util.require_ffmpeg()
    except ffmpeg_util.FFmpegError as e:
        st.error(str(e))
        st.stop()

    api_key = os.environ.get("OPENAI_API_KEY") or ""
    if not api_key:
        st.info(
            "Optional: add `OPENAI_API_KEY` to `.env` for richer titles/descriptions/tags and vision-based scene text; "
            "otherwise the app uses simple text fallbacks."
        )

    _wizard_jump_bar()

    music_files = _list_music()
    music_options = ["None"] + [p.name for p in music_files]

    # --- Step 1 · Upload ---
    _step_anchor(1)
    with st.container(border=True):
        _step_header(
            1,
            "Upload",
            f"Choose a file (max {MAX_DURATION_SEC:.0f}s). Your file is kept until you replace it.",
        )
        uploaded = st.file_uploader(
            "Video file",
            type=["mp4", "mov", "webm", "mkv", "avi"],
            help=f"Maximum length {MAX_DURATION_SEC:.0f} seconds for this version.",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            st.session_state[_SS_UPLOAD_BYTES] = uploaded.getvalue()
            st.session_state[_SS_UPLOAD_NAME] = uploaded.name
        else:
            st.session_state[_SS_UPLOAD_BYTES] = None
            st.session_state[_SS_UPLOAD_NAME] = None

        if st.session_state.get(_SS_UPLOAD_BYTES):
            st.success(f"Ready: **{st.session_state.get(_SS_UPLOAD_NAME, 'video')}**")
        else:
            st.info("Choose a video file to continue.")
        st.caption("Supported: MP4, MOV, WebM, MKV, AVI")
        _step_footer(1)

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # --- Step 2 · Options ---
    _step_anchor(2)
    with st.container(border=True):
        _step_header(
            2,
            "Options",
            "Music, subtitles language, and optional on-screen line.",
        )
        if music_files:
            if st.session_state.get(_SS_MUSIC_PICK) not in music_options:
                st.session_state[_SS_MUSIC_PICK] = "None"
            st.selectbox(
                "Background music (`assets/music/`)",
                music_options,
                key=_SS_MUSIC_PICK,
            )
        else:
            st.caption("Add tracks under `assets/music/` (e.g. YouTube Audio Library) to mix background music.")
            st.session_state[_SS_MUSIC_PICK] = "None"

        st.slider(
            "Music volume (relative to speech)",
            0.0,
            0.5,
            key=_SS_MUSIC_VOL,
        )

        st.radio(
            "Vertical framing (9:16)",
            options=("crop", "letterbox", "blur_fill"),
            format_func=lambda v: {
                "letterbox": "Letterbox — bars top/bottom or sides; entire frame visible",
                "crop": "Center crop — fills 1080×1920; edges may be cut",
                "blur_fill": "Blur fill — blurred full-frame backdrop behind the video",
            }[v],
            key=_SS_VERTICAL_FIT,
            help="How wider or shorter clips are fitted. Blur fill uses FFmpeg’s boxblur filter.",
        )

        st.text_input(
            "Optional: one line of on-screen scene text (replaces AI scene lines for this export)",
            help="Leave empty for auto-generated timed lines. Title/description/tags are not burned into the video.",
            key=_SS_MANUAL_OVERLAY,
        )

        _sub_default = 1 if st.session_state[_SS_WHISPER_TASK] == "translate" else 0
        sub_choice = st.radio(
            "Subtitles & transcript",
            (
                "Same as spoken language",
                "English (translate speech to English)",
            ),
            index=_sub_default,
            help="Translate forces English subtitles and transcript when speech is not English.",
        )
        st.session_state[_SS_WHISPER_TASK] = (
            "translate" if sub_choice.startswith("English") else "transcribe"
        )

        st.text_input(
            "Optional speech language hint (ISO 639-1, e.g. zh, ja). Empty = auto-detect.",
            key=_SS_SPEECH_HINT,
        )
        st.caption(
            "Burned-in speech caption size: set SHORTSAI_CAPTION_FONT_SIZE in .env (default 14; try 12 for smaller text)."
        )
        _step_footer(2)

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # --- Step 3 · Generate ---
    _step_anchor(3)
    upload_bytes = st.session_state.get(_SS_UPLOAD_BYTES)
    upload_name = st.session_state.get(_SS_UPLOAD_NAME) or "clip.mp4"

    with st.container(border=True):
        _step_header(
            3,
            "Generate",
            "Creates the 9:16 MP4 and metadata. First run may download the Whisper model.",
        )
        if not upload_bytes:
            st.warning("Upload a video in **Step 1** first.")
        else:
            st.subheader("Preview")
            st.video(io.BytesIO(upload_bytes), format="video/mp4")
            if st.button("Generate Short", type="primary", use_container_width=True, key="btn_generate"):
                suffix = Path(upload_name).suffix or ".mp4"
                work = Path(tempfile.mkdtemp(prefix="shortsai_"))
                src = work / f"upload{suffix}"
                src.write_bytes(upload_bytes)

                status = st.status("Generating your Short…", expanded=True)

                def on_progress(msg: str) -> None:
                    status.write(msg)

                music_choice: Path | None = None
                pick = st.session_state.get(_SS_MUSIC_PICK, "None")
                if pick and pick != "None" and music_files:
                    for p in music_files:
                        if p.name == pick:
                            music_choice = p
                            break

                has_audio = True
                try:
                    dur = ffmpeg_util.probe_duration_seconds(src)
                    if dur > MAX_DURATION_SEC + 0.05:
                        status.update(label="Too long", state="error")
                        st.error(
                            f"This clip is **{dur:.1f}s**. Maximum length is **{MAX_DURATION_SEC:.0f}s**. "
                            "Trim the video and try again."
                        )
                        shutil.rmtree(work, ignore_errors=True)
                        st.stop()

                    has_audio = ffmpeg_util.has_audio_stream(src)
                    if not has_audio:
                        status.write("Note: no audio track detected — captions and transcript may be empty.")

                    model = _whisper_model()
                    mp4_bytes, meta = process_upload(
                        src,
                        work_dir=work,
                        whisper=model,
                        whisper_model=os.environ.get("SHORTSAI_WHISPER_MODEL", "base"),
                        openai_api_key=api_key.strip() or None,
                        music_path=music_choice,
                        music_volume=float(st.session_state[_SS_MUSIC_VOL]),
                        manual_overlay_text=(st.session_state.get(_SS_MANUAL_OVERLAY) or "").strip() or None,
                        progress=on_progress,
                        whisper_task=st.session_state[_SS_WHISPER_TASK],
                        whisper_language_hint=(st.session_state.get(_SS_SPEECH_HINT) or "").strip() or None,
                        vertical_fit=cast(VerticalFitMode, st.session_state[_SS_VERTICAL_FIT]),
                    )
                    if music_choice is not None:
                        meta["music_file"] = music_choice.name
                    else:
                        meta["music_file"] = None

                    st.session_state[_SS_WHISPER_CACHE] = meta.get("whisper_cache")
                    st.session_state[_SS_MP4] = mp4_bytes
                    st.session_state[_SS_META] = meta
                    st.session_state[_SS_HAS_AUDIO] = has_audio
                    st.session_state[_CURRENT_STEP] = 4
                    st.session_state[_SCROLL_PENDING] = 4
                    status.update(label="Done", state="complete")
                except ValueError as e:
                    status.update(label="Failed", state="error")
                    st.error(str(e))
                except ffmpeg_util.FFmpegError as e:
                    status.update(label="Failed", state="error")
                    st.error(str(e))
                except Exception as e:
                    status.update(label="Failed", state="error")
                    st.exception(e)
                finally:
                    shutil.rmtree(work, ignore_errors=True)

                if st.session_state.get(_SS_MP4):
                    st.rerun()

        _step_footer(3)

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # --- Step 4 · Results (always visible; may be empty) ---
    _step_anchor(4)
    mp4_bytes: bytes | None = st.session_state[_SS_MP4]
    meta: dict[str, Any] | None = st.session_state[_SS_META]

    with st.container(border=True):
        _step_header(4, "Results", "Download the MP4 and use title / description / tags in YouTube Studio.")

        if mp4_bytes is None or meta is None:
            st.info("Nothing here yet — use **Step 3 · Generate Short** above. After a successful run, results appear here.")
        else:
            c_clear, _ = st.columns([1, 3])
            with c_clear:
                if st.button("Clear results", help="Remove the last export from this page"):
                    st.session_state[_SS_MP4] = None
                    st.session_state[_SS_META] = None
                    st.session_state[_SS_WHISPER_CACHE] = None
                    st.session_state[_SS_HAS_AUDIO] = True
                    st.session_state[_CURRENT_STEP] = 3
                    st.session_state[_SCROLL_PENDING] = 3
                    st.rerun()

            st.success("Your Short is ready. Download the video and copy metadata for YouTube Studio.")

            has_audio = bool(st.session_state.get(_SS_HAS_AUDIO, True))
            transcript = (meta.get("transcript") or "").strip()

            if not has_audio:
                st.warning(
                    "This file had **no audio track**. Burned-in speech captions are usually empty; "
                    "you can still use background music from Step 2."
                )
            elif not transcript:
                st.info(
                    "No speech was detected in the audio (or volume was too low). "
                    "Burned-in captions may be minimal or empty."
                )

            if meta.get("scene_overlay_applied") is False:
                err = meta.get("scene_overlay_error") or ""
                st.warning(
                    "Timed **scene text overlays** were not burned in (FFmpeg/drawtext issue). "
                    f"Speech captions still apply if present. {('Details: ' + err[:280]) if err else ''}"
                )

            st.markdown("##### Output video")
            st.video(mp4_bytes, format="video/mp4")
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "Download MP4",
                    data=mp4_bytes,
                    file_name="shorts_export.mp4",
                    mime="video/mp4",
                    type="primary",
                )
            with dl2:
                st.download_button(
                    "Download metadata.json",
                    data=metadata_to_json_bytes(meta),
                    file_name="metadata.json",
                    mime="application/json",
                )
            srt_dl = (meta.get("speech_srt") or "").strip()
            if srt_dl:
                st.download_button(
                    "Download captions.srt",
                    data=srt_dl.encode("utf-8"),
                    file_name="captions.srt",
                    mime="text/plain",
                )

            with st.expander("Edit speech captions (SRT) & re-export", expanded=False):
                st.caption(
                    "Fix typos in cue text only; keep cue numbers and `00:00:00,000 --> 00:00:00,000` lines intact. "
                    "Re-export skips Whisper if the cache below is present (same upload + Step 2 options)."
                )
                with st.form("reexport_captions_form"):
                    srt_default = (meta.get("speech_srt") or "").strip()
                    edited_srt = st.text_area(
                        "SRT",
                        value=srt_default,
                        height=260,
                        help="Standard SubRip format (UTF-8).",
                    )
                    re_go = st.form_submit_button("Re-export MP4 with edited captions")
                if re_go:
                    cache = st.session_state.get(_SS_WHISPER_CACHE)
                    if not isinstance(cache, dict) or not cache.get("words"):
                        st.error(
                            "Whisper cache is missing (e.g. after **Clear results**). "
                            "Run **Generate Short** in Step 3 once, then edit and re-export here."
                        )
                    elif not edited_srt.strip() or "-->" not in edited_srt:
                        st.error("SRT looks empty or invalid (each cue needs a `-->` timestamp line).")
                    else:
                        ub = st.session_state.get(_SS_UPLOAD_BYTES)
                        if not ub:
                            st.error("Upload bytes missing — go back to Step 1 and select your video again.")
                        else:
                            music_re: Path | None = None
                            pick_r = st.session_state.get(_SS_MUSIC_PICK, "None")
                            if pick_r and pick_r != "None" and music_files:
                                for p in music_files:
                                    if p.name == pick_r:
                                        music_re = p
                                        break
                            work2 = Path(tempfile.mkdtemp(prefix="shortsai_re_"))
                            try:
                                src2 = work2 / f"upload{Path(upload_name).suffix or '.mp4'}"
                                src2.write_bytes(ub)
                                model = _whisper_model()
                                with st.spinner("Re-exporting with your SRT…"):
                                    mp4_b, meta2 = process_upload(
                                        src2,
                                        work_dir=work2,
                                        whisper=model,
                                        whisper_model=os.environ.get("SHORTSAI_WHISPER_MODEL", "base"),
                                        openai_api_key=api_key.strip() or None,
                                        music_path=music_re,
                                        music_volume=float(st.session_state[_SS_MUSIC_VOL]),
                                        manual_overlay_text=(st.session_state.get(_SS_MANUAL_OVERLAY) or "").strip()
                                        or None,
                                        progress=None,
                                        whisper_task=st.session_state[_SS_WHISPER_TASK],
                                        whisper_language_hint=(st.session_state.get(_SS_SPEECH_HINT) or "").strip()
                                        or None,
                                        vertical_fit=cast(VerticalFitMode, st.session_state[_SS_VERTICAL_FIT]),
                                        caption_srt_override=edited_srt.strip(),
                                        reuse_whisper_cache=cache,
                                    )
                                if music_re is not None:
                                    meta2["music_file"] = music_re.name
                                else:
                                    meta2["music_file"] = None
                                st.session_state[_SS_WHISPER_CACHE] = meta2.get("whisper_cache")
                                st.session_state[_SS_MP4] = mp4_b
                                st.session_state[_SS_META] = meta2
                                st.success("Re-export complete.")
                                st.rerun()
                            except Exception as e:
                                st.exception(e)
                            finally:
                                shutil.rmtree(work2, ignore_errors=True)

            st.markdown("##### Title")
            title = meta.get("title") or ""
            st.markdown(f"### {title}" if title else "_No title_")
            st.caption("Paste into YouTube Studio → video details")

            st.markdown("##### Description")
            st.write(meta.get("description") or "")

            tags = meta.get("tags")
            if isinstance(tags, list):
                tags_str = ", ".join(str(t).strip() for t in tags if str(t).strip())
            else:
                tags_str = ""
            st.markdown("##### Tags")
            st.write(tags_str if tags_str else "_None_")

            with st.expander("Full metadata (JSON, without long transcript)"):
                st.json(
                    {
                        k: v
                        for k, v in meta.items()
                        if k not in ("transcript", "whisper_cache", "speech_srt")
                    }
                )

            with st.expander("Transcript"):
                st.write(meta.get("transcript", "") or "_Empty_")

        _step_footer(4)

    pending = st.session_state.pop(_SCROLL_PENDING, None)
    if pending is not None:
        _scroll_to_step_after_render(int(pending))


if __name__ == "__main__":
    main()
