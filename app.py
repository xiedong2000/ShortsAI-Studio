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
from shortsai.metadata_llm import strip_overlay_quotes
from shortsai.pipeline import (
    HARD_MAX_DURATION_SEC,
    VerticalFitMode,
    max_duration_sec_from_env,
    OverlayPosition,
    coerce_overlay_position,
    default_overlay_line_times,
    metadata_to_json_bytes,
    overlay_times_from_meta,
    process_upload,
    resolve_caption_font_size,
    resolve_scene_overlay_font_size,
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
_SS_VISION_ONSCREEN_SUBS = "vision_onscreen_subs"
_SS_VISION_ONSCREEN_EN = "vision_onscreen_en"
_SS_AI_HOOK = "ai_hook_cold_open"
_SS_AI_NARRATION = "ai_narration"
_SS_NARRATION_VOL = "narration_vol"
_CURRENT_STEP = "current_step"
_SS_OVERLAY_DEFAULT = "overlay_default_pos"
_SS_OVERLAY_POSITIONS_CSV = "overlay_positions_csv"
_SS_CAPTION_FONT = "caption_font_size_ui"
_SS_SCENE_FONT = "scene_font_size_ui"

WIZARD_STEP_LABELS = ["Upload", "Options", "Generate", "Results"]


def _wizard_stepper_styles() -> str:
    """Circle step buttons: Streamlit puts help= on the native title= attribute (machine token only)."""
    return """
<style>
button[title^="shortsai_step_"] {
  display: block !important;
  margin-left: auto !important;
  margin-right: auto !important;
}
.shortsai-bar-inline {
  width: 100%;
  height: 4px;
  background: #e5e7eb;
  border-radius: 2px;
  margin-top: 0.85rem;
}
button[title^="shortsai_step_"][title$="_done"] {
  border-radius: 50% !important;
  width: 2.35rem !important;
  height: 2.35rem !important;
  min-width: 2.35rem !important;
  min-height: 2.35rem !important;
  max-width: 2.35rem !important;
  max-height: 2.35rem !important;
  padding: 0 !important;
  font-weight: 800 !important;
  font-size: 0.95rem !important;
  background: #dcfce7 !important;
  color: #166534 !important;
  border: 2px solid #86efac !important;
}
button[title^="shortsai_step_"][title$="_current"] {
  border-radius: 50% !important;
  width: 2.35rem !important;
  height: 2.35rem !important;
  min-width: 2.35rem !important;
  min-height: 2.35rem !important;
  max-width: 2.35rem !important;
  max-height: 2.35rem !important;
  padding: 0 !important;
  font-weight: 800 !important;
  font-size: 0.95rem !important;
  background: #2563eb !important;
  color: #fff !important;
  border: 2px solid #1d4ed8 !important;
  box-shadow: 0 2px 10px rgba(37, 99, 235, 0.35) !important;
}
button[title^="shortsai_step_"][title$="_todo"] {
  border-radius: 50% !important;
  width: 2.35rem !important;
  height: 2.35rem !important;
  min-width: 2.35rem !important;
  min-height: 2.35rem !important;
  max-width: 2.35rem !important;
  max-height: 2.35rem !important;
  padding: 0 !important;
  font-weight: 700 !important;
  font-size: 0.95rem !important;
  background: #f9fafb !important;
  color: #9ca3af !important;
  border: 2px solid #e5e7eb !important;
}
.shortsai-step-wrap { max-width: 28rem; margin: 0 auto 0.35rem auto; }
.shortsai-step-meta { text-align: center; color: #6b7280; font-size: 0.85rem; margin-top: 0.35rem; }
.shortsai-step-title { text-align: center; font-size: 1.35rem; font-weight: 700; margin: 0.1rem 0 0.75rem 0; color: #111827; }
</style>
"""


def _inject_scroll_top_fab(current_step: int) -> None:
    """Fixed-position control: scroll to #shortsai-page-top (same approach as step anchors)."""
    cs = max(1, min(4, int(current_step)))
    components.html(
        f"""
<script>
(function () {{
  function scrollToTopAnchor() {{
    const ids = ["shortsai-page-top"];
    const roots = [window.parent.document, document];
    for (const doc of roots) {{
      if (!doc) continue;
      for (const id of ids) {{
        const el = doc.getElementById(id);
        if (el) {{
          try {{
            el.scrollIntoView({{ behavior: "smooth", block: "start" }});
            return true;
          }} catch (err) {{}}
        }}
      }}
    }}
    return false;
  }}
  function mount(hostDoc) {{
    if (!hostDoc || !hostDoc.body) return;
    const id = 'shortsai-scroll-top-fab';
    const old = hostDoc.getElementById(id);
    if (old) old.remove();
    const wrap = hostDoc.createElement('div');
    wrap.id = id;
    wrap.style.cssText =
      'position:fixed;bottom:calc(1.1rem + env(safe-area-inset-bottom,0px));right:1rem;z-index:999999;font-family:system-ui,-apple-system,sans-serif;';
    const btn = hostDoc.createElement('button');
    btn.type = 'button';
    btn.setAttribute('aria-label', 'Scroll to top; current step {cs}');
    btn.style.cssText =
      'width:3.35rem;height:3.35rem;border-radius:50%;border:none;background:#2563eb;color:#fff;font-weight:700;cursor:pointer;box-shadow:0 4px 16px rgba(37,99,235,0.45);display:flex;flex-direction:column;align-items:center;justify-content:center;line-height:1;padding:0.2rem 0 0.15rem 0;gap:1px;';
    btn.innerHTML = '<span style="font-size:1.05rem;line-height:1;">↑</span><span style="font-size:0.95rem;line-height:1;">{cs}</span>';
    btn.addEventListener('click', function (e) {{
      e.preventDefault();
      if (!scrollToTopAnchor()) {{
        try {{ window.parent.scrollTo({{ top: 0, behavior: "smooth" }}); }} catch (e2) {{}}
        try {{ window.scrollTo({{ top: 0, behavior: "smooth" }}); }} catch (e3) {{}}
      }}
    }});
    wrap.appendChild(btn);
    hostDoc.body.appendChild(wrap);
  }}
  try {{
    mount(window.parent.document);
  }} catch (e) {{
    mount(document);
  }}
}})();
</script>
""",
        height=0,
        width=0,
    )


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
            "At the top of the page: **✓** = steps before the current one; **blue circle** = where you are. "
            "Tap a step to jump. The **↑** button (bottom-right) scrolls to the top and shows the step number. "
            "**☰** opens this panel on mobile."
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
    if _SS_VISION_ONSCREEN_SUBS not in st.session_state:
        st.session_state[_SS_VISION_ONSCREEN_SUBS] = False
    if _SS_VISION_ONSCREEN_EN not in st.session_state:
        st.session_state[_SS_VISION_ONSCREEN_EN] = False
    if _SS_AI_HOOK not in st.session_state:
        st.session_state[_SS_AI_HOOK] = False
    if _SS_AI_NARRATION not in st.session_state:
        st.session_state[_SS_AI_NARRATION] = False
    if _SS_NARRATION_VOL not in st.session_state:
        from shortsai.narration import narration_volume_from_env

        st.session_state[_SS_NARRATION_VOL] = narration_volume_from_env()
    if _CURRENT_STEP not in st.session_state:
        st.session_state[_CURRENT_STEP] = 1
    if _SS_OVERLAY_DEFAULT not in st.session_state:
        st.session_state[_SS_OVERLAY_DEFAULT] = coerce_overlay_position(
            os.environ.get("SHORTSAI_OVERLAY_POSITION")
        )
    if _SS_OVERLAY_POSITIONS_CSV not in st.session_state:
        st.session_state[_SS_OVERLAY_POSITIONS_CSV] = ""
    if _SS_CAPTION_FONT not in st.session_state:
        st.session_state[_SS_CAPTION_FONT] = resolve_caption_font_size(None)
    if _SS_SCENE_FONT not in st.session_state:
        st.session_state[_SS_SCENE_FONT] = resolve_scene_overlay_font_size(None)


def _sync_font_sizes_from_meta(meta: dict[str, Any]) -> None:
    cap = meta.get("caption_font_size")
    sc = meta.get("scene_overlay_font_size")
    fpr = ("font_ui", cap, sc)
    if st.session_state.get("_font_ui_fpr") == fpr:
        return
    st.session_state["_font_ui_fpr"] = fpr
    if isinstance(cap, (int, float)):
        st.session_state[_SS_CAPTION_FONT] = resolve_caption_font_size(int(cap))
    if isinstance(sc, (int, float)):
        st.session_state[_SS_SCENE_FONT] = resolve_scene_overlay_font_size(int(sc))


def _export_font_size_kwargs() -> dict[str, int]:
    return {
        "caption_font_size": int(st.session_state[_SS_CAPTION_FONT]),
        "scene_overlay_font_size": int(st.session_state[_SS_SCENE_FONT]),
    }


def _sync_narration_from_meta(meta: dict[str, Any]) -> None:
    """Sync Step 2 narration widgets from export metadata (call before Step 2 renders)."""
    narr = meta.get("ai_narration")
    if not isinstance(narr, dict) or not narr.get("applied"):
        return
    segs = narr.get("segment_lines")
    n_segs = len(segs) if isinstance(segs, list) else 0
    fpr = ("narr_ui", n_segs, narr.get("script"), narr.get("volume"))
    if st.session_state.get("_narr_ui_fpr") == fpr:
        return
    st.session_state["_narr_ui_fpr"] = fpr
    st.session_state[_SS_AI_NARRATION] = True
    vol = narr.get("volume")
    if isinstance(vol, (int, float)):
        st.session_state[_SS_NARRATION_VOL] = float(vol)


def _narration_reexport_kwargs(meta: dict[str, Any] | None) -> dict[str, Any]:
    """Keep AI narration on re-export when Step 2 checkbox is off but prior export had it."""
    use = bool(st.session_state.get(_SS_AI_NARRATION))
    narr_meta: dict[str, Any] | None = None
    if meta is not None:
        raw = meta.get("ai_narration")
        if isinstance(raw, dict) and raw.get("applied"):
            use = True
            narr_meta = raw
    return {
        "ai_narration": use,
        "ai_narration_meta": narr_meta,
        "narration_volume": float(st.session_state[_SS_NARRATION_VOL]),
    }


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


def _parse_overlay_positions_csv(s: str) -> list[OverlayPosition] | None:
    raw = (s or "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return None
    return [coerce_overlay_position(p) for p in parts]


def _scene_positions_for_reexport(meta: dict[str, Any], ov_lines: list[str]) -> list[OverlayPosition] | None:
    """Widget state if initialized, else positions stored on last export metadata."""
    if not ov_lines:
        return None
    fpr = ("ov_pos_ui", tuple(ov_lines))
    if st.session_state.get("_ov_pos_ui_fpr") == fpr:
        try:
            return [
                cast(OverlayPosition, st.session_state[f"opos_sb_{i}"]) for i in range(len(ov_lines))
            ]
        except KeyError:
            pass
    raw = meta.get("on_screen_overlay_positions")
    if isinstance(raw, list) and len(raw) == len(ov_lines):
        return [coerce_overlay_position(str(x)) for x in raw]
    return None


def _read_scene_editors(
    n: int, duration_sec: float
) -> tuple[list[str], list[OverlayPosition], list[tuple[float, float]]] | None:
    """
    Read per-line text, vertical band, and timing from Step 4 widgets. Empty text rows are dropped
    together with their band and time window.
    """
    if n <= 0:
        return None
    try:
        raw_t = [str(st.session_state[f"otext_sb_{i}"]).strip() for i in range(n)]
        raw_p = [cast(OverlayPosition, st.session_state[f"opos_sb_{i}"]) for i in range(n)]
        raw_times = [
            (
                float(st.session_state[f"ostart_sb_{i}"]),
                float(st.session_state[f"oend_sb_{i}"]),
            )
            for i in range(n)
        ]
    except KeyError:
        return None
    triples: list[tuple[str, OverlayPosition, tuple[float, float]]] = []
    for t, p, tw in zip(raw_t, raw_p, raw_times):
        cleaned = strip_overlay_quotes(t).strip()
        if cleaned:
            triples.append((cleaned, p, tw))
    if not triples:
        return None
    return (
        [a for a, _, _ in triples],
        [b for _, b, _ in triples],
        [c for _, _, c in triples],
    )


def _whisper_cache_for_reexport(meta: dict[str, Any] | None) -> dict[str, Any] | None:
    """Session cache first, then metadata from the last successful export."""
    cache = st.session_state.get(_SS_WHISPER_CACHE)
    if isinstance(cache, dict):
        return cache
    if meta and isinstance(meta.get("whisper_cache"), dict):
        return meta["whisper_cache"]
    return None


def _reuse_whisper_cache_arg(meta: dict[str, Any] | None) -> dict[str, Any] | None:
    """Only pass a cache when it has word timings (skips Whisper). Otherwise pipeline re-transcribes."""
    cache = _whisper_cache_for_reexport(meta)
    if not isinstance(cache, dict):
        return None
    words = cache.get("words")
    if isinstance(words, list) and len(words) > 0:
        return cache
    return None


def _sync_whisper_cache_from_meta(meta: dict[str, Any]) -> None:
    wc = meta.get("whisper_cache")
    if isinstance(wc, dict):
        st.session_state[_SS_WHISPER_CACHE] = wc


def _overlay_times_fingerprint(meta: dict[str, Any]) -> tuple[tuple[float, float], ...]:
    raw = meta.get("on_screen_overlay_times")
    if not isinstance(raw, list):
        return ()
    out: list[tuple[float, float]] = []
    for item in raw:
        try:
            if isinstance(item, dict):
                out.append((float(item["start"]), float(item["end"])))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append((float(item[0]), float(item[1])))
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(out)


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
    """Demo-style step rail: Streamlit buttons (no full page load); circles via title-scoped CSS."""
    cur = max(1, min(4, int(st.session_state.get(_CURRENT_STEP, 1))))
    st.session_state[_CURRENT_STEP] = cur

    st.markdown(_wizard_stepper_styles(), unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 5, 1])
    with mid:
        cols = st.columns([1, 0.4, 1, 0.4, 1, 0.4, 1])
        for idx in range(4):
            step = idx + 1
            role = "done" if step < cur else "current" if step == cur else "todo"
            label = "✓" if role == "done" else str(step)
            btype = "primary" if role == "current" else "secondary"
            # `help` becomes `title=`; keep value exactly `shortsai_step_{n}_{role}` for CSS selectors.
            hint = f"shortsai_step_{step}_{role}"
            with cols[idx * 2]:
                if st.button(
                    label,
                    key=f"wiz_step_{step}",
                    type=btype,
                    use_container_width=False,
                    help=hint,
                ):
                    st.session_state[_CURRENT_STEP] = step
                    st.session_state[_SCROLL_PENDING] = step
                    st.rerun()
            if idx < 3:
                with cols[idx * 2 + 1]:
                    st.markdown('<div class="shortsai-bar-inline"></div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="shortsai-step-wrap">'
        f'<div class="shortsai-step-meta">Step {cur} of 4</div>'
        f'<div class="shortsai-step-title">{WIZARD_STEP_LABELS[cur - 1]}</div></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Tap a **step circle** above to jump (same page). **← Back** / **Next →** under each section still work. "
        "Use the **↑** floating button (bottom-right) to scroll back to the top."
    )

    l1, l2, l3, l4 = st.columns(4)
    for col, lbl in zip([l1, l2, l3, l4], WIZARD_STEP_LABELS):
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

    _meta_boot = st.session_state.get(_SS_META)
    if isinstance(_meta_boot, dict):
        _sync_narration_from_meta(_meta_boot)

    st.title("ShortsAI Studio")
    st.markdown(
        '<div id="shortsai-page-top" style="scroll-margin-top:4.5rem;height:1px;width:100%;margin:0;padding:0;"></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Turn a clip into a vertical Short: burned-in speech captions, optional music, scene text overlays, "
        "and YouTube-ready title, description, and tags."
    )
    st.caption(
        "Follow the steps below. The **top step bar** shows progress (✓ done, blue = current); "
        "the **↑** control at the bottom-right jumps to the top and shows your step number."
    )

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
            f"Choose a file (max {max_duration_sec_from_env():.0f}s). Your file is kept until you replace it.",
        )
        uploaded = st.file_uploader(
            "Video file",
            type=["mp4", "mov", "webm", "mkv", "avi"],
            help=(
                f"Maximum length {max_duration_sec_from_env():.0f}s "
                f"(default 60; set SHORTSAI_MAX_DURATION_SEC up to {HARD_MAX_DURATION_SEC:.0f} in .env)."
            ),
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

        st.radio(
            "On-screen scene text — default vertical band",
            options=("upper", "middle", "lower"),
            format_func=lambda v: {
                "upper": "Upper — horizontal center, upper third",
                "middle": "Middle — horizontal center (default)",
                "lower": "Lower — horizontal center, lower third",
            }[v],
            key=_SS_OVERLAY_DEFAULT,
            help="Used for each timed red scene line when you do not set a per-line value (Step 4) "
            "or optional comma list below. Does not move speech captions.",
        )
        st.text_input(
            "Optional: per-line vertical bands (comma-separated)",
            key=_SS_OVERLAY_POSITIONS_CSV,
            placeholder="e.g. upper, middle, lower",
            help="Same order as timed scene lines after export (often 1–4). Words: upper, middle, lower. "
            "Shorter lists pad with the default band above; extra entries are ignored.",
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
        st.checkbox(
            "AI hook cold open (~5s best moment first, then the clip from the start)",
            key=_SS_AI_HOOK,
            help=(
                "Uses OpenAI vision to pick a short eye-catching segment from your upload and "
                f"plays it before the rest (total length still ≤ {max_duration_sec_from_env():.0f}s). "
                "Needs OPENAI_API_KEY; "
                "without a key, uses a simple mid-clip heuristic. Set SHORTSAI_AI_HOOK_SEC in .env "
                "to change hook length (3–8s)."
            ),
        )
        if st.session_state.get(_SS_AI_HOOK) and not api_key:
            st.caption(
                "No API key — hook placement uses a simple heuristic, not vision scoring."
            )
        st.checkbox(
            "AI narration voiceover (English script + OpenAI TTS; ducks original speech)",
            key=_SS_AI_NARRATION,
            help=(
                "Splits the clip into equal time segments (~few lines per minute), writes one line per "
                "segment from vision, and speaks it during that window (the finale line is aligned toward "
                "the end). Not beat-synced to every action. Requires OPENAI_API_KEY."
            ),
        )
        if st.session_state.get(_SS_AI_NARRATION) and not api_key:
            st.caption("Set OPENAI_API_KEY in .env to enable AI narration.")
        st.slider(
            "Narration volume (voiceover loudness)",
            0.5,
            2.5,
            step=0.05,
            key=_SS_NARRATION_VOL,
            help=(
                "Boosts the AI narrator in the final mix (original speech stays ducked). "
                "Default ~1.45; set SHORTSAI_NARRATION_VOL in .env for a permanent default."
            ),
        )
        st.checkbox(
            "Music / little speech: build burned-in captions from on-screen text (OpenAI vision; needs API key)",
            key=_SS_VISION_ONSCREEN_SUBS,
            help="When Whisper finds no or very few words, read visible lyrics/titles from the source video and "
            "burn them as timed captions. Does not replace the red AI scene lines.",
        )
        st.checkbox(
            "Use English for those vision captions (translate if on-screen text is not English)",
            key=_SS_VISION_ONSCREEN_EN,
            disabled=not bool(st.session_state.get(_SS_VISION_ONSCREEN_SUBS)),
            help="Runs a second API step (text-only) so captions are English—vision alone often copies Chinese from the video.",
        )
        st.caption(
            "Speech caption and red scene text sizes: adjust in **Step 4** after export, or set "
            "SHORTSAI_CAPTION_FONT_SIZE / SHORTSAI_SCENE_OVERLAY_FONT_SIZE in .env before generating."
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
                    max_dur = max_duration_sec_from_env()
                    if dur > max_dur + 0.05:
                        status.update(label="Too long", state="error")
                        st.error(
                            f"This clip is **{dur:.1f}s**. Maximum length is **{max_dur:.0f}s** "
                            f"(set `SHORTSAI_MAX_DURATION_SEC` up to **{HARD_MAX_DURATION_SEC:.0f}** in `.env` "
                            "to allow longer clips, then restart the app). Trim the video or raise the limit."
                        )
                        shutil.rmtree(work, ignore_errors=True)
                        st.stop()

                    has_audio = ffmpeg_util.has_audio_stream(src)
                    if not has_audio:
                        status.write("Note: no audio track detected — captions and transcript may be empty.")

                    model = _whisper_model()
                    csv_pos = _parse_overlay_positions_csv(
                        str(st.session_state.get(_SS_OVERLAY_POSITIONS_CSV) or "")
                    )
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
                        vision_onscreen_subtitles=bool(st.session_state.get(_SS_VISION_ONSCREEN_SUBS)),
                        vision_onscreen_subtitles_english=bool(st.session_state.get(_SS_VISION_ONSCREEN_EN)),
                        overlay_position=cast(OverlayPosition, st.session_state[_SS_OVERLAY_DEFAULT]),
                        overlay_positions=csv_pos,
                        ai_hook_cold_open=bool(st.session_state.get(_SS_AI_HOOK)),
                        ai_narration=bool(st.session_state.get(_SS_AI_NARRATION)),
                        narration_volume=float(st.session_state[_SS_NARRATION_VOL]),
                        **_export_font_size_kwargs(),
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
            _sync_whisper_cache_from_meta(meta)
            _sync_font_sizes_from_meta(meta)
            ov_lines_result = list(meta.get("on_screen_overlay_lines") or [])
            clip_dur = float(meta.get("duration_seconds") or max_duration_sec_from_env())
            fpr_ov = (
                "ov_pos_ui",
                tuple(ov_lines_result),
                _overlay_times_fingerprint(meta),
            )
            if st.session_state.get("_ov_pos_ui_fpr") != fpr_ov:
                st.session_state["_ov_pos_ui_fpr"] = fpr_ov
                prev_ov = meta.get("on_screen_overlay_positions")
                prev_times = overlay_times_from_meta(
                    meta.get("on_screen_overlay_times"),
                    count=len(ov_lines_result),
                    duration_sec=clip_dur,
                )
                if prev_times is None and ov_lines_result:
                    prev_times = default_overlay_line_times(len(ov_lines_result), clip_dur)
                for j in range(16):
                    st.session_state.pop(f"opos_sb_{j}", None)
                    st.session_state.pop(f"otext_sb_{j}", None)
                    st.session_state.pop(f"ostart_sb_{j}", None)
                    st.session_state.pop(f"oend_sb_{j}", None)
                ddef = coerce_overlay_position(st.session_state.get(_SS_OVERLAY_DEFAULT, "middle"))
                for i in range(len(ov_lines_result)):
                    d = ddef
                    if isinstance(prev_ov, list) and i < len(prev_ov):
                        d = coerce_overlay_position(str(prev_ov[i]))
                    st.session_state[f"opos_sb_{i}"] = d
                    st.session_state[f"otext_sb_{i}"] = strip_overlay_quotes(
                        str(ov_lines_result[i])
                    )
                    if prev_times and i < len(prev_times):
                        st.session_state[f"ostart_sb_{i}"] = float(prev_times[i][0])
                        st.session_state[f"oend_sb_{i}"] = float(prev_times[i][1])
                    else:
                        st.session_state[f"ostart_sb_{i}"] = 0.0
                        st.session_state[f"oend_sb_{i}"] = clip_dur

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

            hook_info = meta.get("ai_hook")
            if isinstance(hook_info, dict) and hook_info.get("applied"):
                st.info(
                    f"**AI hook** prepended {hook_info.get('hook_sec', '?')}s from source "
                    f"{hook_info.get('source_start', '?')}s–{hook_info.get('source_end', '?')}s "
                    f"({hook_info.get('method', '')}: {hook_info.get('reason', '')})."
                )
            narr_info = meta.get("ai_narration")
            if isinstance(narr_info, dict) and narr_info.get("applied"):
                src = narr_info.get("source") or "scene_timed"
                segs = narr_info.get("segment_lines")
                if isinstance(segs, list) and segs:
                    with st.expander(
                        f"AI narration ({narr_info.get('voice', 'alloy')}, {src}) — "
                        f"{len(segs)} timed lines",
                        expanded=False,
                    ):
                        for seg in segs:
                            if isinstance(seg, dict):
                                st.caption(
                                    f"{seg.get('start', '?')}s – {seg.get('end', '?')}s"
                                )
                                st.write(seg.get("text", ""))
                else:
                    st.info(
                        f"**AI narration** (voice: {narr_info.get('voice', 'alloy')}) — "
                        f"{str(narr_info.get('script', ''))[:200]}"
                    )
            elif isinstance(narr_info, dict) and narr_info.get("error"):
                st.caption(f"AI narration was not applied: {narr_info.get('error', '')[:200]}")

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

            with st.expander("Adjust caption & scene text sizes", expanded=False):
                st.caption(
                    "Change burned-in **speech caption** size and **red scene line** size, then re-export. "
                    "Whisper is skipped when the word cache is present."
                )
                st.slider(
                    "Speech caption size",
                    min_value=10,
                    max_value=44,
                    key=_SS_CAPTION_FONT,
                    help="Burned-in SRT subtitle font (ASS FontSize). Default 10.",
                )
                st.slider(
                    "Red scene text size",
                    min_value=36,
                    max_value=100,
                    key=_SS_SCENE_FONT,
                    help="Timed on-screen scene lines (drawtext). Default 64.",
                )
                if st.button("Re-export MP4 with new text sizes", key="btn_reexport_font_sizes"):
                    ub_fs = st.session_state.get(_SS_UPLOAD_BYTES)
                    if not ub_fs:
                        st.error(
                            "Upload missing — select your video again in **Step 1**, or run "
                            "**Generate Short** in Step 3."
                        )
                    else:
                        music_fs: Path | None = None
                        pick_fs = st.session_state.get(_SS_MUSIC_PICK, "None")
                        if pick_fs and pick_fs != "None" and music_files:
                            for p in music_files:
                                if p.name == pick_fs:
                                    music_fs = p
                                    break
                        work_fs = Path(tempfile.mkdtemp(prefix="shortsai_re_font_"))
                        try:
                            src_fs = work_fs / f"upload{Path(upload_name).suffix or '.mp4'}"
                            src_fs.write_bytes(ub_fs)
                            model = _whisper_model()
                            lines_fs: list[str] | None = None
                            pos_fs: list[OverlayPosition] | None = None
                            times_fs: list[tuple[float, float]] | None = None
                            if ov_lines_result:
                                ed_fs = _read_scene_editors(len(ov_lines_result), clip_dur)
                                if ed_fs:
                                    lines_fs, pos_fs, times_fs = ed_fs
                                else:
                                    lines_fs = [
                                        strip_overlay_quotes(str(x))
                                        for x in ov_lines_result
                                        if strip_overlay_quotes(str(x))
                                    ]
                                    pos_fs = _scene_positions_for_reexport(meta, ov_lines_result)
                                    times_fs = overlay_times_from_meta(
                                        meta.get("on_screen_overlay_times"),
                                        count=len(lines_fs or []),
                                        duration_sec=clip_dur,
                                    )
                            if times_fs and any(t1 <= t0 for t0, t1 in times_fs):
                                st.error("Each scene line needs **End (s)** greater than **Start (s)**.")
                            else:
                                srt_keep_fs = (meta.get("speech_srt") or "").strip()
                                with st.spinner("Re-exporting with new text sizes…"):
                                    mp4_fs, meta_fs = process_upload(
                                        src_fs,
                                        work_dir=work_fs,
                                        whisper=model,
                                        whisper_model=os.environ.get(
                                            "SHORTSAI_WHISPER_MODEL", "base"
                                        ),
                                        openai_api_key=api_key.strip() or None,
                                        music_path=music_fs,
                                        music_volume=float(st.session_state[_SS_MUSIC_VOL]),
                                        manual_overlay_text=None,
                                        progress=None,
                                        whisper_task=st.session_state[_SS_WHISPER_TASK],
                                        whisper_language_hint=(
                                            st.session_state.get(_SS_SPEECH_HINT) or ""
                                        ).strip()
                                        or None,
                                        vertical_fit=cast(
                                            VerticalFitMode, st.session_state[_SS_VERTICAL_FIT]
                                        ),
                                        caption_srt_override=srt_keep_fs if srt_keep_fs else None,
                                        reuse_whisper_cache=_reuse_whisper_cache_arg(meta),
                                        vision_onscreen_subtitles=bool(
                                            st.session_state.get(_SS_VISION_ONSCREEN_SUBS)
                                        ),
                                        vision_onscreen_subtitles_english=bool(
                                            st.session_state.get(_SS_VISION_ONSCREEN_EN)
                                        ),
                                        overlay_position=cast(
                                            OverlayPosition, st.session_state[_SS_OVERLAY_DEFAULT]
                                        ),
                                        overlay_positions=pos_fs,
                                        scene_overlay_lines_override=lines_fs,
                                        scene_overlay_times_override=times_fs,
                                        ai_hook_cold_open=bool(st.session_state.get(_SS_AI_HOOK)),
                                        ai_hook_meta=meta.get("ai_hook")
                                        if isinstance(meta.get("ai_hook"), dict)
                                        else None,
                                        **_narration_reexport_kwargs(meta),
                                        **_export_font_size_kwargs(),
                                    )
                                if music_fs is not None:
                                    meta_fs["music_file"] = music_fs.name
                                else:
                                    meta_fs["music_file"] = None
                                st.session_state[_SS_WHISPER_CACHE] = meta_fs.get("whisper_cache")
                                st.session_state[_SS_MP4] = mp4_fs
                                st.session_state[_SS_META] = meta_fs
                                st.success("Re-export complete.")
                                st.rerun()
                        except Exception as e:
                            st.exception(e)
                        finally:
                            shutil.rmtree(work_fs, ignore_errors=True)

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
                    if not st.session_state.get(_SS_UPLOAD_BYTES):
                        st.error(
                            "Upload missing — select your video again in **Step 1**, or run "
                            "**Generate Short** in Step 3."
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
                                srt_scene_lines: list[str] | None = None
                                pos_r: list[OverlayPosition] | None = None
                                times_r: list[tuple[float, float]] | None = None
                                srt_manual: str | None = None
                                if ov_lines_result:
                                    ed_srt = _read_scene_editors(
                                        len(ov_lines_result), clip_dur
                                    )
                                    if ed_srt:
                                        srt_scene_lines, pos_r, times_r = ed_srt
                                    else:
                                        pos_r = _scene_positions_for_reexport(
                                            meta, ov_lines_result
                                        )
                                        srt_scene_lines = [
                                            strip_overlay_quotes(str(x))
                                            for x in ov_lines_result
                                            if strip_overlay_quotes(str(x))
                                        ]
                                if not srt_scene_lines:
                                    srt_manual = (st.session_state.get(_SS_MANUAL_OVERLAY) or "").strip() or None
                                with st.spinner("Re-exporting with your SRT…"):
                                    mp4_b, meta2 = process_upload(
                                        src2,
                                        work_dir=work2,
                                        whisper=model,
                                        whisper_model=os.environ.get("SHORTSAI_WHISPER_MODEL", "base"),
                                        openai_api_key=api_key.strip() or None,
                                        music_path=music_re,
                                        music_volume=float(st.session_state[_SS_MUSIC_VOL]),
                                        manual_overlay_text=srt_manual,
                                        progress=None,
                                        whisper_task=st.session_state[_SS_WHISPER_TASK],
                                        whisper_language_hint=(st.session_state.get(_SS_SPEECH_HINT) or "").strip()
                                        or None,
                                        vertical_fit=cast(VerticalFitMode, st.session_state[_SS_VERTICAL_FIT]),
                                        caption_srt_override=edited_srt.strip(),
                                        reuse_whisper_cache=_reuse_whisper_cache_arg(meta),
                                        vision_onscreen_subtitles=bool(
                                            st.session_state.get(_SS_VISION_ONSCREEN_SUBS)
                                        ),
                                        vision_onscreen_subtitles_english=bool(
                                            st.session_state.get(_SS_VISION_ONSCREEN_EN)
                                        ),
                                        overlay_position=cast(
                                            OverlayPosition, st.session_state[_SS_OVERLAY_DEFAULT]
                                        ),
                                        overlay_positions=pos_r,
                                        scene_overlay_lines_override=srt_scene_lines,
                                        scene_overlay_times_override=times_r,
                                        ai_hook_cold_open=bool(st.session_state.get(_SS_AI_HOOK)),
                                        ai_hook_meta=meta.get("ai_hook")
                                        if isinstance(meta.get("ai_hook"), dict)
                                        else None,
                                        **_narration_reexport_kwargs(meta),
                                        **_export_font_size_kwargs(),
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

            if ov_lines_result:
                with st.expander(
                    "Scene text — edit wording, timing & vertical position",
                    expanded=False,
                ):
                    st.caption(
                        f"Edit **red timed** scene lines, **start/end seconds** (0–{clip_dur:.1f}s), "
                        "and vertical band. Clear text to drop a beat. Re-export skips Whisper when the cache is present."
                    )
                    for i in range(len(ov_lines_result)):
                        st.text_area(
                            f"Line {i + 1} — text",
                            key=f"otext_sb_{i}",
                            height=68,
                            help="Leave empty to remove this slot.",
                        )
                        tc1, tc2 = st.columns(2)
                        with tc1:
                            st.number_input(
                                "Start (s)",
                                min_value=0.0,
                                max_value=clip_dur,
                                step=0.1,
                                format="%.1f",
                                key=f"ostart_sb_{i}",
                            )
                        with tc2:
                            st.number_input(
                                "End (s)",
                                min_value=0.0,
                                max_value=clip_dur,
                                step=0.1,
                                format="%.1f",
                                key=f"oend_sb_{i}",
                            )
                        st.selectbox(
                            "Vertical band",
                            options=("upper", "middle", "lower"),
                            key=f"opos_sb_{i}",
                            format_func=lambda v: {
                                "upper": "Upper — center",
                                "middle": "Middle — center",
                                "lower": "Lower — center",
                            }[v],
                        )
                        st.markdown("---")

                    if st.button("Re-export MP4 with edited scene text", key="btn_reexport_scene_pos"):
                        ub2 = st.session_state.get(_SS_UPLOAD_BYTES)
                        if not ub2:
                            st.error(
                                "Upload missing — select your video again in **Step 1**, or run "
                                "**Generate Short** in Step 3."
                            )
                        else:
                            reuse_cache = _reuse_whisper_cache_arg(meta)
                            if reuse_cache is None and (meta.get("speech_srt") or "").strip():
                                st.caption(
                                    "No speech word cache — re-export will reuse your saved captions "
                                    "and may take longer if speech is re-analyzed."
                                )
                            music_re2: Path | None = None
                            pick2 = st.session_state.get(_SS_MUSIC_PICK, "None")
                            if pick2 and pick2 != "None" and music_files:
                                for p in music_files:
                                    if p.name == pick2:
                                        music_re2 = p
                                        break
                            work3 = Path(tempfile.mkdtemp(prefix="shortsai_re_scene_"))
                            try:
                                src3 = work3 / f"upload{Path(upload_name).suffix or '.mp4'}"
                                src3.write_bytes(ub2)
                                model = _whisper_model()
                                ed3 = _read_scene_editors(len(ov_lines_result), clip_dur)
                                if ed3 is None:
                                    st.error("Could not read scene text from the form.")
                                else:
                                    lines_use, pos_use, times_use = ed3
                                    if any(t1 <= t0 for t0, t1 in times_use):
                                        st.error(
                                            "Each line needs **End (s)** greater than **Start (s)**."
                                        )
                                    else:
                                        srt_keep = (meta.get("speech_srt") or "").strip()
                                        with st.spinner("Re-exporting scene text…"):
                                            mp4_b3, meta3 = process_upload(
                                                src3,
                                                work_dir=work3,
                                                whisper=model,
                                                whisper_model=os.environ.get(
                                                    "SHORTSAI_WHISPER_MODEL", "base"
                                                ),
                                                openai_api_key=api_key.strip() or None,
                                                music_path=music_re2,
                                                music_volume=float(
                                                    st.session_state[_SS_MUSIC_VOL]
                                                ),
                                                manual_overlay_text=None,
                                                progress=None,
                                                whisper_task=st.session_state[_SS_WHISPER_TASK],
                                                whisper_language_hint=(
                                                    st.session_state.get(_SS_SPEECH_HINT) or ""
                                                ).strip()
                                                or None,
                                                vertical_fit=cast(
                                                    VerticalFitMode,
                                                    st.session_state[_SS_VERTICAL_FIT],
                                                ),
                                                caption_srt_override=srt_keep if srt_keep else None,
                                                reuse_whisper_cache=reuse_cache,
                                                vision_onscreen_subtitles=bool(
                                                    st.session_state.get(_SS_VISION_ONSCREEN_SUBS)
                                                ),
                                                vision_onscreen_subtitles_english=bool(
                                                    st.session_state.get(
                                                        _SS_VISION_ONSCREEN_EN
                                                    )
                                                ),
                                                overlay_position=cast(
                                                    OverlayPosition,
                                                    st.session_state[_SS_OVERLAY_DEFAULT],
                                                ),
                                                overlay_positions=pos_use,
                                                scene_overlay_lines_override=lines_use,
                                                scene_overlay_times_override=times_use,
                                                ai_hook_cold_open=bool(
                                                    st.session_state.get(_SS_AI_HOOK)
                                                ),
                                                ai_hook_meta=meta.get("ai_hook")
                                                if isinstance(meta.get("ai_hook"), dict)
                                                else None,
                                                **_narration_reexport_kwargs(meta),
                                                **_export_font_size_kwargs(),
                                            )
                                        if music_re2 is not None:
                                            meta3["music_file"] = music_re2.name
                                        else:
                                            meta3["music_file"] = None
                                        st.session_state[_SS_WHISPER_CACHE] = meta3.get(
                                            "whisper_cache"
                                        )
                                        st.session_state[_SS_MP4] = mp4_b3
                                        st.session_state[_SS_META] = meta3
                                        st.success("Re-export complete.")
                                        st.rerun()
                            except Exception as e:
                                st.exception(e)
                            finally:
                                shutil.rmtree(work3, ignore_errors=True)

            st.caption(
                "Copy the fields below into YouTube Studio → your short → **Details**. "
                "They are generated from speech (or from video frames if there was little/no speech)—not from this tip line."
            )
            src = meta.get("metadata_source")
            if isinstance(src, str) and src:
                st.caption(f"Metadata source: `{src}`")
            bsrc = meta.get("burned_subtitle_source")
            if isinstance(bsrc, str) and bsrc:
                label = (
                    "Burned-in captions (AI narration text)"
                    if bsrc == "ai_narration"
                    else f"Burned-in caption source: `{bsrc}`"
                )
                st.caption(label)

            st.markdown("##### Title")
            title = meta.get("title") or ""
            st.markdown(f"### {title}" if title else "_No title_")

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

    cur_step = max(1, min(4, int(st.session_state.get(_CURRENT_STEP, 1))))
    _inject_scroll_top_fab(cur_step)


if __name__ == "__main__":
    main()
