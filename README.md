# ShortsAI-Studio

AI-powered YouTube Shorts generator that transforms short videos into engaging content with automatic captions, optional background music, and GPT-generated titles, descriptions, and hashtags.

## Background music: YouTube Audio Library

All bundled or suggested background tracks for this project should come from the **YouTube Audio Library** so licensing stays aligned with Shorts uploads and YouTube’s rules.

1. Open [YouTube Studio](https://studio.youtube.com) → **Audio library** (Music tab).
2. Download tracks you want (MP3 or other offered formats).
3. Place files in `assets/music/` (see that folder’s README). The pipeline will mix them under speech with adjustable volume.

**Important:** Some tracks require **attribution** in the video description. The library shows requirements per track—copy the required credit into your generated metadata JSON or description before publishing.

Do not use random “royalty-free” packs from unclear sources in the default workflow; stick to YouTube Audio Library unless you add a separate, explicitly licensed source later.

## Quick start (v1)

**Prerequisites**

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html): `ffmpeg` and `ffprobe` must be available (see [Fix ffmpeg Error](#fix-ffmpeg-error) if you encounter "Could not find ffmpeg and ffprobe" on first run).

**1. Setup environment**

```bash
cd ShortsAI-Studio
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
# Windows: copy .env.example .env
# macOS/Linux: cp .env.example .env
```

**Upgrade pip before `pip install -r requirements.txt`:** Fresh venvs on some systems ship with very old pip (for example 19.x). That can break installs (for example `tokenizers` / `pyproject.toml` errors). The **1. Setup environment** commands above already run `python -m pip install --upgrade pip setuptools wheel` before `pip install -r requirements.txt`.

**Python 3.8 (not recommended):** The project targets **Python 3.10+**. If you still use **3.8**, `requirements.txt` includes a conditional line so `tokenizers` stays on a release that provides Windows wheels (`<0.21` when `python_version < "3.9"`). Upgrading pip first (above) is still required.

**venv on Windows:** Creating `.venv` runs `ensurepip` (installing pip) and can sit with no output for a minute or two. Let it finish; pressing Ctrl+C raises `KeyboardInterrupt` and leaves a broken venv. If that happens, delete the `.venv` folder and run `python -m venv .venv` again. If it keeps failing, use `python -m venv .venv --without-pip`, then `.\.venv\Scripts\python.exe -m ensurepip --upgrade`.

**2. Install ffmpeg** (if not already installed)

Check if ffmpeg is installed:
```bash
ffmpeg -version
ffprobe -version
```

If not found, install it:
- **Windows:** `winget install Gyan.FFmpeg`, then **close and reopen** your terminal to refresh PATH.
- **macOS:** `brew install ffmpeg`
- **Linux (Ubuntu):** `sudo apt-get install ffmpeg`

**3. Configure (optional)**

Edit `.env` if you want:
- **GPT metadata:** Add your `OPENAI_API_KEY`
- **Whisper tuning:**
  - `SHORTSAI_WHISPER_MODEL` — default `base` (`tiny` is faster, `small` more accurate).
  - `SHORTSAI_WHISPER_TASK` — default `transcribe` (subtitles match the spoken language). Set to **`translate`** to get **English** subtitles and transcript when the audio is Chinese or any other language Whisper supports. In the Streamlit app you can choose this per export (**Subtitles & transcript**); the UI defaults match these env vars.
  - `SHORTSAI_WHISPER_LANGUAGE` — optional ISO 639-1 hint (e.g. `zh`) if auto-detection is wrong; leave unset for auto-detect. Also available in the app (optional text field); leave empty there for auto-detect (empty overrides `.env` for that run).
  - `SHORTSAI_WHISPER_DEVICE` / `SHORTSAI_WHISPER_COMPUTE` — e.g. `cuda` and `float16` if you have a GPU.
- **ffmpeg not on PATH for Streamlit:** Set `SHORTSAI_FFMPEG_DIR` to the folder that contains **both** `ffmpeg` and `ffprobe` (on Windows, both `.exe`). This is useful right after `winget install Gyan.FFmpeg` when your IDE has not picked up the updated PATH yet—Winget often adds shims under `%LOCALAPPDATA%\Microsoft\WinGet\Links` (adjust the drive/username as needed). Alternatively set `FFMPEG_PATH` and `FFPROBE_PATH` to each executable. See `.env.example` for commented placeholders and [Fix ffmpeg Error](#fix-ffmpeg-error) below.

**4. Run the app**

```bash
streamlit run app.py
```

The UI opens in your browser. Upload a clip (up to **60 seconds**) → get a **1080×1920** MP4 with burned-in captions and a **metadata.json** (title, description, tags, transcript, etc.). Add `.mp3` files under `assets/music/` to mix a YouTube Audio Library track.

**Text Overlays**: Optionally add on-screen text overlays including title, description, hashtags, and attribution directly on the video (perfect for YouTube Shorts).

**5. CLI smoke test (`shorts_generator.py`, Day 1–2 MVP)**

Same pipeline as the UI, for demos, scripts, or CI. From the project root, use the **venv** so dependencies (`python-dotenv`, `faster-whisper`, etc.) are available:

```bash
# Windows (PowerShell)
.\.venv\Scripts\python.exe shorts_generator.py -i path\to\your_clip.mp4
# Or: .\.venv\Scripts\activate   then   python shorts_generator.py ...
```

If you run plain `python` from a global install that never had `pip install -r requirements.txt`, you will see `ModuleNotFoundError` for `dotenv` or other packages—install into that Python or use `.venv` as above.

Writes `<clip_stem>_shorts.mp4` and `<clip_stem>_shorts.json` next to the input (override with `-o` / `--metadata`). On success the script prints the two output paths (last two lines). `python shorts_generator.py --help` works even without deps (shows usage only).

List bundled library filenames (under `assets/music/`):

```bash
.\.venv\Scripts\python.exe shorts_generator.py --list-music
```

Common options (same interpreter as above, e.g. `.venv\Scripts\python.exe`):

```bash
.\.venv\Scripts\python.exe shorts_generator.py -i clip.mp4 -o out/demo.mp4 --music first
.\.venv\Scripts\python.exe shorts_generator.py -i clip.mp4 --music "Exact Track Name.mp3"
.\.venv\Scripts\python.exe shorts_generator.py -i clip.mp4 --music C:\path\to\track.mp3
.\.venv\Scripts\python.exe shorts_generator.py -i clip.mp4 --translate --language-hint zh
.\.venv\Scripts\python.exe shorts_generator.py -i clip.mp4 --keep-work-dir -q
```

`--music first` picks the first sorted file in `assets/music/` (same idea as picking a track in the Streamlit dropdown).

`--keep-work-dir` leaves the temp folder in place and logs its path for debugging FFmpeg or Whisper issues. Set `OPENAI_API_KEY` in `.env` for GPT metadata and vision scene lines; otherwise the pipeline uses text fallbacks.

**MVP checklist (one demo video):**

1. `ffmpeg -version` and `ffprobe -version` succeed (or set `SHORTSAI_FFMPEG_DIR` / `FFMPEG_PATH` / `FFPROBE_PATH` in `.env`).
2. Run `.\.venv\Scripts\python.exe shorts_generator.py -i your_clip.mp4` (clip under 60s, with audible speech if you want burned-in captions).
3. Confirm: MP4 plays, captions visible (if speech was detected), optional `--music` if you pass a file, open the JSON for title/description/tags/transcript.

---

## Fix ffmpeg error

If you see: **"Could not find ffmpeg and ffprobe"** when running `streamlit run app.py`, use one of these solutions:

### Option 1: Install ffmpeg globally (recommended)

**Windows:**
```bash
winget install Gyan.FFmpeg
# Then close terminal and reopen it (important!)
ffmpeg -version  # verify
ffprobe -version  # verify
```

Then restart Streamlit:
```bash
streamlit run app.py
```

### Option 2: Point to ffmpeg in `.env`

If you have ffmpeg installed elsewhere or `winget` didn't work:

1. Find the folder containing `ffmpeg.exe` and `ffprobe.exe` (e.g., `C:\ffmpeg\bin`, or after **winget** `Gyan.FFmpeg`, often `%LOCALAPPDATA%\Microsoft\WinGet\Links` on Windows—both shims live in that folder).
2. Open `.env` and add:
   ```
   SHORTSAI_FFMPEG_DIR=C:\ffmpeg\bin
   ```
   Or set each explicitly:
   ```
   FFMPEG_PATH=C:\ffmpeg\bin\ffmpeg.exe
   FFPROBE_PATH=C:\ffmpeg\bin\ffprobe.exe
   ```
   Example using the typical WinGet shim directory (replace `YourUser` if needed):
   ```
   SHORTSAI_FFMPEG_DIR=C:\Users\YourUser\AppData\Local\Microsoft\WinGet\Links
   ```
3. Save and restart Streamlit.

---

## Planned pipeline (MVP)

1. Upload short video (&lt; 60s).
2. Transcribe speech → timed captions.
3. Optional: mix a track from `assets/music/` (YouTube Audio Library downloads).
4. Export vertical 9:16 MP4 + JSON sidecar (title, description, tags, attribution if required).
