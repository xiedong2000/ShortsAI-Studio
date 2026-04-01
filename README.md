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
pip install -r requirements.txt
# Windows: copy .env.example .env
# macOS/Linux: cp .env.example .env
```

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
  - `SHORTSAI_WHISPER_DEVICE` / `SHORTSAI_WHISPER_COMPUTE` — e.g. `cuda` and `float16` if you have a GPU.

**4. Run the app**

```bash
streamlit run app.py
```

The UI opens in your browser. Upload a clip (up to **60 seconds**) → get a **1080×1920** MP4 with burned-in captions and a **metadata.json** (title, description, tags, transcript, etc.). Add `.mp3` files under `assets/music/` to mix a YouTube Audio Library track.

**Text Overlays**: Optionally add on-screen text overlays including title, description, hashtags, and attribution directly on the video (perfect for YouTube Shorts).

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

1. Find the folder containing `ffmpeg.exe` and `ffprobe.exe` (e.g., `C:\ffmpeg\bin`)
2. Open `.env` and add:
   ```
   SHORTSAI_FFMPEG_DIR=C:\ffmpeg\bin
   ```
   Or set each explicitly:
   ```
   FFMPEG_PATH=C:\ffmpeg\bin\ffmpeg.exe
   FFPROBE_PATH=C:\ffmpeg\bin\ffprobe.exe
   ```
3. Save and restart Streamlit.

---

## Planned pipeline (MVP)

1. Upload short video (&lt; 60s).
2. Transcribe speech → timed captions.
3. Optional: mix a track from `assets/music/` (YouTube Audio Library downloads).
4. Export vertical 9:16 MP4 + JSON sidecar (title, description, tags, attribution if required).
