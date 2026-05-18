from __future__ import annotations

from shortsai.text_sanitize import strip_overlay_quotes
from shortsai.transcribe import WordSpan


def _format_ts(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def _wrap_line(s: str, max_chars: int) -> list[str]:
    s = s.strip()
    if not s:
        return []
    words = s.split()
    lines: list[str] = []
    cur: list[str] = []
    length = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur and length + add > max_chars:
            lines.append(" ".join(cur))
            cur = [w]
            length = len(w)
        else:
            cur.append(w)
            length += add
    if cur:
        lines.append(" ".join(cur))
    return lines


def words_to_srt(
    words: list[WordSpan],
    *,
    max_duration: float = 4.0,
    max_chars: int = 36,
) -> str:
    """Group word-level timings into readable subtitle cues."""
    if not words:
        return ""

    cues: list[tuple[float, float, str]] = []
    buf: list[str] = []
    t0 = words[0].start
    t1 = words[0].end

    def flush() -> None:
        nonlocal buf, t0, t1
        if not buf:
            return
        line = strip_overlay_quotes(" ".join(buf).strip())
        if line:
            cues.append((t0, t1, line))
        buf = []

    for w in words:
        candidate = " ".join(buf + [w.text]).strip()
        dur = w.end - t0

        if buf and (dur > max_duration or len(candidate) > max_chars):
            flush()
            t0 = w.start
            t1 = w.end
            buf = [w.text]
        else:
            if not buf:
                t0 = w.start
            buf.append(w.text)
            t1 = w.end

    flush()

    blocks: list[str] = []
    for i, (a, b, txt) in enumerate(cues, start=1):
        wrapped = _wrap_line(txt, max_chars)
        body = "\n".join(wrapped) if wrapped else txt
        blocks.append(f"{i}\n{_format_ts(a)} --> {_format_ts(b)}\n{body}\n")

    return "\n".join(blocks)


def equal_segments_srt(lines: list[str], duration_sec: float, *, max_chars: int = 42) -> str:
    """
    Build SRT with one cue per segment, splitting [0, duration_sec] into equal windows.
    Used when timings come from video segments (e.g. vision-read on-screen text), not Whisper words.
    """
    cleaned = [
        strip_overlay_quotes(str(x).strip())
        for x in lines
        if strip_overlay_quotes(str(x).strip())
    ]
    if not cleaned or duration_sec <= 0:
        return ""
    n = len(cleaned)
    blocks: list[str] = []
    idx = 1
    for i, txt in enumerate(cleaned):
        t0 = duration_sec * i / n
        if i == n - 1:
            t1 = duration_sec
        else:
            t1 = max(t0 + 0.05, duration_sec * (i + 1) / n - 0.02)
        wrapped = _wrap_line(txt, max_chars)
        body = "\n".join(wrapped) if wrapped else txt
        blocks.append(f"{idx}\n{_format_ts(t0)} --> {_format_ts(t1)}\n{body}\n")
        idx += 1
    return "\n".join(blocks)


def segment_timed_srt(
    segments: list[tuple[float, float, str]],
    *,
    max_chars: int = 36,
) -> str:
    """Build SRT from explicit (start_sec, end_sec, text) segments (e.g. AI narration lines)."""
    blocks: list[str] = []
    idx = 1
    for t0, t1, raw in segments:
        txt = strip_overlay_quotes(str(raw).strip())
        if not txt or txt == "-":
            continue
        end = max(t0 + 0.08, t1)
        wrapped = _wrap_line(txt, max_chars)
        body = "\n".join(wrapped) if wrapped else txt
        blocks.append(f"{idx}\n{_format_ts(t0)} --> {_format_ts(end)}\n{body}\n")
        idx += 1
    return "\n".join(blocks)
