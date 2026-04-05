from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from faster_whisper import WhisperModel


@dataclass
class WordSpan:
    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    language: str
    text: str
    words: list[WordSpan]


def transcribe(
    audio_path: Path,
    *,
    model: WhisperModel | None = None,
    model_name: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    task: Literal["transcribe", "translate"] = "transcribe",
) -> TranscriptResult:
    if model is None:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language or None,
        task=task,
        word_timestamps=True,
        vad_filter=True,
    )

    segments: list = list(segments_iter)
    words: list[WordSpan] = []
    full_parts: list[str] = []

    for seg in segments:
        full_parts.append(seg.text.strip())
        if getattr(seg, "words", None):
            for w in seg.words:
                t = (w.word or "").strip()
                if not t:
                    continue
                words.append(WordSpan(start=w.start, end=w.end, text=t))
        else:
            words.append(
                WordSpan(start=seg.start, end=seg.end, text=seg.text.strip())
            )

    text = " ".join(p for p in full_parts if p).strip()
    if not words and text:
        words = [WordSpan(start=0.0, end=max(info.duration, 0.1), text=text)]

    # Whisper "translate" always targets English text; keep metadata accurate for downstream.
    out_language = "en" if task == "translate" else (info.language or "en")

    return TranscriptResult(
        language=out_language,
        text=text,
        words=words,
    )
