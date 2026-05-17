from __future__ import annotations

import re

# Double quotes + guillemets + common wrappers (models/OCR often wrap lines in "..." or „…").
_OVERLAY_QUOTE_GLOBAL = frozenset(
    '"'
    "`"
    "\u201c\u201d"  # “ ”
    "\u201e\u201f"  # „ ‟
    "\uff02"  # ＂ fullwidth
    "\u00ab\u00bb"  # « »
    "\u2039\u203a"  # ‹ ›
    "\u2033"  # ″ double prime (often misread as ")
    "\u300c\u300d"  # 「 」
    "\u300e\u300f"  # 『 』
    "\u275d\u275e"  # ❝ ❞
    "\u301d\u301e"  # 〝 〞
)
# Singles: strip only from line ends so internal apostrophes (It's) stay intact.
_OVERLAY_QUOTE_EDGE_SINGLE = frozenset("'\u2018\u2019\u201a\u201b")

_HTML_QUOTE_ENTITIES = (
    ("&quot;", '"'),
    ("&#34;", '"'),
    ("&#x22;", '"'),
    ("&ldquo;", "\u201c"),
    ("&rdquo;", "\u201d"),
    ("&#8220;", "\u201c"),
    ("&#8221;", "\u201d"),
)


def strip_overlay_quotes(s: str) -> str:
    """Remove wrapping and decorative quotes from on-screen / subtitle lines."""
    s = (s or "").strip()
    if not s:
        return ""

    for entity, _ in _HTML_QUOTE_ENTITIES:
        s = s.replace(entity, "")

    # Model strings sometimes include JSON-style escapes before quotes are stripped.
    s = re.sub(r'\\+["\']', "", s)

    s = "".join(c for c in s if c not in _OVERLAY_QUOTE_GLOBAL)
    s = s.strip()

    while s and s[0] in _OVERLAY_QUOTE_EDGE_SINGLE:
        s = s[1:].strip()
    while s and s[-1] in _OVERLAY_QUOTE_EDGE_SINGLE:
        s = s[:-1].strip()

    # Residual ASCII " from partial escapes or odd OCR.
    s = s.replace('"', "").strip()
    return s


def sanitize_overlay_lines(lines: list[str]) -> list[str]:
    """Strip quotes from each line and drop empties."""
    return [x for x in (strip_overlay_quotes(t) for t in lines) if x]
