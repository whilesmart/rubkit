from __future__ import annotations

import json
import re

from .models import Judgement, Subject

ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")
EDGES = " \t\n\"'“”‘’«»"


def text_of(subject: Subject) -> str:
    """The words a judge reads and quotes from."""
    content = subject.content
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def _plain(text: str) -> str:
    return " ".join(text.split()).casefold()


def found_in(quote: str, text: str) -> bool:
    """Whether every part of a quote, split at ellipses, appears in the text, ignoring case and spacing."""
    parts = [_plain(part.strip(EDGES)) for part in ELLIPSIS.split(quote.strip(EDGES))]
    parts = [part for part in parts if part]
    plain = _plain(text)
    return bool(parts) and all(part in plain for part in parts)


def grounded(judgement: Judgement, subject: Subject) -> bool:
    """Whether a judgement backs its score with evidence quoted word for word from the subject."""
    text = text_of(subject)
    return bool(judgement.evidence) and all(found_in(quote, text) for quote in judgement.evidence)
