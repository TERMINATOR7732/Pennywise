"""
core/multilingual.py
Language detection and multilingual pipeline helper.

Detects the language of the user's input, translates it to English
for the RAG+LLM pipeline, then translates the answer back to the
user's preferred language.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from ibm.translator import IBMTranslator, LANGUAGE_NAMES, get_translator

# ---------------------------------------------------------------------------
# Simple heuristic language detection (works offline, no extra dependency)
# ---------------------------------------------------------------------------
# Unicode ranges for Indian script characters
_SCRIPT_RANGES: dict[str, tuple[int, int]] = {
    "hi": (0x0900, 0x097F),  # Devanagari → Hindi / Marathi
    "bn": (0x0980, 0x09FF),  # Bengali
    "gu": (0x0A80, 0x0AFF),  # Gujarati
    "pa": (0x0A00, 0x0A7F),  # Gurmukhi → Punjabi
    "or": (0x0B00, 0x0B7F),  # Odia
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "te": (0x0C00, 0x0C7F),  # Telugu
    "kn": (0x0C80, 0x0CFF),  # Kannada
    "ml": (0x0D00, 0x0D7F),  # Malayalam
    "ur": (0x0600, 0x06FF),  # Arabic / Urdu
}

# Marathi uses the same Devanagari script as Hindi, so we default "hi" range
# to Hindi unless the user explicitly selects Marathi in the UI.


def _detect_language_by_script(text: str) -> Optional[str]:
    """Return an ISO 639-1 code based on the dominant Unicode script block."""
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for lang, (lo, hi) in _SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1

    if not counts:
        return None  # Assume ASCII / Latin → English

    return max(counts, key=lambda k: counts[k])


@dataclass
class LanguageContext:
    """Carries the detected / selected language through a request."""

    user_lang: str        # ISO 639-1 code of user's language
    query_en: str         # User query translated to English
    original_query: str   # User query in original language
    display_name: str     # Human-readable language name


class MultilingualPipeline:
    """
    Thin orchestration layer that:
    1. Detects or accepts the user's language
    2. Translates the query to English (for RAG + LLM)
    3. Translates the answer back to the user's language
    """

    def __init__(self, translator: Optional[IBMTranslator] = None) -> None:
        self._translator = translator or get_translator()

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_and_translate_query(
        self,
        query: str,
        selected_lang: Optional[str] = None,
    ) -> LanguageContext:
        """
        Detect language from script heuristics, override with *selected_lang*
        if provided.  Translate query to English for downstream processing.

        Args:
            query:         Raw user query (any language).
            selected_lang: Explicit language code from UI selector.

        Returns:
            LanguageContext with translated query and metadata.
        """
        # Determine language
        if selected_lang:
            lang = selected_lang
        else:
            lang = _detect_language_by_script(query) or "en"

        logger.debug("Language detected/selected: {}", lang)

        # Translate to English
        query_en = self._translator.to_english(query, source_lang=lang)

        return LanguageContext(
            user_lang=lang,
            query_en=query_en,
            original_query=query,
            display_name=LANGUAGE_NAMES.get(lang, lang.upper()),
        )

    def translate_answer(self, answer: str, lang_ctx: LanguageContext) -> str:
        """
        Translate the English LLM answer back to the user's language.

        Args:
            answer:   English answer from the LLM.
            lang_ctx: LanguageContext from detect_and_translate_query().

        Returns:
            Answer in user's language.
        """
        return self._translator.from_english(answer, target_lang=lang_ctx.user_lang)

    @staticmethod
    def supported_languages() -> dict[str, str]:
        """Returns a {code: display_name} dict for the UI language picker."""
        return {k: v for k, v in LANGUAGE_NAMES.items()}


# Module-level singleton
_pipeline_instance: Optional[MultilingualPipeline] = None


def get_multilingual_pipeline() -> MultilingualPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = MultilingualPipeline()
    return _pipeline_instance
