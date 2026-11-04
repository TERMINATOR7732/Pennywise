"""
ibm/translator.py
Multilingual translation wrapper using IBM Watson Language Translator
(Lite plan – free on IBM Cloud).

Falls back to a passthrough (no translation) when the service is not
configured, so the rest of the app works without a Translator key.
"""
from __future__ import annotations

from typing import Optional

from loguru import logger

from core.config import get_settings

try:
    from ibm_watson import LanguageTranslatorV3
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    _TRANSLATOR_SDK = True
except ImportError:  # pragma: no cover
    _TRANSLATOR_SDK = False
    logger.warning(
        "ibm-watson SDK not installed. "
        "Translation disabled. Install with: pip install ibm-watson"
    )


# Full display names for the language picker UI
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "हिन्दी (Hindi)",
    "mr": "मराठी (Marathi)",
    "ta": "தமிழ் (Tamil)",
    "te": "తెలుగు (Telugu)",
    "bn": "বাংলা (Bengali)",
    "gu": "ગુજરાતી (Gujarati)",
    "kn": "ಕನ್ನಡ (Kannada)",
    "pa": "ਪੰਜਾਬੀ (Punjabi)",
    "ur": "اردو (Urdu)",
    "ml": "മലയാളം (Malayalam)",
    "or": "ଓଡ଼ିଆ (Odia)",
}


class IBMTranslator:
    """
    Translates text to/from English using IBM Watson Language Translator.

    When credentials are absent, all translate_* methods return the input
    unchanged so the app degrades gracefully.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._translator: Optional[object] = None
        self._enabled = False
        self._init()

    def _init(self) -> None:
        if not _TRANSLATOR_SDK:
            return
        if not self._settings.is_translator_enabled:
            logger.info(
                "IBM Watson Language Translator not configured – "
                "operating in English-only mode."
            )
            return

        try:
            authenticator = IAMAuthenticator(
                self._settings.language_translator_apikey
            )
            self._translator = LanguageTranslatorV3(
                version="2018-05-01",
                authenticator=authenticator,
            )
            self._translator.set_service_url(
                self._settings.language_translator_url
            )
            self._enabled = True
            logger.info("IBM Watson Language Translator initialised.")
        except Exception as exc:
            logger.error("Failed to initialise Language Translator: {}", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def to_english(self, text: str, source_lang: str) -> str:
        """Translate *text* from *source_lang* to English."""
        if not self._enabled or source_lang == "en":
            return text
        return self._translate(text, source=source_lang, target="en")

    def from_english(self, text: str, target_lang: str) -> str:
        """Translate *text* from English to *target_lang*."""
        if not self._enabled or target_lang == "en":
            return text
        return self._translate(text, source="en", target=target_lang)

    def _translate(self, text: str, source: str, target: str) -> str:
        try:
            result = (
                self._translator
                .translate(text=text, source=source, target=target)
                .get_result()
            )
            return result["translations"][0]["translation"]
        except Exception as exc:
            logger.error(
                "Translation failed ({} -> {}): {}", source, target, exc
            )
            return text  # graceful fallback

    @staticmethod
    def language_display_name(code: str) -> str:
        return LANGUAGE_NAMES.get(code, code.upper())


# Module-level singleton
_translator_instance: Optional[IBMTranslator] = None


def get_translator() -> IBMTranslator:
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = IBMTranslator()
    return _translator_instance
