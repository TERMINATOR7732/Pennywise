"""
ibm/watsonx_client.py
Wrapper around ibm-watsonx-ai for text generation and embeddings
using IBM Granite models on IBM Cloud Lite (free tier).

Uses the /ml/v1/text/chat (chat completions) API which is required
for granite-4-h-small and other modern Granite models.

IBM Cloud Lite rate limit: 2 requests/second.
Retry logic: on 429, wait 5-20s with exponential back-off.
"""
from __future__ import annotations

import time
from typing import List, Optional

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import get_settings

try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.wml_client_error import ApiRequestFailure
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False
    ApiRequestFailure = Exception
    logger.warning("ibm-watsonx-ai not installed. Run: pip install ibm-watsonx-ai")


def _is_rate_limit(exc: BaseException) -> bool:
    """Return True if the exception is a 429 rate-limit response."""
    msg = str(exc)
    return "429" in msg or "rate_limit" in msg


def _is_retryable(exc: BaseException) -> bool:
    """Retry on rate limits and transient server errors (5xx), not on 4xx auth errors."""
    msg = str(exc)
    if "429" in msg or "rate_limit" in msg:
        return True
    if "500" in msg or "502" in msg or "503" in msg or "504" in msg:
        return True
    return False


class WatsonXClient:
    """
    Wraps IBM WatsonX.ai for:
    - Text generation via chat API  (IBM Granite 4 models)
    - Text embeddings               (IBM Granite embedding models)
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._model: Optional[object] = None
        self._embed_model: Optional[object] = None
        self._initialised = False
        self._last_call_ts: float = 0.0   # tracks last API call for rate-limit spacing

    def _ensure_init(self) -> None:
        if self._initialised:
            return
        if not _SDK_AVAILABLE:
            raise RuntimeError("ibm-watsonx-ai SDK not installed. Run: pip install ibm-watsonx-ai")

        api_key = self._settings.ibm_cloud_api_key
        if not api_key or api_key == "your_ibm_cloud_api_key_here":
            raise ValueError(
                "IBM_CLOUD_API_KEY is missing. Update your .env file."
            )

        credentials = Credentials(
            url=self._settings.watsonx_url,
            api_key=api_key,
        )

        self._model = ModelInference(
            model_id=self._settings.granite_model_id,
            credentials=credentials,
            project_id=self._settings.watsonx_project_id,
        )

        self._embed_model = ModelInference(
            model_id=self._settings.granite_embedding_model,
            credentials=credentials,
            project_id=self._settings.watsonx_project_id,
        )

        self._initialised = True
        logger.info(
            "WatsonXClient initialised | model={} | embed={}",
            self._settings.granite_model_id,
            self._settings.granite_embedding_model,
        )

    # ── Text Generation (chat API) ────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=2, min=5, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def generate(self, prompt: str) -> str:
        """
        Generate a response from IBM Granite using the chat completions API.

        Args:
            prompt: Full formatted prompt string (system + context + question).

        Returns:
            Generated text as a string.
        """
        self._ensure_init()

        # Respect IBM Lite rate limit: min 0.6s between calls
        now = time.perf_counter()
        since_last = now - self._last_call_ts
        if since_last < 0.6:
            time.sleep(0.6 - since_last)

        start = time.perf_counter()
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self._model.chat(
                messages=messages,
                params={
                    "max_tokens": self._settings.max_new_tokens,
                    "temperature": self._settings.temperature,
                    "top_p": self._settings.top_p,
                },
            )
            self._last_call_ts = time.perf_counter()
            text = response["choices"][0]["message"]["content"]
            elapsed = time.perf_counter() - start
            logger.debug("WatsonX chat | {:.2f}s | chars={}", elapsed, len(text))
            return text.strip()
        except Exception as exc:
            if _is_rate_limit(exc):
                logger.warning("Rate limit hit — waiting 10s before retry…")
                time.sleep(10)
            raise

    # ── Embeddings ────────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using IBM Granite embedding model."""
        self._ensure_init()
        result = self._embed_model.embed_documents(texts=texts)
        return result

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self.embed([text])[0]

    # ── Health check ──────────────────────────────────────────────────────────

    def health_check(self) -> dict:
        try:
            self._ensure_init()
            test = self.generate("Say OK in one word.")
            return {
                "status": "ok",
                "model": self._settings.granite_model_id,
                "response": test[:80],
            }
        except Exception as exc:
            logger.error("WatsonX health check failed: {}", exc)
            return {"status": "error", "error": str(exc)}


_client_instance: Optional[WatsonXClient] = None


def get_watsonx_client() -> WatsonXClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = WatsonXClient()
    return _client_instance
