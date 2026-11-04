"""
tests/test_agent.py
Unit tests for Pennywise – Financial Literacy Agent.
Tests run without real IBM credentials by mocking the WatsonX client.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.config import get_settings
from core.multilingual import MultilingualPipeline, _detect_language_by_script
from rag.knowledge_base import FinancialKnowledgeBase, SEED_KNOWLEDGE


# ── Language detection tests ──────────────────────────────────────────────────

class TestLanguageDetection:
    def test_devanagari_detected_as_hindi(self):
        assert _detect_language_by_script("यूपीआई से पैसे कैसे भेजें?") == "hi"

    def test_tamil_script_detected(self):
        assert _detect_language_by_script("இது எப்படி வேலை செய்கிறது?") == "ta"

    def test_bengali_script_detected(self):
        assert _detect_language_by_script("আমি কীভাবে টাকা পাঠাব?") == "bn"

    def test_english_returns_none(self):
        assert _detect_language_by_script("How do I send money via UPI?") is None

    def test_empty_string_returns_none(self):
        assert _detect_language_by_script("") is None


# ── Knowledge base tests ──────────────────────────────────────────────────────

class TestKnowledgeBase:
    def test_seed_knowledge_not_empty(self):
        assert len(SEED_KNOWLEDGE) >= 10

    def test_seed_knowledge_has_required_fields(self):
        for item in SEED_KNOWLEDGE:
            assert "topic" in item
            assert "content" in item
            assert "source" in item
            assert len(item["content"]) > 50

    def test_upi_topic_exists(self):
        topics = {item["topic"] for item in SEED_KNOWLEDGE}
        assert "UPI" in topics

    def test_scam_topic_exists(self):
        topics = {item["topic"] for item in SEED_KNOWLEDGE}
        assert any("scam" in t.lower() for t in topics)


# ── Multilingual pipeline tests ───────────────────────────────────────────────

class TestMultilingualPipeline:
    def test_english_passthrough(self):
        pipeline = MultilingualPipeline()
        ctx = pipeline.detect_and_translate_query(
            "How do I send money via UPI?", selected_lang="en"
        )
        assert ctx.user_lang == "en"
        assert ctx.query_en == "How do I send money via UPI?"

    def test_language_override(self):
        pipeline = MultilingualPipeline()
        ctx = pipeline.detect_and_translate_query("some text", selected_lang="hi")
        assert ctx.user_lang == "hi"

    def test_detect_hindi_from_script(self):
        pipeline = MultilingualPipeline()
        ctx = pipeline.detect_and_translate_query("यूपीआई क्या है?")
        assert ctx.user_lang == "hi"

    def test_supported_languages_not_empty(self):
        langs = MultilingualPipeline.supported_languages()
        assert len(langs) >= 10
        assert "en" in langs
        assert "hi" in langs


# ── Config tests ──────────────────────────────────────────────────────────────

class TestConfig:
    def test_default_model_id_is_granite(self):
        settings = get_settings()
        assert "granite" in settings.granite_model_id.lower()

    def test_supported_language_list(self):
        settings = get_settings()
        langs = settings.supported_language_list
        assert "en" in langs
        assert len(langs) >= 3

    def test_generation_params_keys(self):
        settings = get_settings()
        params = settings.generation_params
        assert "max_new_tokens" in params
        assert "temperature" in params


# ── Agent integration test (mocked) ──────────────────────────────────────────

class TestFinancialAgentMocked:
    @patch("agent.financial_agent.get_watsonx_client")
    @patch("agent.financial_agent.get_knowledge_base")
    def test_chat_returns_response(self, mock_kb, mock_wx):
        from agent.financial_agent import FinancialLiteracyAgent
        from langchain_core.documents import Document

        # Mock WatsonX generate
        mock_wx_instance = MagicMock()
        mock_wx_instance.generate.return_value = (
            "To send money via UPI, open your UPI app, enter the recipient's "
            "UPI ID, enter the amount, and confirm with your PIN."
        )
        mock_wx.return_value = mock_wx_instance

        # Mock knowledge base
        mock_kb_instance = MagicMock()
        mock_kb_instance.similarity_search.return_value = [
            Document(
                page_content="UPI allows instant payments using UPI ID.",
                metadata={"topic": "UPI", "source": "NPCI"},
            )
        ]
        mock_kb.return_value = mock_kb_instance

        agent = FinancialLiteracyAgent()
        response = agent.chat("How do I send money via UPI?", lang="en")

        assert response.error is None
        assert len(response.answer) > 10
        assert response.language == "en"
        assert response.retrieved_docs == 1

    @patch("agent.financial_agent.get_watsonx_client")
    @patch("agent.financial_agent.get_knowledge_base")
    def test_chat_returns_error_response_on_exception(self, mock_kb, mock_wx):
        from agent.financial_agent import FinancialLiteracyAgent

        mock_wx_instance = MagicMock()
        mock_wx_instance.generate.side_effect = RuntimeError("API error")
        mock_wx.return_value = mock_wx_instance

        mock_kb_instance = MagicMock()
        mock_kb_instance.similarity_search.return_value = []
        mock_kb.return_value = mock_kb_instance

        agent = FinancialLiteracyAgent()
        response = agent.chat("test question", lang="en")

        assert response.error is not None
        assert "error" in response.answer.lower() or "sorry" in response.answer.lower()
