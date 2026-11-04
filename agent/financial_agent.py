"""
agent/financial_agent.py
Core RAG + IBM Granite agent for Digital Financial Literacy.

Pipeline per query:
  1. Detect / translate input language  (MultilingualPipeline)
  2. Retrieve relevant documents        (FinancialKnowledgeBase)
  3. Build a grounded prompt            (build_prompt)
  4. Generate answer via IBM Granite    (WatsonXClient)
  5. Translate answer to user language  (MultilingualPipeline)
  6. Return FinancialResponse dataclass
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.documents import Document
from loguru import logger

from core.config import get_settings
from core.multilingual import LanguageContext, get_multilingual_pipeline
from ibm.watsonx_client import get_watsonx_client
from rag.knowledge_base import get_knowledge_base

# ---------------------------------------------------------------------------
# System prompt – instructions for IBM Granite
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are Pennywise, a friendly and trustworthy AI agent specialising in
    digital financial literacy for Indian users.

    Your role:
    - Help users understand UPI, online banking, avoiding financial scams,
      interest rates, budgeting, savings, insurance, and government schemes.
    - Always cite sources when referencing data (e.g., "As per RBI guidelines…").
    - Warn users about common scams clearly and firmly.
    - Keep answers concise, clear, and jargon-free.
    - If you don't know the answer, say so honestly and suggest the user
      consult RBI (rbi.org.in), SEBI (sebi.gov.in), or their bank directly.
    - Never provide personalised investment advice or specific stock picks.
    - Always prioritise user safety and financial well-being.

    Respond ONLY based on the context provided below. Do not make up figures
    or regulatory details not present in the context.
    """
).strip()

_RAG_PROMPT_TEMPLATE = textwrap.dedent(
    """
    {system_prompt}

    --- RELEVANT KNOWLEDGE BASE EXCERPTS ---
    {context}
    --- END OF EXCERPTS ---

    User question: {question}

    Answer (in English, clear and helpful):
    """
).strip()

_FALLBACK_PROMPT_TEMPLATE = textwrap.dedent(
    """
    {system_prompt}

    User question: {question}

    The knowledge base had no relevant excerpts for this specific question.
    Please provide a general, safe, and accurate answer based on widely
    accepted financial literacy principles. If uncertain, recommend the user
    consult RBI (rbi.org.in) or their bank.

    Answer (in English, clear and helpful):
    """
).strip()


@dataclass
class FinancialResponse:
    """Structured response returned by the agent."""

    answer: str                          # Answer in user's language
    answer_en: str                       # Answer in English (for debugging)
    query: str                           # Original user query
    language: str                        # ISO 639-1 code
    language_name: str                   # Human-readable language name
    sources: List[dict] = field(default_factory=list)
    retrieved_docs: int = 0
    model_used: str = ""
    error: Optional[str] = None


class FinancialLiteracyAgent:
    """
    End-to-end RAG agent for digital financial literacy.

    Usage:
        agent = FinancialLiteracyAgent()
        response = agent.chat("UPI se paise kaise bhejein?", lang="hi")
        print(response.answer)
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._kb = get_knowledge_base()
        self._watsonx = get_watsonx_client()
        self._pipeline = get_multilingual_pipeline()

    # ── Main entry point ──────────────────────────────────────────────────────

    def chat(
        self,
        query: str,
        lang: Optional[str] = None,
    ) -> FinancialResponse:
        """
        Process a user query and return a grounded financial literacy answer.

        Args:
            query: User's question (any supported language).
            lang:  Optional explicit language code ('hi', 'ta', 'en', …).

        Returns:
            FinancialResponse with the answer and metadata.
        """
        try:
            # Step 1 – Language detection / translation
            lang_ctx: LanguageContext = (
                self._pipeline.detect_and_translate_query(query, selected_lang=lang)
            )
            logger.info(
                "Query received | lang={} | query_en='{}'",
                lang_ctx.user_lang,
                lang_ctx.query_en[:60],
            )

            # Step 2 – Retrieve relevant documents
            docs: List[Document] = self._kb.similarity_search(
                lang_ctx.query_en,
                k=self._settings.top_k_results,
            )
            logger.debug("Retrieved {} documents from knowledge base.", len(docs))

            # Step 3 – Build prompt
            prompt = self._build_prompt(lang_ctx.query_en, docs)

            # Step 4 – Generate answer
            answer_en = self._watsonx.generate(prompt)

            # Step 5 – Translate answer back
            answer = self._pipeline.translate_answer(answer_en, lang_ctx)

            # Step 6 – Build response
            sources = self._extract_sources(docs)
            return FinancialResponse(
                answer=answer,
                answer_en=answer_en,
                query=query,
                language=lang_ctx.user_lang,
                language_name=lang_ctx.display_name,
                sources=sources,
                retrieved_docs=len(docs),
                model_used=self._settings.granite_model_id,
            )

        except Exception as exc:
            logger.error("Agent error: {}", exc, exc_info=True)
            return FinancialResponse(
                answer=(
                    "I'm sorry, I encountered an error processing your request. "
                    "Please try again or consult rbi.org.in for financial guidance."
                ),
                answer_en="",
                query=query,
                language=lang or "en",
                language_name="Unknown",
                error=str(exc),
            )

    # ── Prompt builders ───────────────────────────────────────────────────────

    def _build_prompt(self, question_en: str, docs: List[Document]) -> str:
        if not docs:
            return _FALLBACK_PROMPT_TEMPLATE.format(
                system_prompt=_SYSTEM_PROMPT,
                question=question_en,
            )

        context_parts = []
        for i, doc in enumerate(docs, 1):
            topic = doc.metadata.get("topic", "General")
            source = doc.metadata.get("source", "")
            context_parts.append(
                f"[{i}] Topic: {topic}\n"
                f"    Source: {source}\n"
                f"    {doc.page_content}"
            )
        context = "\n\n".join(context_parts)

        return _RAG_PROMPT_TEMPLATE.format(
            system_prompt=_SYSTEM_PROMPT,
            context=context,
            question=question_en,
        )

    @staticmethod
    def _extract_sources(docs: List[Document]) -> List[dict]:
        seen = set()
        sources = []
        for doc in docs:
            key = (doc.metadata.get("topic", ""), doc.metadata.get("source", ""))
            if key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "topic": doc.metadata.get("topic", ""),
                        "source": doc.metadata.get("source", ""),
                    }
                )
        return sources


# Module-level singleton
_agent_instance: Optional[FinancialLiteracyAgent] = None


def get_agent() -> FinancialLiteracyAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = FinancialLiteracyAgent()
    return _agent_instance
