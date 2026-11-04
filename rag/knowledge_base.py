"""
rag/knowledge_base.py
RAG knowledge base for Digital Financial Literacy.

Responsibilities:
  1. Seed the in-memory / FAISS vector store with curated financial literacy
     content (UPI, scams, interest rates, budgeting, insurance, etc.)
  2. Accept new documents (PDF, DOCX, plain text, web pages).
  3. Expose a similarity_search() method used by the agent.

Uses LangChain's FAISS integration.  When IBM WatsonX embeddings are
available they are used; otherwise falls back to a lightweight
sentence-transformers model so the app still works locally.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from core.config import get_settings

# ---------------------------------------------------------------------------
# Try IBM WatsonX embeddings first, fall back to sentence-transformers
# ---------------------------------------------------------------------------
try:
    from langchain_ibm import WatsonxEmbeddings
    _IBM_EMBED = True
except ImportError:
    _IBM_EMBED = False

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    _HF_EMBED = True
except ImportError:
    _HF_EMBED = False

try:
    from langchain_community.vectorstores import FAISS
    _FAISS = True
except ImportError:
    _FAISS = False
    logger.warning("faiss-cpu not installed. RAG vector store unavailable.")

# ---------------------------------------------------------------------------
# Curated seed knowledge for financial literacy
# ---------------------------------------------------------------------------
SEED_KNOWLEDGE: List[dict] = [
    # ── UPI ──────────────────────────────────────────────────────────────────
    {
        "topic": "UPI",
        "content": (
            "Unified Payments Interface (UPI) is a real-time payment system "
            "developed by the National Payments Corporation of India (NPCI). "
            "To send money via UPI: open your UPI app (PhonePe, Google Pay, "
            "Paytm, BHIM), enter the recipient's UPI ID or scan their QR code, "
            "enter the amount, and authenticate with your UPI PIN. "
            "Payments are instant and free for users. "
            "Never share your UPI PIN or OTP with anyone – not even bank staff."
        ),
        "source": "NPCI / RBI Guidelines",
    },
    {
        "topic": "UPI Safety",
        "content": (
            "UPI safety tips: (1) Never accept collect requests from unknown "
            "numbers – scammers send fake collect requests. (2) QR codes are "
            "only for receiving money; scanning a QR code to 'receive' payment "
            "is a scam. (3) Keep your UPI PIN confidential. (4) Enable "
            "transaction alerts. (5) Verify the beneficiary's name before "
            "confirming payment. (6) Use official bank apps only."
        ),
        "source": "RBI Kehta Hai Campaign",
    },
    # ── Online Scams ─────────────────────────────────────────────────────────
    {
        "topic": "Online Scams",
        "content": (
            "Common online financial scams in India: "
            "(1) KYC update scam – fake calls asking for Aadhaar/PAN to update "
            "KYC; banks never ask for this over phone/SMS. "
            "(2) Lottery / prize scams – 'you won Rs 10 lakh' messages asking "
            "for a processing fee. "
            "(3) Job offer scams – asking for registration fees. "
            "(4) Screen-sharing scams – asking you to install AnyDesk/TeamViewer. "
            "(5) SIM swap fraud – cloning your SIM to intercept OTPs. "
            "If defrauded, report immediately to cybercrime.gov.in or call 1930."
        ),
        "source": "Cybercrime.gov.in / RBI",
    },
    {
        "topic": "Phishing",
        "content": (
            "Phishing is a cyber-attack where criminals impersonate banks or "
            "government bodies via fake emails, SMS, or websites to steal "
            "credentials. "
            "Red flags: urgent language ('act now'), spelling errors, "
            "mismatched URLs (e.g., 'sbi-alert.com' instead of 'sbi.co.in'). "
            "Always type your bank's URL directly; never click links in SMS/email. "
            "Official bank domains end in .co.in or .com – verify with "
            "https://rbi.org.in/Scripts/bs_viewcontent.aspx?Id=3736"
        ),
        "source": "RBI / CERT-In",
    },
    # ── Interest Rates ────────────────────────────────────────────────────────
    {
        "topic": "Safe Interest Rate for a Loan",
        "content": (
            "RBI-regulated interest rate benchmarks (2024): "
            "Home loans: 8.40%–9.50% p.a. (floating, linked to MCLR/RLLR). "
            "Personal loans: 10.5%–24% p.a. (based on credit score). "
            "Gold loans: 7%–12% p.a. "
            "Microfinance loans (MFI): capped at MCLR + 10% for small loans. "
            "Warning: any lender quoting above 36% p.a. for a personal loan, "
            "or asking for upfront 'processing fee' before disbursement, is "
            "likely predatory or illegal. Always check the RBI-regulated lender "
            "list at https://rbi.org.in."
        ),
        "source": "RBI / NHB Guidelines",
    },
    {
        "topic": "Fixed Deposit Rates",
        "content": (
            "Fixed Deposit (FD) interest rates in India (2024 indicative): "
            "Public-sector banks: 6.5%–7.25% for 1–3 years. "
            "Small Finance Banks: up to 9% p.a. (higher risk). "
            "Interest is taxable as per your income slab (TDS at 10% if PAN "
            "provided, 20% otherwise). "
            "DICGC insures deposits up to ₹5 lakh per depositor per bank. "
            "For senior citizens: additional 0.25%–0.50% p.a. is common."
        ),
        "source": "RBI / DICGC",
    },
    # ── Budgeting ─────────────────────────────────────────────────────────────
    {
        "topic": "Budgeting Basics",
        "content": (
            "The 50-30-20 budgeting rule: allocate 50% of take-home income to "
            "needs (rent, groceries, utilities), 30% to wants (dining, "
            "entertainment), and 20% to savings and debt repayment. "
            "Steps to create a budget: (1) Track all income sources. "
            "(2) List fixed expenses. (3) Estimate variable expenses. "
            "(4) Identify discretionary spending. "
            "(5) Set a savings goal (minimum 3–6 months emergency fund). "
            "Free budgeting apps available in India: ET Money, Walnut, Money "
            "View."
        ),
        "source": "Financial planning best practices",
    },
    {
        "topic": "Emergency Fund",
        "content": (
            "An emergency fund is 3–6 months of living expenses kept in a "
            "liquid instrument like a savings account or liquid mutual fund. "
            "Purpose: covers job loss, medical emergencies, or unexpected "
            "repairs without taking on debt. "
            "Where to park: high-interest savings accounts (>= 4% p.a.) or "
            "liquid/overnight mutual funds (better post-tax returns than "
            "savings accounts for most investors). "
            "Do NOT invest emergency funds in equities – they are not liquid "
            "enough during a crisis."
        ),
        "source": "AMFI / SEBI Investor Education",
    },
    # ── Credit Score ──────────────────────────────────────────────────────────
    {
        "topic": "Credit Score",
        "content": (
            "A CIBIL score (or credit score) ranges from 300 to 900. "
            "750+ is considered excellent and qualifies for the best loan rates. "
            "Key factors: payment history (35%), credit utilisation (30%), "
            "length of credit history (15%), credit mix (10%), new enquiries (10%). "
            "To improve your score: pay EMIs on time, keep credit card "
            "utilisation below 30%, avoid multiple loan applications at once, "
            "and check your free annual credit report at https://www.cibil.com."
        ),
        "source": "CIBIL / RBI",
    },
    # ── Mutual Funds ─────────────────────────────────────────────────────────
    {
        "topic": "Mutual Funds SIP",
        "content": (
            "A Systematic Investment Plan (SIP) allows you to invest a fixed "
            "amount in a mutual fund at regular intervals (monthly, weekly). "
            "Benefits: rupee cost averaging, compounding, low minimum amount "
            "(as low as ₹100/month). "
            "SEBI regulates mutual funds in India. "
            "All mutual funds carry market risk – equity funds are for a "
            "5+ year horizon. "
            "Direct plans (without distributor commission) give ~0.5%–1% "
            "higher returns than regular plans. "
            "Invest through AMFI-registered platforms (ET Money, Zerodha Coin, "
            "MF Central) or directly through the AMC website."
        ),
        "source": "SEBI / AMFI",
    },
    # ── Insurance ────────────────────────────────────────────────────────────
    {
        "topic": "Health Insurance",
        "content": (
            "Health insurance covers hospitalisation, surgeries, and sometimes "
            "pre/post-hospitalisation expenses. "
            "Minimum recommended cover: ₹5 lakh for individuals, ₹10–15 lakh "
            "for families. "
            "Ayushman Bharat (PM-JAY) provides ₹5 lakh/year free cover to "
            "eligible Below-Poverty-Line families. "
            "Key terms: premium (annual cost), deductible (amount you pay "
            "before insurance kicks in), co-pay, sum insured, waiting period "
            "(usually 2–4 years for pre-existing diseases). "
            "IRDAI regulates all insurers – verify insurer registration at "
            "https://www.irdai.gov.in."
        ),
        "source": "IRDAI / NHA",
    },
    {
        "topic": "Term Life Insurance",
        "content": (
            "Term insurance pays a death benefit to nominees if the insured "
            "passes away during the policy term. No maturity benefit. "
            "Recommended cover: 10–15× your annual income. "
            "Pure term plans are the cheapest form of life insurance "
            "(e.g., ₹1 crore cover for ~₹700–900/month at age 30). "
            "Avoid mixing insurance and investment (endowment, ULIP). "
            "Buy directly from insurer or IRDAI-approved aggregators to save "
            "on commissions. "
            "Always disclose medical history accurately; non-disclosure leads "
            "to claim rejection."
        ),
        "source": "IRDAI",
    },
    # ── Government Schemes ────────────────────────────────────────────────────
    {
        "topic": "Jan Dhan Yojana",
        "content": (
            "Pradhan Mantri Jan Dhan Yojana (PMJDY) provides zero-balance bank "
            "accounts to unbanked citizens. "
            "Benefits: free RuPay debit card with ₹2 lakh accidental insurance, "
            "₹10,000 overdraft facility after 6 months of satisfactory operation, "
            "access to government DBT (Direct Benefit Transfer) subsidies. "
            "To open: visit any bank branch or Business Correspondent (BC) "
            "with Aadhaar and one more identity proof."
        ),
        "source": "PMJDY / Ministry of Finance",
    },
    {
        "topic": "PPF – Public Provident Fund",
        "content": (
            "PPF is a government-backed long-term savings scheme. "
            "Interest rate: 7.1% p.a. (compounded annually, set by GoI quarterly). "
            "Lock-in: 15 years (partial withdrawal from 7th year). "
            "Tax benefit: EEE status – contribution deductible under Sec 80C "
            "(up to ₹1.5 lakh/year), interest and maturity are tax-free. "
            "Annual investment: minimum ₹500, maximum ₹1.5 lakh. "
            "Open at any post office, SBI, or authorised bank. "
            "Safe: sovereign guarantee, no market risk."
        ),
        "source": "Ministry of Finance / India Post",
    },
]


class FinancialKnowledgeBase:
    """
    In-memory FAISS vector store pre-loaded with financial literacy content.
    Supports adding custom documents (PDF, DOCX, text, URL).
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._vectorstore = None
        self._embeddings = None
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── Embedding backend ─────────────────────────────────────────────────────

    def _get_embeddings(self):
        if self._embeddings is not None:
            return self._embeddings

        settings = self._settings

        if _IBM_EMBED and settings.ibm_cloud_api_key not in (
            "", "your_ibm_cloud_api_key_here"
        ):
            try:
                self._embeddings = WatsonxEmbeddings(
                    model_id=settings.granite_embedding_model,
                    url=settings.watsonx_url,
                    apikey=settings.ibm_cloud_api_key,
                    project_id=settings.watsonx_project_id,
                )
                logger.info("Using IBM WatsonX embeddings (model={}).", settings.granite_embedding_model)
                return self._embeddings
            except Exception as exc:
                logger.warning("IBM embeddings failed ({}); falling back to local.", exc)

        if _HF_EMBED:
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )
            logger.info("Using local HuggingFace sentence-transformers embeddings.")
            return self._embeddings

        raise RuntimeError(
            "No embedding backend available. "
            "Install ibm-watsonx-ai or sentence-transformers."
        )

    # ── Build / Load vector store ─────────────────────────────────────────────

    def build(self) -> None:
        """Build the vector store from seed knowledge."""
        if not _FAISS:
            raise RuntimeError("faiss-cpu is not installed.")

        docs: List[Document] = []
        for item in SEED_KNOWLEDGE:
            chunks = self._splitter.split_text(item["content"])
            for chunk in chunks:
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "topic": item["topic"],
                            "source": item["source"],
                        },
                    )
                )

        embeddings = self._get_embeddings()
        self._vectorstore = FAISS.from_documents(docs, embeddings)
        logger.info(
            "Knowledge base built: {} chunks from {} seed documents.",
            len(docs),
            len(SEED_KNOWLEDGE),
        )

    def save(self, path: Optional[str] = None) -> None:
        """Persist vector store to disk."""
        if self._vectorstore is None:
            raise RuntimeError("Build the knowledge base first.")
        save_path = path or self._settings.vector_store_path
        Path(save_path).mkdir(parents=True, exist_ok=True)
        self._vectorstore.save_local(save_path)
        logger.info("Vector store saved to {}", save_path)

    def load(self, path: Optional[str] = None) -> bool:
        """Load a previously saved vector store. Returns True on success."""
        load_path = path or self._settings.vector_store_path
        if not Path(load_path).exists():
            return False
        try:
            embeddings = self._get_embeddings()
            self._vectorstore = FAISS.load_local(
                load_path,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("Vector store loaded from {}", load_path)
            return True
        except Exception as exc:
            logger.error("Failed to load vector store: {}", exc)
            return False

    def ensure_ready(self) -> None:
        """Load from disk if available; otherwise build from seed."""
        if self._vectorstore is not None:
            return
        if not self.load():
            self.build()
            self.save()

    # ── Search ────────────────────────────────────────────────────────────────

    def similarity_search(
        self, query: str, k: Optional[int] = None
    ) -> List[Document]:
        """
        Retrieve the top-k most relevant documents for *query*.

        Args:
            query: User question (in English after translation).
            k:     Number of results; defaults to settings.top_k_results.

        Returns:
            List of LangChain Document objects.
        """
        self.ensure_ready()
        if self._vectorstore is None:
            return []
        k = k or self._settings.top_k_results
        return self._vectorstore.similarity_search(query, k=k)

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> None:
        """Add new text chunks to the live vector store."""
        self.ensure_ready()
        embeddings = self._get_embeddings()
        if self._vectorstore is None:
            self._vectorstore = FAISS.from_texts(
                texts, embeddings, metadatas=metadatas
            )
        else:
            self._vectorstore.add_texts(texts, metadatas=metadatas)


# Module-level singleton
_kb_instance: Optional[FinancialKnowledgeBase] = None


def get_knowledge_base() -> FinancialKnowledgeBase:
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = FinancialKnowledgeBase()
    return _kb_instance
