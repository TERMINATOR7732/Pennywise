"""
app.py
Streamlit front-end for Pennywise – AI Agent for Digital Financial Literacy.
Powered by IBM Granite (WatsonX.ai) + FAISS RAG.
"""
from __future__ import annotations

# Load .env FIRST — before any IBM/agent imports read os.environ
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# On Streamlit Cloud, secrets live in st.secrets — push them into os.environ
# so that pydantic-settings (which reads os.environ) picks them up.
import os
try:
    import streamlit as _st
    for _key in ("IBM_CLOUD_API_KEY", "WATSONX_PROJECT_ID",
                 "LANGUAGE_TRANSLATOR_APIKEY", "LANGUAGE_TRANSLATOR_URL"):
        if _key in _st.secrets and not os.environ.get(_key):
            os.environ[_key] = _st.secrets[_key]
except Exception:
    pass

# Clear any stale cached settings from previous runs
try:
    from core.config import get_settings
    get_settings.cache_clear()
except Exception:
    pass

import time
from typing import Optional

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Pennywise – Digital Financial Literacy",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

    /* ── Main content area: always dark text on light bg ─────────────── */
    .main, .block-container, [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
        color: #1f2328 !important;
    }

    /* Ensure all Streamlit markdown/text inside main is dark */
    .main p, .main span, .main div, .main li, .main h1, .main h2, .main h3,
    .main h4, .main h5, .main h6, .main label, .main small {
        color: #1f2328 !important;
    }

    /* ── Sidebar ──────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] { background: #1a1a2e !important; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    [data-testid="stSidebar"] button { background: #2a2a4e !important; color: #e0e0e0 !important; border: 1px solid #3a3a5e !important; }
    [data-testid="stSidebar"] button:hover { background: #3b82f6 !important; color: #ffffff !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: #e0e0e0 !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #e0e0e0 !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 { color: #ffffff !important; }
    /* Selectbox dropdown text */
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: #1f2328 !important; }

    /* ── Header banner ────────────────────────────────────────────────── */
    .pw-header {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        padding: 1.4rem 2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        color: white !important;
    }
    .pw-header h1 { font-size: 2rem; margin: 0; color: white !important; }
    .pw-header p  { font-size: 0.92rem; margin: 0.3rem 0 0; opacity: 0.85; color: white !important; }

    /* ── Chat bubbles ─────────────────────────────────────────────────── */
    .user-bubble {
        background: #e8f0fe !important;
        border-left: 4px solid #3b82f6;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.6rem 0;
        color: #1a1a2e !important;
    }
    .user-bubble * { color: #1a1a2e !important; }

    .bot-bubble {
        background: #f0fdf4 !important;
        border-left: 4px solid #22c55e;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.6rem 0;
        color: #1a1a2e !important;
    }
    .bot-bubble * { color: #1a1a2e !important; }

    /* ── Source tags ──────────────────────────────────────────────────── */
    .source-tag {
        display: inline-block;
        background: #e0e7ff;
        color: #3730a3 !important;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin: 2px 3px;
    }

    /* ── Disclaimer box ───────────────────────────────────────────────── */
    .warn-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 0.84rem;
        margin-bottom: 0.8rem;
        color: #856404 !important;
    }
    .warn-box * { color: #856404 !important; }
    .warn-box a { color: #0f3460 !important; }

    /* ── Chat input ───────────────────────────────────────────────────── */
    [data-testid="stChatInput"] textarea {
        background-color: #ffffff !important;
        color: #1f2328 !important;
        border: 1px solid #d0d7de !important;
    }
    [data-testid="stChatInput"] textarea::placeholder { color: #6e7781 !important; }
    [data-testid="stChatInputContainer"] {
        background-color: #f7f8fa !important;
        border-top: 1px solid #e5e7eb;
    }
    /* chat_input submit button */
    [data-testid="stChatInput"] button { color: #1f2328 !important; }

    /* ── st.info / st.warning / st.success boxes ──────────────────────── */
    [data-testid="stAlert"] { color: #1f2328 !important; }
    [data-testid="stAlert"] p { color: #1f2328 !important; }
    [data-testid="stInfo"] { color: #1f2328 !important; }

    /* ── Expander ─────────────────────────────────────────────────────── */
    [data-testid="stExpander"] { color: #1f2328 !important; }
    [data-testid="stExpander"] summary { color: #1f2328 !important; }
    [data-testid="stExpander"] p { color: #1f2328 !important; }

    /* ── Divider & small text ─────────────────────────────────────────── */
    small { color: #57606a !important; }

    /* ── IBM badge ────────────────────────────────────────────────────── */
    .ibm-badge {
        font-size: 0.72rem;
        color: #9ca3af !important;
        text-align: center;
        margin-top: 1.5rem;
        padding-top: 0.6rem;
        border-top: 1px solid #374151;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_ready" not in st.session_state:
    st.session_state.agent_ready = False
if "agent_error" not in st.session_state:
    st.session_state.agent_error = None


# ── Lazy agent loader (only called when user sends first message) ─────────────
@st.cache_resource(show_spinner=False)
def _load_agent():
    """Load agent once and cache. Called on first query, not on page load."""
    try:
        from agent.financial_agent import get_agent
        ag = get_agent()
        return ag, None
    except Exception as exc:
        return None, str(exc)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💰 Pennywise")
    st.markdown("*AI Agent for Digital Financial Literacy*")
    st.divider()

    # Language selector
    from ibm.translator import LANGUAGE_NAMES
    lang_options = {v: k for k, v in LANGUAGE_NAMES.items()}
    selected_display = st.selectbox(
        "🌐 Language",
        options=list(lang_options.keys()),
        index=0,
    )
    selected_lang_code = lang_options[selected_display]

    st.divider()
    st.markdown("### 📚 Quick Topics")
    quick_questions = {
        "🏧 Send money via UPI":         "How do I send money via UPI?",
        "⚠️ Common UPI scams":           "What are common UPI scams I should avoid?",
        "📈 Safe loan interest rate":    "What is a safe interest rate for a personal loan?",
        "💰 Start a monthly budget":     "How do I create a monthly budget?",
        "📊 Improve CIBIL score":        "How can I improve my CIBIL credit score?",
        "🏛 PM Jan Dhan Yojana":         "What is Pradhan Mantri Jan Dhan Yojana?",
        "📑 What is SIP?":               "What is a Systematic Investment Plan (SIP)?",
        "🛡 Term insurance basics":      "How much term life insurance do I need?",
    }
    for label, q in quick_questions.items():
        if st.button(label, use_container_width=True):
            st.session_state["_quick"] = q
            st.rerun()

    st.divider()
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        '<div class="ibm-badge">Powered by <b>IBM Granite</b> on WatsonX.ai<br>IBM Cloud Lite · Free Tier</div>',
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="pw-header">
        <h1>💰 Pennywise</h1>
        <p>AI-powered Digital Financial Literacy &mdash;
        UPI &bull; Scam Prevention &bull; Budgeting &bull;
        Loans &bull; Insurance &bull; Government Schemes</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="warn-box">⚠️ <b>Disclaimer:</b> Pennywise provides general '
    'financial literacy information only — not personalised financial advice. '
    'Always verify at <a href="https://rbi.org.in" target="_blank">rbi.org.in</a>.</div>',
    unsafe_allow_html=True,
)

# ── Welcome / empty state ─────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("### 👋 Welcome to Pennywise!")
    st.markdown(
        "I can help you with:\n"
        "- 📲 **UPI payments** – how to send, receive, and stay safe\n"
        "- 🚨 **Scam prevention** – identify and avoid online financial fraud\n"
        "- 💳 **Loans & interest rates** – what's safe, what's predatory\n"
        "- 📊 **Budgeting** – the 50-30-20 rule and saving strategies\n"
        "- 🏛 **Government schemes** – Jan Dhan, PPF, Ayushman Bharat\n"
        "- 📑 **Mutual funds & SIP** – basics for new investors\n\n"
        "**Type your question below or pick a topic from the sidebar →**"
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("🌐 **Multilingual**\nAsk in Hindi, Marathi, Tamil, Bengali & more")
    with c2:
        st.info("🔒 **Grounded & Safe**\nAnswers based on RBI / SEBI / NPCI guidelines")
    with c3:
        st.info("🤖 **IBM Granite AI**\nPowered by WatsonX.ai – Cloud Lite (free)")

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">👤 <b>You:</b><br>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="bot-bubble">🤖 <b>Pennywise:</b><br>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        if msg.get("sources"):
            tags = "".join(
                f'<span class="source-tag">📌 {s["topic"]} – {s["source"]}</span>'
                for s in msg["sources"]
            )
            st.markdown(f"<small>Sources: {tags}</small>", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
quick = st.session_state.pop("_quick", None)
user_input: Optional[str] = st.chat_input(
    "Ask anything about UPI, scams, loans, budgeting… (any language)"
) or quick

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Load agent on first use
    with st.spinner("🔌 Connecting to IBM WatsonX…" if not st.session_state.agent_ready else "🤔 Thinking…"):
        agent, err = _load_agent()
        if err:
            st.session_state.agent_error = err
        else:
            st.session_state.agent_ready = True

    if st.session_state.agent_error:
        answer = (
            f"⚠️ Could not connect to IBM WatsonX: `{st.session_state.agent_error}`\n\n"
            "Please check your `IBM_CLOUD_API_KEY` and `WATSONX_PROJECT_ID` in the `.env` file."
        )
        sources = []
        model_used = ""
        elapsed = 0.0
        retrieved = 0
    else:
        with st.spinner("🤔 Thinking… (may take 10–30s on IBM Lite)"):
            from agent.financial_agent import FinancialResponse
            t0 = time.perf_counter()
            response: FinancialResponse = agent.chat(
                query=user_input,
                lang=selected_lang_code,
            )
            elapsed = time.perf_counter() - t0

        if response.error:
            err = response.error
            if "429" in err or "rate_limit" in err:
                answer = (
                    "⏳ **IBM WatsonX rate limit reached** (Lite plan: 2 req/s).\n\n"
                    "Please wait **10–15 seconds** and send your message again."
                )
            elif "CLOUD_API_KEY" in err or "credentials" in err.lower() or "apikey" in err.lower():
                answer = (
                    "⚠️ **Credentials error.** Please check your `.env` file:\n"
                    "- `IBM_CLOUD_API_KEY` must be a valid 44-character IBM Cloud key\n"
                    "- `WATSONX_PROJECT_ID` must be your WatsonX project UUID"
                )
            else:
                answer = f"❌ Error: {err}\n\nPlease try again in a few seconds."
            sources = []
        else:
            answer = response.answer
            sources = response.sources

        model_used = response.model_used
        retrieved = response.retrieved_docs

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

    if model_used:
        with st.expander(
            f"ℹ️ {elapsed:.1f}s · lang={selected_display} · "
            f"docs retrieved={retrieved} · model={model_used}"
        ):
            if "response" in dir() and response.answer_en and response.language != "en":
                st.markdown(f"**Answer in English:**\n\n{response.answer_en}")
            if sources:
                st.markdown("**Sources used:**")
                for s in sources:
                    st.markdown(f"- **{s['topic']}** – _{s['source']}_")

    st.rerun()
