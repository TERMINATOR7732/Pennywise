# 💰 Pennywise – AI Agent for Digital Financial Literacy

> **Problem Statement #7** | Edunet / IBM SkillsBuild Internship  
> Powered by **IBM Granite** on **IBM WatsonX.ai (Cloud Lite – Free Tier)**

---

## 🎯 Overview

**Pennywise** is a RAG-powered (Retrieval-Augmented Generation) conversational AI agent that makes digital financial literacy accessible to every Indian citizen — regardless of language or technical background.

It retrieves reliable, curated content from RBI, NPCI, SEBI, IRDAI, and government portals, then uses **IBM Granite** (via WatsonX.ai) to generate grounded, safe, and accurate answers.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🤖 **IBM Granite LLM** | `ibm/granite-13b-chat-v2` on WatsonX.ai (Lite) |
| 🔍 **RAG Pipeline** | FAISS vector store + LangChain, 13 seed topics |
| 🌐 **Multilingual** | 12 Indian languages via IBM Watson Language Translator |
| 📲 **UPI Guidance** | Step-by-step UPI payment & safety instructions |
| 🚨 **Scam Prevention** | KYC scams, phishing, QR scams, SIM swap |
| 💳 **Loan Advice** | Safe interest rates, predatory lender red flags |
| 📊 **Budgeting** | 50-30-20 rule, emergency fund, savings strategies |
| 🏛 **Govt Schemes** | Jan Dhan Yojana, PPF, Ayushman Bharat |
| 📑 **Investments** | Mutual funds, SIP, FD rates, CIBIL score |
| 🛡 **Insurance** | Health & term life insurance basics |

---

## 🏗 Architecture

```
User (any language)
       │
       ▼
┌─────────────────────────┐
│  Streamlit UI  (app.py) │
└────────────┬────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│   FinancialLiteracyAgent               │
│   (agent/financial_agent.py)           │
│                                        │
│  ① MultilingualPipeline               │
│     • IBM Watson Language Translator  │
│     • Script-based heuristic detect   │
│     • Translate query → English       │
│                                        │
│  ② FinancialKnowledgeBase (FAISS)    │
│     • Similarity search (top-k docs)  │
│     • IBM WatsonX Embeddings          │
│       (slate-125m-english-rtrvr)      │
│                                        │
│  ③ Prompt Builder                     │
│     • System prompt + context docs    │
│                                        │
│  ④ IBM Granite (WatsonX.ai)          │
│     • granite-13b-chat-v2             │
│     • Grounded text generation        │
│                                        │
│  ⑤ MultilingualPipeline              │
│     • Translate answer → user lang    │
└────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & install dependencies

```bash
git clone <repo-url>
cd pennywise
pip install -r requirements.txt
```

### 2. Configure IBM Cloud credentials

Edit the `.env` file (already created) and fill in your real values:

```env
IBM_CLOUD_API_KEY=<your-ibm-cloud-api-key>
WATSONX_PROJECT_ID=<your-watsonx-project-id>
LANGUAGE_TRANSLATOR_APIKEY=<your-lt-api-key>       # optional
LANGUAGE_TRANSLATOR_URL=<your-lt-service-url>       # optional
```

#### How to get IBM Cloud Lite credentials (free)

1. Sign up at [cloud.ibm.com](https://cloud.ibm.com) (no credit card required for Lite plan)
2. **API Key** → IAM → API Keys → Create → Copy
3. **WatsonX Project ID** → [dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com) → New Project → Settings → Copy Project ID
4. **Language Translator** (optional) → IBM Cloud Catalog → Language Translator → Create Lite instance → Credentials

### 3. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
pennywise/
├── .env                        ← IBM Cloud credentials (DO NOT commit)
├── app.py                      ← Streamlit UI entry point
├── requirements.txt
│
├── core/
│   ├── config.py               ← Pydantic-Settings configuration
│   └── multilingual.py         ← Language detection + IBM Translator
│
├── ibm/
│   ├── watsonx_client.py       ← IBM Granite text generation + embeddings
│   └── translator.py           ← IBM Watson Language Translator wrapper
│
├── rag/
│   └── knowledge_base.py       ← FAISS vector store + seed knowledge
│
├── agent/
│   └── financial_agent.py      ← RAG + LLM orchestration pipeline
│
├── data/
│   └── vector_store/           ← Persisted FAISS index (auto-created)
│
└── tests/
    └── test_agent.py           ← Unit tests
```

---

## 🌐 Supported Languages

| Code | Language |
|------|----------|
| `en` | English |
| `hi` | हिन्दी (Hindi) |
| `mr` | मराठी (Marathi) |
| `ta` | தமிழ் (Tamil) |
| `te` | తెలుగు (Telugu) |
| `bn` | বাংলা (Bengali) |
| `gu` | ગુજરાતી (Gujarati) |
| `kn` | ಕನ್ನಡ (Kannada) |
| `pa` | ਪੰਜਾਬੀ (Punjabi) |
| `ml` | മലയാളം (Malayalam) |
| `ur` | اردو (Urdu) |
| `or` | ଓଡ଼ିଆ (Odia) |

> **Tip:** The app works in English without a Language Translator key.  
> Configure `LANGUAGE_TRANSLATOR_APIKEY` to unlock all 12 languages.

---

## 🔬 IBM Services Used (All Free Lite Tier)

| Service | Purpose |
|---------|---------|
| **IBM WatsonX.ai** | Hosts IBM Granite LLM for text generation |
| **IBM Granite 13B Chat v2** | Primary language model |
| **IBM Slate 125M English** | Embedding model for RAG |
| **IBM Watson Language Translator** | Multilingual translation |
| **IBM Cloud IAM** | API key authentication |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## ⚠️ Disclaimer

Pennywise provides general financial literacy information only.  
It is **not** a registered financial adviser.  
For personalised financial or investment advice, consult a SEBI-registered adviser or your bank.  
Always verify important financial information at official sources:  
- RBI: [rbi.org.in](https://www.rbi.org.in)  
- SEBI: [sebi.gov.in](https://www.sebi.gov.in)  
- NPCI: [npci.org.in](https://www.npci.org.in)  
- Cybercrime: [cybercrime.gov.in](https://cybercrime.gov.in) | Helpline: **1930**

---

*Built with ❤️ using IBM Granite + WatsonX.ai for the Edunet Digital Financial Literacy Challenge*
