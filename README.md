# Hi, I'm Amit Verma 👋

**Lead Data Scientist** · Nearly 12 years building ML & GenAI systems across energy, telecom, life sciences, and healthcare · Azure & AWS · Agentic AI, RAG, and transformer NLP at production scale

---

## 🧭 The arc

I've spent nearly 12 years turning messy real-world data into models that ship. I started in **energy analytics** forecasting green-energy generation and loan disbursement trends, moved into **telecom ML engineering** predicting network incidents and automating trade-compliance audits, then led **transformer-based NLP** at a life-sciences company, and now build **agentic GenAI systems** for healthcare operations at Optum.

Across all of it, one thread: classical ML fundamentals (the math matters), production engineering (pipelines that don't break at 3am), and now LLM systems that reason over real enterprise data.

## 🔭 What I'm building now (Optum)

- **Agentic AI for call-center intelligence** — multi-agent **LangChain** system that diagnoses avoidable call transfers, retrieves solutions from internal knowledge bases, and generates self-service scripts + agent-assist recommendations
- **Large-scale NER pipeline on Azure** — processing ~250k call transcripts per run using **Anthropic Claude Sonnet** models on Azure, with async Python I/O and results written back to Snowflake
- **Time-series forecasting at scale** — ARIMA/SARIMAX across 15 business segments, automated into monthly capacity planning

## 🤖 Agentic AI — LangChain & LangGraph in production

| Project | What it does | Stack |
|---------|-------------|-------|
| **Call Leads** (Optum) | Hourly LangGraph pipeline: conditional fan-out — ≤1k records → single narrative pass; >1k → 5 parallel map jobs over 1k-summary chunks. Map extracts common narrative themes, reduce clubs them, a ranking agent returns the top 10 anomalies/insights/trends | LangGraph, LangChain, Claude Sonnet (Azure) |

## 🧠 What I'm good at

| Area | In practice |
|------|-------------|
| **GenAI & Agents** | RAG architectures, prompt engineering, LangChain/LangGraph multi-agent systems, LLM fine-tuning |
| **Cloud AI** | Azure OpenAI, AWS Bedrock, SageMaker, serverless inference |
| **Data Engineering** | Snowflake, Spark, Kafka, ELT pipelines, async Python at scale |
| **Classical ML** | Statsmodels forecasting, XGBoost, scikit-learn, transformer fine-tuning |
| **Transformer NLP** | BERT-based sentiment/multilabel/summarization, SpaCy, Hugging Face |

## 🛠️ Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Go](https://img.shields.io/badge/Go-00ADD8?style=flat&logo=go&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-0078D4?style=flat&logo=microsoftazure&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-232F3E?style=flat&logo=amazonaws&logoColor=white)
![Snowflake](https://img.shields.io/badge/Snowflake-29B5E8?style=flat&logo=snowflake&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-000000?style=flat&logo=langchain&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)

## 📌 Selected work across the years

| Project | What it does | Impact |
|---------|-------------|--------|
| **Agentic Transfer Obviation** (Optum) | Multi-agent **LangChain** system for root-cause + self-service generation | 12% fewer avoidable transfers in 2 months |
| **Drug NER Pipeline** (Optum, Azure) | Claude Sonnet NER over 250k transcripts + Snowflake | 60–90 min per full run |
| **Call Transfer Forecasting** (Optum) | ARIMA/SARIMAX across 15 segments | Automated into monthly reporting DB |
| **Customer Experience NLP** (Agilent) | BERT sentiment, multilabel, summarization + Django/FastAPI | Three transformer models in production |
| **Network Incident Prediction** (Orange) | ETL over historical logs + XGBoost classifier | 79% accuracy predicting issue types |
| **Green Energy Forecasting** (IREDA) | ML models on Joint Meter Readings + Tableau reports | Depreciation and generation prediction |

## 📫 Find me

[LinkedIn](https://www.linkedin.com/in/amit-verma-240992) · [Email](mailto:amitverma.dce@gmail.com) · [GitHub](https://github.com/AmitVermaDce)

---

*"Nearly 12 years of building systems that learn — from fuzzy clustering research to multi-agent LLM pipelines."*
