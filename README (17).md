<div align="center">

# 🎙️ Voice RAG Agent

**Ask your PDFs anything — get answers you can read *and* hear.**

A voice-enabled RAG application powered by Groq's blazing-fast inference: Llama 3.3 70B for reasoning, Orpheus TTS for lifelike speech, and Qdrant for vector search.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-F55036)](https://groq.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC244C)](https://qdrant.tech/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Features](#-features) • [Demo](#-how-it-works) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Roadmap](#-roadmap)

</div>

---

## ✨ Features

- 🎤 **Voice + Text answers** — every response is spoken aloud with Orpheus TTS and downloadable as WAV
- ⚡ **Groq-fast inference** — Llama 3.3 70B generates grounded answers in seconds on LPU hardware
- 📄 **Smart PDF processing** — recursive chunking (1000 chars, 200 overlap) preserves context across chunk boundaries
- 💸 **Zero embedding cost** — FastEmbed runs locally, no embedding API bills
- 🔍 **Semantic search** — Qdrant Cloud retrieves the most relevant chunks via cosine similarity
- 🗣️ **4 natural voices** — troy, hannah, austin, autumn (with `[cheerful]`-style vocal direction support)
- 📚 **Multi-document support** — upload multiple PDFs with source tracking in every answer

## 🏗️ Architecture

```
                        ┌─────────────── INDEXING ───────────────┐
                        │                                        │
  📄 PDF ──► PyPDFLoader ──► Chunking ──► FastEmbed ──► 🗄️ Qdrant │
                        │   (1000/200)    (local)                │
                        └────────────────────────────────────────┘

                        ┌─────────────── QUERYING ───────────────┐
                        │                                        │
  ❓ Question ──► FastEmbed ──► Qdrant top-3 ──► 🦙 Llama 3.3 70B │
                        │        (cosine)          (Groq)        │
                        │                             │          │
  🔊 WAV audio ◄── Orpheus TTS (Groq) ◄── text answer ┘          │
                        └────────────────────────────────────────┘
```

**The RAG flow in one line:** your question is embedded → the 3 most similar document chunks are retrieved → Llama answers *using only those chunks* → Orpheus speaks the answer.

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/RishiMaddheshiya/voice_rag_agent.git
cd voice_rag_agent
pip install -r requirements.txt
```

### 2. Get your keys (both free)

| Key | Where |
|---|---|
| Groq API key | [console.groq.com/keys](https://console.groq.com/keys) |
| Qdrant URL + API key | [cloud.qdrant.io](https://cloud.qdrant.io/) → create a free cluster |

Create a `.env` file:

```bash
GROQ_API_KEY='gsk_...'
QDRANT_URL='https://your-cluster.aws.cloud.qdrant.io:6333'
QDRANT_API_KEY='your-qdrant-api-key'
```

> ⚠️ **One-time step:** accept the Orpheus model terms in the [Groq Console](https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english) before your first query, or the TTS call will return a `model_terms_required` error.

### 3. Run

```bash
streamlit run rag_voice.py
```

Enter your keys in the sidebar → upload a PDF → ask away. 🎉

## ⚙️ How It Works

| Stage | What happens |
|---|---|
| **Ingest** | PDF → text via PyPDFLoader, split into overlapping chunks with source metadata |
| **Embed** | Each chunk → 384-dim vector via FastEmbed, running locally on your machine |
| **Store** | Vectors + payloads upserted into a Qdrant Cloud collection (cosine distance) |
| **Retrieve** | Your question is embedded with the same model; top-3 nearest chunks are fetched |
| **Generate** | Chunks + question go to `llama-3.3-70b-versatile` on Groq with spoken-word-friendly instructions |
| **Speak** | The answer is synthesized by `canopylabs/orpheus-v1-english` and returned as WAV |

## 🧰 Tech Stack

**Groq** (LLM + TTS) · **Qdrant Cloud** (vector DB) · **FastEmbed** (local embeddings) · **Streamlit** (UI) · **LangChain** (text splitting) · **OpenAI Agents SDK** (agent orchestration via Groq's OpenAI-compatible endpoint)

## 🗺️ Roadmap

- [ ] 🎙️ Voice *input* — ask questions by speaking (Whisper on Groq)
- [ ] 💬 Chat history with follow-up questions
- [ ] 🌐 Support for DOCX, TXT, and web pages
- [ ] 🐳 Dockerfile for one-command deployment

## 🙏 Acknowledgements

Inspired by the voice RAG tutorial in [awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps) — rebuilt and migrated from OpenAI to the Groq stack (Llama 3.3 + Orpheus TTS).

## 📄 License

MIT © [Rishi Maddheshiya](https://github.com/RishiMaddheshiya)

---

<div align="center">

**If this project helped you, drop a ⭐ — it helps others find it!**

</div>
