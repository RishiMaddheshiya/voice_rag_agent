# 🎙️ Voice RAG Agent — Powered by Groq

A voice-enabled Retrieval-Augmented Generation (RAG) application built with Groq and Streamlit. Upload PDF documents, ask questions in natural language, and receive both text and spoken audio answers — powered by Llama 3.3 70B for reasoning and Orpheus TTS for lifelike speech, all running on Groq's ultra-fast inference.

## Features

- Voice-enabled RAG system built on Groq's OpenAI-compatible API
- PDF document processing with smart chunking (1000 chars, 200 overlap)
- Qdrant Cloud as the vector database for fast cosine-similarity search
- Local embeddings via FastEmbed — zero embedding API cost
- Lifelike text-to-speech with Orpheus (`canopylabs/orpheus-v1-english`)
- Multiple voice options: troy, hannah, austin, autumn
- Clean Streamlit interface with audio playback and WAV download
- Multiple document uploads with source tracking

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Llama 3.3 70B via Groq |
| Text-to-Speech | Orpheus v1 English via Groq |
| Vector Database | Qdrant Cloud |
| Embeddings | FastEmbed (local) |
| UI | Streamlit |
| Orchestration | OpenAI Agents SDK + LangChain text splitters |

## Getting Started

1. Clone this repository and install dependencies:
```bash
git clone <your-repo-url>
cd voice-rag-groq
pip install -r requirements.txt
```

2. Set up your API keys:
- Get your free [Groq API key](https://console.groq.com/keys)
- Create a [Qdrant Cloud](https://cloud.qdrant.io/) free cluster and copy its URL and API key
- Create a `.env` file:
```bash
GROQ_API_KEY='your-groq-api-key'
QDRANT_URL='https://your-cluster.cloud.qdrant.io:6333'
QDRANT_API_KEY='your-qdrant-api-key'
```
> Note: Accept the Orpheus model terms once in the [Groq Console](https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english) before first use.

3. Run the app:
```bash
streamlit run rag_voice.py
```

4. Open the URL shown in the console, enter your keys in the sidebar, upload a PDF, and start asking questions.

## How It Works

1. **Document Processing:** Uploaded PDFs are split into overlapping chunks using LangChain's RecursiveCharacterTextSplitter. Each chunk is embedded locally with FastEmbed and stored in Qdrant with source metadata.

2. **Query Processing:** Your question is embedded with the same model, and the top 3 most similar chunks are retrieved from Qdrant. These chunks plus your question are sent as context to Llama 3.3 70B on Groq, which generates a clear, spoken-word-friendly answer grounded in your document.

3. **Voice Generation:** The text answer is converted to speech using Groq's Orpheus TTS. Choose your preferred voice, play the audio directly in the browser, or download it as a WAV file.

## Why Groq?

Groq's LPU-based inference generates answers in seconds, and its OpenAI-compatible endpoint means both chat completion and speech synthesis run through a single API key — with a generous free tier.

## License

MIT
