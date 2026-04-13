# Chat-With-PDF-App

A Streamlit app for having conversations with PDF documents. Upload any PDF, ask questions about its content, or generate a concise summary — powered by LangChain and OpenAI.

**Live demo:** deploy locally in under 2 minutes.

---

## What it does

Two modes, one upload:

**Chat with PDF** — Ask any question about your document and get a direct, cited answer. Uses vector embeddings to retrieve the most relevant sections before generating a response.

**Summarization** — Get a structured summary of the full document without reading it yourself.

---

## How it works

```
PDF upload
    │
    ▼
LangChain PDF loader → text chunking → OpenAI embeddings
    │
    ▼
ChromaDB vector store (local, persisted across sessions)
    │
    ▼
User question → similarity search → relevant chunks → GPT → answer
```

The vector store is saved to `chroma_storage/` so re-uploading the same PDF skips re-embedding.

---

## Setup

### Prerequisites

- Python 3.10+
- OpenAI API key

### Install

```bash
git clone https://github.com/Prabhuteja799/Chat-With-PDF-App.git
cd Chat-With-PDF-App
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

Create a `.env` file or export directly:

```bash
export OPENAI_API_KEY=your_key_here
```

### Run

```bash
streamlit run Home.py
# → http://localhost:8501
```

---

## Project structure

```
Chat-With-PDF-App/
├── Home.py              ← Streamlit entry point + landing page
├── pages/
│   ├── chat.py          ← Chat with PDF page
│   └── summarize.py     ← Summarization page
├── chroma_storage/      ← Persisted vector embeddings
├── requirements.txt
└── logo.png
```

---

## Tech stack

| Layer | Tech |
|---|---|
| UI | Streamlit (multi-page) |
| PDF parsing | LangChain `PyPDFLoader` |
| Embeddings | OpenAI `text-embedding-ada-002` |
| Vector store | ChromaDB (local persistence) |
| LLM | OpenAI GPT (`gpt-3.5-turbo` / `gpt-4`) |
| Orchestration | LangChain `RetrievalQA` chain |

---

## Future improvements

- Multi-PDF support (ask across several documents at once)
- Source page citation in answers
- Streaming responses
- Hosted deployment with user sessions
