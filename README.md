# 📊 Fintech Document Q&A (RAG)

Ask natural-language questions about real financial filings (10-Ks) and get grounded answers with a locally-run LLM — no OpenAI API key, no cost, fully private. Built with a FastAPI backend and a Streamlit frontend.

## What it does

Upload financial documents (10-K filings), ask a question in plain English — e.g. *"What was JPMorgan's net revenue?"* — and get an answer generated **only from the actual document text**, not the model's general knowledge. This is Retrieval-Augmented Generation (RAG): the system retrieves the most relevant chunks of the document first, then asks the LLM to answer using only that retrieved context.

## Why RAG, and why local

Financial numbers can't be hallucinated — a model answering from "general knowledge" about a company is dangerous in this domain. RAG grounds every answer in real retrieved text. Running everything locally via Ollama (rather than an API) means zero cost, full data privacy (real filings never leave your machine), and no dependency on external services.

## Demo

Ask things like:
- "What was JPMorgan's net revenue?"
- "What are NVIDIA's main risk factors?"
- "Summarize Goldman Sachs' business segments."

## Architecture

```
PDF documents (10-Ks)
       ↓
  Load & chunk (LangChain + PyPDF)
       ↓
  Embed locally (Ollama: nomic-embed-text)
       ↓
  Store in vector DB (Chroma)
       ↓
  User question → retrieve top-k relevant chunks
       ↓
  LLM answers using retrieved context (Ollama: qwen3:8b)
       ↓
  FastAPI backend (/ask endpoint) → Streamlit frontend
```

## Tech Stack

**LLM/RAG:** LangChain, ChromaDB, Ollama (qwen3:8b + nomic-embed-text)
**Backend:** FastAPI
**Frontend:** Streamlit
**Document Processing:** PyPDF, LangChain text splitters

## Data Source

Real 10-K filings sourced directly from [SEC EDGAR](https://www.sec.gov/edgar/search/) — JPMorgan Chase, Goldman Sachs, and NVIDIA.

## Project Structure

```
fintech-rag/
├── ingest.py          # Loads PDFs, chunks text, builds the vector database
├── query.py            # Standalone script for testing RAG queries
├── main.py             # FastAPI backend serving the /ask endpoint
├── app.py               # Streamlit frontend
├── documents/            # Source 10-K PDFs
├── requirements.txt
└── chroma_db/              # Vector database (generated, not committed — see .gitignore)
```

## Running It

**1. Set up Ollama** (local LLM, free, no API key):
```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

**2. Install dependencies:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Add your documents** — place PDFs in `documents/`

**4. Build the vector database:**
```bash
python ingest.py
```

**5. Run the backend** (Terminal 1):
```bash
uvicorn main:app --reload
```

**6. Run the frontend** (Terminal 2, same folder, venv active):
```bash
streamlit run app.py
```

Visit the Streamlit URL it opens (usually `http://localhost:8501`).

## Engineering Notes

- **Batched embedding**: Embedding all chunks in a single request overwhelmed the local Ollama server. Fixed by batching chunk embedding in groups of 50, which is both more memory-stable and standard practice for large-document ingestion.
- **Chunk overlap**: Used 200-character overlap between chunks to avoid severing context at chunk boundaries (e.g., a sentence split mid-fact across two chunks).

## Future Improvements

- [ ] Chat-style conversational interface (multi-turn, not single Q&A)
- [ ] Source citations — show which document/page an answer came from
- [ ] Document selector — query a specific filing instead of all documents mixed
- [ ] Dockerize for deployment
- [ ] Deploy live for a public demo URL

## Author

**Megha Dalsania** — [LinkedIn](www.linkedin.com/in/megha-dalsania-b04b90267) · [GitHub](https://github.com/Megha301278)
