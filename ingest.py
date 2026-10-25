import os
import shutil

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_DIR = "chroma_db"

# filename in documents/  ->  company name used for filtering
# (check that each filename really belongs to the company next to it)
DOCS = {
    "jpm-20251231.pdf": "JPMorgan",
    "10-K.pdf": "Goldman Sachs",
    "nvda-20260125.pdf": "NVIDIA",
}

# Rebuild from scratch so re-running never duplicates chunks
if os.path.exists(CHROMA_DIR):
    shutil.rmtree(CHROMA_DIR)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DIR)

batch_size = 50
for filename, company in DOCS.items():
    docs = PyPDFLoader(f"documents/{filename}").load()
    for doc in docs:
        doc.metadata["company"] = company
    chunks = splitter.split_documents(docs)
    print(f"{company}: {len(docs)} pages -> {len(chunks)} chunks")

    for i in range(0, len(chunks), batch_size):
        vectorstore.add_documents(chunks[i:i + batch_size])
        print(f"  embedded {min(i + batch_size, len(chunks))} / {len(chunks)}")

print(f"Done. Vector store saved at ./{CHROMA_DIR}/")