from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("documents/jpm-20251231.pdf")
docs = loader.load()

print(f"Loaded {len(docs)} pages")
print(docs[0].page_content[:500])  # print the first 500 characters of page 1

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")
print(chunks[0].page_content)

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create an empty Chroma database first
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_db",
)

# Add chunks in small batches instead of all at once
batch_size = 50
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    vectorstore.add_documents(batch)
    print(f"Embedded {min(i + batch_size, len(chunks))} / {len(chunks)} chunks")

print(f"Stored {len(chunks)} chunks in Chroma at ./chroma_db/")