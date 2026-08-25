from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("documents/jpm-20251231.pdf")
docs = loader.load()

print(f"Loaded {len(docs)} pages")
print(docs[0].page_content[:500])  # print the first 500 characters of page 1

from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")
print(chunks[0].page_content)