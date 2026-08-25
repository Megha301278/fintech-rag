from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# Reconnect to the SAME Chroma database ingest.py already built
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Set up the local chat model
llm = ChatOllama(model="qwen3:8b", temperature=0)

# The prompt template: tells the LLM HOW to use the retrieved chunks
system_prompt = (
    "You are a financial document assistant. Answer the question using "
    "ONLY the context below. If the answer isn't in the context, say so. "
    "Do not make up numbers.\n\nContext: {context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Chain it together: retrieve -> stuff into prompt -> ask LLM
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# Ask a question
question = "What was JPMorgan's net revenue?"
result = rag_chain.invoke({"input": question})

print("Question:", question)
print("\nAnswer:", result["answer"])
