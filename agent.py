import re

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

llm = ChatOllama(model="qwen3:8b", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# Companies the agent knows about, and how people might refer to them
COMPANIES = {
    "JPMorgan": r"\b(jpmorgan|jp morgan|jpm|chase)\b",
    "Goldman Sachs": r"\b(goldman sachs|goldman|gs)\b",
    "NVIDIA": r"\b(nvidia|nvda)\b",
}

NOT_FOUND = "I couldn't find enough information in the filings to answer that."


# ---------- helpers ----------
def clean(text: str) -> str:
    """Strip <think>...</think> reasoning that qwen3 may include."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ask_llm(prompt: str) -> str:
    return clean(llm.invoke(prompt).content)


def detect_companies(question: str) -> list[str]:
    q = question.lower()
    return [name for name, pattern in COMPANIES.items() if re.search(pattern, q)]


def search(question: str, company: str | None = None, k: int = 4):
    """Vector search, optionally restricted to one company's filing."""
    if company:
        return vectorstore.similarity_search(question, k=k, filter={"company": company})
    return vectorstore.similarity_search(question, k=k)


def format_sources(chunks) -> list[dict]:
    return [
        {
            "company": c.metadata.get("company", "unknown"),
            "page": c.metadata.get("page", 0) + 1,
            "excerpt": c.page_content[:200].strip() + "...",
        }
        for c in chunks
    ]


# ---------- step 1: query reformulation ----------
def reformulate_query(question: str) -> str:
    """Rewrite vague questions into clearer, more searchable ones."""
    prompt = f"""You are a query optimization assistant for a financial document search system.

Given a user's question, decide if it's specific enough to search effectively.
If it's already clear and specific, return it UNCHANGED.
If it's vague, rewrite it to be more specific and searchable, using financial terminology where appropriate.

Only return the final question text, nothing else - no explanation.

User question: {question}

Optimized question:"""
    return ask_llm(prompt)


# ---------- step 2: grade retrieval, retry once ----------
def chunks_are_relevant(question: str, chunks) -> bool:
    if not chunks:
        return False
    context = "\n\n".join(c.page_content[:500] for c in chunks)
    prompt = f"""You are grading search results for a financial document Q&A system.

Question: {question}

Retrieved text:
{context}

Does the retrieved text contain information that helps answer the question (even partially)?
Answer with exactly one word: yes or no."""
    return ask_llm(prompt).lower().startswith("yes")


def retrieve_with_retry(question: str, company: str | None = None, steps: list | None = None):
    """Returns (chunks, query_used, retried). Empty chunks = nothing relevant found."""
    log = steps if steps is not None else []
    scope = company or "all filings"

    chunks = search(question, company)
    if chunks_are_relevant(question, chunks):
        log.append(f"Searched {scope}: retrieved {len(chunks)} relevant chunks")
        return chunks, question, False

    log.append(f"Searched {scope}: results looked weak")
    new_query = reformulate_query(question)
    log.append(f'Reformulated query: "{new_query}"')
    chunks = search(new_query, company)
    if chunks_are_relevant(question, chunks):
        log.append(f"Retry found {len(chunks)} relevant chunks")
        return chunks, new_query, True

    log.append("Retry still found nothing relevant")
    return [], new_query, True


# ---------- generation ----------
def generate_answer(question: str, chunks) -> str:
    context = "\n\n".join(c.page_content for c in chunks)
    prompt = f"""You are a financial document assistant. Answer the question using ONLY the context below. If the answer isn't in the context, say so. Do not make up numbers.

Context:
{context}

Question: {question}

Answer:"""
    return ask_llm(prompt)


def generate_comparison(question: str, chunks_by_company: dict) -> str:
    sections = []
    for company, chunks in chunks_by_company.items():
        text = "\n\n".join(c.page_content for c in chunks) if chunks else "(no relevant text found)"
        sections.append(f"=== {company} ===\n{text}")
    context = "\n\n".join(sections)
    prompt = f"""You are a financial document assistant comparing companies. Use ONLY the context below, which is grouped by company. If a company's section has no relevant text, say you couldn't find it instead of guessing. Do not make up numbers. Keep the comparison short and structured.

Context:
{context}

Question: {question}

Comparison:"""
    return ask_llm(prompt)


# ---------- step 3: self-critique ----------
def critique_answer(answer: str, chunks) -> str:
    """Fact-check the answer against its own sources. Returns high / medium / low."""
    context = "\n\n".join(c.page_content for c in chunks)
    prompt = f"""You are fact-checking an answer against source text.

Source text:
{context}

Answer to check:
{answer}

Is every number and claim in the answer directly supported by the source text?
Reply with exactly one word: supported, partial, or unsupported."""
    verdict = ask_llm(prompt).lower()
    if verdict.startswith("supported"):
        return "high"
    if verdict.startswith("partial"):
        return "medium"
    return "low"


# ---------- the agent ----------
def answer_comparison(question: str, companies: list[str], steps: list) -> dict:
    chunks_by_company, retried_any = {}, False
    for company in companies:
        chunks, _, retried = retrieve_with_retry(question, company, steps)
        chunks_by_company[company] = chunks
        retried_any = retried_any or retried
        if not chunks:
            steps.append(f"No relevant evidence found for {company}")

    all_chunks = [c for chunks in chunks_by_company.values() for c in chunks]
    if not all_chunks:
        steps.append("No relevant evidence for any company: declining to answer")
        return {"answer": NOT_FOUND, "confidence": "low", "retried": retried_any,
                "search_query": question, "sources": [], "steps": steps}

    answer = generate_comparison(question, chunks_by_company)
    steps.append("Generated comparison from the retrieved chunks")
    confidence = critique_answer(answer, all_chunks)
    # If one company has no evidence, never report "high"
    if any(not chunks for chunks in chunks_by_company.values()) and confidence == "high":
        confidence = "medium"
    steps.append(f"Fact-checked against sources: {confidence} confidence")
    return {"answer": answer, "confidence": confidence, "retried": retried_any,
            "search_query": question, "sources": format_sources(all_chunks), "steps": steps}


def answer_question(question: str) -> dict:
    steps = []
    companies = detect_companies(question)
    if len(companies) >= 2:
        steps.append(f"Comparison mode: {' vs '.join(companies)}")
        return answer_comparison(question, companies, steps)

    company = companies[0] if companies else None
    steps.append(f"Search scoped to {company}" if company else "No company named: searching all filings")
    chunks, query_used, retried = retrieve_with_retry(question, company, steps)
    if not chunks:
        steps.append("No relevant evidence found: declining to answer")
        return {"answer": NOT_FOUND, "confidence": "low", "retried": retried,
                "search_query": query_used, "sources": [], "steps": steps}

    answer = generate_answer(question, chunks)
    steps.append("Generated answer from the retrieved chunks")
    confidence = critique_answer(answer, chunks)
    steps.append(f"Fact-checked against sources: {confidence} confidence")
    return {"answer": answer, "confidence": confidence, "retried": retried,
            "search_query": query_used, "sources": format_sources(chunks), "steps": steps}


if __name__ == "__main__":
    tests = [
        "What was JPMorgan's net revenue?",
        "Compare JPMorgan and Goldman Sachs on profitability",
        "What is NVIDIA's favorite pizza topping?",
    ]
    for q in tests:
        r = answer_question(q)
        print(f"Q: {q}")
        print(f"A: {r['answer'][:300]}")
        print(f"Confidence: {r['confidence']} | Retried: {r['retried']} | Sources: {len(r['sources'])}")
        for s in r["steps"]:
            print(f"  - {s}")
        print()