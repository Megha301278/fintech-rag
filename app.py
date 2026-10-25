import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/ask"
REPO_URL = "https://github.com/Megha301278/fintech-rag"
GREETING = (
    "Ask me about the JPMorgan, Goldman Sachs, or NVIDIA 10-K filings. "
    "Mention two companies and I'll compare them."
)
EXAMPLES = [
    "What was JPMorgan's net revenue?",
    "What are NVIDIA's main risk factors?",
    "Compare JPMorgan and Goldman Sachs on profitability",
    "What is NVIDIA's favorite pizza topping?",
]
CONFIDENCE_LABEL = {
    "high": ":green[● High confidence]",
    "medium": ":orange[● Medium confidence]",
    "low": ":red[● Low confidence]",
}

st.set_page_config(page_title="Fintech Document Q&A", page_icon="📈")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]

# ---------- sidebar ----------
with st.sidebar:
    st.header("📈 Filings loaded")
    st.markdown("- JPMorgan (JPM)\n- Goldman Sachs (GS)\n- NVIDIA (NVDA)")

    st.divider()
    st.subheader("Try asking")
    for example in EXAMPLES:
        if st.button(example):
            st.session_state.pending = example

    st.divider()
    show_steps = st.toggle("Show agent steps", value=True)
    if st.button("Clear chat"):
        st.session_state.messages = [{"role": "assistant", "content": GREETING}]
        st.rerun()

    st.divider()
    st.info("Runs fully locally on Ollama. No API key needed, and your documents never leave your machine.")
    st.markdown(f"[View the source code]({REPO_URL})")
    st.caption("For information only, not investment advice.")

# ---------- header ----------
st.title("📈 Fintech Document Q&A")
st.write(
    "An agentic RAG app over SEC 10-K filings. It shows its thoughts and actions as it works: "
    "searching, grading its own retrieval, retrying with a better query, and fact-checking its answer."
)


def render_extras(msg: dict) -> None:
    """Confidence, agent steps, and sources under an assistant answer."""
    confidence = msg.get("confidence")
    if confidence:
        retried = " · search retried" if msg.get("retried") else ""
        st.caption(CONFIDENCE_LABEL[confidence] + retried)
    if show_steps and msg.get("steps"):
        with st.expander("🧠 Agent steps"):
            for i, step in enumerate(msg["steps"], start=1):
                st.markdown(f"{i}. {step}")
    if msg.get("sources"):
        with st.expander(f"📄 Sources ({len(msg['sources'])})"):
            for source in msg["sources"]:
                st.markdown(f"**{source['company']} · page {source['page']}**")
                st.caption(source["excerpt"])


# ---------- chat history ----------
for msg in st.session_state.messages:
    avatar = "📈" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            render_extras(msg)

# ---------- new question ----------
prompt = st.chat_input("What was JPMorgan's net revenue?")
if prompt is None:
    prompt = st.session_state.pop("pending", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant", avatar="📈"):
        try:
            with st.spinner("Searching, grading, answering, fact-checking..."):
                response = requests.post(API_URL, json={"question": prompt}, timeout=600)
                response.raise_for_status()
                result = response.json()
        except requests.exceptions.ConnectionError:
            st.error("Can't reach the backend. Start it with `uvicorn main:app --reload` in another terminal.")
            st.stop()
        except requests.exceptions.RequestException as error:
            st.error(f"The backend returned an error: {error}")
            st.stop()

        reply = {
            "role": "assistant",
            "content": result["answer"],
            "confidence": result["confidence"],
            "retried": result["retried"],
            "steps": result["steps"],
            "sources": result["sources"],
        }
        st.write(reply["content"])
        render_extras(reply)

    st.session_state.messages.append(reply)