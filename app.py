import streamlit as st
import requests

st.set_page_config(page_title="Fintech Document Q&A", page_icon="📊")

st.title("📊 Fintech Document Q&A")
st.write("Ask anything about the uploaded financial documents (JPMorgan, Goldman Sachs, NVIDIA).")

question = st.text_input("Your question:")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={"question": question}
        )
        answer = response.json()["answer"]
    st.write(answer)
