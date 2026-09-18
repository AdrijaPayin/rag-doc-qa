"""
Streamlit front-end for the RAG Document QA API.

This is a thin UI layer -- it doesn't do any retrieval or generation
itself. It just sends questions to your FastAPI backend's /ask
endpoint and displays the response nicely.

Run locally with:
    streamlit run streamlit_app.py

Before running, make sure the FastAPI backend is running (locally
via `uvicorn app.main:app --reload`, or point API_URL below at your
deployed Render URL).
"""

import streamlit as st
import requests

# --- Configuration ---
# Change this to your deployed Render URL once it's live, e.g.:
# API_URL = "https://your-app.onrender.com"
API_URL = st.secrets.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Document QA (RAG)", page_icon="📄", layout="centered")

st.title("📄 Document Question Answering")
st.caption("Ask a question and get an answer grounded in your documents, powered by RAG.")

# --- Sidebar: backend status + settings ---
with st.sidebar:
    st.subheader("Settings")
    api_url_input = st.text_input("Backend API URL", value=API_URL)
    top_k = st.slider("Number of chunks to retrieve (top_k)", min_value=1, max_value=10, value=5)

    st.divider()
    st.subheader("Backend status")
    try:
        health = requests.get(f"{api_url_input}/health", timeout=5)
        if health.status_code == 200:
            st.success("Backend is reachable")
        else:
            st.warning(f"Backend responded with status {health.status_code}")
    except requests.exceptions.RequestException:
        st.error("Cannot reach backend. Is it running?")

# --- Main question input ---
question = st.text_input(
    "Your question",
    placeholder="e.g. What is Retrieval-Augmented Generation?",
)

ask_clicked = st.button("Ask", type="primary")

if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving relevant chunks and generating an answer..."):
            try:
                response = requests.post(
                    f"{api_url_input}/ask",
                    json={"question": question, "top_k": top_k},
                    timeout=60,
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the backend: {e}")
                response = None

        if response is not None:
            if response.status_code == 200:
                data = response.json()

                st.subheader("Answer")
                st.write(data["answer"])

                col1, col2 = st.columns(2)
                col1.metric("Retrieval latency", f"{data['retrieval_latency_ms']} ms")
                col2.metric("Total latency", f"{data['total_latency_ms']} ms")

                if data.get("sources"):
                    st.subheader("Sources")
                    st.write(", ".join(data["sources"]))

                if data.get("retrieved_chunks"):
                    with st.expander("View retrieved chunks"):
                        for i, chunk in enumerate(data["retrieved_chunks"], start=1):
                            st.markdown(f"**Chunk {i}** — `{chunk['source']}` (score: {chunk['score']:.3f})")
                            st.text(chunk["text"])
                            st.divider()

            elif response.status_code == 503:
                st.error(
                    "The document index isn't loaded on the backend yet. "
                    "Run scripts/ingest.py on the backend and restart it."
                )
            elif response.status_code == 400:
                st.warning(response.json().get("detail", "Bad request."))
            else:
                st.error(f"Unexpected error: {response.status_code} — {response.text}")

st.divider()
st.caption("Backend: FastAPI + FAISS + Sentence Transformers + Gemini")
