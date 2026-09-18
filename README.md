# 📄 Document Question-Answering System using RAG

## 💡 What is this?

This project is an intelligent question-answering system that lets you ask
questions and get answers grounded in a specific set of documents — instead
of relying purely on what a language model already "knows" from training.

It's built on a technique called **Retrieval-Augmented Generation (RAG)**,
which works in two stages:

- 🔍 **Retrieve** — When you ask a question, the system searches through the
  documents to find the most relevant pieces of information.
- ✍️ **Generate** — It then feeds those relevant pieces to a language model,
  which crafts a clear, natural-language answer *based only on that
  retrieved context*.

## 🎯 Why RAG instead of just asking an LLM directly?

Language models can sometimes "hallucinate" — confidently stating things
that aren't true. RAG reduces this risk by grounding every answer in real,
retrievable source material. It also means the system can answer questions
about documents the model was never trained on, without needing to retrain
anything — just update the document collection.

## ⚙️ How it works under the hood

1. 📚 Documents are split into small, overlapping chunks
2. 🧠 Each chunk is converted into a vector embedding via the Gemini
   embeddings API
3. 🗂️ Embeddings are stored in a FAISS index for fast similarity search
4. ❓ A user's question is embedded the same way
5. 🔗 The most similar chunks are retrieved
6. 🤖 Those chunks + the question are sent to Gemini, which generates a
   grounded, context-aware answer

## 📊 Verified results

Measured on an 11-question evaluation set spanning both source documents
(see `scripts/eval_dataset.py` and `scripts/evaluate.py`):

- ✅ **100% answer accuracy**
- ✅ **100% Retrieval Recall@5**
- 🧪 **93%+ test coverage** (Pytest)

## 🚀 Live app

**Try it here:** https://rag-doc-app-g8tutjvhjr4z9bg5rje5dt.streamlit.app/

## ✨ Highlights

- 📄 Document-grounded answers, not generic LLM guesses
- 🔎 Vector similarity search via FAISS
- 🌐 Interactive web interface for easy, no-code use
- 🧩 Beginner-friendly, modular codebase
