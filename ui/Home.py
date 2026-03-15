import streamlit as st

st.set_page_config(page_title="Qibo AI Assistant", layout="wide", page_icon="🌌")

st.title("🌌 Qibo AI Assistant")
st.markdown("""
Welcome to QiboAgent UI, the AI assistant for **Qibo** — the open-source Python framework for quantum computing.

### Available Pages

- **📚 RAG Q&A** — Ask questions about the Qibo codebase using retrieval-augmented generation. Run generated code and visualize plots directly in the browser.
- **🔧 Agent Issues** — Provide a Qibo GitHub issue number and let an autonomous agent analyze the problem, inspect the source code, and propose a patch.

Select a page from the sidebar to get started.
""")