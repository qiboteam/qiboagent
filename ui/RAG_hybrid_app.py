import streamlit as st
import sys
import os
import subprocess
import tempfile
from pathlib import Path

# --- Path Configuration ---
# Calculate the absolute path to the 'root' directory
ROOT_DIR = Path(__file__).parent.parent
# Add 'python_scripts' to the system path so we can import RAG_hybrid
sys.path.append(str(ROOT_DIR / "python_scripts"))

from RAG_hybrid import (
    load_json_settings, initialize_llm, parse_repo, 
    create_code_chunks, create_doc_chunks, deduplicate_documents,
    build_vectorstore, HybridRetriever, create_qa_chain, extract_code
)

st.set_page_config(page_title="Qibo AI Assistant", layout="wide")

st.title("🌌 Qibo Quantum RAG Assistant")

# --- Backend Initialization ---
@st.cache_resource
def setup_rag():
    # Use the ROOT_DIR to find settings and data consistently
    settings = load_json_settings(str(ROOT_DIR / "settings_json" / "settings.json"))
    llm = initialize_llm(settings)
    
    qibo_dir = ROOT_DIR / "qiboKnow" / "qibo"
    code_root = str(qibo_dir / "src/qibo")
    docs_root = str(qibo_dir / "doc")
    
    parsed_items = parse_repo(code_root)
    all_docs = deduplicate_documents(create_code_chunks(parsed_items) + create_doc_chunks(docs_root))
    
    # Persist the vectorstore in the root directory
    vect = build_vectorstore(create_code_chunks(parsed_items), create_doc_chunks(docs_root), 
                             persist=str(ROOT_DIR / "kb_chroma"))
    
    retriever = HybridRetriever(all_docs, vect)
    return create_qa_chain(llm, retriever)

qa_chain = setup_rag()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Code Execution Logic
        if message["role"] == "assistant":
            code = extract_code(message["content"])
            if code:
                if st.button(f"▶️ Run This Qibo Code", key=f"run_{i}"):
                    with st.status("Executing on Local Backend...", expanded=True):
                        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
                            tmp.write(code)
                            tmp_path = tmp.name
                        try:
                            # Run the code using the current environment's Python
                            result = subprocess.run(
                                [sys.executable, tmp_path], 
                                capture_output=True, text=True, timeout=30
                            )
                            if result.stdout:
                                st.subheader("Output")
                                st.code(result.stdout)
                            if result.stderr:
                                st.subheader("Debug/Errors")
                                st.warning(result.stderr)
                        except Exception as e:
                            st.error(f"Execution Error: {e}")
                        finally:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)

if prompt := st.chat_input("Ask a Qibo question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating code..."):
            response = qa_chain.invoke(prompt)
            answer = response["answer"]
            st.markdown(answer)
            
            with st.expander("Retrieved Documentation (Grounding)"):
                for doc in response["context"]:
                    st.caption(f"File: {doc.metadata.get('file')}")
                    st.text(doc.page_content[:400] + "...")

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()
