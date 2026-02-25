import streamlit as st
import sys
import os
import subprocess
import tempfile
import glob
from pathlib import Path
import requests
import json

# --- Path Configuration ---
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR / "python_scripts"))

from RAG_hybrid import (
    load_json_settings, initialize_llm, parse_repo, 
    create_code_chunks, create_doc_chunks, deduplicate_documents,
    build_vectorstore, HybridRetriever, create_qa_chain, extract_code
)

st.set_page_config(page_title="Qibo AI Assistant", layout="wide")

st.title("Qibo Quantum RAG Assistant")

# --- Fetch Available Models from Ollama ---
@st.cache_resource
def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            model_names = [model["name"] for model in models_data.get("models", [])]
            return model_names if model_names else ["No models available"]
        else:
            return ["Error: Could not fetch models"]
    except requests.exceptions.RequestException:
        return ["Ollama not running - using default settings"]

# --- Sidebar for Model Selection ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    available_models = get_ollama_models()
    selected_model = st.selectbox(
        "Select LLM Model",
        available_models,
        help="Choose a model from your Ollama instance"
    )
    
    st.info(f"**Current Model:** {selected_model}")
    
    if st.button("🔄 Refresh Models"):
        st.cache_resource.clear()
        st.rerun()

# --- Plot Capture Preamble ---
PLOT_CAPTURE_PREAMBLE = '''
import matplotlib as _mpl
_mpl.use("Agg")
import matplotlib.pyplot as _plt
import atexit as _atexit
import os as _os

_PLOT_DIR = {plot_dir!r}

def _save_all_figures():
    for _i, _num in enumerate(_plt.get_fignums()):
        _fig = _plt.figure(_num)
        _path = _os.path.join(_PLOT_DIR, f"figure_{{_i}}.png")
        _fig.savefig(_path, dpi=150, bbox_inches="tight")
    _plt.close("all")

_original_show = _plt.show
def _patched_show(*args, **kwargs):
    _save_all_figures()
_plt.show = _patched_show

_atexit.register(_save_all_figures)
'''

# --- Backend Initialization ---
@st.cache_resource
def setup_rag(model_name):
    settings = load_json_settings(str(ROOT_DIR / "settings_json" / "settings.json"))
    # Override model name in settings if needed
    settings["model"] = model_name
    llm = initialize_llm(settings)
    
    qibo_dir = ROOT_DIR / "qiboKnow" / "qibo"
    code_root = str(qibo_dir / "src/qibo")
    docs_root = str(qibo_dir / "doc")
    
    parsed_items = parse_repo(code_root)
    all_docs = deduplicate_documents(create_code_chunks(parsed_items) + create_doc_chunks(docs_root))
    
    vect = build_vectorstore(create_code_chunks(parsed_items), create_doc_chunks(docs_root), 
                             persist=str(ROOT_DIR / "kb_chroma"))
    
    retriever = HybridRetriever(all_docs, vect)
    return create_qa_chain(llm, retriever)

qa_chain = setup_rag(selected_model)

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "executed_blocks" not in st.session_state:
    st.session_state.executed_blocks = {}

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Code Execution Logic
        if message["role"] == "assistant":
            code = extract_code(message["content"])
            if code:
                run_key = f"run_{i}"
                if st.button(f"▶️ Run This Qibo Code", key=run_key):
                    plot_dir = tempfile.mkdtemp(prefix="qibo_plots_")
                    preamble = PLOT_CAPTURE_PREAMBLE.format(plot_dir=plot_dir)
                    full_code = preamble + "\n\n" + code

                    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
                        tmp.write(full_code)
                        tmp_path = tmp.name
                    try:
                        result = subprocess.run(
                            [sys.executable, tmp_path], 
                            capture_output=True, text=True, timeout=60
                        )
                        plot_paths = sorted(glob.glob(os.path.join(plot_dir, "figure_*.png")))

                        st.session_state.executed_blocks[run_key] = {
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                            "plot_paths": plot_paths,
                        }
                    except subprocess.TimeoutExpired:
                        st.session_state.executed_blocks[run_key] = {
                            "stdout": "",
                            "stderr": "⏱️ Timeout: execution exceeded 60 seconds.",
                            "plot_paths": [],
                        }
                    except Exception as e:
                        st.session_state.executed_blocks[run_key] = {
                            "stdout": "",
                            "stderr": f"Execution Error: {e}",
                            "plot_paths": [],
                        }
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    st.rerun()

                if run_key in st.session_state.executed_blocks:
                    res = st.session_state.executed_blocks[run_key]
                    if res["stdout"]:
                        st.subheader("📄 Output")
                        st.code(res["stdout"])
                    if res["plot_paths"]:
                        st.subheader("📊 Generated Plots")
                        for j, img_path in enumerate(res["plot_paths"]):
                            if os.path.exists(img_path):
                                st.image(img_path, caption=f"Figure {j + 1}", use_container_width=True)
                                with open(img_path, "rb") as f:
                                    st.download_button(
                                        label=f"⬇️ Download Figure {j + 1}",
                                        data=f,
                                        file_name=f"qibo_plot_{j + 1}.png",
                                        mime="image/png",
                                        key=f"{run_key}_dl_{j}",
                                    )
                    if res["stderr"]:
                        st.subheader("⚠️ Debug/Errors")
                        st.warning(res["stderr"])

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