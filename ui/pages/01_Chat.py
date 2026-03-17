import streamlit as st
import sys
import os
import subprocess
import tempfile
import glob
from pathlib import Path
import requests
import time
from htbuilder.units import rem
from htbuilder import div, styles

# --- Path Configuration ---
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR / "python_scripts"))

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
MEMORY_WINDOW_DEFAULT = 6

from RAG_hybrid import (
    load_json_settings, initialize_llm, parse_repo,
    create_code_chunks, create_doc_chunks, deduplicate_documents,
    build_vectorstore, HybridRetriever, create_qa_chain, extract_code,
    load_vectorstore, load_retriever, save_retriever, check_qibo_repo_availability
)

st.set_page_config(page_title="Qibo AI Assistant", layout="wide", page_icon="⚛️")

# --- Fetch Available Models from Ollama ---
@st.cache_resource(show_spinner="Searching Ollama models ...")
def get_ollama_models():
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if response.status_code == 200:
            models_data = response.json()
            model_names = [model["name"] for model in models_data.get("models", [])]
            return model_names if model_names else []
        return []
    except requests.exceptions.RequestException:
        try:
            subprocess.Popen(["ollama", "serve"])
            time.sleep(3)
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                model_names = [model["name"] for model in models_data.get("models", [])]
                return model_names if model_names else []
        except Exception:
            pass
        return []

# --- Backend Initialization ---
@st.cache_resource(show_spinner="Loading RAG database...")
def get_retriever():
    try:
        status = check_qibo_repo_availability(str(ROOT_DIR))
        if not status:
            st.error("Qibo repository not found.")
    except Exception as e:
        st.error(f"Error checking Qibo repository: {e}")
        st.stop()

    
    settings = load_json_settings(str(ROOT_DIR / "settings_json" / "settings.json"))
    qibo_dir = ROOT_DIR / "qiboKnow" / "qibo"
    code_root = str(qibo_dir / "src/qibo")
    docs_root = str(qibo_dir / "doc")
    retriever_cache_path = str(ROOT_DIR / "retriever_cache.pkl")
    vectorstore_path = str(ROOT_DIR / "kb_chroma")

    if settings.get("rebuild", True):
        parsed_items = parse_repo(code_root)
        code_chunks = create_code_chunks(parsed_items)
        doc_chunks = create_doc_chunks(docs_root)
        all_docs = deduplicate_documents(code_chunks + doc_chunks)
        vs = build_vectorstore(code_chunks, doc_chunks, persist=vectorstore_path)
        retriever = HybridRetriever(all_docs, vs)
        save_retriever(retriever, retriever_cache_path)
    else:
        vs = load_vectorstore(vectorstore_path)
        retriever = load_retriever(retriever_cache_path, vs)
        if retriever is None or vs is None:
            parsed_items = parse_repo(code_root)
            code_chunks = create_code_chunks(parsed_items)
            doc_chunks = create_doc_chunks(docs_root)
            all_docs = deduplicate_documents(code_chunks + doc_chunks)
            vs = build_vectorstore(code_chunks, doc_chunks, persist=vectorstore_path)
            retriever = HybridRetriever(all_docs, vs)
            save_retriever(retriever, retriever_cache_path)
    return retriever

def update_qa_chain(model_name):
    settings = load_json_settings(str(ROOT_DIR / "settings_json" / "settings.json"))
    settings["llm"]["model_name"] = model_name
    llm = initialize_llm(settings)
    retriever = get_retriever()
    return create_qa_chain(llm, retriever)

# --- Context Window Utilities ---
def get_model_context_window(model_name: str) -> int:
    """Retrieve the context window size for the given model from Ollama."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/show",
            json={"name": model_name},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            model_info = data.get("model_info", {})
            for key, value in model_info.items():
                if "context_length" in key.lower():
                    return int(value)
            params = data.get("parameters", "")
            for line in params.splitlines():
                if "num_ctx" in line:
                    return int(line.split()[-1])
    except Exception:
        pass
    return 4096  # fallback default

def estimate_token_count(text: str) -> int:
    """Rough token count estimate: ~4 characters per token on average."""
    return len(text) // 4

def build_prompt_with_history(prompt: str, messages: list, memory_window: int) -> str:
    """
    Enriches the current question with the most recent conversation history with first user message pinned.
    This text is passed as {question} in the RAG_hybrid.py prompt template.
    The chain system prompt remains unchanged.
    """
    recent_messages = messages[-(memory_window * 2):]
    first_user_msg = next((msg for msg in messages if msg["role"] == "user"), None)

    if first_user_msg is not None and first_user_msg not in recent_messages:
        history = [first_user_msg] + recent_messages
    else:
        history = recent_messages

    if not history:
        return prompt
    

    history_text = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n\n"
    return (
        f"Conversation history so far:\n\n"
        f"{history_text}"
        f"Current question: {prompt}\n\n"
    )

def get_context_usage(messages: list, memory_window: int, context_window: int) -> dict:
    """Calculate the estimated context window usage based on current history."""

    recent_messages = messages[-(memory_window * 2):]
    first_user_msg = next((msg for msg in messages if msg["role"] == "user"), None)

    if first_user_msg is not None and not any(msg is first_user_msg for msg in recent_messages):
        history = [first_user_msg] + recent_messages
    else:
        history = recent_messages

    history_text = " ".join(m["content"] for m in history)
    used_tokens = estimate_token_count(history_text)
    percentage = min((used_tokens / context_window) * 100, 100)
    return {
        "used": used_tokens,
        "total": context_window,
        "percentage": percentage,
    }

# --- Global State Initialization ---
if "model_confirmed" not in st.session_state:
    st.session_state.model_confirmed = False

available_models = get_ollama_models()

# Failsafe if Ollama is unreachable
if not available_models:
    st.error("Ollama is starting or no models were found. Please wait a moment and click Refresh.")
    if st.button("🔄 Refresh Models List"):
        get_ollama_models.clear()
        st.rerun()
    st.stop()

# --- Welcome Screen (Pre-Chat) ---
if not st.session_state.model_confirmed:
    st.html(div(style=styles(font_size=rem(5), line_height=1))["⚛️"])

    title_row = st.container(horizontal=True, vertical_alignment="bottom")
    with title_row:
        st.title("Qibo Quantum RAG Assistant", anchor=False, width="stretch")

    st.info("Please select a model to initialize the workspace.")

    with st.container():
        selected_initial_model = st.selectbox("Available Ollama Models", available_models)

        SUGGESTIONS = {
            "What is Qibo?": (
                "What is Qibo, what is it great at, and what can I do with it?"
            ),
            "How do i build a circuit?": (
                "How do I build a basic quantum circuit in Qibo? Show me an example."
            ),
            "How can i switch backends'": (
                "How do i set numpy backend and complex64 as simulation data type in Qibo library?"
            ),
            "How do i use VQE?": (
                "How do I use the Variational Quantum Eigensolver (VQE) in Qibo?"
            ),
        }

        selected_suggestion = st.pills(
            label="Examples",
            label_visibility="collapsed",
            options=SUGGESTIONS.keys(),
            key="selected_suggestion_welcome",
        )

    if st.button("🚀 Start Chat", type="primary"):
        st.session_state.selected_model = selected_initial_model
        with st.spinner("Initializing RAG pipeline ..."):
            st.session_state.qa_chain = update_qa_chain(selected_initial_model)
        st.session_state.context_window_size = get_model_context_window(selected_initial_model)
        st.session_state.memory_window = MEMORY_WINDOW_DEFAULT
        st.session_state.model_confirmed = True
        if selected_suggestion:
            st.session_state.pending_suggestion = SUGGESTIONS[selected_suggestion]
        st.rerun()
    st.stop()

# --- Sidebar ---
def on_model_change():
    new_model = st.session_state.model_dropdown
    st.session_state.selected_model = new_model
    st.session_state.qa_chain = update_qa_chain(new_model)
    st.session_state.context_window_size = get_model_context_window(new_model)

with st.sidebar:
    st.header("⚙️ Configuration")

    try:
        current_index = available_models.index(st.session_state.selected_model)
    except ValueError:
        current_index = 0

    st.selectbox(
        "Select LLM Model",
        available_models,
        index=current_index,
        key="model_dropdown",
        on_change=on_model_change,
        help="Choose a model from your Ollama instance"
    )
    st.caption(f"**Current Model:** {st.session_state.selected_model}")

    # Initialise context window if not already set
    if "context_window_size" not in st.session_state:
        st.session_state.context_window_size = get_model_context_window(
            st.session_state.selected_model
        )

    # --- Memory Window Slider ---
    st.divider()
    st.subheader("🧠 Memory Settings")
    memory_window = st.slider(
        "Memory Window (exchanges)",
        min_value=1,
        max_value=20,
        value=st.session_state.get("memory_window", MEMORY_WINDOW_DEFAULT),
        step=1,
        help="Number of question/answer exchanges to include as conversation history"
    )
    st.session_state.memory_window = memory_window
    active_exchanges = len(st.session_state.get("messages", [])) // 2
    st.caption(f"💬 Exchanges in memory: **{min(active_exchanges, memory_window)}/{memory_window}**")

    # --- Context Window Monitor ---
    st.divider()
    st.subheader("📊 Context Window Monitor")
    ctx_size = st.session_state.context_window_size
    st.caption(f"**Model context window:** `{ctx_size:,}` tokens")

    if st.session_state.get("messages"):
        usage = get_context_usage(
            st.session_state.messages,
            memory_window,
            ctx_size
        )
        pct = usage["percentage"]

        if pct < 60:
            color, status = "🟢", "OK"
        elif pct < 80:
            color, status = "🟡", "Warning"
        else:
            color, status = "🔴", "Critical"

        st.progress(pct / 100, text=f"{color} {status}: {pct:.1f}% used")
        st.caption(f"~{usage['used']:,} / {usage['total']:,} estimated tokens")

        if pct >= 80:
            st.warning(
                "⚠️ Context window almost full! "
                "Consider reducing the **Memory Window** or resetting the session."
            )
    else:
        st.progress(0.0, text="🟢 OK: 0% used")
        st.caption("_No messages in memory yet_")

    st.divider()

    if st.button("🔄 Refresh Models List"):
        get_ollama_models.clear()
        st.rerun()

    if st.button("🛑 Reset Session"):
        st.session_state.clear()
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

qa_chain = st.session_state.qa_chain

# --- Header ---
st.html(div(style=styles(font_size=rem(5), line_height=1))["⚛️"])

title_row = st.container(horizontal=True, vertical_alignment="bottom")

with title_row:
    st.title("Qibo Quantum RAG Assistant", anchor=False, width="stretch")

    def clear_conversation():
        keys_to_clear = ["messages", "executed_blocks", "pending_suggestion"]
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]

    st.button(
        "Restart",
        icon=":material/refresh:",
        on_click=clear_conversation,
    )

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "executed_blocks" not in st.session_state:
    st.session_state.executed_blocks = {}

# Render conversation history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.container()  # Fix ghost message bug.
        st.markdown(message["content"])

        if message["role"] == "assistant":
            if "context_docs" in message and message["context_docs"]:
                with st.expander("🔍 Retrieved Documentation (Grounding)"):
                    for doc in message["context_docs"]:
                        st.caption(f"📄 File: {doc['file']}")
                        st.text(doc['content'] + "...")

            code = extract_code(message["content"])
            if code:
                run_key = f"run_{i}"
                if st.button("▶️ Run This Qibo Code", key=run_key):
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
                            "stderr": f"Execution error: {e}",
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
                        st.subheader("⚠️ Debug / Errors")
                        st.warning(res["stderr"])

# --- New user message ---
# Handle suggestion carried over from the welcome screen
pending = st.session_state.pop("pending_suggestion", None)
user_message = st.chat_input("Ask a Qibo question...") or pending

if user_message:
    user_message = user_message.replace("$", r"\$")

    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        st.container()  # Fix ghost message bug.

        with st.spinner("Researching..."):
            # Enrich the question with conversation history.
            # The system prompt defined in create_qa_chain() remains unchanged.
            enriched_prompt = build_prompt_with_history(
                user_message,
                st.session_state.messages[:-1],  # exclude the message just appended
                st.session_state.get("memory_window", MEMORY_WINDOW_DEFAULT)
            )

        with st.spinner("Thinking..."):
            response = qa_chain.invoke(enriched_prompt)

        with st.container():
            answer = response["answer"]
            st.markdown(answer)

            context_docs = []
            for doc in response.get("context", []):
                file_path = doc.metadata.get("file", doc.metadata.get("source", "Unknown File"))
                context_docs.append({
                    "file": file_path,
                    "content": doc.page_content[:400]
                })

            if context_docs:
                with st.expander("🔍 Retrieved Documentation (Grounding)"):
                    for doc in context_docs:
                        st.caption(f"📄 File: {doc['file']}")
                        st.text(doc['content'] + "...")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "context_docs": context_docs
            })
            st.rerun()