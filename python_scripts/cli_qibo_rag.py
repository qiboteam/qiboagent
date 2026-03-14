import streamlit as st # This import is not needed for CLI, but might be present in RAG_hybrid.py for other reasons. If so, it should be ignored for CLI.
import sys
import os
import subprocess
import tempfile
import glob
from pathlib import Path
import requests
import json
import argparse
import logging

# Configure logging for better output control
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Assume ROOT_DIR is the project root when script is run from project root ---
# If run from a subdirectory, this might need adjustment.
ROOT_DIR = Path(__file__).parent.parent.parent
# Adjust sys.path to allow importing from python_scripts and other project modules
if str(ROOT_DIR / "python_scripts") not in sys.path:
    sys.path.append(str(ROOT_DIR / "python_scripts"))
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# --- Import core RAG logic ---
try:
    from RAG_hybrid import (
        load_json_settings, initialize_llm, parse_repo,
        create_code_chunks, create_doc_chunks, deduplicate_documents,
        build_vectorstore, HybridRetriever, create_qa_chain, extract_code
    )
    # This might be needed for the setup_rag function if it's directly callable
    # If setup_rag is defined in RAG_hybrid.py, it should be accessible.
    # If it depends on Streamlit context like st.cache_resource, it might need adaptation.
    # Let's assume for now it's adaptable or we can replicate its logic.
except ImportError as e:
    logging.error(f"Failed to import RAG components: {e}")
    logging.error("Please ensure you are running this script from the project root and all dependencies are installed via Poetry.")
    sys.exit(1)

# --- Plot Capture Preamble (Copied from Streamlit app) ---
PLOT_CAPTURE_PREAMBLE = '''
import matplotlib as _mpl
_mpl.use("Agg")
import matplotlib.pyplot as _plt
import atexit as _atexit
import os as _os
import glob as _glob

_PLOT_DIR = {plot_dir!r}

def _save_all_figures():
    # Ensure the plot directory exists
    _os.makedirs(_PLOT_DIR, exist_ok=True)
    
    # Clear existing figures to avoid duplicates if run multiple times in one session
    _plt.clf() 
    _plt.close('all') # Close all figures to ensure a clean slate

    # Collect existing figures to save. This is a bit of a hack.
    # In a real scenario, we might want to manage figure objects more directly.
    # For now, we rely on matplotlib's internal figure management.
    
    # Store current figure numbers
    current_fig_nums = _plt.get_fignums()
    
    # Re-open figures to ensure they are valid and save them.
    # This is tricky; ideally, we'd have access to the figure objects directly.
    # A more robust approach would be to pass figure objects or modify the functions that create them.
    # For this CLI, we'll attempt to save figures that *might* have been created.
    # A simpler approach for CLI might be to just rely on stdout/stderr and avoid plots initially,
    # or save plots to a predictable temp location.

    # Let's simplify: assume the user wants to save plots from the *current* state
    # and we'll save them to the provided directory.
    
    saved_count = 0
    for i, num in enumerate(_plt.get_fignums()):
        try:
            fig = _plt.figure(num)
            path = _os.path.join(_PLOT_DIR, f"figure_{i}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            saved_count += 1
        except Exception as e:
            logging.error(f"Failed to save figure {num}: {e}")
    
    # Close all figures after saving
    _plt.close('all')
    
    # Return the list of saved plot paths
    return _glob.glob(f"{_PLOT_DIR}/figure_*.png")

# Monkey patch plt.show to capture figures before they are displayed (or not)
_original_show = _plt.show
def _patched_show(*args, **kwargs):
    # When show() is called, it usually displays the figures.
    # For CLI, we want to capture them instead of displaying them.
    # We'll call our save function and then potentially close them.
    saved_files = _save_all_figures()
    # Optionally, return the saved files list or just let the process finish.
    # For CLI, we just need to ensure plots are saved.
    # We don't want to call the original _plt.show() which might try to open a GUI.
    logging.info(f"Captured {len(saved_files)} plots to {_PLOT_DIR}")
    
_plt.show = _patched_show

# Register the save function to be called on exit
# This ensures plots are saved even if the script exits unexpectedly or without calling show()
_atexit.register(_save_all_figures)

'''

# --- Helper Functions for CLI ---

def get_ollama_models(ollama_url="http://localhost:11434"):
    """Fetches available models from the Ollama API."""
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            model_names = [model["name"] for model in models_data.get("models", [])]
            return model_names if model_names else ["No models available"]
        else:
            logging.warning(f"Ollama API returned status {response.status_code}")
            return [f"Error: Could not fetch models (Status: {response.status_code})"]
    except requests.exceptions.RequestException as e:
        logging.warning(f"Could not connect to Ollama at {ollama_url}: {e}")
        return ["Ollama not running or unreachable"]

def initialize_rag_system(model_name, settings_path, ollama_base_url="http://localhost:11434"):
    """Initializes the RAG system, similar to Streamlit's setup_rag."""
    logging.info(f"Initializing RAG system with model: {model_name}")
    try:
        settings = load_json_settings(str(settings_path))
        settings["model"] = model_name # Override model name from settings
        # Ensure Ollama URL is set if model implies Ollama (this might need more sophisticated logic based on 'model' string)
        # For now, we assume model_name implicitly means Ollama if it's from the Ollama list.
        # If the LLM initialization directly uses a URL, it should be passed.
        # Let's adapt initialize_llm if necessary, or assume it handles Ollama integration.
        
        # The streamlit app passes 'settings["model"] = model_name' to initialize_llm.
        # It also uses 'requests.get("http://localhost:11434/api/tags")' to get model names.
        # This implies that if a model name is from Ollama's list, it should be configured for Ollama.
        # If initialize_llm needs the base URL, it should be passed or configured.
        # For simplicity, let's assume initialize_llm can take the model_name and it infers Ollama.
        # If RAG_hybrid.py requires more parameters for Ollama, we'd need to adjust.
        
        # Replicating the logic from RAG_hybrid.py's setup_rag
        llm = initialize_llm(settings, ollama_base_url=ollama_base_url)
        
        qibo_dir = ROOT_DIR / "qiboKnow" / "qibo"
        code_root = str(qibo_dir / "src/qibo")
        docs_root = str(qibo_dir / "doc")
        kb_persist_dir = str(ROOT_DIR / "kb_chroma")
        
        parsed_items = parse_repo(code_root)
        all_docs = deduplicate_documents(create_code_chunks(parsed_items) + create_doc_chunks(docs_root))
        
        # Check if vectorstore needs to be built or if it can be loaded
        # For CLI, it's safer to build it if it doesn't exist or if settings change.
        # For simplicity, we will build it every time for now, unless 'persist' is handled.
        # The Streamlit app uses 'persist=str(ROOT_DIR / "kb_chroma")', so it should persist.
        vect = build_vectorstore(create_code_chunks(parsed_items), create_doc_chunks(docs_root), 
                                 persist=kb_persist_dir)
        
        retriever = HybridRetriever(all_docs, vect)
        qa_chain = create_qa_chain(llm, retriever)
        
        logging.info("RAG system initialized successfully.")
        return qa_chain, kb_persist_dir # Return persist dir for potential plotting
        
    except Exception as e:
        logging.error(f"Error initializing RAG system: {e}", exc_info=True)
        return None, None

def execute_code_block(code, plot_dir, timeout=60):
    """Executes a Python code block, capturing stdout, stderr, and matplotlib plots."""
    logging.info(f"Executing code block in {plot_dir}")
    
    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)
    
    # Prepare preamble for plot capture
    # The PLOT_CAPTURE_PREAMBLE expects a 'plot_dir' variable to be formatted into it.
    preamble = PLOT_CAPTURE_PREAMBLE.format(plot_dir=plot_dir)
    full_code = preamble + "

" + code

    # Create a temporary Python file to execute
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp_file:
        tmp_file.write(full_code)
        tmp_path = tmp_file.name

    stdout_str, stderr_str = "", ""
    plot_paths = []
    exit_code = -1
    
    try:
        # Use sys.executable to ensure the same Python interpreter is used
        process = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=timeout
        )
        stdout_str = process.stdout
        stderr_str = process.stderr
        exit_code = process.returncode
        
        # After execution, check for saved plots
        # We need to find files matching the pattern in the plot_dir
        # The _save_all_figures function in preamble should have saved them.
        # We need to check this AFTER the subprocess completes.
        # The prompt originally used glob.glob, let's stick to that.
        plot_paths = sorted(_glob.glob(os.path.join(plot_dir, "figure_*.png")))
        
        logging.info(f"Code execution finished. Exit code: {exit_code}, Plots found: {len(plot_paths)}")
        
    except subprocess.TimeoutExpired as e:
        stderr_str = f"⏱️ Timeout: execution exceeded {timeout} seconds."
        logging.error(stderr_str)
    except Exception as e:
        stderr_str = f"Execution Error: {e}"
        logging.error(f"Error during code execution: {e}", exc_info=True)
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    return stdout_str, stderr_str, plot_paths, exit_code

# --- Main CLI Logic ---
def main():
    parser = argparse.ArgumentParser(description="Qibo Quantum RAG Assistant CLI")
    parser.add_argument(
        "--model",
        type=str,
        help="LLM model name to use (e.g., 'llama3', 'mistral'). If not provided, will prompt user."
    )
    parser.add_argument(
        "--settings",
        type=str,
        default=str(ROOT_DIR / "settings_json" / "settings.json"),
        help="Path to the settings JSON file."
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Base URL for the Ollama API."
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not attempt to execute code that generates plots."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging."
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=str(ROOT_DIR / "kb_chroma"),
        help="Directory to persist the vector store. Will be created if it doesn't exist."
    )
    
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # --- Model Selection ---
    selected_model = args.model
    if not selected_model:
        logging.info("Fetching available Ollama models...")
        available_models = get_ollama_models(args.ollama_url)
        if "Error" in available_models[0] or "No models available" in available_models[0]:
            logging.error(f"Could not find Ollama models: {available_models[0]}")
            logging.info("Please ensure Ollama is running and models are pulled, or specify a model using --model.")
            sys.exit(1)
        
        print("Available LLM Models:")
        for i, model in enumerate(available_models):
            print(f"{i+1}. {model}")
        
        while True:
            try:
                choice = input("Select a model number or enter model name: ")
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(available_models):
                        selected_model = available_models[idx]
                        break
                    else:
                        print("Invalid number. Please try again.")
                elif choice in available_models:
                    selected_model = choice
                    break
                else:
                    print(f"'{choice}' is not a valid model name or number. Please try again.")
            except EOFError: # Handle Ctrl+D
                print("
Exiting.")
                sys.exit(0)
            except KeyboardInterrupt: # Handle Ctrl+C
                print("
Exiting.")
                sys.exit(0)
    
    logging.info(f"Using model: {selected_model}")

    # --- Initialize RAG System ---
    qa_chain, vector_store_persist_dir = initialize_rag_system(
        selected_model, 
        args.settings, 
        args.ollama_url
    )

    if qa_chain is None:
        logging.error("Failed to initialize QA chain. Exiting.")
        sys.exit(1)

    # --- Create temporary directory for plots if not running with --no-plots ---
    plot_dir = None
    if not args.no_plots:
        try:
            plot_dir = tempfile.mkdtemp(prefix="qibo_cli_plots_", dir=ROOT_DIR / "tmp") # Use project's tmp dir
            logging.info(f"Plots will be saved to: {plot_dir}")
        except Exception as e:
            logging.warning(f"Could not create temporary directory for plots: {e}. Plots will not be saved.")
            args.no_plots = True # Disable plots if temp dir creation fails

    # --- Interactive Chat Loop ---
    print("
Qibo Quantum RAG Assistant CLI (type 'quit' or 'exit' to end)")
    print("--------------------------------------------------------------")
    
    while True:
        try:
            user_prompt = input("Ask a Qibo question: ")
            if user_prompt.lower() in ["quit", "exit"]:
                break

            if not user_prompt.strip():
                continue
            
            logging.info(f"User prompt: {user_prompt}")
            
            # Invoke the QA chain
            # The Streamlit app invokes it with prompt, and expects {'answer': ..., 'context': ...}
            try:
                response = qa_chain.invoke({"input": user_prompt}) # Assuming chain expects {"input": ...}
                answer = response.get("answer", "No answer found.")
                context_docs = response.get("context", [])

                print(f"
Assistant:
{answer}")

                if context_docs:
                    print("
--- Retrieved Documentation (Grounding) ---")
                    for i, doc in enumerate(context_docs):
                        # Accessing metadata might vary based on how docs are structured
                        # Assuming doc.metadata['file'] or doc.metadata['source']
                        source_file = doc.metadata.get('file', doc.metadata.get('source', 'Unknown Source'))
                        print(f"[{i+1}] Source: {source_file}")
                        # Truncate content for display
                        content_preview = doc.page_content.strip().replace('
', ' ')
                        if len(content_preview) > 200:
                            content_preview = content_preview[:200] + "..."
                        print(f"    Content: {content_preview}")
                    print("-------------------------------------------")

                # --- Code Execution ---
                # Check if the answer contains code blocks that can be executed
                code_blocks = extract_code(answer)
                if code_blocks and not args.no_plots:
                    run_code_choice = input("Do you want to execute the extracted Qibo code block? (y/N): ")
                    if run_code_choice.lower() == 'y':
                        # Assume the first code block is the one to run for simplicity in CLI
                        code_to_run = code_blocks[0] 
                        
                        stdout, stderr, plots, exit_code = execute_code_block(code_to_run, plot_dir)
                        
                        if stdout:
                            print("
--- Code Execution Output (stdout) ---")
                            print(stdout)
                        if stderr:
                            print("
--- Code Execution Errors/Debug (stderr) ---")
                            print(f"⚠️ {stderr}")
                        if plots:
                            print(f"
--- Generated Plots saved to: {plot_dir} ---")
                            for i, plot_path in enumerate(plots):
                                print(f"  - {os.path.basename(plot_path)}")
                            print("--------------------------------------------")
                            
                        if exit_code != 0:
                            print(f"Code execution finished with errors (exit code: {exit_code}).")
                        else:
                            print("Code executed successfully.")

            except Exception as e:
                logging.error(f"Error during RAG chain invocation or response processing: {e}", exc_info=True)
                print(f"
An error occurred: {e}")
                
        except (EOFError, KeyboardInterrupt):
            print("
Exiting.")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
            print(f"
An unexpected error occurred: {e}")
            break

    # --- Cleanup ---
    if plot_dir and os.path.exists(plot_dir) and not args.no_plots:
        logging.info(f"Attempting to clean up temporary plot directory: {plot_dir}")
        try:
            # Optionally, prompt user before deleting plots, or just leave them
            # For CLI, often leaving them in a designated temp dir is fine.
            # If we want to delete them:
            # import shutil
            # shutil.rmtree(plot_dir)
            # logging.info(f"Temporary plot directory {plot_dir} removed.")
            pass # Keep plots for inspection in the temp dir
        except Exception as e:
            logging.warning(f"Could not clean up plot directory {plot_dir}: {e}")

    print("
Qibo RAG CLI session ended.")

if __name__ == "__main__":
    main()
