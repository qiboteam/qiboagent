#!/usr/bin/env python3
"""
Hyperparameter optimization for a Retrieval-Augmented Generation (RAG) pipeline
"""

import logging
import json
import shutil
from pathlib import Path
from typing import List
import re
import tempfile
import subprocess
import datetime
import gc
import argparse
import sys
import nbformat
from tqdm import tqdm
import optuna

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings



logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

for name in ("httpx", "httpcore"):
    logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger(name).propagate = False
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_ollama").setLevel(logging.WARNING)
logging.getLogger("ollama").setLevel(logging.WARNING)


def load_documents(root: str) -> List[Document]:
    """Load documents from the specified root directory, including .md, .py, .rst, and .ipynb files."""
    root_path = Path(root)
    if not root_path.exists():
        logger.error(f"Data folder does not exist: {root}")
        return []

    loader_md = DirectoryLoader(str(root_path), glob=["**/*.md*"], show_progress=True)
    docs_md = loader_md.load()

    loader_py = DirectoryLoader(str(root_path), glob=["**/*.py"], show_progress=True)
    docs_py = loader_py.load()

    loader_rst = DirectoryLoader(str(root_path), glob=["**/*.rst"],
                                 loader_cls=TextLoader,
                                 loader_kwargs={"encoding": "utf-8"},
                                 show_progress=True)
    docs_rst = loader_rst.load()

    nb_docs: List[Document] = []
    ipynb_files = list(root_path.rglob("*.ipynb"))
    for nb_path in ipynb_files:
        try:
            nb = nbformat.read(str(nb_path), as_version=4)
            for idx, cell in enumerate(nb.cells):
                if cell.cell_type in ("markdown", "code"):
                    content = cell.source
                    meta = {"source": str(nb_path), "cell_index": idx, "cell_type": cell.cell_type}
                    nb_docs.append(Document(page_content=content, metadata=meta))
        except Exception as e:
            logger.warning(f"Error reading {nb_path}: {e}")

    all_docs: List[Document] = docs_md + docs_py + docs_rst + nb_docs
    logger.info(f"Total documents loaded (before chunking): {len(all_docs)}")
    return all_docs

def load_json_settings(file_path: str) -> dict:
    """Load JSON RAG pipeline settings from the specified file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            settings = json.load(f)
        return settings
    except Exception as e:
        logger.error(f"Error loading JSON settings from {file_path}: {e}")
        return {}


def split_docs(docs: List[Document],
               md_chunk_size: int = 500, md_chunk_overlap: int = 50,
               py_chunk_size: int = 500, py_chunk_overlap: int = 50,
               rst_chunk_size: int = 500, rst_chunk_overlap: int = 50) -> List[Document]:
    """Split documents into chunks with different parameters for each document type."""
    md_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN, chunk_size=md_chunk_size, chunk_overlap=md_chunk_overlap
    )
    py_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=py_chunk_size, chunk_overlap=py_chunk_overlap
    )
    rst_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.RST, chunk_size=rst_chunk_size, chunk_overlap=rst_chunk_overlap
    )
    md_docs, py_docs, rst_docs = [], [], []

    for d in docs:
        src = d.metadata.get("source", "")
        cell_type = d.metadata.get("cell_type")
        if cell_type == "markdown" or src.endswith(".md"):
            md_docs.append(d)
        elif cell_type == "code" or src.endswith(".py"):
            py_docs.append(d)
        elif src.endswith(".rst"):
            rst_docs.append(d)
        else:
            md_docs.append(d)

    md_chunks = md_splitter.split_documents(md_docs) if md_docs else []
    py_chunks = py_splitter.split_documents(py_docs) if py_docs else []
    rst_chunks = rst_splitter.split_documents(rst_docs) if rst_docs else []

    chunks = md_chunks + py_chunks + rst_chunks
    logger.info(f"Total chunks: {len(chunks)} (md: {len(md_chunks)}, py: {len(py_chunks)}, rst: {len(rst_chunks)})")
    return chunks


def build_vectorstore(chunks: List[Document],
                      embedding_model,
                      persist_directory: str,
                      rebuild: bool = True):
    """Build or load a Chroma vector store from document chunks."""
    p = Path(persist_directory) if persist_directory else None

    if rebuild and p and p.exists():
        shutil.rmtree(p)

    if p and p.exists():
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    elif p:
        vectordb = Chroma.from_documents(chunks, embedding_model, persist_directory=persist_directory)
        try:
            vectordb.persist()
        except Exception:
            logger.debug("Chroma.persist() failed; continuing without explicit persist.",
                         exc_info=True)
    else:
        vectordb = Chroma.from_documents(chunks, embedding_model)

    return vectordb


def create_qa_chain(llm, retriever):
    """Create a RetrievalQA chain with a custom prompt for Qibo-related questions."""
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""

INSTRUCTIONS:
You are an **expert quantum developer** specialized exclusively in the **Qibo quantum computing library**.
Your goal is to provide precise, actionable, and contextually grounded answers to the user's questions.

ANSWER STRUCTURE & FORMATTING:
- If the question is not answerable from the context, respond with a short generic answer or "I don't know".
- Do NOT invent any functions, classes, or methods that do not exist in the Qibo library, always refer to the Qibo documentation.
- Write only a single code block in Python if the answer requires code, do NOT write multiple code blocks or other languages, The single code block must be formatted using Markdown (```python ... ```)..
- When providing code, include a brief explanation of what the code does and why it's the correct approach based on the context.
- Always import the necessary modules from Qibo if you use any of its functions or classes.
- For simple or conversational questions (like "What is your name?", "How are you?", "What is Qibo?"), answer briefly and clearly even if not in the context. For example, "My name is QiboLLM."

Context:
{context}

Question: {question}

EXAMPLE:

Question:

Build a qibo circuit of 1 qubit, add an H gate to it, execute it and save the final state to the file `state.npy`.

Answer:

```python

import numpy as np
from qibo import Circuit, gates

c = Circuit(1)
c.add(gates.H(0))

state = c().state()

with open('state.npy', 'wb') as f:

    np.save(f, state)

```

Explanation:
You can append gates to a Circuit object through the method `add` and execute the circuit simply by calling it `c()`, then you can extract the final state by using the `state` method from the resulting object.

Sources:
- qiboKnow/qibo/doc/source/code-examples/examples.rst
- qiboKnow/qibo/src/qibo/models/circuit.py
- qiboKnow/qibo/src/qibo/gates/gates.py
"""
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )
    return qa_chain

def extract_sources_from_docs(docs: List[Document]) -> List[str]:
    """Extract metadata sources from a list of Document objects used for the answer."""
    sources = []
    for d in docs:
        m = d.metadata or {}
        src = (
            m.get("source")
            or m.get("file_path")
            or m.get("filepath")
            or m.get("path")
            or m.get("document_path")
            or m.get("source_file")
            or m.get("id")
            or "<inline>"
        )
        cell = m.get("cell_index")
        if cell is not None:
            sources.append(f"{src} [cell {cell}]")
        else:
            sources.append(str(src))
    return sorted(set(sources))


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def run_batch(qa_chain, retriever, questions: List[str], output_file: str | None = None, save: bool = True):
    """Run a batch of questions through the QA chain. If save=True, save results to output_file."""
    results = []
    for q in tqdm(questions, desc="Processing questions"):
        try:
            response = qa_chain.invoke({"query": q})
            answer = response.get("result") or response.get("output_text", "")
            docs = response.get("source_documents") or []
        except Exception as e:
            logger.error(f"Error answering '{q}': {e}")
            answer, docs = "", []

        if not docs:
            try:
                docs = getattr(retriever, "get_relevant_documents", retriever.invoke)(q)
            except Exception as e:
                logger.error(f"Error retrieving docs for '{q}': {e}")
                docs = []

        sources = extract_sources_from_docs(docs)

        code = extract_code(answer)
        code_output = {"stdout": "", "stderr": ""}
        if code:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            try:
                proc = subprocess.run([sys.executable, tmp_path],
                                      capture_output=True,
                                      text=True, timeout=30,
                                      check=False)
                code_output["stdout"] = (proc.stdout or "").strip()
                code_output["stderr"] = (proc.stderr or "").strip()
            except subprocess.TimeoutExpired:
                code_output["stderr"] = "Execution timeout (30s)"
            except Exception as e:
                code_output["stderr"] = f"Error executing code: {e}"
            finally:
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass

        results.append({
            "question": q,
            "answer": answer,
            "sources": sources,
            "output": code_output
        })

    if save and output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(results)} results to {output_file}")

    return results



def get_pylint_score(
        
    pyfile_path: str,
    disable: str = "missing-module-docstring,missing-final-newline",
    timeout: int = 30,
) -> float:
    """
    Run pylint on a Python file using the current Python environment (venv) 
    and return a normalized score (0.0..1.0). 
    Returns 0.0 if pylint fails or the score cannot be determined.
    """
    cmd = [
        sys.executable, "-m", "pylint", pyfile_path,
        f"--disable={disable}",
        "--score=y",
        "--output-format=text"
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )

        output = proc.stdout + "\n" + proc.stderr
        match = re.search(r"rated at\s+(-?\d+(?:\.\d+)?)/10", output, re.IGNORECASE)
        if match:
            try:
                rating = float(match.group(1))
                return max(0.0, min(1.0, rating / 10.0))
            except ValueError:
                pass

        for line in output.splitlines():
            if "rated" in line.lower():
                for n in re.findall(r"-?\d+(?:\.\d+)?", line):
                    try:
                        val = float(n)
                        if 0 <= val <= 10:
                            return val / 10.0
                    except ValueError:
                        continue

    except subprocess.TimeoutExpired:
        logger.warning("Pylint timeout for file: %s", pyfile_path)
    except FileNotFoundError:
        logger.warning("Pylint executable not found in current Python environment.")
    except Exception as e:
        logger.warning("Pylint execution failed for file %s: %s", pyfile_path, e)

    return 0.0


def auto_scoring(results: list,
                 golden_answers: dict,
                 model_name: str,
                 save: bool = False,
                 scoring_path: str | None = None):
    """Score batch answers. Se save=False non scrive su disco."""
    scores = []
    for result in results:
        question = result["question"]
        golden_answer = golden_answers.get(question, {})
        score = {
            "model": model_name,
            "question": question,
            "correctness": 0,
            "grounding": 0,
            "hallucination": 1,
            "pylint score": 0.0,
            "error": None
        }
        answer = result["answer"]
        raw_expected_output = golden_answer.get("expected_output", "")
        if isinstance(raw_expected_output, str):
            expected_outputs = [raw_expected_output.strip()] if raw_expected_output.strip() else []
        elif isinstance(raw_expected_output, list):
            expected_outputs = [eo.strip() for eo in raw_expected_output if isinstance(eo, str) and eo.strip()]
        else:
            expected_outputs = []

        code = extract_code(answer)

        stdout = _normalize_text(result.get("output", {}).get("stdout", "")) if result.get("output") else ""
        stderr = _normalize_text(result.get("output", {}).get("stderr", "")) if result.get("output") else ""
        if code and not (stdout or stderr):
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            try:
                proc = subprocess.run([sys.executable, tmp_path],
                                      capture_output=True, text=True,
                                      timeout=30,
                                      check=False)
                stdout = _normalize_text(proc.stdout or "")
                stderr = _normalize_text(proc.stderr or "")
            except subprocess.TimeoutExpired:
                stderr = "Execution timeout (30s)"
            except Exception as e:
                stderr = f"Error executing code: {e}"
            finally:
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass

        combined = (stdout + " " + stderr).strip()
        if stderr:
            score["error"] = stderr

        if expected_outputs and any(_normalize_text(eo) in combined for eo in expected_outputs):
            score["correctness"] = 1

        if any(k in stderr for k in ("AttributeError", "NameError", "ImportError", "ModuleNotFoundError")):
            score["hallucination"] = 0

        if code:
            try:
                with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
                    tmp.write(code)
                    tmp_path = tmp.name
                score["pylint score"] = get_pylint_score(
                    tmp_path,
                    disable="missing-module-docstring,missing-final-newline"
                )
            finally:
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass

        sources = result.get("sources", [])
        expected_sources = golden_answer.get("expected_sources", [])
        for src in expected_sources:
            if any(src in s for s in sources):
                score["grounding"] += 1
        score["grounding"] /= max(1, len(expected_sources))

        scores.append(score)

    if save:
        path = scoring_path or f"scoring_json/scoring_{model_name.replace('/', '_')}.json"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)

    return scores

def run_once(settings: dict, save: bool = False):
    """
    Run a RAG pipeline with the given settings.
    """
    # Prepare data directory
    data_dir = settings.get("data_dir", "./qiboKnow")
    qibo_dir = Path(data_dir) / "qibo"
    if not qibo_dir.exists():
        logger.info("Cloning Qibo repository into %s", qibo_dir)
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/qiboteam/qibo.git", str(qibo_dir)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info("Qibo repository cloned successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone Qibo repository: {e.stderr}")
            sys.exit(1)
    else:
        logger.info("Qibo directory already exists at %s", qibo_dir)

    # configuration
    persist_dir = settings.get("persist_dir", "./chroma_qiboKnow")
    questions_file = settings.get("questions_file", "./settings_json/questions.json")
    golden_file = settings.get("golden_file", "./settings_json/golden_answers.json")
    rebuild = settings.get("rebuild", False)

    #embedding_model = OllamaEmbeddings(model=settings.get("embeddings_model", "all-minilm:22m"))
    embedding_model = HuggingFaceEmbeddings(model_name=settings.get("embeddings_model", "all-MiniLM-L6-v2"),
                                            model_kwargs={'device': 'cpu'})

    print("Using embedding model:", embedding_model)

    # Build or load vectorstore
    if persist_dir and Path(persist_dir).exists() and not rebuild:
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    else:
        # Load and chunk documents
        docs = load_documents(data_dir)
        if not docs:
            logger.critical("No documents loaded, exiting.")
            return []
        
        # Text splitter settings
        ts = settings.get("text_splitter", {})
        md_chunk_size = ts.get("md_chunk_size", 500)
        md_chunk_overlap = ts.get("md_chunk_overlap", 50)
        py_chunk_size = ts.get("py_chunk_size", 500)
        py_chunk_overlap = ts.get("py_chunk_overlap", 50)
        rst_chunk_size = ts.get("rst_chunk_size", 500)
        rst_chunk_overlap = ts.get("rst_chunk_overlap", 50)
        
        chunks = split_docs(
            docs,
            md_chunk_size=md_chunk_size,
            md_chunk_overlap=md_chunk_overlap,
            py_chunk_size=py_chunk_size,
            py_chunk_overlap=py_chunk_overlap,
            rst_chunk_size=rst_chunk_size,
            rst_chunk_overlap=rst_chunk_overlap
        )
        
        vectordb = build_vectorstore(chunks, embedding_model, persist_dir, rebuild=rebuild)

    # Retriever settings
    retriever_settings = settings.get("retriever", {})
    search_type = retriever_settings.get("search_type", "similarity")
    search_k = retriever_settings.get("search_k", 4)
    search_kwargs = {"k": search_k}
    
    # Create retriever
    retriever = vectordb.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )

    # Initialize LLM and QA chain
    llm = OllamaLLM(model=settings["llm"]["model_name"])
    qa_chain = create_qa_chain(llm, retriever)

    # Load questions and golden answers
    try:
        with open(questions_file, "r", encoding="utf-8") as f:
            questions = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load questions from {questions_file}: {e}")
        return []

    try:
        with open(golden_file, "r", encoding="utf-8") as f:
            golden_answers = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load golden answers from {golden_file}: {e}")
        return []

    output_file = f"./answers_json/answers_{settings['llm']['model_name'].replace('/', '_')}.json" if save else None
    results = run_batch(qa_chain, retriever, questions, output_file=output_file, save=save)

    scores = auto_scoring(results, golden_answers, settings["llm"]["model_name"], save=save)
    if vectordb is not None:
        del vectordb
        gc.collect()
    return scores


def extract_code(answer: str) -> str:
    """
    Extract the first Python code block from the answer.
    """
    code_blocks = re.findall(r"```python(.*?)```", answer, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    code_blocks = re.findall(r"```(.*?)```", answer, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    return ""

def objective(trial):
    """Objective function for Optuna hyperparameter optimization."""
    # Always use the same parameter space for all trials
    md_chunk_size = trial.suggest_int("md_chunk_size", 200, 1000, step=100)
    md_chunk_overlap = trial.suggest_int("md_chunk_overlap", 20, 200, step=10)  
    py_chunk_size = trial.suggest_int("py_chunk_size", 200, 1000, step=100)
    py_chunk_overlap = trial.suggest_int("py_chunk_overlap", 20, 200, step=20) 
    rst_chunk_size = trial.suggest_int("rst_chunk_size", 200, 1000, step=100)
    rst_chunk_overlap = trial.suggest_int("rst_chunk_overlap", 20, 200, step=10)
    search_type = trial.suggest_categorical("search_type", ["similarity", "mmr"])
    search_k = trial.suggest_int("search_k", 2, 9)

    # Load base settings from JSON and update with trial parameters
    settings = load_json_settings("./settings_json/hyp_settings.json")
    settings.setdefault("text_splitter", {})
    settings["text_splitter"]["md_chunk_size"] = md_chunk_size
    settings["text_splitter"]["md_chunk_overlap"] = md_chunk_overlap
    settings["text_splitter"]["py_chunk_size"] = py_chunk_size
    settings["text_splitter"]["py_chunk_overlap"] = py_chunk_overlap
    settings["text_splitter"]["rst_chunk_size"] = rst_chunk_size
    settings["text_splitter"]["rst_chunk_overlap"] = rst_chunk_overlap

    # Force rebuild for each trial and disable persistence
    settings["persist_dir"] = None
    settings["rebuild"] = True
    
    # Update retriever settings
    settings.setdefault("retriever", {})
    settings["retriever"]["search_type"] = search_type
    settings["retriever"]["search_k"] = search_k

    # Run the pipeline with current parameters
    scores = run_once(settings, save=False)
    if not scores:
        return 0.0
    
    # Calculate and return average correctness score
    avg_correctness = sum(s["correctness"] for s in scores) / len(scores)
    return avg_correctness

def list_existing_studies(storage):
    """List all existing studies in the storage"""
    try:
        studies = optuna.study.get_all_study_summaries(storage=storage)
        if studies:
            logger.info("Existing studies:")
            for study_summary in studies:
                logger.info(f"  - {study_summary.study_name}: {study_summary.n_trials} trials")
        else:
            logger.info("No existing studies found")
        return studies
    except Exception as e:
        logger.warning(f"Could not list studies: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for RAG")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--study-name", type=str, default="rag-opt", help="Optuna study name")
    parser.add_argument("--resume", action="store_true", help="Resume existing study")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_rag.db", help="Optuna study storage")
    parser.add_argument("--list-studies", action="store_true", default=False, help="List existing studies and exit")
    args = parser.parse_args()

    if args.list_studies:
        list_existing_studies(args.storage)
        return 0

    # Check if resuming and if the storage file exists
    storage = args.storage
    load_if_exists = args.resume
    if load_if_exists and "sqlite:///" in storage:
        db_path = storage.replace("sqlite:///", "")
        if not Path(db_path).exists():
            logger.warning(f"No existing study found at {db_path}. Starting new study.")
            load_if_exists = False

    # Optuna study setup
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
    
    try:
        existing_studies = list_existing_studies(storage)
        existing_studies_names = [study.study_name for study in existing_studies]

        if args.study_name in existing_studies_names and not load_if_exists:
            logger.warning(f"Study '{args.study_name}' already exists. Do you want to overwrite it? [y/n]")
            while True:
                user_input = input()
                if user_input.lower() == "n":
                    logger.info("Exiting without changes.")
                    return 0
                if user_input.lower() == "y":
                    logger.info(f"Overwriting study '{args.study_name}'.")
                    optuna.delete_study(study_name=args.study_name, storage=storage)
                    break
                logger.info("Please enter 'y' or 'n'.")
                    
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=args.study_name,
            storage=storage,
            load_if_exists=load_if_exists
        )
        # Enqueue a baseline trial if no trials exist
        if len(study.trials) == 0:
            study.enqueue_trial({
                "md_chunk_size": 500,
                "md_chunk_overlap": 50,
                "py_chunk_size": 700,
                "py_chunk_overlap": 80,
                "rst_chunk_size": 400,
                "rst_chunk_overlap": 50,
                "search_type": "mmr",
                "search_k": 8
            })
            logger.info("Enqueued baseline trial with fixed parameters")
        
        if load_if_exists and study.trials:
            logger.info(f"Resuming study with {len(study.trials)} existing trials")
            try:
                logger.info(f"Current best value: {study.best_value}")
                logger.info(f"Current best params: {study.best_params}")
            except Exception:
                logger.info("No best trial yet")
                
        logger.info(f"Starting optimization: trials={args.trials} seed={args.seed}")
        pbar = tqdm(total=args.trials, desc="Optuna trials", unit="trial")
        
        # Run optimization with progress bar and checkpointing
        try:
            for _ in range(args.trials):
                try:
                    study.optimize(objective, n_trials=1)
                    try:
                        best_val = study.best_value
                        pbar.set_postfix(best=f"{best_val:.4f}")
                    except Exception:
                        pass
                    pbar.update(1)
                    # Save checkpoint after each trial
                    with open(f"optuna_results_{args.study_name}_latest.json", "w", encoding="utf-8") as f:
                        json.dump({
                            "best_value": study.best_value if hasattr(study, "best_value") else None,
                            "best_params": study.best_params if hasattr(study, "best_params") else None,
                            "completed_trials": len(study.trials),
                            "timestamp": str(datetime.datetime.now())
                        }, f, indent=2, ensure_ascii=False) 
                except KeyboardInterrupt:
                    logger.warning("Optimization interrupted by user!")
                    break
        except KeyboardInterrupt:
            logger.warning("Optimization interrupted by user!")
        finally:
            pbar.close()

        # Save final results
        with open(f"optuna_results_{args.study_name}.json", "w", encoding="utf-8") as f:
            json.dump({
                "best_value": study.best_value if hasattr(study, "best_value") else None,
                "best_params": study.best_params if hasattr(study, "best_params") else None,
                "trials": [
                    {
                        "number": t.number,
                        "value": t.value if t.value is not None else None,
                        "params": t.params,
                        "state": str(t.state)
                    } for t in study.trials
                ]
            }, f, indent=2, ensure_ascii=False)

        try:
            best = study.best_trial
            logger.info("Best value (avg_correctness): %.4f", best.value)
            logger.info("Best params: %s", best.params)
        except Exception:
            logger.warning("No best trial available.")
            
    except Exception as e:
        logger.error(f"Study creation or optimization error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
