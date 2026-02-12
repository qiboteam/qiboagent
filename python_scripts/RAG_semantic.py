#!/usr/bin/env python3
"""
RAG pipeline for Qibo documentation using LangChain, with auto-scoring of answers. no HybridSearch
"""

import logging
import json
import shutil
from pathlib import Path
from typing import List
import re
import tempfile
import subprocess, requests
import sys
import nbformat
import os
from difflib import SequenceMatcher
os.environ["ANONYMIZED_TELEMETRY"] = "False"


# langchain imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ------------------------ DOCUMENT LOADING AND PREPROCESSING ------------------------ #

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

    loader_rst = DirectoryLoader(str(root_path), glob=["**/*.rst"], loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}, show_progress=True)
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


# In this pipeline we will split documents into chunks based on their type, 
# using different chunk sizes and overlaps for markdown, python, and rst files.
def split_docs(docs: List[Document]) -> List[Document]:
    """Split documents into chunks based on their type (markdown, python, rst)."""
    md_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN, chunk_size=750, chunk_overlap=100)
    py_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=700, chunk_overlap=180)
    rst_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.RST, chunk_size=700, chunk_overlap=60)

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
    logger.info(f"Total chunks created: {len(chunks)}")
    return chunks


def build_vectorstore(chunks: List[Document], embedding_model, persist_directory: str, rebuild: bool = False):
    """Build or load a Chroma vector store from document chunks."""
    p = Path(persist_directory)
    if rebuild and p.exists():
        shutil.rmtree(p)

    if p.exists():
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    else:
        vectordb = Chroma.from_documents(chunks, embedding_model, persist_directory=persist_directory)
        try:
            vectordb.persist()
        except Exception:
            pass
    return vectordb

#------------------------- RAG CHAIN CREATION ------------------------ #

def create_qa_chain(llm, retriever):
    """Create a RAG chain using LCEL (LangChain Expression Language)."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an **expert quantum developer** specialized exclusively in the **Qibo quantum computing library**.
Your goal is to provide precise, actionable, and contextually grounded answers to the user's questions.

INSTRUCTIONS:
- If the question is not answerable from the context, respond with a short generic answer or "I don't know".
- Do NOT invent any functions, classes, or methods that do not exist in the Qibo library, always refer to the Qibo documentation.
- Write only A SINGLE BLOCK CODE in Python if the answer requires code, do NOT write multiple code blocks or other languages. The single code block must be formatted using Markdown (```python ... ```).
- When providing code, include a brief explanation of what the code does and why it's the correct approach based on the context.
- Always import the necessary modules from Qibo if you use any of its functions or classes, always search in documentation for built-in methods and functions to evaluate something (example: for evaluating hamiltonian eigenvalues there is the method eigenvalues()).
- For simple or conversational questions (like "What is your name?", "How are you?", "What is Qibo?"), answer briefly and clearly even if not in the context. For example, "My name is QiboLLM."

EXAMPLE:
Question: Build a qibo circuit of 1 qubit, add an H gate to it, execute it and save the final state to the file `state.npy`.

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

Context:
{context}"""),
        ("human", "with the provided instructions answer this question:\n{question}")
    ])
    

    
    generate = (
        {
            "context": retriever | RunnableLambda(lambda docs: "\n\n".join(doc.page_content for doc in docs)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableLambda(lambda query: {
    "answer": generate.invoke(query),
    "context": retriever.invoke(query)
    })

def run_batch(qa_chain, retriever, questions: List[str], output_file: str):
    """Run a batch of questions through the QA chain and save results to a JSON file."""
    
    system_template = """You are an **expert quantum developer** specialized exclusively in the **Qibo quantum computing library**.
Your goal is to provide precise, actionable, and contextually grounded answers to the user's questions.

INSTRUCTIONS:
- If the question is not answerable from the context, respond with a short generic answer or "I don't know".
- Do NOT invent any functions, classes, or methods that do not exist in the Qibo library, always refer to the Qibo documentation.
- Write only A SINGLE BLOCK CODE in Python if the answer requires code, do NOT write multiple code blocks or other languages. The single code block must be formatted using Markdown (```python ... ```).
- When providing code, include a brief explanation of what the code does and why it's the correct approach based on the context.
- Always import the necessary modules from Qibo if you use any of its functions or classes, always search in documentation for built-in methods and functions to evaluate something (example: for evaluating hamiltonian eigenvalues there is the method eigenvalues()).
- For simple or conversational questions (like "What is your name?", "How are you?", "What is Qibo?"), answer briefly and clearly even if not in the context. For example, "My name is QiboLLM."

EXAMPLE:
Question: Build a qibo circuit of 1 qubit, add an H gate to it, execute it and save the final state to the file `state.npy`.

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

Context:
{context}"""

    results = []
    start_index = 47
    for idx, q in enumerate(tqdm(questions[start_index:], desc="Processing questions"), start=start_index + 1):
        try:
            response = qa_chain.invoke(q)
            answer = response.get("answer", "")
            docs = response.get("context", [])
            
            context = "\n\n".join(doc.page_content for doc in docs)
            system_prompt_filled = system_template.format(context=context)
            
            # PRINT COMPLETE PROMPT TO SEE RAG CONTEXT AND INSTRUCTIONS
            print(f"\n{'='*80}", flush=True)
            print(f"QUESTION {idx}/{len(questions)}", flush=True)
            print(f"{'='*80}", flush=True)
            print(f"\n--- SYSTEM PROMPT ---\n", flush=True)
            print(system_prompt_filled, flush=True)
            print(f"\n{'-'*80}", flush=True)
            print(f"\n--- USER QUESTION ---\n", flush=True)
            print(q, flush=True)
            print(f"\n{'='*80}\n", flush=True)
            
        except Exception as e:
            logger.error(f"Error answering '{q}': {e}")
            answer, docs = "", []

        if not docs:
            try:
                docs = retriever.invoke(q)
            except Exception as e:
                logger.error(f"Error retrieving docs for '{q}': {e}")
                docs = []

        sources = []
        for d in docs:
            src = d.metadata.get("source") or d.metadata.get("file_path") or "<inline>"
            cell = d.metadata.get("cell_index")
            if cell is not None:
                sources.append(f"{src} [cell {cell}]")
            else:
                sources.append(src)
        sources = sorted(set(sources))

        code = extract_code(answer)
        code_output = {"stdout": "", "stderr": ""}
        if code:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            try:
                proc = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=30)
                code_output["stdout"] = proc.stdout.strip()
                code_output["stderr"] = proc.stderr.strip()
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

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(results)} results to {output_file}")
    return results

#------------------------- AUTO-SCORING ------------------------ #

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
            timeout=timeout
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


def _normalize_and_extract(text: str) -> str:
    """
    Normalize text by:
    1. Extracting meaningful content (remove log prefixes, timestamps)
    2. Normalizing whitespace
    3. Standardizing numeric formats
    """
    if not text:
        return ""
    
    # Remove log prefixes
    log_patterns = [
        r'\[.*?\|.*?\|.*?\]:\s*',  # [Qibo x.x.x|INFO|timestamp]:
        r'\[INFO\]\s*',          
        r'\[WARNING\]\s*',          
        r'\[ERROR\]\s*',          
        r'\[DEBUG\]\s*',          
    ]
    for pattern in log_patterns:
        text = re.sub(pattern, '', text)

    # Remove timestamps and dates
    text = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', '', text)

    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text.strip())

    # Standardize complex numbers in arrays
    # 0.        +0.j -> 0.+0.j
    text = re.sub(r'(\d)\s+\+', r'\1+', text)
    text = re.sub(r'\+\s+(\d)', r'+\1', text)

    # Remove spaces around special characters in arrays
    text = re.sub(r'\s*\[\s*', '[', text)
    text = re.sub(r'\s*\]\s*', ']', text)
    text = re.sub(r'\s*j\s*', 'j', text)
    
    return text.strip()


def _fuzzy_contains(expected: str, actual: str, threshold: float = 0.85) -> bool:
    """
    Check if expected is contained in actual using fuzzy matching.
    Returns True if:
    1. Exact substring match
    2. Fuzzy similarity above threshold
    """
    
    expected_norm = _normalize_and_extract(expected)
    actual_norm = _normalize_and_extract(actual)
    
    # Exact substring match 
    if expected_norm in actual_norm:
        return True
    
    # Reverse substring match
    if actual_norm in expected_norm:
        return True

    # Fuzzy matching to handle small differences
    # E.g.: "0.70710678" vs "0.707107"
    ratio = SequenceMatcher(None, expected_norm, actual_norm).ratio()
    return ratio >= threshold


def auto_scoring(results: list, golden_answers: dict, model_name: str, save: bool = False, scoring_path: str | None = None):
    """Score batch answers with robust text comparison. Always executes extracted code."""
    scores = []
    for idx, result in enumerate(results, start=1):
        question = result["question"]
        golden_answer = golden_answers.get(question, {})
        score = {
            "model": model_name,
            "question_number": idx, 
            "question": question,
            "answer": result["answer"],
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

        stdout = ""
        stderr = ""
    
        if code:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            try:
                logger.debug(f"Executing code for question {idx}: {question[:50]}...")
                proc = subprocess.run(
                    [sys.executable, tmp_path], 
                    capture_output=True, 
                    text=True, 
                    timeout=30
                )
                stdout = proc.stdout or ""
                stderr = proc.stderr or ""
                logger.debug(f"Code execution completed for question {idx}")
            except subprocess.TimeoutExpired:
                stderr = "Execution timeout (30s)"
                logger.warning(f"Code execution timeout for question {idx}")
            except Exception as e:
                stderr = f"Error executing code: {e}"
                logger.error(f"Code execution error for question {idx}: {e}")
            finally:
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass
        else:
            stdout = result.get("output", {}).get("stdout", "") if result.get("output") else ""
            stderr = result.get("output", {}).get("stderr", "") if result.get("output") else ""

        combined = (stdout + " " + stderr).strip()
        
        if stderr and not any(k in stderr for k in ("DeprecationWarning", "FutureWarning", "UserWarning")):
            score["error"] = stderr

        if expected_outputs:
            for eo in expected_outputs:
                if _fuzzy_contains(eo, combined, threshold=0.85):
                    score["correctness"] = 1
                    break

        if any(k in stderr for k in ("AttributeError", "NameError", "ImportError", "ModuleNotFoundError")):
            score["hallucination"] = 0

        # Pylint score
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
        if expected_sources:
            for src in expected_sources:
                if any(src in s for s in sources):
                    score["grounding"] += 1
            score["grounding"] /= len(expected_sources)

        scores.append(score)

    if save:
        path = scoring_path or f"./scoring/scoring_semantic_json/scoring_{model_name.replace('/', '_')}.json"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved scoring results to {path}")

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

def check_ollama_llm_availability(model_name: str, base_url: str) -> bool:
    """Check if an Ollama model is available; if not, attempt to pull it."""
    # Check if Ollama server is reachable
    try:
        # get list of available models, ollama api: https://docs.ollama.com/api
        response = requests.get(f"{base_url}/api/tags", timeout=5)
    except (requests.ConnectionError, requests.Timeout):
        logger.error(f"Cannot connect to Ollama at {base_url}. Is the server running?")
        return False

    # Check if model is already downloaded
    model_short = model_name.split(":")[0]
    models = response.json().get("models", [])
    available = [m["name"].split(":")[0] for m in models]
    if model_short in available:
        logger.info(f"Model '{model_name}' is available.")
        return True

    # Attempt to pull the model via CLI
    logger.info(f"Model '{model_name}' not found locally. Pulling...")
    result = subprocess.run(["ollama", "pull", model_name])
    if result.returncode != 0:
        logger.error(f"Invalid model name '{model_name}'. Check available models at https://ollama.com/library")
        return False

    logger.info(f"Model '{model_name}' pulled successfully.")
    return True


def initialize_llm(settings: dict):
    """Initialize LLM based on settings configuration."""
    if settings["llm"]["provider"] == "ollama":
        model_name = settings["llm"]["model_name"]
        base_url = settings["llm"]["base_url"]
        
        # Check if model is available, pull if necessary
        if not check_ollama_llm_availability(model_name, base_url):
            raise RuntimeError(f"Model {model_name} is not available and could not be pulled.")
        
        return OllamaLLM(model=model_name, base_url=base_url, temperature=0.0)
    elif settings["llm"]["provider"] == "google_genai":
        return ChatGoogleGenerativeAI(model=settings["llm"]["model_name"],
                                      google_api_key=settings["llm"]["api_key"],
                                      temperature=0.0,
                                      convert_system_message_to_human=True)
    # Add more providers as needed
    elif settings["llm"]["provider"] == "mistralai":
        return ChatMistralAI(model=settings["llm"]["model_name"], temperature=0.0, api_key=settings["llm"]["api_key"])
    elif settings["llm"]["provider"] == "openai":
        return ChatOpenAI(model_name=settings["llm"]["model_name"], temperature=0.0, openai_api_key=settings["llm"]["api_key"])
    elif settings["llm"]["provider"] == "anthropic":
        return ChatAnthropic(model=settings["llm"]["model_name"], temperature=0.0, anthropic_api_key=settings["llm"]["api_key"])
    else:
        raise ValueError(f"Unsupported LLM provider: {settings['llm']['provider']}")

#------------------------- MAIN ------------------------ #

def main():
    settings = load_json_settings("./settings_json/settings.json")
    data_dir = settings["data_dir"] if "data_dir" in settings else "./qiboKnow"
    qibo_dir = Path(data_dir) / "qibo"
    # Clone Qibo repository if not already present
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
    persist_dir = settings["persist_dir"] if "persist_dir" in settings else "./chroma_qibo_knowledge"
    questions_file = settings.get("questions_file", "./settings_json/questions.json")
    output_file = f"./answers/answers_semantic_json/answers_{settings['llm']['model_name'].replace('/', '_')}.json"
    golden_file = settings.get("golden_file", "./settings_json/golden_answers.json")
    rebuild = settings.get("rebuild", False)
    scores_file = f"./scoring/scoring_semantic_json/scoring_{settings['llm']['model_name'].replace('/', '_')}.json"

    llm = initialize_llm(settings)
    # Initialize embedding model based on settings configuration
    if settings["embedding_model"]["type"] == "huggingface":
        embedding_model = HuggingFaceEmbeddings(model_name=settings["embedding_model"]["model_name"], model_kwargs={"device": "cpu"})
    elif settings["embedding_model"]["type"] == "ollama":
        embedding_model = OllamaEmbeddings(model=settings["embedding_model"]["model_name"])

    # rebuild vectorstore if persist dir does not exist or if rebuild flag is set to True, otherwise load existing vectorstore from disk
    if Path(persist_dir).exists() and not rebuild:
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    else:
        logger.info("Loading and processing documents...")
        docs = load_documents(data_dir)
        if not docs:
            logger.critical("No documents loaded, exiting.")
            return
        chunks = split_docs(docs)
        vectordb = build_vectorstore(chunks, embedding_model, persist_dir, rebuild=rebuild)

    # Create retriever with specified search type and parameters from settings (we used MMR)
    retriever = vectordb.as_retriever(search_type=settings["retriever"]["search_type"], search_kwargs={"k": settings["retriever"]["search_k"],
                                                                                                        "fetch_k": settings["retriever"]["fetch_k"],
                                                                                                        "lambda_mult": settings["retriever"]["mmr_lambda"]})

    print(f"semantic RAG run with {settings['llm']['model_name']}")

    qa_chain = create_qa_chain(llm, retriever)

    # Load questions
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = run_batch(qa_chain, retriever, questions, output_file)
    
    with open(golden_file, "r", encoding="utf-8") as f:
        golden_answers = json.load(f)

    scores = auto_scoring(results, golden_answers, settings["llm"]["model_name"], save=True, scoring_path=scores_file)
        
    total_questions = len(scores)
    correct_answers = sum(1 for score in scores if score["correctness"] == 1)
    accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        
    logger.info(f"Model used: {settings['llm']['model_name']}")
    logger.info(f"Accuracy (correctness): {accuracy:.2%} ({correct_answers}/{total_questions})")

if __name__ == "__main__":
    main()
