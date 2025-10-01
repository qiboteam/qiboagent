#!/usr/bin/env python3
"""
RAG pipeline for Qibo documentation using LangChain, with auto-scoring of answers.
"""

import logging
import json
import shutil
from pathlib import Path
from typing import List
import re
import tempfile
import subprocess
import sys
import nbformat

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM




logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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


def split_docs(docs: List[Document]) -> List[Document]:
    """Split documents into chunks based on their type (markdown, python, rst)."""
    md_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN, chunk_size=500, chunk_overlap=50)
    py_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=700, chunk_overlap=80)
    rst_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.RST, chunk_size=400, chunk_overlap=50)

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
"""
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )
    return qa_chain


def run_batch(qa_chain, retriever, questions: List[str], output_file: str):
    """Run a batch of questions through the QA chain and save results to a JSON file."""
    results = []
    for q in questions:
        try:
            relevant_docs = retriever.invoke(q)
        except Exception as e:
            logger.error(f"Error retrieving docs for '{q}': {e}")
            relevant_docs = []

        try:
            response = qa_chain.invoke({"query": q})
            answer = response.get("result", "")
        except Exception as e:
            logger.error(f"Error answering '{q}': {e}")
            answer = ""

        sources = []
        for d in relevant_docs:
            src = d.metadata.get("source", "<inline>")
            cell = d.metadata.get("cell_index")
            if cell is not None:
                sources.append(f"{src} [cell {cell}]")
            else:
                sources.append(src)

        results.append({
            "question": q,
            "answer": answer,
            "sources": list(sorted(set(sources)))
        })

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


def auto_scoring(results: list, golden_answers: dict, model_name: str):
    """Score batch answers: execute code, match expected output, detect simple hallucinations, compute pylint score, and grounding count."""
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
            "pylint score": 0
        }
        answer = result["answer"]
        raw_expected_output = golden_answer.get("expected_output", "")
        # Normalize expected_output into a list of non-empty strings
        if isinstance(raw_expected_output, str):
            expected_outputs = [raw_expected_output.strip()] if raw_expected_output.strip() else []
        elif isinstance(raw_expected_output, list):
            expected_outputs = [eo.strip() for eo in raw_expected_output if isinstance(eo, str) and eo.strip()]
        else:
            expected_outputs = []
        code = extract_code(answer)

        if code:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            try:
                proc = subprocess.run(
                    ["python3", tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                stdout = proc.stdout.strip()
                stderr = proc.stderr.strip()
                combined = (stdout + "\n" + stderr).strip()
                #logger.info(f"[{question}] STDOUT len={len(stdout)} STDERR len={len(stderr)}")
                if expected_outputs and any(eo in combined for eo in expected_outputs):
                    score["correctness"] = 1
                # Hallucination signals: invented symbols / modules
                if ("AttributeError" in stderr) or ("NameError" in stderr) or ("ImportError" in stderr) or ("ModuleNotFoundError" in stderr):
                    score["hallucination"] = 0

                score["pylint score"] = get_pylint_score(
                    tmp_path,
                    disable="missing-module-docstring,missing-final-newline" # maybe add import-error
                )

            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout executing code for question: {question}")
            except Exception as e:
                logger.warning(f"Error executing code: {e}")
            finally:
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass

        # Grounding
        sources = result.get("sources", [])
        expected_sources = golden_answer.get("expected_sources", [])
        for src in expected_sources:
            if any(src in s for s in sources):
                score["grounding"] += 1

        score["grounding"] /= max(1, len(expected_sources))  # normalize to [0..1]
        scores.append(score)

    Path("scoring_json").mkdir(exist_ok=True)
    with open(f"scoring_json/scoring_{model_name.replace('/', '_')}.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    return


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


def main():
    settings = load_json_settings("./settings_json/settings.json")
    data_dir = settings["data_dir"] if "data_dir" in settings else "./qiboKnow"
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
    persist_dir = settings["persist_dir"] if "persist_dir" in settings else "./chroma_qiboKnow"
    questions_file = "./settings_json/questions.json"
    output_file = f"./answers_json/answers_{settings['llm']['model_name'].replace('/', '_')}.json"
    golden_file = "./settings_json/golden_answers.json"
    rebuild = settings.get("rebuild", False)

    embedding_model = OllamaEmbeddings(model="all-minilm:22m")

    if Path(persist_dir).exists() and not rebuild:
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    else:
        docs = load_documents(data_dir)
        if not docs:
            logger.critical("No documents loaded, exiting.")
            return
        chunks = split_docs(docs)
        vectordb = build_vectorstore(chunks, embedding_model, persist_dir, rebuild=rebuild)

    retriever = vectordb.as_retriever(search_type=settings["retriever"]["search_type"], search_kwargs={"k": settings["retriever"]["search_k"]})
    
    llm = OllamaLLM(model=settings["llm"]["model_name"])

    qa_chain = create_qa_chain(llm, retriever)

    # Load questions
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = run_batch(qa_chain, retriever, questions, output_file)
    
    with open(golden_file, "r", encoding="utf-8") as f:
        golden_answers = json.load(f)
    auto_scoring(results, golden_answers, settings["llm"]["model_name"])

if __name__ == "__main__":
    main()
