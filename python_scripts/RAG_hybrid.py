#!/usr/bin/env python3
"""
RAG pipeline for Qibo documentation using LangChain, with auto-scoring of answers,
integrated with Python parsing and HybridRetriever.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Optional
import ast
import os
import hashlib
import re
import time
import tempfile
import subprocess
import sys
from difflib import SequenceMatcher
from tqdm import tqdm
import nbformat
from rank_bm25 import BM25Okapi

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ------------------------ UTILITIES ------------------------ #

# All hyperparameters are written in the setting file settings.json
def load_json_settings(file_path: str) -> dict:
    """Load JSON RAG pipeline settings from the specified file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            settings = json.load(f)
        return settings
    except Exception as e:
        logger.error(f"Error loading JSON settings from {file_path}: {e}")
        return {}

def initialize_llm(settings: dict):
    """Initialize LLM based on settings configuration."""
    if settings["llm"]["provider"] == "ollama":
        # Be careful with your Ollama server base_url
        return OllamaLLM(model=settings["llm"]["model_name"], base_url=settings["llm"]["base_url"], temperature=0.0)
    elif settings["llm"]["provider"] == "google_genai":
        return ChatGoogleGenerativeAI(model=settings["llm"]["model_name"], temperature=0.0, google_api_key=settings["llm"]["api_key"])
    else:
        raise ValueError(f"Unsupported LLM provider: {settings['llm']['provider']}")
    
def safe_relative(path: Path, root: Path) -> str:
    """convert path to relative to root if possible, else return absolute path as string."""
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)

def extract_by_lines(source: str, start: int, end: int) -> str:
    """Extract lines from source between start and end"""
    return "\n".join(source.splitlines()[start - 1:end])

def preprocess_text(text: str) -> List[str]:
    """Preprocess text for BM25 by lowercasing and removing punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()

def find_def_header(lines: List[str]) -> List[str]:
    "extract function definition including decorators"
    header = []
    paren_balance = 0
    started = False
    for ln in lines:
        s = ln.strip()
        if s.startswith("def ") or s.startswith("async def "):
            started = True
        if started:
            header.append(ln)
            paren_balance += ln.count("(") - ln.count(")")
            if paren_balance <= 0 and ln.rstrip().endswith(":"):
                break
    return header if header else [lines[0]]

def split_markdown_by_headings(text: str) -> List[str]:
    """Split markdown text into chunks based on top-level headings (#)."""
    chunks = []
    current = []
    for line in text.splitlines():
        if line.startswith("#") and current:
            chunks.append("\n".join(current).strip())
            current = []
        current.append(line)
    if current:
        chunks.append("\n".join(current).strip())
    return chunks

# ------------------------ PARSING PYTHON ------------------------ #

def parse_repo(root: str) -> List[Dict[str, str]]:
    """Parse Python files in the given root directory and extract code items."""
    # scan all .py files under root directory
    logger.info(f"Parsing Python files under: {root}")
    root_path = Path(root)
    py_files = list(root_path.rglob("*.py"))
    items = []

    # Parse each file
    for py_file in tqdm(py_files, desc="Parsing repo"):
        try:
            # Extract module header and AST
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
            module_doc = ast.get_docstring(tree)
            module_header = f'"""{module_doc}"""' if module_doc else ""
            rel = safe_relative(py_file, root_path)
            file_tag = f"{rel} {module_header}".strip()

            # Extract classes and functions
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Extract functions
                    if node.name.startswith("_"): continue
                    item = extract_function(node, source, file_tag, None)
                    if item: items.append(item)
                    # Extract classes
                elif isinstance(node, ast.ClassDef):
                    if node.name.startswith("_"): continue
                    class_item = extract_class(node, source, file_tag)
                    if class_item: items.append(class_item)
                    # Extract class methods
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if sub.name.startswith("_"): continue
                            item = extract_function(sub, source, file_tag, node.name)
                            if item: items.append(item)
        except Exception as e:
            logger.warning(f"Skipping {py_file}: {e}")

    logger.info(f"Total code items extracted: {len(items)}")
    return items

def extract_class(node: ast.ClassDef, source: str, file_tag: str) -> Optional[Dict[str, str]]:
    """extract info about class and its __init__ method"""
    try:
        start, end = node.lineno, node.end_lineno
        block = extract_by_lines(source, start, end)
        lines = block.splitlines()
        class_def_line = lines[0]
        cls_doc = ast.get_docstring(node)
        signature_text = class_def_line
        if cls_doc:
            signature_text += f'\n    """{cls_doc}"""'
        init_src = ""
        for sub in node.body:
            if isinstance(sub, ast.FunctionDef) and sub.name == "__init__":
                init_src = extract_by_lines(source, sub.lineno, sub.end_lineno).strip()
        if not init_src: init_src = "# No __init__ method"
        return {
            "file_header": file_tag,
            "item_type": f"class {node.name}",
            "signature": signature_text,
            "source_code": init_src
        }
    except:
        return None

def extract_function(node: ast.FunctionDef, source: str, file_tag: str, parent: Optional[str]) -> Optional[Dict[str, str]]:
    """extract info about function or method"""
    try:
        start, end = node.lineno, node.end_lineno
        full = extract_by_lines(source, start, end)
        lines = full.splitlines()
        decorators = [f"@{ast.get_source_segment(source, d)}" for d in node.decorator_list if ast.get_source_segment(source, d)]
        header_lines = find_def_header(lines)
        signature_parts = decorators + ["\n".join(header_lines)]
        doc = ast.get_docstring(node)
        if doc: signature_parts.append(f'    """{doc}"""')
        signature = "\n".join(signature_parts)
        body = "\n".join(lines[len(header_lines):]).strip()
        item_type = f"method of class {parent}" if parent else "function"
        return {
            "file_header": file_tag,
            "item_type": item_type,
            "signature": signature,
            "source_code": body
        }
    except:
        return None

# ------------------------ CHUNKING ------------------------ #

def create_code_chunks(items: List[Dict[str, str]]) -> List[Document]:
    """Create Document chunks for code items, grouping methods under their classes when possible."""
    docs = []
    class_map = {}
    for item in items:
        if item["item_type"].startswith("class "):
            name = item["item_type"].replace("class ", "")
            class_map[name] = {"class_item": item, "methods": []}
    for item in items:
        if item["item_type"].startswith("method of class "):
            cls = item["item_type"].split("method of class ")[1]
            if cls in class_map: class_map[cls]["methods"].append(item)

    for cls, data in class_map.items():
        citem = data["class_item"]
        methods = data["methods"]
        content = [
            f"# Class {cls}",
            "",
            "## Signature",
            "```python",
            citem["signature"],
            "```",
            "",
            "## Constructor",
            "```python",
            citem["source_code"],
            "```",
        ]
        if methods:
            content.append("## Methods")
            for m in methods:
                sig_line = m["signature"].splitlines()[0]
                content.append(f"- `{sig_line}`")
        docs.append(Document(page_content="\n".join(content),
                             metadata={"source_type": "code", "content": "class", "class_name": cls, "file": citem["file_header"]}))

    for item in [i for i in items if i["item_type"].startswith("method") or i["item_type"]=="function"]:
        if item["item_type"].startswith("method of class "):
            cls = item["item_type"].split("method of class ")[1]
            docs.append(Document(page_content="\n".join([
                f"# Method of {cls}",
                "```python",
                item["signature"],
                "```",
                "",
                "## Body",
                "```python",
                item["source_code"],
                "```"
            ]), metadata={"source_type":"code","content":"method","class_name":cls,"file":item["file_header"]}))
        else:
            docs.append(Document(page_content="\n".join([
                "# Function",
                "```python",
                item["signature"],
                "```",
                "",
                "## Body",
                "```python",
                item["source_code"],
                "```"
            ]), metadata={"source_type":"code","content":"function","file":item["file_header"]}))
    return docs

# Doc chunks are created with the classic splitting method except for markdown files where we split by top-level headings to preserve more context in each chunk.
def create_doc_chunks(root: str) -> List[Document]:
    """Create Document chunks for documentation files"""
    docs = []
    root_path = Path(root)
    md_files = list(root_path.rglob("*.md"))
    rst_files = list(root_path.rglob("*.rst"))
    rst_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

    for md in md_files:
        text = md.read_text(encoding="utf-8")
        for i, c in enumerate(split_markdown_by_headings(text)):
            docs.append(Document(page_content=c, metadata={"source_type":"documentation","content":"markdown","file":str(md.relative_to(root_path)),"index":i}))

    for rst in rst_files:
        text = rst.read_text(encoding="utf-8")
        for i, p in enumerate(rst_splitter.split_text(text)):
            docs.append(Document(page_content=p, metadata={"source_type":"documentation","content":"rst","file":str(rst.relative_to(root_path)),"index":i}))
    return docs

def deduplicate_documents(docs: List[Document]) -> List[Document]:
    """ Deduplicate documents based on the hash of their content. This is important to avoid duplicates between code and doc chunks, especially for docstrings that appear in both code and documentation."""
    seen = set()
    unique_docs = []
    for doc in docs:
        h = hashlib.sha256(doc.page_content.strip().encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_docs.append(doc)
    removed = len(docs) - len(unique_docs)
    logger.info(f"Removed duplicates: {removed}")
    return unique_docs

# ------------------------ VECTORSTORE ------------------------ #

def build_vectorstore(code_docs, doc_docs, persist="./kb_chroma"):
    """ Build a Chroma vectorstore from code and documentation documents, using HuggingFace embeddings. The vectorstore is persisted to disk for future use."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    logger.info("Building vectorstore...")
    vs = Chroma.from_documents(
        code_docs + doc_docs,
        embedding=embeddings,
        persist_directory=persist,
    )
    return vs

# ------------------------ HYBRID RETRIEVER ------------------------ #

class HybridRetriever:
    def __init__(self, docs: List[Document], embedding_vectorstore, bm25_weight: float = 0.5):
        self.docs = docs
        self.vs = embedding_vectorstore
        self.bm25_weight = bm25_weight
        
        # Hash map for documents
        logger.info("Indexing documents for Hybrid Retrieval...")
        self.doc_map = {
            self._hash_doc(doc): i 
            for i, doc in enumerate(self.docs)
        }
        
        # Building BM25 index
        logger.info("Building BM25 index...")
        self.corpus_tokens = [preprocess_text(doc.page_content) for doc in docs]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def _hash_doc(self, doc: Document) -> str:
        """Generate a unique hash for the document based on its content."""
        return hashlib.sha256(doc.page_content.strip().encode("utf-8")).hexdigest()

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        # BM25 Retrieval
        query_tokens = preprocess_text(query)
        raw_bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Vector Search
        # Retrieve ALL documents ordered by semantic similarity
        embedding_results = self.vs.similarity_search(query, k=len(self.docs))
        
        # we could also use MMR like in semantic but since we want to combine the scores in a hybrid way we will do the 
        # re-ranking ourselves based on the original BM25 scores and the rank of the vector search results, 
        # without applying MMR filtering to the vector search results. 
        # This way we can leverage all the information from both retrieval methods and combine them in a more flexible way.
        # max_marginal_relevance_search(query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5) -> list[Document]
        
        # A. Normalization BM25 (0.0 to 1.0)
        max_bm25 = max(raw_bm25_scores) if len(raw_bm25_scores) > 0 else 1.0
        if max_bm25 == 0: max_bm25 = 1.0 # Avoid division by zero
        
        normalized_bm25 = [score / max_bm25 for score in raw_bm25_scores]

        # B. Normalization Vector Scores (based on rank)
        n_results = len(embedding_results)
        vec_score_map = {}
        
        for rank, doc in enumerate(embedding_results):
            doc_hash = self._hash_doc(doc)
            normalized_rank_score = 1.0 - (rank / n_results)
            vec_score_map[doc_hash] = normalized_rank_score

       
        final_results = []
        
        for i, doc in enumerate(self.docs):
            score_bm25 = normalized_bm25[i]
            doc_hash = self._hash_doc(doc)
            score_vec = vec_score_map.get(doc_hash, 0.0)
            
            hybrid_score = (
                (self.bm25_weight * score_bm25) + 
                ((1 - self.bm25_weight) * score_vec)
            )
            
            final_results.append((hybrid_score, doc))

        # Sorting and Returning
        final_results.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in final_results[:k]]

    def invoke(self, query: str, k: int = 4) -> List[Document]:
        return self.retrieve(query, k=k)

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

RAG CONTEXT:\n
{context}"""),
        ("human", "with the provided instructions answer this question:\n{question}")
    ])
    
    context_retriever = RunnableLambda(lambda q: retriever.invoke(q))
    generate = (
        {
            "context": context_retriever | RunnableLambda(lambda docs: "\n\n".join(doc.page_content for doc in docs)),
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

RAG CONTEXT:\n
{context}"""

    results = []
    
    start_from = 48
    for idx, q in enumerate(tqdm(questions[start_from-1:], desc="Processing questions"), start=start_from):
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

# ------------------------ SCORING FUNCTIONS ------------------------ #

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
        path = scoring_path or f"./scoring/scoring_hybrid_json/scoring_{model_name.replace('/', '_')}.json"
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

# ------------------------ MAIN ------------------------ #

def main():
    settings = load_json_settings("./settings_json/settings.json")
    data_dir = settings.get("data_dir", "./qiboKnow")
    qibo_dir = Path(data_dir) / "qibo"
    # Clone Qibo repository if not already present
    if not qibo_dir.exists():
        logger.info("Cloning Qibo repository...")
        subprocess.run(["git","clone","https://github.com/qiboteam/qibo.git",str(qibo_dir)], check=True)
    persist_dir = settings.get("persist_dir", "./kb_chroma")
    questions_file = settings.get("questions_file", "./settings_json/questions_2.json")
    output_file = f"./answers/answers_hybrid_json/answers_{settings['llm']['model_name'].replace('/', '_')}.json"
    golden_file = settings.get("golden_file", "./settings_json/golden_answers_2.json")
    rebuild = settings.get("rebuild", False)

    # LLM
    llm = initialize_llm(settings)

    # Loading and parsing items
    # one can avoid the rebuild by putting this block under the if rebuild condition 
    # and loading the already created vectorstore from disk if it exists,
    # but for simplicity we will just always parse and create the vectorstore in this example.
    code_root = str(qibo_dir / "src/qibo")
    docs_root = str(qibo_dir / "doc")
    parsed_items = parse_repo(code_root)
    code_docs = create_code_chunks(parsed_items)
    doc_docs = create_doc_chunks(docs_root)
    all_docs = deduplicate_documents(code_docs + doc_docs)

    # --- Vectorstore ---
    vect = build_vectorstore(code_docs, doc_docs, persist=persist_dir)

    # --- Hybrid retriever ---
    retriever = HybridRetriever(all_docs, vect, bm25_weight=0.5)

    # --- QA chain ---
    qa_chain = create_qa_chain(llm, retriever)

    # --- Run batch ---
    with open(questions_file,"r",encoding="utf-8") as f:
        questions = json.load(f)
    results = run_batch(qa_chain, retriever, questions, output_file)

    # --- Auto-scoring ---
    with open(golden_file,"r",encoding="utf-8") as f:
        golden_answers = json.load(f)
    scores_file = f"./scoring/scoring_hybrid_json/scoring_{settings['llm']['model_name'].replace('/', '_')}.json"
    scores = auto_scoring(results, golden_answers, settings['llm']['model_name'], save=True, scoring_path=scores_file)

    total_questions = len(scores)
    correct_answers = sum(1 for s in scores if s["correctness"]==1)
    accuracy = correct_answers/total_questions if total_questions>0 else 0.0
    logger.info(f"Model used: {settings['llm']['model_name']}")
    logger.info(f"Accuracy (correctness): {accuracy:.2%} ({correct_answers}/{total_questions})")

if __name__=="__main__":
    main()
