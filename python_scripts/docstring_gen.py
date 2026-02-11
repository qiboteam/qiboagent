#!/usr/bin/env python3
"""
RAG pipeline for docstring generation with Hybrid Retrieval and AST-based Python chunking
"""
import logging
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import re
import nbformat
import os
import ast
import hashlib
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ------------------------ UTILITIES ------------------------ #

def load_json_settings(file_path: str) -> dict:
    """Load JSON RAG pipeline settings from the specified file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            settings = json.load(f)
        return settings
    except Exception as e:
        logger.error(f"Error loading JSON settings from {file_path}: {e}")
        return {}

def preprocess_text(text: str) -> List[str]:
    """Preprocess text for BM25 by lowercasing and removing punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()

def safe_relative(path: Path, root: Path) -> str:
    """convert path to relative to root if possible, else return absolute path as string."""
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)

def extract_by_lines(source: str, start: int, end: int) -> str:
    """Extract lines from source between start and end"""
    return "\n".join(source.splitlines()[start - 1:end])

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
    if removed > 0:
        logger.info(f"Removed duplicates: {removed}")
    return unique_docs

def build_vectorstore(chunks: List[Document], embedding_model, persist_directory: str, rebuild: bool = False):
    """ Build a Chroma vectorstore from code and documentation documents, using HuggingFace embeddings. The vectorstore is persisted to disk for future use."""
    p = Path(persist_directory)
    if rebuild and p.exists():
        shutil.rmtree(p)

    if p.exists():
        logger.info("Loading existing VectorStore...")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    else:
        logger.info("Creating new VectorStore...")
        vectordb = Chroma.from_documents(chunks, embedding_model, persist_directory=persist_directory)
    return vectordb


def _extract_class_header(source: str, class_node: ast.ClassDef) -> str:

    class_lines = []
    class_source = ast.get_source_segment(source, class_node)
    if not class_source: return ""
    lines = class_source.splitlines()
    for line in lines:
        class_lines.append(line)
        if line.rstrip().endswith(':'): break
    
    for method in class_node.body:
        if isinstance(method, ast.FunctionDef) and method.name == '__init__':
            init_source = ast.get_source_segment(source, method)
            if init_source:
                init_lines = init_source.splitlines()
                for line in init_lines:
                    class_lines.append(f"    {line}")
                    if line.rstrip().endswith(':'): break
            break
    return "\n".join(class_lines)

# ------------------------ PARSING PYTHON ------------------------ #

def parse_repo(root: str, exclude_file: str = "") -> List[Dict[str, str]]:
    """Parse Python files in the given root directory and extract code items."""
    logger.info(f"Parsing Python files under: {root}")
    root_path = Path(root)
    py_files = list(root_path.rglob("*.py"))
    
    if exclude_file:
        exclude_name = Path(exclude_file).name
        py_files = [f for f in py_files if f.name != exclude_name]
    
    items = []
    for py_file in tqdm(py_files, desc="Parsing repo"):
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
            module_doc = ast.get_docstring(tree)
            module_header = f'"""{module_doc}"""' if module_doc else ""
            rel = safe_relative(py_file, root_path)
            file_tag = f"{rel} {module_header}".strip()

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("_"): continue
                    item = extract_function(node, source, file_tag, None)
                    if item: items.append(item)
                elif isinstance(node, ast.ClassDef):
                    if node.name.startswith("_"): continue
                    class_item = extract_class(node, source, file_tag)
                    if class_item: items.append(class_item)
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

def parse_target_file(file_path: str) -> List[Dict[str, str]]:
    """Parse the target Python file to extract public functions and classes along with their source code. """
    if not Path(file_path).exists():
        logger.error(f"File does not exist: {file_path}")
        return []
    logger.info(f"Parsing target file: {file_path}")
    items = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source)

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('_'): continue
                source_segment = ast.get_source_segment(source, node)
                if source_segment:
                    items.append({"name": node.name, "type": "function", "source": source_segment})
            elif isinstance(node, ast.ClassDef):
                if node.name.startswith('_'): continue
                class_header = _extract_class_header(source, node)
                if class_header:
                    items.append({"name": node.name, "type": "class", "source": class_header})
                for method_node in node.body:
                    if isinstance(method_node, ast.FunctionDef):
                        if method_node.name.startswith('_') and method_node.name != '__init__': continue
                        if method_node.name == '__init__': continue
                        method_source = ast.get_source_segment(source, method_node)
                        if method_source:
                            items.append({
                                "name": f"{node.name}.{method_node.name}",
                                "type": "method",
                                "source": method_source,
                                "class_name": node.name
                            })
        logger.info(f"Found {len(items)} public items in {file_path}")
        return items
    except Exception as e:
        logger.error(f"Failed to parse {file_path}: {e}")
        return []

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
    
    nb_docs = []
    ipynb_files = list(root_path.rglob("*.ipynb"))
    for nb_path in ipynb_files:
        try:
            nb = nbformat.read(str(nb_path), as_version=4)
            for idx, cell in enumerate(nb.cells):
                if cell.cell_type in ("markdown", "code"):
                    content = cell.source
                    meta = {"source_type": "documentation", "content": cell.cell_type, "file": str(nb_path.relative_to(root_path)), "cell_index": idx}
                    nb_docs.append(Document(page_content=content, metadata=meta))
        except Exception as e:
            logger.warning(f"Error reading {nb_path}: {e}")
    
    return docs + nb_docs

# ------------------------ HYBRID RETRIEVER ------------------------ #

class HybridRetriever:
    def __init__(self, docs: List[Document], embedding_vectorstore, bm25_weight: float = 0.5):
        self.docs = docs
        self.vs = embedding_vectorstore
        self.bm25_weight = bm25_weight
        
        logger.info("Indexing documents for Hybrid Retrieval...")
        self.doc_map = {
            self._hash_doc(doc): i 
            for i, doc in enumerate(self.docs)
        }
        
        logger.info("Building BM25 index...")
        self.corpus_tokens = [preprocess_text(doc.page_content) for doc in docs]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def _hash_doc(self, doc: Document) -> str:
        return hashlib.sha256(doc.page_content.strip().encode("utf-8")).hexdigest()

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        query_tokens = preprocess_text(query)
        raw_bm25_scores = self.bm25.get_scores(query_tokens)
        
        embedding_results = self.vs.similarity_search(query, k=len(self.docs))
        
        if len(raw_bm25_scores) > 0:
            max_bm25 = max(raw_bm25_scores) 
            if max_bm25 == 0: max_bm25 = 1.0
            normalized_bm25 = [score / max_bm25 for score in raw_bm25_scores]
        else:
            normalized_bm25 = []

        n_results = len(embedding_results)
        vec_score_map = {}
        
        for rank, doc in enumerate(embedding_results):
            doc_hash = self._hash_doc(doc)
            normalized_rank_score = 1.0 - (rank / n_results)
            vec_score_map[doc_hash] = normalized_rank_score

        final_results = []
        
        for i, doc in enumerate(self.docs):
            score_bm25 = normalized_bm25[i] if i < len(normalized_bm25) else 0.0
            doc_hash = self._hash_doc(doc)
            score_vec = vec_score_map.get(doc_hash, 0.0)
            
            hybrid_score = (
                (self.bm25_weight * score_bm25) + 
                ((1 - self.bm25_weight) * score_vec)
            )
            
            if score_vec > 0 or score_bm25 > 0.5: 
                final_results.append((hybrid_score, doc))

        final_results.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in final_results[:k]]

    def invoke(self, query: str, k: int = 4) -> List[Document]:
        return self.retrieve(query, k=k)

# ------------------------ SETUP ------------------------ #

def initialize_llm(settings: dict):
    provider = settings["llm"]["provider"]
    model_name = settings["llm"]["model_name"]
    api_key = settings["llm"].get("api_key", "")

    if provider == "ollama":
        return OllamaLLM(model=model_name, base_url=settings["llm"]["base_url"], temperature=0.0)
    elif provider == "google_genai":
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.0)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

# ------------------------ GENERATION ------------------------ #

def create_generation_chain(llm):
    """Query the LLM with a prompt that includes the code item and retrieved context to generate a docstring. The prompt template is loaded from an external file for easy customization."""
    
    template_content = """You are an expert in writing docstrings for the Qibo library, a quantum computing framework in Python. 

Your task is to generate clear, correct, and concise **Python docstrings** for Qibo functions, classes, and methods using the given source code and context.

<RULES>
1. Use triple double quotes (\"\"\") for docstrings.
2. Begin the docstring with a short one-line summary describing the function/class/method purpose.
3. Include structured sections for:
    - Args: (or Parameters:)
    - Returns
    - (Optional) Example: only if useful (write .. testcode:: \n <the code example snippet> ).
    - **Do not include sections for Raises**.
    If there are no Args or Returns, omit those sections.
4. When writing math expressions use .rst syntax (e.g. :math:: E = mc^2), when writing values use double backticks (e.g. ``0.5`` or :math:``0.5``). 
5. When Args and Returns contains Qibo-specific types, include cross-references (e.g. Returns:\n    list: Set of one-qubit, :class:`qibo.gates.CNOT`)
6. **Do not include any explanations, markdown, or additional text outside the docstring**.
</RULES>

<ANSWER EXAMPLE> (you must follow this format exactly)

\"\"\" Plot the statistical distribution of measurement results.

Args:
    results (dict): A dictionary containing measurement results where keys are bitstrings
                    and values are their corresponding counts.
    ax (matplotlib.axes.Axes, optional): Matplotlib Axes object to plot on. If None,
                                         a new figure and axes will be created.
    title (str, optional): Title of the plot. Default is " ".
    figsize (tuple, optional): Size of the figure. Default is (8, 6).

Returns:
    matplotlib.axes.Axes: The Axes object containing the plot.
\"\"\"
</ANSWER EXAMPLE>

<YOUR TASK>

Now write docstring for the following code item:

</YOUR TASK>

<CODE ITEM>

Name: {item_name}

Source code:
{direct_source_code}

</CODE ITEM>

<RAG CONTEXT>
{rag_context}
</RAG CONTEXT>
"""
    prompt = ChatPromptTemplate.from_template(template_content)
    
    chain = prompt | llm | StrOutputParser()
    return chain

def generate_docstring(chain, item_name: str, direct_source_code: str, rag_context: str) -> Dict:
    """Generate a docstring for the given code item using the provided chain. The input includes the item name, its source code, and the retrieved context from the RAG pipeline.""" 
    inputs = {
        "item_name": item_name,
        "direct_source_code": direct_source_code,
        "rag_context": rag_context
    }
    logger.info(f"Generating docstring for: {item_name}")
    result = chain.invoke(inputs)
    docstring = _extract_docstring_from_answer(result)
    return {"item_name": item_name, "source_code": direct_source_code, "docstring": docstring, "answer_raw": result}

def _extract_docstring_from_answer(answer: str) -> str:
    if not answer: return ""
    answer = answer.strip()
    answer = re.sub(r'^(Assistant:|Human:|System:|AI:)\s*', '', answer, flags=re.IGNORECASE)
    
    patterns = [r'"""(.*?)"""', r"'''(.*?)'''"]
    for p in patterns:
        match = re.search(p, answer, re.DOTALL)
        if match:
            doc = match.group(1)
            doc = re.sub(r'\n---+\s*$', '', doc.strip())
            return "\n".join([line.rstrip() for line in doc.splitlines()])
            
    return answer.strip()

def write_docstrings_to_file(docstrings: List[Dict[str, str]], output_file: str):
    """Write the generated docstrings back to a Python file, preserving the original code structure. """
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for item in docstrings:
                source_lines = item['source_code'].splitlines()
                if not source_lines: continue
                
                decl_end_idx = 0
                for i, line in enumerate(source_lines):
                    if line.rstrip().endswith(':'):
                        decl_end_idx = i
                        break
                
                for i in range(decl_end_idx + 1):
                    f.write(source_lines[i] + '\n')
                
                indent = '        ' if item.get('type') == 'method' else '    '
                if item['docstring']:
                    f.write(f'{indent}"""\n')
                    for line in item['docstring'].splitlines():
                        f.write(f'{indent}{line}\n')
                    f.write(f'{indent}"""\n')
                f.write(f'{indent}pass\n\n\n')
        logger.info(f"Docstrings written to {output_file}")
    except Exception as e:
        logger.error(f"Failed to write docstrings to file: {e}")

# ------------------------ MAIN ------------------------ #

def main():
    try:
        settings = load_json_settings("../settings_json/settings.json")
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        return

    script_dir = Path(__file__).parent.resolve()   
    base_dir = script_dir.parent                   
    
    # Get docstring involved the file path 
    # Set in settings.json the file path of the Python file you want to generate docstrings for.
    raw_py_path = settings.get("py_file_path", "../docstrings/result_visualization.py")
    if raw_py_path.startswith("../"):
        clean_path = raw_py_path.replace("../", "", 1)
        py_file_path = base_dir / clean_path
    else:
        py_file_path = Path(raw_py_path).resolve()

    if not py_file_path.exists():
        logger.error(f"File does not exist: {py_file_path}")
        return

    output_dir = base_dir / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    py_file_name = py_file_path.stem
    model_name = settings['llm']['model_name'].replace(":", "-") 
    output_file = output_dir / f"{py_file_name}_{model_name}.json"
    py_output_file = output_dir / f"{py_file_name}_{model_name}.py"
    data_dir = settings.get("data_dir", str(base_dir / "qiboKnow"))
    persist_dir = settings.get("persist_dir", str(base_dir / "chroma_qibo_docs"))
    
    logger.info(f"Input file: {py_file_path}")
    logger.info(f"Output dir: {output_dir}")  
    rebuild = settings.get("rebuild", False)

    logger.info("Initializing Embeddings...")
    if settings["embedding_model"]["type"] == "huggingface":
        embedding_model = HuggingFaceEmbeddings(
            model_name=settings["embedding_model"]["model_name"], 
            model_kwargs={"device": "cuda" if "cuda" in os.environ.get("PATH", "") else "cpu"}
        )
    elif settings["embedding_model"]["type"] == "ollama":
        embedding_model = OllamaEmbeddings(model=settings["embedding_model"]["model_name"])
    else:
        raise ValueError("Unsupported embedding model")

    logger.info("Parsing code and documentation...")
    # exclude the target file in order to avoid data contamination
    code_items = parse_repo(data_dir, exclude_file=str(py_file_path))
    code_chunks = create_code_chunks(code_items)
    doc_chunks = create_doc_chunks(data_dir)
    
    all_chunks = code_chunks + doc_chunks
    unique_chunks = deduplicate_documents(all_chunks)

    vectordb = build_vectorstore(unique_chunks, embedding_model, persist_dir, rebuild=rebuild)

    search_k = settings["retriever"].get("search_k", 4)
    logger.info(f"Initializing Hybrid Retriever with k={search_k}...")
    
    retriever = HybridRetriever(
        docs=unique_chunks,
        embedding_vectorstore=vectordb,
        bm25_weight=0.5
    )

    llm = initialize_llm(settings)
    chain = create_generation_chain(llm)

    functions_dict = parse_target_file(str(py_file_path))
    docstring_results = []

    # Generate the docstrings for each function and class in file
    for func in tqdm(functions_dict, desc="Generating docstrings"):
        # One can change the query on order to retrieve different context.
        query = f"How is {func['name']} used in Qibo? {func['name']} {func.get('class_name', '')}"
        relevant_docs = retriever.retrieve(query, k=search_k)
        rag_context = "\n\n".join([d.page_content for d in relevant_docs])
        res = generate_docstring(chain, func['name'], func['source'], rag_context)
        res['type'] = func['type'] 
        docstring_results.append(res)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(docstring_results, f, indent=4)
    
    write_docstrings_to_file(docstring_results, output_file=str(py_output_file))

if __name__ == "__main__":
    main()
