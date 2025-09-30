#!/usr/bin/env python3
"""
RAG pipeline using LangChain
- load documents
- split in chunks
- create embeddings
- create vectorstore
- create retrieval QA chain
- interactive Q&A loop
"""

import logging
import subprocess
import textwrap
import shutil
from pathlib import Path
from typing import List

import nbformat

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate




logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_documents(root: str) -> List[Document]:
    """
    Loads documents from a specified directory, including .md, .py, .rst, and .ipynb files, 
    while preserving metadata such as the source file path and cell indices for notebooks.

    Args:
        root (str): The root directory from which to load the documents.

    Returns:
        List[Document]: A list of Document objects, each representing a file or notebook cell 
        with its content and associated metadata.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        nbformat.reader.NotJSONError: If a notebook file is not a valid JSON.
        Exception: For other errors encountered while reading files or processing notebooks.
    """
    root_path = Path(root)
    if not root_path.exists():
        print(f"Data folder does not exist: {root}")
        return []


    print(f"Loading .md files from {root}")
    loader_md = DirectoryLoader(str(root_path), glob=["**/*.md*"], show_progress=True)
    docs_md = loader_md.load()
    print(f"Loaded {len(docs_md)} .md documents")

    print(f"Loading .py files from {root}")
    loader_py = DirectoryLoader(str(root_path), glob=["**/*.py"], show_progress=True)
    docs_py = loader_py.load()
    print(f"Loaded {len(docs_py)} .py documents")

    print(f"Loading .rst files from {root}")
    loader_rst = DirectoryLoader(str(root_path), glob=["**/*.rst"], loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}, show_progress=True)
    docs_rst = loader_rst.load()
    print(f"Loaded {len(docs_rst)} .rst documents")

    # Custom loader for notebooks: extract each cell as a Document with metadata.source, maybe one can choose to load combined cells?
    nb_docs: List[Document] = []
    ipynb_files = list(root_path.rglob("*.ipynb"))
    if ipynb_files:
        print(f"Found {len(ipynb_files)} .ipynb files; extracting cells...")
        for nb_path in ipynb_files:
            try:
                nb = nbformat.read(str(nb_path), as_version=4)
                for idx, cell in enumerate(nb.cells):
                    if cell.cell_type in ("markdown", "code"):
                        content = cell.source
                        meta = {"source": str(nb_path), "cell_index": idx, "cell_type": cell.cell_type}
                        nb_docs.append(Document(page_content=content, metadata=meta))
            except Exception as e:
                print(f"Error reading {nb_path}: {e}")

    all_docs: List[Document] = docs_md + docs_py + docs_rst + nb_docs
    print(f"Total documents loaded (before chunking): {len(all_docs)}")
    return all_docs


def split_docs(docs: List[Document]) -> List[Document]:
    """
    Splits a list of Document objects into smaller chunks while preserving metadata such as the source file path 
    and cell type. Documents are classified and processed based on their file extension or cell type.

    Args:
        docs (List[Document]): A list of Document objects to be split into smaller chunks.

    Returns:
        List[Document]: A list of smaller Document chunks, each with its associated metadata.

    Raises:
        ValueError: If the input list of documents is empty.
        Exception: For errors encountered during the splitting process.
    """
    md_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN, chunk_size=500, chunk_overlap=50)
    py_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=700, chunk_overlap=80)
    rst_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.RST, chunk_size=400, chunk_overlap=50)

    md_docs = []
    py_docs = []
    rst_docs = []

    for d in docs:
        src = d.metadata.get("source", "")
        cell_type = d.metadata.get("cell_type")
        if cell_type == "markdown":
            md_docs.append(d)
        elif cell_type == "code":
            py_docs.append(d)
        elif src.endswith(".md"):
            md_docs.append(d)
        elif src.endswith(".py"):
            py_docs.append(d)
        elif src.endswith(".rst"):
            rst_docs.append(d)
        else:
            # fallback: treat as markdown
            md_docs.append(d)

    print(f"Splitting: {len(md_docs)} .md, {len(py_docs)} .py, {len(rst_docs)} .rst documents")
    md_chunks = md_splitter.split_documents(md_docs) if md_docs else []
    py_chunks = py_splitter.split_documents(py_docs) if py_docs else []
    rst_chunks = rst_splitter.split_documents(rst_docs) if rst_docs else []

    chunks = md_chunks + py_chunks + rst_chunks
    print(f"Total chunks created: {len(chunks)}")
    return chunks


def choose_embeddings():
    """
    Allows the user to interactively choose between HuggingFace or Ollama embeddings.

    The function prompts the user to select an embedding model. If HuggingFace embeddings 
    are selected, the function initializes and returns a HuggingFaceEmbeddings object. 
    If Ollama embeddings are selected, the function initializes and returns an OllamaEmbeddings object. 
    In case of errors or unavailability of the selected model, it falls back to HuggingFace embeddings 
    if possible.

    Returns:
        Embedding model object: An instance of HuggingFaceEmbeddings or OllamaEmbeddings.

    Raises:
        ImportError: If neither HuggingFace nor Ollama embeddings can be imported.
        Exception: For other errors encountered during the initialization of embeddings.
    """

    while True:
        ok = input("Use HuggingFace embeddings? (y/n) [n]: ").strip().lower() or "n"
        if ok == "y":
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
                print("Using HuggingFaceEmbeddings: sentence-transformers/all-MiniLM-L6-v2 (cpu)")
                return emb
            except Exception as e:
                print(f"Error importing HuggingFaceEmbeddings: {e}")
                continue
        elif ok == "n":
            try:
                from langchain_ollama import OllamaEmbeddings
                emb = OllamaEmbeddings(model="all-minilm:22m")
                print("Using OllamaEmbeddings: all-minilm:22m")
                return emb
            except Exception as e:
                print(f"Error importing OllamaEmbeddings: {e}")
                # if Ollama is unavailable, fall back to HF
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
                    print("Fallback to HuggingFaceEmbeddings available")
                    return emb
                except Exception:
                    print("No embedding method available; install langchain_huggingface or langchain_ollama.")
                    raise
        else:
            print("Please enter 'y' or 'n'.")


def build_vectorstore(chunks: List[Document], embedding_model, persist_directory: str, rebuild: bool = False):
    """
    Builds or loads a Chroma vectorstore from a list of document chunks.

    This function creates a new vectorstore if the specified persistent directory does not exist
    or if the `rebuild` flag is set to True. Otherwise, it loads the existing vectorstore from the
    specified directory. The vectorstore is used to store embeddings for efficient similarity search.

    Args:
        chunks (List[Document]): A list of Document objects to be embedded and stored in the vectorstore.
        embedding_model: The embedding model used to generate vector representations of the document chunks.
        persist_directory (str): The directory where the vectorstore will be saved or loaded from.
        rebuild (bool, optional): If True, deletes the existing vectorstore and creates a new one. Defaults to False.

    Returns:
        Chroma: The Chroma vectorstore object, either newly created or loaded from the persistent directory.

    Raises:
        Exception: If there are issues during the creation, saving, or loading of the vectorstore.
    """

    p = Path(persist_directory)
    # Rebuild vectorstore if requested
    if rebuild and p.exists():
        print(f"Rebuild requested: removing persistent folder {persist_directory}")
        shutil.rmtree(p)

    # if the persistent directory exists, load it; otherwise create a new one using Chroma
    if p.exists():
        print(f"Loading existing vectorstore from {persist_directory}")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    else:
        print(f"Creating new Chroma vectorstore at {persist_directory}")
        vectordb = Chroma.from_documents(chunks, embedding_model, persist_directory=persist_directory, )
        try:
            vectordb.persist()
        except Exception:
            pass

    # Print the number of vectors in the store 
    try:
        count = vectordb._collection.count()
        print(f"Vectorstore created/loaded with {count} vectors")
    except Exception:
        print("Unable to read vector count (depends on Chroma version)")
    return vectordb


def choose_ollama_model():
    """
    Allows the user to select an Ollama model interactively.

    This function lists all available Ollama models by executing the `ollama list` command
    and prompts the user to select a model by name. If no input is provided, a default model
    name is used.

    Returns:
        str: The name of the selected Ollama model.

    Raises:
        FileNotFoundError: If the `ollama` command is not found.
        subprocess.SubprocessError: If there is an error executing the `ollama list` command.
    """
    try:
        res = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
        if res.stdout:
            print("\nAvailable Ollama models:\n")
            print(res.stdout)
        else:
            logger.warning("'ollama list' did not return output or Ollama is not installed.")
    except FileNotFoundError:
        logger.warning("'ollama' command not found. Make sure Ollama is installed if you want to use Ollama models.")

    model_name = input("Enter the Ollama model name (default: llama3.2:1b): ").strip() or "llama3.2:1b"
    return model_name


def create_qa_chain(llm, retriever):
    prompt_template = PromptTemplate(input_variables=["context", "question"], template="""
You are an expert in the Qibo quantum computing library.
Use the following context to answer the question. Always include references to file names, cell indices, and functions when relevant.

IMPORTANT:
- If the question is not answerable from the context, respond with a short generic answer or "I don't know".
- Do NOT invent any functions, classes, or methods that do not exist in the Qibo library.
- For simple or conversational questions (like "What is your name?", "How are you?", "What is Qibo?"), answer briefly and clearly even if not in the context. For example, "My name is QiboLLM."

Context:
{context}

Question: {question}

Answer in detail, with code examples if relevant. At the end, include a short 'SOURCES' section listing the files and cell indices used (one per line).
""")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )
    return qa_chain


def interactive_loop(qa_chain, retriever):
    print("\n📚 Ask the Qibo knowledge base! Type '/bye' to exit. Type '/sources' to change source settings.\n")
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() == "/bye":
            break
        # Get the relevant documents (to show sources)
        try:
            relevant_docs = retriever.get_relevant_documents(question)
        except Exception as e:
            logger.error("Error getting relevant documents: %s", e)
            relevant_docs = []

        if relevant_docs:
            print("\n📂 SOURCES (before the answer):")
            seen = set()
            for d in relevant_docs:
                src = d.metadata.get("source", "<inline>")
                cell = d.metadata.get("cell_index")
                if cell is not None:
                    label = f"{src} [cell {cell}]"
                else:
                    label = src
                if label not in seen:
                    print(" -", label)
                    seen.add(label)
        else:
            print("\n📂 SOURCES: no sources found")

        # Call the chain
        try:
            response = qa_chain.invoke({"query": question})
            result = response.get("result", "")
        except Exception as e:
            logger.error("Error running the chain: %s", e)
            continue

        print("\n🤖 Answer:")
        if "```python" in result:
            parts = result.split("```python")
            explanation = parts[0].strip()
            code = parts[1].strip() if len(parts) > 1 else ""

            if explanation:
                print("\n📖 Explanation:")
                print(textwrap.fill(explanation, width=100))

            if code:
                print("\n💻 Python code:")
                print(f"```python\n{code}\n```")
        else:
            print(textwrap.fill(result, width=100))

        # Also print the list of sources at the end if the chain didn't include them
        print("\n")


def main():
    print("=== RAG Qibo - interactive runner ===")

    data_dir = input("Documents directory (default: ./qiboKnow): ").strip() or "./qiboKnow"
    persist_dir = input("Chroma persistent directory (default: ./chroma_qiboKnow): ").strip() or "./chroma_qiboKnow"
    rebuild = input("Rebuild the vectorstore from scratch? (y/N) [if you change embedding model you must rebuild]: ").strip().lower() == "y"
    embedding_model = choose_embeddings()


    vectorstore_path = Path(persist_dir)
    if vectorstore_path.exists() and not rebuild:
        print(f"Loading existing vectorstore from {persist_dir}...")
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding_model) 
    else:
        docs = load_documents(data_dir)
        if not docs:
            logger.critical("No documents loaded, exiting.")
            return

        chunks = split_docs(docs)


        vectordb = build_vectorstore(chunks, embedding_model, persist_dir, rebuild=rebuild)

    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8})

    model_name = choose_ollama_model()

    try:
        from langchain_ollama import OllamaLLM
        llm = OllamaLLM(model=model_name)
    except Exception as e:
        logger.error("Error creating OllamaLLM: %s", e)
        logger.info("Try installing/enabling Ollama or provide a compatible local model.")
        return

    qa_chain = create_qa_chain(llm, retriever)

    interactive_loop(qa_chain, retriever)


if __name__ == "__main__":
    main()
