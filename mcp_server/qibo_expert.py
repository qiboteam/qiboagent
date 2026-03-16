import sys
import os
from pathlib import Path
from fastmcp import FastMCP

# --- Path Configuration ---
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR / "python_scripts"))

from RAG_hybrid import (
    load_json_settings, initialize_llm, parse_repo,
    create_code_chunks, create_doc_chunks, deduplicate_documents,
    build_vectorstore, HybridRetriever, create_qa_chain,
    load_vectorstore, load_retriever, save_retriever, check_qibo_repo_availability
)

# Initialize MCP Server
mcp = FastMCP("Qibo Expert")

# --- Global State for Lazy Initialization ---
_QA_CHAIN = None

def get_qa_chain():
    """Lazily initialize and return the QA chain."""
    global _QA_CHAIN
    if _QA_CHAIN is not None:
        return _QA_CHAIN

    # Check Qibo repo availability
    status = check_qibo_repo_availability(str(ROOT_DIR))
    if not status:
        raise RuntimeError("Qibo repository not found.")

    settings = load_json_settings(str(ROOT_DIR / "settings_json" / "settings.json"))
    
    # Override LLM settings if needed or use defaults from settings.json
    # In a real scenario, you might want to pass these via environment variables or CLI args.
    model_name = os.environ.get("QIBO_MODEL_NAME", settings["llm"]["model_name"])
    settings["llm"]["model_name"] = model_name
    
    qibo_dir = ROOT_DIR / "qiboKnow" / "qibo"
    code_root = str(qibo_dir / "src/qibo")
    docs_root = str(qibo_dir / "doc")
    retriever_cache_path = str(ROOT_DIR / "retriever_cache.pkl")
    vectorstore_path = str(ROOT_DIR / "kb_chroma")

    # Initialization logic similar to UI
    if settings.get("rebuild", False) or not Path(vectorstore_path).exists():
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
        if retriever is None:
             # Fallback if cache is corrupted
            parsed_items = parse_repo(code_root)
            code_chunks = create_code_chunks(parsed_items)
            doc_chunks = create_doc_chunks(docs_root)
            all_docs = deduplicate_documents(code_chunks + doc_chunks)
            vs = build_vectorstore(code_chunks, doc_chunks, persist=vectorstore_path)
            retriever = HybridRetriever(all_docs, vs)
            save_retriever(retriever, retriever_cache_path)

    llm = initialize_llm(settings)
    _QA_CHAIN = create_qa_chain(llm, retriever)
    return _QA_CHAIN

@mcp.tool()
def ask_qibo_expert(question: str) -> str:
    """
    Asks the Qibo Expert for help with quantum computing software implementation using Qibo.
    It uses a RAG pipeline to provide answers grounded in the Qibo documentation and source code.
    
    Args:
        question: The question about Qibo or quantum computing implementation.
    """
    qa_chain = get_qa_chain()
    # For now, we don't handle history here as it's a single-shot tool call from the CLI
    # But we could implement history if the CLI supports it via some context.
    response = qa_chain.invoke(question)
    
    answer = response.get("answer", "No answer generated.")
    
    # We can also append sources if we want more detail
    docs = response.get("context", [])
    if docs:
        sources = "\n\nSources Used:\n"
        for i, doc in enumerate(docs[:3]): # top 3 sources
            file_name = doc.metadata.get("file", "Unknown")
            sources += f"- {file_name}\n"
        answer += sources
        
    return answer

if __name__ == "__main__":
    # Start the MCP server
    mcp.run()
