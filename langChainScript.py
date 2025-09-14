import subprocess
from pathlib import Path
import textwrap

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import Language
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def main():
    
    dir_path = './qiboKnow'  # directory path

    """load data from directory"""

    print('Loading .md files from ', dir_path)
    loader_md = DirectoryLoader(dir_path, glob=["**/*.md"], show_progress=True)  
    docs_md = loader_md.load()
    print(f"Loaded {len(docs_md)} .md documents.")

    print('\nLoading .py files from ', dir_path)
    loader_py = DirectoryLoader(dir_path, glob=["**/*.py"], show_progress=True)
    docs_py = loader_py.load()
    print(f"Loaded {len(docs_py)} .py documents.")

    # add .ipynb support?

    print("\nLoading .rst files from ", dir_path)
    loader_rst = DirectoryLoader(dir_path, glob=["**/*.rst"], loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}, show_progress=True)
    docs_rst = loader_rst.load()
    print(f"Loaded {len(docs_rst)} .rst documents.")

    """split documents in chunks"""
    print("\nSplitting documents into chunks...")
    # Markdown splitter
    md_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN, chunk_size=400, chunk_overlap=40)
    md_docs = md_splitter.create_documents([doc.page_content for doc in docs_md])  
    print(f"\nSplit .md files into {len(md_docs)} chunks.")

    # Python splitter
    py_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=700, chunk_overlap=80)
    python_docs = py_splitter.create_documents([doc.page_content for doc in docs_py]) 
    print(f"Split .py files into {len(python_docs)} chunks.")

    # RST splitter
    rst_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.RST, chunk_size=400, chunk_overlap=50)
    rst_docs = rst_splitter.create_documents([doc.page_content for doc in docs_rst])
    print(f"Split .rst files into {len(rst_docs)} chunks.")

    """embeddings model selection"""
    ok = input("\nuse HuggingFace embeddings (y/n) \nif you choose n, Ollama embedding will be used\n").strip().lower()
    while True:
        if ok == 'y':
            from langchain_huggingface import HuggingFaceEmbeddings
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
            break
        elif ok == 'n':
            from langchain_ollama import OllamaEmbeddings
            embedding_model = OllamaEmbeddings(model="qibot-3b") # to be modified
            break
        else:
            print("please enter y or n")
    

    """create vectorstore"""

    persist_directory = './chroma_qiboKnow'
    if Path(persist_directory).exists():
        print(f"\nVectorstore already exists at {persist_directory}. Loading existing vectorstore...")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    else:
        print(f"\nCreating Chroma vector store at {persist_directory}...")
        vectordb = Chroma.from_documents(md_docs + python_docs + rst_docs, embedding_model, persist_directory=persist_directory)
        print(f"\nChroma vector store created at {persist_directory} with {vectordb._collection.count()} vectors.")

    """istantiate a retriever"""
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8}) # other options: https://python.langchain.com/docs/how_to/vectorstore_retriever/

    """Select Ollama model"""
    print("\nAvailable Ollama models:")
    subprocess.run(["ollama", "list"])
    model_name = input("\nEnter Ollama model name (default: llama3.2:1b): ").strip() or "llama3.2:1b"
    llm = OllamaLLM(model=model_name)

    """Create chain for Q&A"""
    prompt_template = PromptTemplate(input_variables=["context", "question"], template="""
    You are an expert in the Qibo quantum computing library. 
    Use only the following context to answer the question. 
    If you don't know, say you don't know. 
    Always include references to file names and functions.

    Context:
    {context}

    Question: {question}

    Answer in detail, with code examples if relevant.
    """
    )    

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )

    """Interactive loop"""
    print("\n📚 Ask questions about your knowledge base! Type '/bye' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() == "/bye":
            break
        response = qa_chain.invoke({"query": question})
        
        result = response.get("result", "")
        #query = response.get("query", "")
        
        print("\n🤖 **Bot Response:**")
        if "```python" in result:
            # Split the response into explanation and code
            parts = result.split("```python")
            explanation = parts[0].strip()
            code = parts[1].strip() if len(parts) > 1 else ""
            
            if explanation:
                print("\n📖 **Explanation:**")
                print(textwrap.fill(explanation, width=80))
            
            if code:
                print("\n💻 **Python Code:**")
                print(f"```python\n{code}\n```")
        else:
            print(textwrap.fill(result, width=80))
        
        print("\n")




if __name__ == "__main__":
    main()



