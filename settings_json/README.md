## Configuration - settings.json

### General Settings
- **data_dir**: Knowledge-Base Directory
- **persist_dir**: ChromaDB Persistence Directory (vector-store directory)
- **rebuild**: Rebuild the vector store from the knowledge base

### Embedding Model
- **type**: Embedding model type: "huggingface" or "Ollama"
- **model_name**: Embedding model name

### LLM
- **provider**: LLM provider
- **model_name**: LLM model name
- **base_url**: Base URL for the OLLAMA API (only for local models like Ollama)
- **api_key**: LLM API key for propietary models
- **reasoning**: enable model reasoning

### Workflow
- **start_from_phase**: Starting phase (example: with 2 you will start from the second agent)
- **skip_cleanup**: Clean the generated code files when restart

### Retriever
- **search_type**: Retrieval strategy parameters
- **search_k**: see https://reference.langchain.com/python/langchain_core/vectorstores/#langchain_core.vectorstores.base.VectorStore.as_retriever
- **fetch_k**: Number of documents to fetch
- **mmr_lambda**: Maximum Marginal Relevance lambda parameter

### Files
- **questions_file**: File containing the questions about Qibo to be answered by the LLM
- **golden_file**: File containing the golden answers to the questions for evaluation
- **py_file_docstring**: files for which docstrings are to be generated