# QiboAgent

This repository contains all the Python scripts used for the QiboAgent paper, along with a containerized UI interface.

## 🚀 Quickstart: QiboAgent UI (Docker)

If you want to run the QiboAgent UI without managing local Python environments, you can use Docker. This setup automatically provisions both the frontend UI and a local Ollama inference server.

### Prerequisites

* **Docker** and **Docker Compose** installed on your machine.

### Deployment Steps

1. **Clone the repository:**
    ```bash
    git clone https://github.com/qiboteam/qiboagent.git
    cd qiboagent
    ```

2. **Build and start the containers:**
    ```bash
    docker compose up -d --build
    ```

3. **Access the UI:**
    Navigate to **http://localhost:8501** in your web browser.

4. **Cleanup:**
    To stop the application (your models remain saved):
    ```bash
    docker compose down
    ```

## 💻 Local Development & Installation

If you prefer to run the RAG and agentic pipelines directly or do local development, all dependencies are managed by [Poetry](https://python-poetry.org/docs/#installation).

### Setup

```bash
git clone https://github.com/qiboteam/qiboagent.git
cd qiboagent
poetry install
```

To run a local model without Docker, you must install [Ollama](https://ollama.com/) on your host machine and pull the model you want to use. If you use a remote Ollama server, be sure to set the correct base URL in the settings.

## 📜 Scripts

In the `python_scripts/` directory, you will find all the necessary scripts to reproduce the RAG and agentic pipelines.

To run a script, use Poetry:

```bash
poetry run python python_scripts/script_name.py
```

### Quantum Computing Q&A

There is a script for each experimental RAG pipeline featuring autoscoring of the answers. Be sure to correctly set the knowledge base path; the scripts will automatically clone the Qibo repository if it is not found in the specified path.

* **no_RAG.py**: Query the LLM without RAG.
* **RAG_semantic.py**: Query the LLM using a semantic RAG pipeline.
* **RAG_hybrid.py**: Query the LLM using a hybrid RAG pipeline.

### Docstring Generation

* **docstring_gen.py**: Generates docstrings for the Qibo library using a hybrid RAG pipeline. You can set the file path of the code for which you want to generate docstrings in the `settings.json` file.

### Agentic Workflows

* **agent.py**: Agentic workflow for Issue resolution within the Qibo library.
* **agentic_core.py**: Agentic workflow used to generate the `qibo_core` module.

## ⚙️ Settings

All pipeline settings are contained in the `settings/settings.json` file. Here, you can configure parameters for the vector store, the LLM, and the agentic workflows.

In the same directory, you will find files containing the questions and golden answers used for evaluating the RAG pipelines.
