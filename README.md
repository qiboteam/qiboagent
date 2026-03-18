# QiboAgent

[![arXiv](https://img.shields.io/badge/arXiv-2603.15538-b31b1b.svg)](https://arxiv.org/abs/2603.15538)

This repository contains all the Python scripts used for the QiboAgent paper, along with a UI and a Model Context Protocol (MCP) server exposing the Qibo Expert RAG pipeline as a tool.

## 💻 Installation

To set up the project, all dependencies are managed by [Poetry](https://python-poetry.org/docs/#installation).

```bash
git clone https://github.com/qiboteam/qiboagent.git 
cd qiboagent
poetry install
```

**Important:** [Ollama](https://ollama.com/) is strictly required to run the local models for this project. You must install it on your host machine and pull the required models. If you use a remote Ollama server, be sure to set the correct base URL in the settings.

## 🖥️ UI

After installing the dependencies, you can launch the UI with:

```bash
poetry run streamlit run ui/Home.py
```

A browser window will open with the interface. From the sidebar, you can choose to either chat with the Qibo Expert RAG model or use an agent to resolve an issue within the Qibo codebase.

## 🛠️ MCP Server (Model Context Protocol)

The Qibo Expert is also available as an MCP server, allowing you to use it as a tool in Gemini CLI or other MCP-compatible clients.

### Run the MCP Server

```bash
poetry run python mcp_server/qibo_expert.py
```

### Registration with MCP Clients

You can add this server to your favorite MCP-compatible client to expose the `ask_qibo_expert` tool. Note that you should replace `<ABSOLUTE_PATH_TO_REPO>` with the actual path to this repository on your machine.

#### Gemini CLI
Add the server using the `mcp add` command:
```bash
gemini mcp add qibo-expert poetry run python <ABSOLUTE_PATH_TO_REPO>/mcp_server/qibo_expert.py
```

#### Claude Code (Anthropic CLI)
Register the tool globally using:
```bash
claude mcp add qibo-expert --scope user poetry run python <ABSOLUTE_PATH_TO_REPO>/mcp_server/qibo_expert.py
```

#### OpenCode
Add the following to your `~/.opencode/config.json` (global) or `opencode.json` (project):
```json
{
  "mcp": {
    "qibo-expert": {
      "type": "local",
      "command": "poetry",
      "args": ["run", "python", "<ABSOLUTE_PATH_TO_REPO>/mcp_server/qibo_expert.py"],
      "enabled": true
    }
  }
}
```

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

All pipeline settings are contained in the `settings_json/settings.json` file. Here, you can configure parameters for the vector store, the LLM, and the agentic workflows.

In the same directory, you will find files containing the questions and golden answers used for evaluating the RAG pipelines.
