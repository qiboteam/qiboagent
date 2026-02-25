# QiboLLM

This repository contains all the python scripts used for QiboLLM project.

## Installation 

All dependencies are managed by [Poetry](https://python-poetry.org/docs/#installation), to get set run 

```bash
git clone https://github.com/qiboteam/qibollm.git
cd qibollm

poetry install
```

To run local model you also have to install [Ollama](https://ollama.com/) and pull the model you want to use.
If you use an Ollama server be careful to set the correct base url in settings.


## Scripts

in ```python_script/``` there are all the necessary script to reproduce the RAG and agentic pipelines.

to run the scripts simply run 

```bash
poetry run python python_scripts/script_name.py
```

### Quantum Computing Q&A

There is a script for each experimental RAG pipeline with autoscoring of the answers, be sure to set correctly the knowledge base path, the scripts will automatically clone Qibo repository if not found in the specified path.

- ```no_RAG.py``` contains the script where LLM can be queried without RAG

- ```RAG_semantic.py``` contains the script where LLM can be queried with semantic RAG pipeline

- ```RAG_hybrid```contains contains the script where LLM can be queried with hybrid RAG pipeline

### Docstring generation

in ```docstring_gen.py``` there is the script to generate docstrings for the Qibo library using hybrid RAG pipeline. you can set the file path of the code for which you want to generate docstring in the `setting.json` file.

### Agentic workflows

For the agentic workflows:

- ```agent.py``` contains the script used for the agentic workflow for Issue resolution in the Qibo library. 

- ```agentic_core.py``` contains the script for the agentic workflow used to generate the qibo_core module

## Settings

All the settings for the pipelines are contained in the `settings/settings.json` file, where you can set all the parameters for the vector store, the LLM and the agentic workflow. In the same directory there are the files with the questions and golden answers for the evaluation of the RAG pipelines.