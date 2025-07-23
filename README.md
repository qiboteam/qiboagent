# Qibollm

Qibollm aims to develop an AI coding assistant with expertise in the Qibo codebase.

## Installation

This project relies on the following frameworks:

- [**Ollama**](https://ollama.com/) – Easily run large language models (LLMs) locally.
- [**OpenWebUI**](https://openwebui.com/) – A user-friendly interface for interacting with LLMs.

You have two main options to run this project:

- **Using the Docker image**
- **Installing locally with Python and Bash scripts**

In both cases, you will need to install **Ollama** first: [https://ollama.com/download](https://ollama.com/download)


### Python installation

You can install Open WebUI using `pip`, the Python package manager.  
Make sure you're using **Python 3.11**, as other versions may cause compatibility issues.

```console 
pip install open-webui
```

after that, you can run the following command to start the Open WebUI server:

```console
open-webui run serve
```
Once started, the Open WebUI server will be available at [http://localhost:8080](http://localhost:8080).

### Model installation

In order to use an LLM, you can choose your model from the Ollama model list.

1. Visit the [Ollama model list](https://ollama.com/models) to see available models.
2. Use the following command to pull the desired model:
```console
ollama pull <model_name>
```

after pulling the model you can use it from terminal with the following command:

```console
ollama run <model_name>
```

if you want to use the model with Open WebUI, you can run the following command:

```console
open-webui run serve --model <model_name>
```
