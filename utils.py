import requests
import uuid 
from pathlib import Path
import subprocess

def upload_file(token, file_path, base_path="."):
    """
    Uploads a file to a specified API endpoint.

    Args:
        token (str): Bearer authentication token for the API.
        file_path (str | Path): Path to the file to upload.

    Returns:
        dict: JSON response from the API containing details of the uploaded file.

    Raises:
        requests.exceptions.HTTPError: If the HTTP request fails.
        ValueError: If the response does not contain the file ID.
    """

    # For more details refer to the API documentation
    # (https://docs.openwebui.com/getting-started/api-endpoints/  RAG section)
    url = 'http://localhost:3000/api/v1/files/'
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json'
    }

    # Use file path as metadata
    file_path = Path(file_path)
    relative_path = str(file_path.relative_to(base_path))

    with open(file_path, 'rb') as f:
        files = {
            'file': (file_path.name, f, 'application/octet-stream')
        }
        metadata =  {"relative_path": relative_path}
        response = requests.post(url, headers=headers, files=files, data=metadata, timeout=60)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        # include server body for easier debugging
        raise requests.exceptions.HTTPError(f"{e}\nServer response: {response.text}") from e
    return response.json()

def add_file_to_knowledge(token, knowledge_id, file_id):
    """
    Adds a file to a specific knowledge base via the API.

    Args:
        token (str): Bearer authentication token for the API.
        knowledge_id (str | int): The ID of the target knowledge base.
        file_id (str | int): The ID of the file to add.

    Returns:
        dict | None: JSON response from the API on success (HTTP 200), or None if the file
        is a duplicate or an error occurred.

    Raises:
        requests.exceptions.HTTPError: If the HTTP request fails and is not handled.
    """
    url = f'http://localhost:3000/api/v1/knowledge/{knowledge_id}/file/add'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {'file_id': file_id}
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 400: # bad request, possibly duplicate
        try:
            error_msg = response.json().get('detail', '')
        except Exception:
            error_msg = response.text
        if 'duplicate content' in error_msg.lower():
            # Ignore silently (content already linked)
            return None
        else:
            print(f"Error while linking file: {error_msg}")
            return None
    else:
        try:
            error_msg = response.json().get('detail', '')
        except Exception:
            error_msg = response.text
        print(f"HTTP {response.status_code} while linking file: {error_msg}")
        return None

def get_or_create_knowledge(token, knowledge_name, description=None):
    """
    Retrieves the ID of an existing knowledge base by name, or creates new one if it does not exist.

    Args:
        token (str): Bearer authentication token for the API.
        knowledge_name (str): The name of the knowledge base to retrieve or create.

    Returns:
        str | int: The ID of the found or newly created knowledge base.

    Raises:
        requests.exceptions.HTTPError: If the HTTP request fails.
        Exception: If the API response is invalid or creation fails.
    """
    list_url = 'http://localhost:3000/api/v1/knowledge/' # list existing knowledge bases
    create_url = 'http://localhost:3000/api/v1/knowledge/create' # create a new knowledge base
    headers = {'Authorization': f'Bearer {token}'}

    # Check if the knowledge base already exists
    try:
        resp = requests.get(list_url, headers=headers)
        resp.raise_for_status()
        for knowledge in resp.json():
            if knowledge.get("name") == knowledge_name:
                print(f"Found existing knowledge '{knowledge_name}' (id={knowledge['id']})")
                return knowledge["id"]
    except requests.exceptions.HTTPError as e:
        print(f"Error listing knowledge bases: {e}")
        print(f"Response: {resp.text}")
        raise

    # Create a new knowledge base if not found
    headers['Content-Type'] = 'application/json'
    # If no description provided, ask the user
    if description is None:
        description = input(f"Enter a description for knowledge base '{knowledge_name}' (optional): ").strip()
    data = {
        'name': knowledge_name,
        'description': description or f'Knowledge base for {knowledge_name}'
    }
    
    try:
        resp = requests.post(create_url, headers=headers, json=data)
        resp.raise_for_status()
        knowledge_id = resp.json()["id"]
        print(f"Created new knowledge '{knowledge_name}' (id={knowledge_id})")
        return knowledge_id
    except requests.exceptions.HTTPError as e:
        print(f"Error creating knowledge base: {e}")
        print(f"Response: {resp.text}")
        print(f"Request data: {data}")
        raise

def pull_repo(repo_url, target_dir):
    """
    Clones a Git repository into the specified target directory.

    Args:
        repo_url (str): The URL of the Git repository to clone.
        target_dir (str | Path): The local directory where the repository will be cloned.

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If the git clone command fails.
    """
    # Check if target_dir already exists to avoid overwriting
    target_dir = Path(target_dir)
    if target_dir.exists():
        print(f"Error: Directory '{target_dir}' already exists. Aborting clone.")
        return
    try:
        subprocess.run(['git', 'clone', repo_url, str(target_dir)], check=True)
        print(f"Repository cloned to {target_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")

def pull_model(model_name):
    """
    Pulls a model from the Ollama repository using the provided model name.

    Args:
        model_name (str): The name of the model to pull.

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If the ollama pull command fails.
    """
    try:
        subprocess.run(['ollama', 'pull', model_name], check=True)
        print(f"Model '{model_name}' pulled successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error pulling model: {e}")

def list_models():
    """
    Lists all available models in the local Ollama environment.

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If the ollama list command fails.
    """
    subprocess.run(['ollama', 'list'], check=True)
    print("")

def remove_model(model_name):
    """
    Removes a model from the local Ollama environment.

    Args:
        model_name (str): The name of the model to remove.

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If the ollama rm command fails.
    """
    try:
        subprocess.run(['ollama', 'rm', model_name], check=True)
        print(f"Model '{model_name}' removed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error removing model: {e}")

def list_knowledge_base(token):
    """
    Lists all existing knowledge bases for the given API token.

    Args:
        token (str): Bearer authentication token for the API.

    Returns:
        None

    Raises:
        requests.exceptions.HTTPError: If the HTTP request fails.
    """
    url = 'http://localhost:3000/api/v1/knowledge/' # list existing knowledge bases
    headers = {'Authorization': f'Bearer {token}'}

    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        knowledge_bases = resp.json()
        if not knowledge_bases:
            print("No knowledge bases found.")
            return
        print("\nExisting Knowledge Bases:")
        for knowledge in knowledge_bases:
            print(f"- ID: {knowledge['id']}, Name: {knowledge['name']}, Description: {knowledge.get('description', 'N/A')}\n")
    except requests.exceptions.HTTPError as e:
        print(f"Error listing knowledge bases: {e}")
        print(f"Response: {resp.text}")
        raise

def list_custom_models(token):
    """
    Lists all custom models for the given API token.

    Args:
        token (str): Bearer authentication token for the API.

    Returns:
        None

    Raises:
        requests.exceptions.HTTPError: If the HTTP request fails.
    """
    url = 'http://localhost:3000/api/v1/models/'  # list existing models
    headers = {'Authorization': f'Bearer {token}'}

    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        models = resp.json()
        if not models:
            print("No custom models found.")
            return
        print("\nExisting Custom Models:")
        for model in models:
            print(f"- ID: {model['id']}, Name: {model['name']}, Description: {model.get('description', 'N/A')}\n")
    except requests.exceptions.HTTPError as e:
        print(f"Error listing custom models: {e}")
        print(f"Response: {resp.text}")
        raise

def validate_token(token, base_url='http://localhost:3000'):
    """Quickly validate the API token by calling the knowledge list endpoint.

    Returns True if the token is accepted (HTTP 200), False otherwise.
    """
    url = f"{base_url}/api/v1/knowledge/"
    headers = {'Authorization': f'Bearer {token}'}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return True
        # common auth failure codes
        if resp.status_code in (401, 403):
            print(f"Token validation failed: {resp.status_code} - Unauthorized or forbidden.")
            return False
        print(f"Token validation returned HTTP {resp.status_code}: {resp.text}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Token validation error (connection or timeout): {e}")
        return False

def delete_knowledge_base_by_id(token, knowledge_id, base_url='http://localhost:3000'):
    """
    Delete a knowledge base by its ID using Open WebUI API.
    Endpoint: DELETE /api/v1/knowledge/{id}/delete
    """
    url = f"{base_url}/api/v1/knowledge/{knowledge_id}/delete"
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json'
    }
    try:
        resp = requests.delete(url, headers=headers, timeout=30)
        if resp.status_code in (200, 202, 204):
            print(f"Knowledge base '{knowledge_id}' deleted successfully.")
            return True
        else:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            print(f"Failed to delete knowledge '{knowledge_id}' (HTTP {resp.status_code}):{detail}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Connection error while deleting knowledge '{knowledge_id}': {e}")
        return False


def delete_knowledge_base_by_name(token, knowledge_name, base_url='http://localhost:3000'):
    """
    Find a knowledge base by name and delete it by ID.
    """
    list_url = f"{base_url}/api/v1/knowledge/"
    headers = {'Authorization': f'Bearer {token}', 'Accept': 'application/json'}
    try:
        r = requests.get(list_url, headers=headers, timeout=30)
        r.raise_for_status()
        items = r.json() or []
        match = next(
            (kb for kb in items if str(kb.get('name', '')).strip().lower() == knowledge_name.strip().lower()),
            None
        )
        if not match:
            print(f"Knowledge named '{knowledge_name}' not found.")
            return False
        return delete_knowledge_base_by_id(token, match['id'], base_url=base_url)
    except requests.exceptions.RequestException as e:
        print(f"Error resolving knowledge by name '{knowledge_name}': {e}")
        return False


def create_custom_model(token, model_name, base_model, knowledge_id, prompt=None, description=None):
    """
    Creates a custom model in Open WebUI linking a knowledge base and (optional) prompt.

    Args:
        token (str)
        model_name (str)
        base_model (str)
        knowledge_id (str)
        prompt (str | None): System / initial prompt to guide the model.
        description (str | None)
    """
    url = 'http://localhost:3000/api/v1/models/create'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    model_id = str(uuid.uuid4())

    meta = {
        "profile_image_url": "/static/favicon.png",
        "description": description or f"Custom model '{model_name}' based on '{base_model}'",
        "tags": [],
        "suggestion_prompts": None,
        "capabilities": {
            "vision": False,
            "file_upload": False,
            "web_search": False,
            "image_generation": False,
            "code_interpreter": False,
            "citations": False
        },
        "knowledge": [
            {"id": knowledge_id}
        ]
    }
    if prompt:
        meta["prompt"] = prompt
    else:
        meta["prompt"] = """You are an expert programming assistant specialized in the Python library Qibo (quantum computing). Your job is to answer technical questions and produce correct, secure, and runnable code using **only** the information provided in the retrieved context ({context}). Follow these rules strictly:

                        1. PRIORITIZE THE CONTEXT (RAG)
                           - Base your answer primarily on the provided {context}. Do not hallucinate facts.
                           - If the answer cannot be determined from the context, say: "I don't have enough information in the knowledge base to answer that with confidence."
                           - When relevant, extract and summarize the most pertinent parts of the context before answering.

                        2. CITATIONS & TRACEABILITY
                           - Every important factual claim must cite at least one source from the context using this format: `[source: <path_or_chunk_id>]`
                           - If you use multiple context chunks, list their sources in order of relevance.
                           - When quoting or paraphrasing a passage from the context, include the source.

                        3. CODE OUTPUT REQUIREMENTS
                           - When returning code, always include any necessary imports and a minimal runnable example.
                           - Provide brief instructions to run the example (Python version, required packages).
                           - Format code in fenced Markdown with the language tag, e.g.:
                             ```python
                             # example code
                             ```
                           - If code snippets are drawn from context chunks, append the source next to the snippet.

                        4. HANDLING UNCERTAINTY & CONFLICTS
                           - If context chunks conflict, present the differing statements, indicate the conflict, and show the sources for each version.
                           - If essential information is missing, list exactly what is missing and how to obtain it (files, functions, or searches to run).

                        5. STYLE, LANGUAGE & STRUCTURE
                           - Answer in English unless the user explicitly requests another language.
                           - Start with a 1–2 sentence **Summary** of the answer.
                           - Then provide a clear **Answer / Solution** with technical details, and finally any **Code** examples.
                           - Use headings, bullet points, and short paragraphs for readability.

                        6. REQUIRED RESPONSE FORMAT (MUST FOLLOW)
                           Always return the following sections in this order:
                           - **Summary:** (1–2 sentences)
                           - **Answer / Solution:** (detailed explanation)
                           - **Code:** (if applicable — fenced code block(s))
                           - **Sources:** each on its own line, format: `[source: path_or_chunk_id]`
                           - **Confidence:** one of `High`, `Medium`, or `Low` with a short justification (1 sentence)

                        7. PERFORMANCE & BEST PRACTICES
                           - When asked about performance or complexity, provide complexity or resource estimates and practical optimization suggestions (e.g., batching, vectorization, alternative APIs).
                           - Favor safe, maintainable code patterns and explicit dependency notes.

                        8. DO NOT
                           - Do not invent APIs, functions, or behavior absent from the context.
                           - Do not provide long unrelated background; keep responses focused on the question and context.

                        When answering, obey the format exactly and include the placeholders where appropriate. Now answer the question: {question}""" 

    data = {
        "id": model_id,
        "name": model_name,
        "base_model_id": base_model,
        "meta": meta,
        "params": {},
        "access_control": None
    }

    try:
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        print(f"Custom model '{model_name}' created successfully.")
        return resp.json()
    except requests.exceptions.HTTPError as e:
        print(f"Error creating custom model: {e}")
        print(f"Response: {resp.text}")
    except requests.exceptions.RequestException as e:
        print(f"Connection error while creating custom model: {e}")

# Open WebUI does not support deleting models by name, only by ID so we need to resolve the name first

def delete_custom_model_by_id(token, model_id, base_url='http://localhost:3000'):
    """
    Delete a custom model by its ID using Open WebUI API.
    Endpoint: DELETE /api/v1/models/model/delete?id=<MODEL_ID>
    """
    url = f"{base_url}/api/v1/models/model/delete"
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json',
        'Content-Type': 'application/json',
    }
    try:
        resp = requests.delete(url, headers=headers, params={'id': model_id}, timeout=30)
        if resp.status_code in (200, 202, 204):
            print(f"Custom model '{model_id}' deleted successfully.")
            return True
        else:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            print(f"Failed to delete model '{model_id}' (HTTP {resp.status_code}): {detail}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Connection error while deleting model '{model_id}': {e}")
        return False


def delete_custom_model(token, model_name, base_url='http://localhost:3000'):
    """
    Find a custom model by name and delete it.
    """
    list_url = f"{base_url}/api/v1/models/"
    headers = {'Authorization': f'Bearer {token}', 'Accept': 'application/json'}
    try:
        r = requests.get(list_url, headers=headers, timeout=30)
        r.raise_for_status()
        models = r.json() or []
        # confronta per nome (case-insensitive)
        match = next((m for m in models if str(m.get('name', '')).strip().lower() == model_name.strip().lower()), None)
        if not match:
            print(f"Model named '{model_name}' not found.")
            return False
        return delete_custom_model_by_id(token, match['id'], base_url=base_url)
    except requests.exceptions.RequestException as e:
        print(f"Error resolving model by name '{model_name}': {e}")
        return False
