import requests # HTTP library for requests
from pathlib import Path # library for filesystem path
from tqdm import tqdm
import subprocess 
import time
import uuid


def upload_file(token, file_path):
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

    # For more details refer to the API documentation (https://docs.openwebui.com/getting-started/api-endpoints/  RAG section)
    url = 'http://localhost:3000/api/v1/files/'
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json'
    }
    with open(file_path, 'rb') as f:
        files = {
            'file': (Path(file_path).name, f, 'application/octet-stream')
        }
        response = requests.post(url, headers=headers, files=files, timeout=60)
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
        dict: JSON response from the API with details about the operation.

    Raises:
        requests.exceptions.HTTPError: If the HTTP request fails.
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
    elif response.status_code == 400:
        # Ignora errore di contenuto duplicato
        try:
            error_msg = response.json().get('detail', '')
        except Exception:
            error_msg = response.text
        if 'duplicate content' in error_msg.lower():
            # Ignora e non stampa nulla
            return None
        else:
            print(f"Errore nell'associazione file: {error_msg}")
            return None
    else:
        try:
            error_msg = response.json().get('detail', '')
        except Exception:
            error_msg = response.text
        print(f"Errore HTTP {response.status_code} nell'associazione file: {error_msg}")
        return None

def get_or_create_knowledge(token, knowledge_name, description=None):
    """
    Retrieves the ID of an existing knowledge base by name, or creates a new one if it does not exist.

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
    subprocess.run(['ollama', 'list'])

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

import requests

# open_web UI does not support deleting knowledge bases via API???
#def delete_knowledge_base(token, knowledge_id, base_url='http://localhost:3000'):
#    """
#    Deletes a knowledge base by its ID using the OpenWebUI API.
#
#    Args:
#        token (str): Bearer authentication token for the API.
#        knowledge_id (str): The ID of the knowledge base to delete.
#        base_url (str): Base URL of the OpenWebUI instance.
#
#    Returns:
#        None
#
#    Raises:
#        requests.exceptions.HTTPError: If the HTTP request fails.
#    """
#    url = f"{base_url}/api/v1/knowledge/{knowledge_id}/delete"
#    headers = {'Authorization': f'Bearer {token}'}
#
#    try:
#        resp = requests.delete(url, headers=headers, timeout=30)
#        resp.raise_for_status()
#        print(f"Knowledge base '{knowledge_id}' deleted successfully.")
#    except requests.exceptions.HTTPError as e:
#        print(f"Error deleting knowledge base '{knowledge_id}': {e}")
#        print(f"Response: {resp.text if resp is not None else 'No response body'}")
#    except requests.exceptions.RequestException as e:
#        print(f"Connection error while deleting knowledge base '{knowledge_id}': {e}")


def create_custom_model(token, model_name, base_model, knowledge_id):
    """
    Creates a custom model in Open WebUI using a specified base model 
    and a linked knowledge base.

    Args:
        token (str): Bearer authentication token for the API.
        model_name (str): The display name for the new custom model.
        base_model (str): The base model to use (e.g., 'llama3.2:1b').
        knowledge_id (str): The ID of the knowledge base to link to this model.

    Returns:
        dict: The JSON response from the API if successful.

    Raises:
        requests.exceptions.HTTPError: If the API request fails.
    """
    url = 'http://localhost:3000/api/v1/models/create'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    # Generate a unique ID for the model
    model_id = str(uuid.uuid4())

    # Build the request payload
    data = {
        "id": model_id,
        "name": model_name,
        "base_model_id": base_model,
        "meta": {
            "profile_image_url": "/static/favicon.png",
            "description": f"Custom model '{model_name}' created via API",
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
        },
        "params": {},  # Empty params object if no extra configuration
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



def main():
    print("\n=== Open WebUI Knowledge Base Manager ===")

    # Ask for API token once at startup and reuse it across the session
    token = None
    while True:
        token = input("\nEnter your API token (Settings > Account in Open WebUI): ").strip()
        if not token:
            print("API token is required to manage knowledge bases. Please enter a valid token.")
            continue
        if not validate_token(token):
            print("Provided token is invalid or the server is unreachable. Please try again.")
            continue
        break

    while True:
        print("\nMenu:")
        print("1. Pull Ollama model")
        print("2. Remove Ollama model")
        print("3. Create knowledge base")
        print("4. Remove knowledge base")
        print("5. List available models and knowledge bases")
        print("6. Create custom model with knowledge")
        print("7. List custom models")
        print("8. Exit")
        choice = input("Select an option (1-8): ").strip()

        if choice == '1':
            print("\nollama available models:")
            list_models()
            model_name = input("Enter the model name to pull (Visit https://ollama.com/models for a list): ").strip()
            if model_name:
                pull_model(model_name)
        
        elif choice == '2':
            print("\nollama available models:")
            list_models()
            model_name = input("Enter the model name to remove (or leave blank to cancel): ").strip()
            if model_name:
                remove_model(model_name)

        elif choice == '3':
            # use token collected at startup
            list_knowledge_base(token)
            knowledge_name = input("Enter knowledge base name (or leave blank to cancel): ").strip()
            if not knowledge_name:
                print("Cancelled knowledge base creation.")
                continue
            print("Enter repository URLs (comma-separated for multiple repos):")
            repo_urls = input("Repository URLs: ").strip()
            if not repo_urls:
                print("Error: No repository URL provided.")
                continue
            repo_url_list = [url.strip() for url in repo_urls.split(',') if url.strip()]
            knowledge_id = get_or_create_knowledge(token, knowledge_name)
            main_dir = Path(knowledge_name)
            main_dir.mkdir(exist_ok=True)
            print(f"Created directory: {main_dir}")
            repo_dirs = []
            for i, repo_url in enumerate(repo_url_list):
                repo_name = repo_url.split('/')[-1].replace('.git', '')
                if not repo_name:
                    repo_name = f"repo_{i}"
                repo_dir = main_dir / repo_name
                pull_repo(repo_url, repo_dir)
                repo_dirs.append(repo_dir)
            all_files = []
            allowed_extensions = {'.py', '.md', '.ipynb'}
            for repo_dir in repo_dirs:
                if repo_dir.exists():
                    filtered_files = [p for p in repo_dir.rglob("*") 
                                    if p.is_file() and p.suffix.lower() in allowed_extensions]
                    all_files.extend(filtered_files)
            if not all_files:
                print("No files found with allowed extensions (.py, .md, .ipynb).")
                continue
            success_count, error_count = 0, 0
            with tqdm(total=len(all_files), desc="Uploading files", unit="file") as pbar:
                for file_path in all_files:
                    try:
                        upload_resp = upload_file(token, file_path)
                        file_id = upload_resp.get('id')
                        if not file_id:
                            raise ValueError("API response missing 'id'")
                        add_file_to_knowledge(token, knowledge_id, file_id)
                        success_count += 1
                    except Exception as e:
                        error_count += 1
                        pbar.write(f"Error processing '{file_path.name}': {e}")
                    # small throttle to avoid overwhelming the server
                    time.sleep(0.15)
                    pbar.set_postfix(success=success_count, errors=error_count)
                    pbar.update(1)
            print(f"\nUpload complete: {success_count} files added, {error_count} errors.")

        elif choice == '4':
            # use token collected at startup
            #list_knowledge_base(token)
            #knowledge_id = input("Enter the ID of the knowledge base to delete: ").strip()
            #if not knowledge_id:
            #    print("Cancelled knowledge base removal.")
            #    continue
            #delete_knowledge_base(token, knowledge_id)
            print("Knowledge base deletion is currently disabled.")

        elif choice == '5':
            print("\nollama available models:")
            list_models()
            # use token collected at startup
            list_knowledge_base(token)

        elif choice == '6':
            print("Ollama available Models:")
            list_models()

            print("Available Knowledge Bases:")
            list_knowledge_base(token)

            custom_model_name = input("Enter a name for the new custom model: ").strip()
            selected_model = input("Enter the base model to use (e.g., llama3, mistral): ").strip()
            selected_knowledge = input("Enter the ID of the knowledge base to use: ").strip()

            if not custom_model_name or not selected_model or not selected_knowledge:
                print("Error: All fields are required.")
                continue
            else:
                create_custom_model(token, custom_model_name, selected_model, selected_knowledge)

        elif choice == '7':
            list_custom_models(token)

        elif choice == '8':
            print("\nExiting...")
            break
        else:
            print("Invalid option. Please select a number between 1 and 8.")


if __name__ == "__main__":
    main()