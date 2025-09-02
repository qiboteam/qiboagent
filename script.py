import requests
from pathlib import Path
from tqdm import tqdm
import subprocess


def upload_file(token, file_path):
    url = 'http://localhost:3000/api/v1/files/'
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json'
    }
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, headers=headers, files=files)
    response.raise_for_status()
    return response.json()

def add_file_to_knowledge(token, knowledge_id, file_id):
    url = f'http://localhost:3000/api/v1/knowledge/{knowledge_id}/file/add'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {'file_id': file_id}
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()

def get_or_create_knowledge(token, knowledge_name):
    list_url = 'http://localhost:3000/api/v1/knowledge/'
    create_url = 'http://localhost:3000/api/v1/knowledge/create'
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
    data = {
        'name': knowledge_name,
        'description': f'Knowledge base for {knowledge_name}' # maybe add description from user input
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
    try:
        subprocess.run(['git', 'clone', repo_url, str(target_dir)], check=True)
        print(f"Repository cloned to {target_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")

def pull_model(model_name):
    try:
        subprocess.run(['ollama', 'pull', model_name], check=True)
        print(f"Model '{model_name}' pulled successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error pulling model: {e}")

def list_models():
    subprocess.run(['ollama', 'list'])


def main():
    print("=== Open WebUI Knowledge Base Uploader ===")
    print()

    # List available models
    print(f"your available models are:")
    list_models()

    # Pull a specific model
    model_name = input("Enter the model name to pull (or leave blank to skip). Visit https://ollama.com/models for a list of available models.\n").strip()
    if model_name:
        pull_model(model_name)
    
    # Ask for API token
    print(f"creating a new knowledge base")
    token = input("Enter your API token: ").strip()
    if not token:
        print("Error: API token is required.")
        return
    
    # Ask for knowledge base name
    knowledge_name = input("Enter knowledge base name: ").strip()
    if not knowledge_name:
        print("Error: Knowledge base name is required.")
        return
    
    # Ask for repository URLs
    print("Enter repository URLs (comma-separated for multiple repos):")
    repo_urls = input("Repository URLs: ").strip()

    if not repo_urls:
        print("Error: No repository URL provided.")
        return

    # Split multiple URLs by comma if provided
    repo_url_list = [url.strip() for url in repo_urls.split(',') if url.strip()]
    
    knowledge_id = get_or_create_knowledge(token, knowledge_name)
    
    # Create main directory with knowledge name
    main_dir = Path(knowledge_name)
    main_dir.mkdir(exist_ok=True)
    print(f"Created directory: {main_dir}")
    
    # Clone all repositories into the knowledge directory
    repo_dirs = []
    for i, repo_url in enumerate(repo_url_list):
        # Extract repo name from URL for better naming
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        if not repo_name:
            repo_name = f"repo_{i}"
        
        repo_dir = main_dir / repo_name
        pull_repo(repo_url, repo_dir)
        repo_dirs.append(repo_dir)

    # Collect all files from all cloned repositories
    all_files = []
    allowed_extensions = {'.py', '.md', '.ipynb'}
    
    for repo_dir in repo_dirs:
        if repo_dir.exists():
            # Filter files by extension
            filtered_files = [p for p in repo_dir.rglob("*") 
                            if p.is_file() and p.suffix.lower() in allowed_extensions]
            all_files.extend(filtered_files)

    if not all_files:
        print("No files found with allowed extensions (.py, .md, .ipynb).")
        return

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
            pbar.set_postfix(success=success_count, errors=error_count)
            pbar.update(1)

    print(f"\nUpload complete: {success_count} files added, {error_count} errors.")


if __name__ == "__main__":
    main()

