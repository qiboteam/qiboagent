import requests # HTTP library for requests
from pathlib import Path # library for filesystem path
from tqdm import tqdm
import subprocess 
import time
import uuid
import utils as ut

def main():
    print("\n=== Open WebUI Custom Model Builder ===")

    # Ask for API token once at startup and reuse it across the session
    token = None
    while True:
        token = input("\nEnter your API token (Settings > Account in Open WebUI): ").strip()
        if not token:
            print("API token is required to manage knowledge bases. Please enter a valid token.")
            continue
        if not ut.validate_token(token):
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
        print("7. Delete custom model")
        print("8. List custom models")
        print("9. Exit")
        choice = input("Select an option (1-9): ").strip()
        print()

        if choice == '1':
            print("\nollama available models:")
            ut.list_models()
            model_name = input("Enter the model name to pull (Visit https://ollama.com/models for a list, leave blank to cancel): ").strip()
            if model_name:
                ut.pull_model(model_name)
            else:
                print("Cancelled model pull.")
        
        elif choice == '2':
            print("\nollama available models:")
            ut.list_models()
            model_name = input("Enter the model name to remove (or leave blank to cancel): ").strip()
            if model_name:
                ut.remove_model(model_name)
            else:
                print("Cancelled model removal.")

        elif choice == '3':
            # use token collected at startup
            ut.list_knowledge_base(token)
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
            knowledge_id = ut.get_or_create_knowledge(token, knowledge_name)
            main_dir = Path(knowledge_name)
            main_dir.mkdir(exist_ok=True)
            print(f"Created directory: {main_dir}")
            repo_dirs = []
            for i, repo_url in enumerate(repo_url_list):
                repo_name = repo_url.split('/')[-1].replace('.git', '')
                if not repo_name:
                    repo_name = f"repo_{i}"
                repo_dir = main_dir / repo_name
                ut.pull_repo(repo_url, repo_dir)
                repo_dirs.append(repo_dir)
            all_files = []
            allowed_extensions = {'.txt','.rst','.py', '.md', '.ipynb'}
            for repo_dir in repo_dirs:
                if repo_dir.exists():
                    filtered_files = [p for p in repo_dir.rglob("*") 
                                    if p.is_file() and p.suffix.lower() in allowed_extensions]
                    all_files.extend(filtered_files)
            if not all_files:
                print("No files found with allowed extensions (.txt, .rst, .py, .md, .ipynb).")
                continue
            success_count, error_count = 0, 0
            with tqdm(total=len(all_files), desc="Uploading files", unit="file") as pbar:
                for file_path in all_files:
                    try:
                        upload_resp = ut.upload_file(token, file_path)
                        file_id = upload_resp.get('id')
                        if not file_id:
                            raise ValueError("API response missing 'id'")
                        ut.add_file_to_knowledge(token, knowledge_id, file_id)
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
            ut.list_knowledge_base(token)
            ident = input("Enter the Knowledge Base ID or Name to delete: ").strip()
            if not ident:
                print("Cancelled knowledge base removal.")
                continue
            try:
                uuid.UUID(ident)
                ut.delete_knowledge_base_by_id(token, ident)
            except ValueError:
                ut.delete_knowledge_base_by_name(token, ident)

        elif choice == '5':
            print("\nOllama available models:")
            ut.list_models()
            ut.list_knowledge_base(token)

        elif choice == '6':
            print("Ollama available Models:")
            ut.list_models()
            ut.list_knowledge_base(token)

            custom_model_name = input("Enter a name for the new custom model: ").strip()
            if not custom_model_name:
                print("Error: Custom model name is required.")
                continue
            selected_model = input("Enter the base model to use (e.g., llama3, mistral): ").strip()
            if not selected_model:
                print("Error: Base model is required.")
                continue
            selected_knowledge = input("Enter the ID of the knowledge base to use: ").strip()
            if not selected_knowledge:
                print("Error: Knowledge base ID is required.")
                continue
            description = input("Enter a description for the custom model (optional): ").strip()
            system_prompt = input("Enter a system prompt for the custom model (optional): ").strip()
            if not system_prompt:
                system_prompt = "You are a code assistant, You are specialized on Qibo Quantum Computing package. Always check the knowledge base and the documentation of Qibo before answering."

            if not custom_model_name or not selected_model or not selected_knowledge:
                print("Error: First three fields are required.")
                continue
            else:
                ut.create_custom_model(token, custom_model_name, selected_model, selected_knowledge, system_prompt=system_prompt, description=description)

        elif choice == '7':
            model_name = input("Enter the name of the custom model to delete (leave blank to cancel): ").strip()
            if model_name:
                ut.delete_custom_model(token, model_name)
            else:
                print("Cancelled model deletion.")

        elif choice == '8':
            ut.list_custom_models(token)

        elif choice == '9':
            print("\nExiting...")
            break
        else:
            print("Invalid option. Please select a number between 1 and 9.")


if __name__ == "__main__":
    main()