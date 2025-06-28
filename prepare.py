import requests
import json

# --- Configuration ---
# Replace with the actual URL of your Open-WebUI instance
OPEN_WEBUI_BASE_URL = "http://localhost:8080" # Example: "http://your-openwebui-ip:8080"

# The name of the model to pull
MODEL_NAME_TO_PULL = "qwen2.5-coder:latest"

# You might need an API key or a session token if your Open-WebUI
# instance requires authentication for API calls (e.g., if signup is disabled
# or if you've already created an admin user).
# If required, obtain this token by logging into Open-WebUI and inspecting
# network requests or checking your user profile for an API key.
AUTH_TOKEN = "sk-0336d9d0379743e7a5f820cd3baaaf47" # Leave empty string if not required

# --- API Endpoint ---
# This is a common endpoint for proxying Ollama model operations through Open-WebUI.
# It typically handles pull, push, and delete operations for models.
OLLAMA_PULL_ENDPOINT = f"{OPEN_WEBUI_BASE_URL}/ollama/api/pull"

# --- Request Headers ---
headers = {
    "Content-Type": "application/json",
}

# Add Authorization header if a token is provided
if AUTH_TOKEN:
    headers["Authorization"] = f"Bearer {AUTH_TOKEN}"

# --- Request Body (Payload) ---
# The payload for pulling a model.
# The 'model' field is required, and 'stream' is often set to true for real-time progress.
# However, this script will demonstrate a non-streaming pull for simplicity,
# as handling streaming responses requires a different approach (e.g., Server-Sent Events).
pull_data = {
    "name": MODEL_NAME_TO_PULL,
    "stream": True # Set to true if you want to handle streaming output (more complex)
}

print(f"Attempting to pull model: {MODEL_NAME_TO_PULL}")
print(f"Sending request to: {OLLAMA_PULL_ENDPOINT}")
print(f"Payload: {json.dumps(pull_data, indent=2)}")
print("\nNote: Model pulling can take a significant amount of time depending on size and internet speed.")
print("The API response will indicate success or failure. Real-time progress is best viewed in Open-WebUI's UI or by setting 'stream': true and handling the stream.")

# --- Make the API Request ---
try:
    response = requests.post(OLLAMA_PULL_ENDPOINT, headers=headers, json=pull_data)

    # --- Handle the Response ---
    if response.status_code == 200:
        print("\nModel pull request sent successfully!")
        response_json = response.json()
        print("Response:")
        print(json.dumps(response_json, indent=2))

        if response_json.get("status") == "success":
            print(f"\nModel '{MODEL_NAME_TO_PULL}' successfully pulled or already exists.")
        else:
            # If 'stream' was false, the 'status' might still be pending or indicate a problem
            print(f"\nModel pull operation might be in progress or encountered an issue. Check Open-WebUI UI for status.")
            print(f"Message: {response_json.get('message', 'No message provided.')}")

    elif response.status_code == 400: # Bad Request - e.g., invalid model name
        print(f"\nError: Bad Request (Status Code: {response.status_code})")
        print("Possible reasons: Invalid model name, malformed request, etc.")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    elif response.status_code == 401: # Unauthorized - e.g., missing or invalid authentication token
        print(f"\nError: Unauthorized (Status Code: {response.status_code})")
        print("Please ensure AUTH_TOKEN is correct and has access rights to pull models.")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    elif response.status_code == 404: # Not Found - e.g., Open-WebUI not running or endpoint changed
        print(f"\nError: Not Found (Status Code: {response.status_code})")
        print("The API endpoint might be incorrect, or Open-WebUI is not accessible at the given URL.")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"\nAn unexpected error occurred. Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))

except requests.exceptions.ConnectionError as e:
    print(f"\nError: Could not connect to Open-WebUI at {OPEN_WEBUI_BASE_URL}.")
    print("Please ensure Open-WebUI is running and accessible at the specified URL.")
    print("Also ensure your Ollama backend is running and connected to Open-WebUI.")
    print(f"Details: {e}")
except requests.exceptions.Timeout:
    print(f"\nError: The request timed out.")
    print("The server took too long to respond.")
except requests.exceptions.RequestException as e:
    print(f"\nAn error occurred during the request: {e}")

