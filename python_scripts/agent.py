"""
Agentic Pipeline for Qibo github issues resolution
"""

import ast
from pathlib import Path
from dataclasses import dataclass
import requests
import os,json
import logging
import json
from rich import print as rprint
from rich.panel import Panel
from rich.syntax import Syntax

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OWNER = "qiboteam"
REPO = "qibo"
ISSUE_NUMBER = 1710 #1710 unfolded matrix product #1699 issue related to non_trainable parameters 
SETTINGS_PATH = "../settings_json/settings.json"
OUTPUT_FILE = "../agent_output.json"
BASE_DIR = Path(__file__).parent / "qibo"

# Two examples of system prompts. It's important to emphasize the general workflow and strategy.
# If you need any special requirements, you can express them in the user prompt.

SYSTEM_PROMPT = """You are a Python Maintainer for the Qibo library (a python framework for quantum computing).
Your ONLY goal is to analyze and fix the specific GitHub issue assigned to you.

You have access to tools to read issues and inspect code.

### CRITICAL RULES (READ CAREFULLY):
1. **SINGLE ISSUE FOCUS**: You are STRICTLY FORBIDDEN from calling `get_github_issue` with any number other than {ISSUE_NUMBER}.
2. **IGNORE DISTRACTIONS**: If `search_code` reveals other issue numbers (e.g., in comments or commit messages), IGNORE THEM. They are noise.
3. **ATTRIBUTES ARE NOT FUNCTIONS**: Do NOT use `read_function_code` to read variables like `self.trainable_gates`. 
   - To see attributes, read the `__init__` method.
   - To see class properties, search for `@property`.
4. **NO SUMMARIES**: Never explain the code. Write the patch.

### STRATEGY & WORKFLOW:

1. **ANALYZE THE ISSUE (MANDATORY START)**:
   - Call `get_github_issue(number={ISSUE_NUMBER})`.
   - Understand the Goal: Is it a missing parameter? A wrong calculation?
   - Identify the target functions.

2. **DEEP CONTEXT & DEPENDENCY TRACING**:
   - Locate the function code using `search_code`.
   - Use `read_file_structure` to see the map of the file (imports, classes, methods) WITHOUT reading the full content.
   - Use `read_function_code` to inspect the specific function logic.
   - **Crucial Step**: Check if the function calls internal helpers linked, if so, read them too.

3. **IMPACT ANALYSIS**:
   - If the issue requires adding a new argument to the main function:
     - You MUST pass this new argument down to the helper function.
     - You MUST update the helper function's definition to accept this argument.
     - You MUST update the helper function's logic to USE this argument.
   - **Do not break the chain**. The main function and the helper function must agree on the arguments.

4. **FINAL OUTPUT**:
   - Provide the FULL code for the modified functions.
   - Do NOT output explanations or summaries of the file structure.
   - Call `ResponseFormat` immediately when ready.

### OUTPUT FORMAT REQUIREMENTS:
- **proposed_patch MUST contain ONLY valid Python code**
- **DO NOT include diff markers** (no `***`, `@@`, `+`, `-`)
- **DO NOT include file path comments inside the patch**
- **Include COMPLETE function definitions** with all decorators and docstrings, INCLUDING any new helper methods.
- **If multiple functions are modified, separate them with blank lines**

### CRITICAL RULES:
- **No Chatter**: Do not explain "I see the issue now". Just act.
- **Search before you guess**: Use `search_code` to find definitions.
- **Retry on Error**: If a tool fails, retry with corrected arguments.
- **Clean Code Only**: The proposed_patch field must be executable Python code.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: YOU MUST USE TOOLS - NO EXCEPTIONS 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DO NOT write explanations like "I will search for..." or "Let me analyze...".
INSTEAD, respond with a tool call in this EXACT format (DO NOT use markdown code blocks (like ```json).):

{{
  "tool_calls": [{{
    "name": "tool_name_here",
    "args": {{"param": "value"}}
  }}]
}}


"""


SYSTEM_PROMPT2 = """You are a Python Maintainer for the Qibo library (python framework for quantum computing).
Your ONLY goal is to analyze and fix the specific GitHub issue assigned to you (# {ISSUE_NUMBER}).

You have access to tools to read issues and inspect code.

### CRITICAL RULES:
1. **SINGLE ISSUE FOCUS**: Work ONLY on issue #{ISSUE_NUMBER}.
2.  **ADAPT SCRIPTS**: Issues often contain "Proof of Concept" (PoC) scripts or benchmarks demonstrating a fix.
    - **EXTRACT** only the algorithmic core (the new logic).
    - **INTEGRATE** this logic into the existing class/method structure of Qibo USING THE TOOLS TO GET CONTEXT.
3.  **HELPER EXTRACTION**: If the proposed fix relies on a helper function defined in the issue, you MUST adopt it. Make it a private method and place it in a utility module if appropriate.
4. **USE THE TOOLS**: Use `search_code` to find functions, `read_file_structure` to understand file layout, and `read_function_code` to inspect logic.
5. **REASONING**: YOU MUST GENERATE DETAILED REASONING STEPS as you work through the issue. EXPLAIN your thought process clearly before producing the final patch.

### STRATEGY & WORKFLOW:

1. **ANALYZE ISSUE & SNIPPETS**:
   - Call `get_github_issue`.
   - **Scan for Code**: specificially look for Python blocks in the issue description.
    - **Identify the Fix**: distinguish between the *benchmark scaffolding* (waste) and the *algorithmic fix*.

2. **LOCATE & INSPECT CONTEXT**:
   - call `search_code` to find the target functions.
   - Call `read_function_code` to read the CURRENT implementation. 
   - **Mental Mapping**: If the issue provides a fix script, map its variables to the actual class attributes.

3. **PLAN THE FIX**:
   - **Main Function Changes**: Identify what changes are needed in the main function, generate a thought process.
   - **Helper Method**: If a helper method is needed:
     - Define its purpose clearly.
     - Ensure it receives all necessary parameters from the main function.

4. **GENERATE THE PATCH**:
   - Provide the FULL code of the functions you modify.
   - If you added a helper method, output BOTH the new helper method AND the modified original method together in the `proposed_patch`.
   - Ensure imports are correct for the new logic.

### OUTPUT FORMAT:
When ready, call `ResponseFormat` with:
- file_path: The path of the modified file.
- explanation: A brief explanation of the changes made.
- proposed_patch: The FULL code of the modified function(s), including any new helper methods.
"""

#-------------------------AGENT TOOLS AND UTILS------------------------ #

@dataclass
class ResponseFormat:
    file_path: str
    explanation: str
    proposed_patch: str

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

# This tool fetches the issue details directly from GitHub, including title, body, and comments. 
@tool("get_github_issue", description="Fetch the details of a GitHub issue including title, body, and comments.")
def get_github_issue(owner: str = OWNER, repo: str = REPO, number: int = ISSUE_NUMBER) -> str:
   """Fetch the details of a GitHub issue including title, body, and comments.
   Args:
       owner (str): Repository owner.
       repo (str): Repository name.
       number (int): Issue number.
   """
   url = f"https://api.github.com/repos/{owner}/{repo}/issues/{number}"
   try:
       r = requests.get(url)
       r.raise_for_status()
       data = r.json()
       
       text = [f"ISSUE TITLE: {data.get('title', '')}", f"ISSUE BODY:\n{data.get('body', '')}"]
       
       if data.get("comments_url"):
           rc = requests.get(data["comments_url"])
           if rc.status_code == 200:
               for c in rc.json():
                   text.append(f"COMMENT BY {c['user']['login']}:\n{c['body']}")
       return "\n\n".join(text)
   except Exception as e:
       logger.error(f"Failed to fetch issue: {e}")
       return ""

# For testing without GitHub API access, you can use a local JSON file with the same structure as the GitHub API response.
# If you are sending a lot of requests to GitHub API, consider using this local file approach to avoid rate limits and speed up testing.

# @tool("get_github_issue", description="Fetch the details of a GitHub issue including title, body, and comments.")
# def get_github_issue(owner: str = OWNER, repo: str = REPO, number: int = ISSUE_NUMBER) -> str:
#     """Fetch the details of a GitHub issue including title, body, and comments.
#     Args:
#         owner (str): Repository owner.
#         repo (str): Repository name.
#         number (int): Issue number.
#     """
#     file_path = f"agent/issue_{number}.json"
    
#     try:
#         if not os.path.exists(file_path):
#             logger.error(f"Local file {file_path} not found.")
#             return ""

#         with open(file_path, "r") as f:
#             data = json.load(f)
        
#         text = [f"ISSUE TITLE: {data.get('title', '')}", f"ISSUE BODY:\n{data.get('body', '')}"]
        
#         #if "local_comments" in data:
#         #    for c in data["local_comments"]:
#         #        text.append(f"COMMENT BY {c['user']['login']}:\n{c['body']}")
        
#         return "\n\n".join(text)
#     except Exception as e:
#         logger.error(f"Failed to read local issue: {e}")
#         return ""

@tool("read_file_structure", description="Reads the structure of a python file (imports, classes, methods signatures) WITHOUT the actual code implementation.")
def read_file_structure(file_path: str) -> str:
    """
    Reads the structure of a python file (imports, classes, methods signatures) 
    WITHOUT the actual code implementation.
    Use this to understand available methods and global variables before selecting 
    specific functions to read with `read_function_code`.

    Args:
        file_path (str): The relative path to the file (e.g., "src/qibo/models/evolution.py").
    """
    try:
        repo_path = BASE_DIR / file_path
        if not repo_path.exists():
            return f"Error: File {file_path} not found."
            
        tree = ast.parse(repo_path.read_text(encoding="utf-8"))
        
        output = []
        
        imports = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.unparse(node))
        if imports:
            output.append("### IMPORTS:")
            output.extend(imports)
            output.append("")

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                output.append(f"class {node.name}(...):")
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        decorators = [f"@{ast.unparse(d)}" for d in item.decorator_list]
                        dec_str = " " + " ".join(decorators) if decorators else ""
                        args = ast.unparse(item.args)
                        output.append(f"    def {item.name}({args}){dec_str}: ...")
                output.append("")
                
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = ast.unparse(node.args)
                output.append(f"def {node.name}({args}): ...")
                
            elif isinstance(node, ast.Assign):
                try:
                    output.append(f"{ast.unparse(node)} (Global Variable)")
                except:
                    pass

        return "\n".join(output)

    except Exception as e:
        return f"Error parsing structure: {e}"

@tool("read_function_code", description="Read the code of specific functions, methods, or classes from a file.")
def read_function_code(file_path: str, function_names: str) -> str:
    """
    Read the code of specific functions, methods, or classes from a file.
    Critically useful to inspect logic without loading the entire file.

    Args:
        file_path (str): The relative path to the file (e.g., "src/qibo/models/circuit.py").
        function_names (str): A comma-separated list of names to extract. 
                              Supports explicit class methods (e.g., "Circuit.set_parameters") 
                              or simple names (e.g., "set_parameters").
    """
    try:
        repo_path = BASE_DIR / file_path
        if not repo_path.exists():
            return f"Error: File '{file_path}' does not exist. Use `list_files` to check the path."
            
        source_code = repo_path.read_text(encoding="utf-8")
        
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return f"Error: File '{file_path}' contains syntax errors and cannot be parsed."

        targets = [name.strip() for name in function_names.split(",")]
        
        node_map = {}
        available_nodes = []

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                node_map[node.name] = node
                available_nodes.append(node.name)
                
                if isinstance(node, ast.ClassDef):
                    for sub_node in node.body:
                        if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            qualified_name = f"{node.name}.{sub_node.name}"
                            node_map[qualified_name] = sub_node
                            available_nodes.append(qualified_name)
                            
                            if sub_node.name not in node_map:
                                node_map[sub_node.name] = sub_node

        found_nodes = {}
        for target in targets:
            if target in node_map:
                found_nodes[target] = node_map[target]

        if not found_nodes:
            available_preview = ", ".join(available_nodes[:20])
            if len(available_nodes) > 20:
                available_preview += ", ..."
            return (f"Error: None of the requested functions {targets} were found in '{file_path}'.\n"
                    f"Did you mean one of these? [{available_preview}]\n"
                    f"HINT: Check for typos or use 'ClassName.method_name' format.")

        result_output = []
        lines = source_code.splitlines()
        
        for name, node in found_nodes.items():
            start_line = node.lineno - 1
            if node.decorator_list:
                start_line = node.decorator_list[0].lineno - 1
            
            end_line = node.end_lineno
            
            code_snippet = "\n".join(lines[start_line:end_line])
            result_output.append(
                f"File: {file_path}\n"
                f"Target: {name}\n"
                f"Type: {type(node).__name__}\n"
                f"```python\n{code_snippet}\n```"
            )
            
        return "\n\n".join(result_output)

    except Exception as e:
        return f"System Error in read_function_code: {str(e)}"
    
@tool("list_files", description="List all files and directories in the specified folder of the repository.")
def list_files(directory: str = ".") -> str:
    """
    List all files and directories in the specified folder of the repository.
    Use this to understand the project structure or find correct file paths.

    Args:
        directory (str): The target directory path relative to the repository root.
    """
    try:
        target_path = BASE_DIR / directory
        
        if not target_path.exists():
            return f"Error: Directory '{directory}' does not exist."
        
        items = []
        for item in target_path.iterdir():
            if item.name.startswith("."): continue 
            prefix = "[DIR] " if item.is_dir() else "[FILE]"
            items.append(f"{prefix} {item.name}")
            
        return f"Contents of '{directory}':\n" + "\n".join(sorted(items))
    except Exception as e:
        return f"Error listing files: {str(e)}"

@tool("search_code", description="Search for a specific keyword (function name, class, variable) across all Python files.")
def search_code(query: str) -> str:
    """
    Search for a specific keyword (function name, class, variable) across all Python files.
    Returns the file path and line number of occurrences.
    Useful when you know the function name from the issue but not the file path.

    Args:
        query (str): The keyword to search for in the codebase.

    example:
        search_code("apply_gate")
    """
    base_path = BASE_DIR
    results = []
    MAX_RESULTS = 20
    
    try:
        for file_path in base_path.rglob("*.py"):
            try:
                content = file_path.read_text(encoding="utf-8").splitlines()
                for i, line in enumerate(content):
                    if query in line:
                        rel_path = file_path.relative_to(base_path)
                        results.append(f"{rel_path}:{i+1}: {line.strip()[:100]}") 
                        if len(results) >= MAX_RESULTS:
                            return "\n".join(results) + "\n... (more results truncated)"
            except Exception:
                continue 
                
        if not results:
            return f"No results found for keyword: '{query}'"
            
        return "\n".join(results)
    except Exception as e:
        return f"Error searching code: {str(e)}"


#---------------------------AGENT INITIALIZATION------------------------ #


def init_agent(model_name: str, system_prompt: str, reasoning: bool = False):
    """Initialize the agent with the specified model, tools, and system prompt."""
    # Be careful with your Ollama server base_url
    model = ChatOllama(model=model_name, temperature=0.0, reasoning=reasoning, base_url=["llm"]["base_url"])
    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=[read_file_structure, get_github_issue, list_files, search_code, read_function_code],
        checkpointer=checkpointer,
        response_format=ToolStrategy(ResponseFormat),
        system_prompt=system_prompt,
        middleware=[handle_tool_errors],
    )
    return agent

def load_json_settings(file_path: str) -> dict:
    """Load JSON RAG pipeline settings from the specified file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            settings = json.load(f)
        return settings
    except Exception as e:
        logger.error(f"Error loading JSON settings from {file_path}: {e}")
        return {}

#---------------------------MAIN EXECUTION------------------------ #

def main():
    settings = load_json_settings(SETTINGS_PATH)
    formatted_system_prompt = SYSTEM_PROMPT2.format(ISSUE_NUMBER=ISSUE_NUMBER)
    agent = init_agent(settings.get('llm', {}).get('model_name', "qwen3-coder:30b"), formatted_system_prompt, reasoning=settings.get('llm', {}).get('reasoning', False))
    config = {"configurable": {"thread_id": "1"}}
    
    #remove notes file if exists
    if os.path.exists(NOTES_FILE):
        os.remove(NOTES_FILE)
    
    rprint(f"[bold green]Starting Agent on Issue #{ISSUE_NUMBER}...[/bold green]")


    # two different user prompts for testing. Adjust as needed
    user_message = (
        f"PROPOSE A FIX for issue {ISSUE_NUMBER} in {OWNER}/{REPO}. "
        "Start by using `get_github_issue` to read the requirements. "
        "- Read the issue: FOLLOW EXACTLY THE STRATEGY USED BY THE USERS IN THE ISSUE TO PROPOSE THE FIX.  "
        "- USE THE TOOLS TO READ THE CODE AND UNDERSTAND THE CONTEXT."
        "- USE THE TOOL `search_code` TO LOCATE THE FUNCTIONS TO MODIFY."
        "- USE THE TOOL `read_function_code` TO READ THE FUNCTION LOGIC."
    )

    user_m = (
        f"PROPOSE A FIX for issue {ISSUE_NUMBER} in {OWNER}/{REPO}."
        "- Start by using `get_github_issue` to read the issue. "
        "- Read the code using the tools `search_code`, `read_function_code` and `read_file_structure` to understand the context."
        "- Modify also any helper functions as needed and include them in the proposed_patch."
    )

    result = agent.invoke(
       {"messages": [{"role": "user", "content": user_m}]},
       config=config,
    )

    # Print the agent's execution log with rich formatting
    rprint("\n[bold green]========== AGENT EXECUTION LOG ==========[/bold green]")
    rprint(f"\n[bold green]========== USING {settings.get('llm', {}).get('model_name')} ==========[/bold green]")
    
    last_message_content = ""
    
    if "messages" in result:
        for msg in result["messages"]:
            if msg.type == "human":
                rprint(Panel(msg.content, title="[bold blue]USER[/bold blue]", border_style="blue"))
            
            elif msg.type == "ai":
                last_message_content = msg.content
                if msg.content:
                    if '"name": "ResponseFormat"' not in msg.content:
                        rprint(f"\n[bold magenta]AI Thought:[/bold magenta] {msg.content}")
                
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        rprint(Panel(
                            f"Args: {tool_call['args']}", 
                            title=f"[bold yellow]Tool Call: {tool_call['name']}[/bold yellow]", 
                            border_style="yellow"
                        ))

            elif msg.type == "tool":
                content_preview = msg.content[:5000] + "..." if len(msg.content) > 5000 else msg.content
                rprint(Panel(
                    content_preview, 
                    title=f"[bold cyan]Tool Output ({msg.name})[/bold cyan]", 
                    border_style="cyan"
                ))

    rprint("\n[bold green]========== FINAL RESULT ==========[/bold green]")
    
    final_patch = None
    final_explanation = "No explanation provided."
    final_file_path = "Unknown"

    if 'structured_response' in result and result['structured_response']:
        response = result['structured_response']
        final_patch = response.proposed_patch
        final_explanation = response.explanation
        final_file_path = response.file_path

    elif last_message_content:
        try:
            import re
            json_match = re.search(r'\{.*\}', last_message_content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                
                if "parameters" in data:
                    params = data["parameters"]
                    final_patch = params.get("patch") or params.get("proposed_patch")
                    final_explanation = params.get("explanation", "Extracted from JSON")
                    final_file_path = params.get("file_path", "Unknown")
                elif "proposed_patch" in data:
                    final_patch = data["proposed_patch"]
                    final_explanation = data.get("explanation", "Extracted from JSON")
                    final_file_path = data.get("file_path", "Unknown")
        except Exception:
            pass

    if final_patch:
        rprint(Panel(
            final_explanation, 
            title="[bold blue]Explanation[/bold blue]", 
            border_style="blue",
            expand=False
        ))
        
        rprint(f"\n[bold yellow]Target File:[/bold yellow] [cyan]{final_file_path}[/cyan]")
        
        code_content = final_patch.replace("\\n", "\n").replace('\\"', '"')
        
        syntax = Syntax(
            code_content, 
            "python", 
            theme="monokai", 
            line_numbers=True, 
            word_wrap=False
        )
        
        rprint("\n[bold yellow]Proposed Patch:[/bold yellow]")
        rprint(syntax)
        
        try:
            with open(OUTPUT_FILE, "w") as f:
                json.dump({
                    "file_path": final_file_path,
                    "explanation": final_explanation,
                    "proposed_patch": code_content
                }, f, indent=4)
            rprint(f"\n[dim]Result saved to {OUTPUT_FILE}[/dim]")
        except Exception as e:
            rprint(f"[red]Error saving file: {e}[/red]")

    else:
        rprint("[red]Agent failed to produce a valid patch[/red]")
        rprint(Panel(last_message_content, title="[bold red]Raw Output (Parsing Failed)[/bold red]", border_style="red"))
        logger.error("Agent failed to produce a valid patch.")

if __name__ == "__main__":
    main()
