"""
Streamlit GUI for the Agentic Pipeline — Qibo GitHub Issue Resolution.
Reuses the backend logic from agent.py (init_agent, tools, prompts).
"""

import streamlit as st
import sys
import json
import re
import time
import requests
from pathlib import Path

# --- Path Configuration ---
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR / "python_scripts"))

# Import everything we need from the existing agent module
from agent import (
    init_agent,
    load_json_settings,
    check_ollama_llm_availability,
    ResponseFormat,
    OWNER, REPO,
    SYSTEM_PROMPT, SYSTEM_PROMPT2,
)

st.set_page_config(page_title="Qibo Agent — Issue Resolver", layout="wide", page_icon="🔧")

st.title("🔧 Qibo Agent — GitHub Issue Resolver")
st.caption("Analyze and propose fixes for Qibo GitHub issues with an autonomous agent")

# ─────────────────────────── Session State Init ───────────────────────────
for key, default in {
    "agent_running": False,
    "agent_steps": [],
    "agent_result": None,
    "agent_error": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────── Fetch Available Models from Ollama ────────────
@st.cache_resource
def get_ollama_models(base_url="http://localhost:11434"):
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            model_names = [model["name"] for model in models_data.get("models", [])]
            return model_names if model_names else ["No models available"]
        else:
            return ["Error: Could not fetch models"]
    except requests.exceptions.RequestException:
        return ["Ollama not running"]

# ─────────────────────────── Sidebar Config ───────────────────────────────
with st.sidebar:
    st.header("⚙️ Agent Configuration")

    settings_path = ROOT_DIR / "settings_json" / "settings.json"
    if settings_path.exists():
        settings = load_json_settings(str(settings_path))
    else:
        settings = {"llm": {"provider": "ollama", "model_name": "qwen3-coder:30b",
                            "base_url": "http://localhost:11434", "reasoning": False}}

    base_url = st.text_input(
        "Ollama Base URL",
        value=settings["llm"].get("base_url", "http://localhost:11434"),
        help="Base URL for your Ollama instance"
    )

    # Fetch available models
    available_models = get_ollama_models(base_url)
    default_model = settings["llm"].get("model_name", "qwen3-coder:30b")
    
    # Set default index
    try:
        default_idx = available_models.index(default_model)
    except ValueError:
        default_idx = 0

    model_name = st.selectbox(
        "Select LLM Model",
        available_models,
        index=default_idx,
        help="Choose a model from your Ollama instance"
    )

    # Refresh button
    if st.button("🔄 Refresh Models", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    use_reasoning = st.checkbox("Enable reasoning", value=settings["llm"].get("reasoning", False))

    st.divider()
    prompt_choice = st.radio("System Prompt", ["Prompt 1 (Strict)", "Prompt 2 (Adaptive)"], index=0)

    st.divider()
    st.markdown("**Target Repository**")
    owner = st.text_input("Owner", value=OWNER)
    repo = st.text_input("Repo", value=REPO)

# ─────────────────────────── Main Input ───────────────────────────────────
col1, col2 = st.columns([1, 3])
with col1:
    issue_number = st.number_input("Issue #", min_value=1, value=1699, step=1)
with col2:
    user_prompt = st.text_area(
        "User prompt (instructions for the agent)",
        value=(
            f"PROPOSE A FIX for issue {issue_number} in {owner}/{repo}. "
            "- Start by using `get_github_issue` to read the issue. "
            "- Read the code using the tools `search_code`, `read_function_code` "
            "and `read_file_structure` to understand the context. "
            "- Modify also any helper functions as needed and include them in the proposed_patch."
        ),
        height=120,
    )

start_btn = st.button("🚀 Start Agent", type="primary", disabled=st.session_state.agent_running,
                       use_container_width=True)

# ─────────────────────────── Execution Log Container ─────────────────────
log_container = st.container()

# ─────────────────────────── Agent Runner ─────────────────────────────────
def render_step(step, idx):
    step_type = step["type"]
    title = step["title"]
    content = step["content"]

    if step_type == "user":
        with st.chat_message("user"):
            st.markdown(content[:500])

    elif step_type == "thought":
        with st.expander(f"🧠 Step {idx} — {title}", expanded=False):
            st.markdown(content)

    elif step_type == "tool_call":
        with st.expander(f"🔧 Step {idx} — {title}", expanded=False):
            st.code(content, language="json")

    elif step_type == "tool_output":
        with st.expander(f"📋 Step {idx} — {title}", expanded=False):
            if len(content) > 3000:
                st.code(content[:3000] + "\n\n... (truncated)", language="text")
            else:
                st.code(content, language="text")

    elif step_type == "status":
        st.info(f"**{title}**: {content}")

    elif step_type == "success":
        st.success(f"**{title}**: {content}")

    elif step_type == "error":
        with st.expander(f"❌ Step {idx} — {title}", expanded=True):
            st.error(content)

    elif step_type == "result":
        st.success(f"**{title}**")


def run_agent(issue_number, user_prompt, model_name, base_url,
              use_reasoning, prompt_choice, owner, repo):
    """
    Runs the agent reusing init_agent() from agent.py and writes
    every step to st.session_state.agent_steps for live rendering.
    """
    steps = st.session_state.agent_steps
    steps.clear()

    def update_log(step_dict):
        steps.append(step_dict)
        with log_container:
            render_step(step_dict, len(steps) - 1)

    # ── 1. Check model availability ──
    update_log({"type": "status", "title": "🔍 Checking model",
                "content": f"Verifying availability of `{model_name}`..."})

    if not check_ollama_llm_availability(model_name, base_url):
        st.session_state.agent_error = f"Model '{model_name}' is not available at {base_url}"
        update_log({"type": "error", "title": "❌ Model Error",
                    "content": st.session_state.agent_error})
        return

    update_log({"type": "success", "title": "✅ Model ready",
                "content": f"`{model_name}` is available."})

    # ── 2. Build settings dict and init agent via agent.py ──
    runtime_settings = {
        "llm": {
            "provider": "ollama",
            "model_name": model_name,
            "base_url": base_url,
            "reasoning": use_reasoning,
        }
    }

    system_prompt_template = SYSTEM_PROMPT if prompt_choice == "Prompt 1 (Strict)" else SYSTEM_PROMPT2
    formatted_prompt = system_prompt_template.format(ISSUE_NUMBER=issue_number)

    try:
        agent = init_agent(
            model_name=model_name,
            system_prompt=formatted_prompt,
            reasoning=use_reasoning,
            settings=runtime_settings,
        )
    except RuntimeError as e:
        st.session_state.agent_error = f"Agent initialization failed: {e}"
        update_log({"type": "error", "title": "❌ Initialization Error",
                    "content": str(e)})
        return

    update_log({"type": "status", "title": "🤖 Agent initialized",
                "content": f"System prompt: {prompt_choice}\nIssue: #{issue_number}"})

    # ── 3. Stream the agent ──
    config = {"configurable": {"thread_id": f"streamlit_{issue_number}_{time.time()}"}}

    last_message_content = ""
    result = None
    seen_message_ids = set()

    try:
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": user_prompt}]},
            config=config,
            stream_mode="values",
        ):
            messages = chunk.get("messages", [])
            if not messages:
                continue

            msg = messages[-1]

            msg_id = getattr(msg, "id", None) or id(msg)
            if msg_id in seen_message_ids:
                continue
            seen_message_ids.add(msg_id)

            if msg.type == "human":
                update_log({"type": "user", "title": "👤 User", "content": msg.content})

            elif msg.type == "ai":
                last_message_content = msg.content

                if msg.content and '"name": "ResponseFormat"' not in msg.content:
                    update_log({"type": "thought", "title": "🧠 AI Reasoning",
                                "content": msg.content})

                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        update_log({
                            "type": "tool_call",
                            "title": f"🔧 Tool Call: `{tc['name']}`",
                            "content": json.dumps(tc["args"], indent=2),
                            "tool_name": tc["name"],
                        })

            elif msg.type == "tool":
                update_log({
                    "type": "tool_output",
                    "title": f"📋 Tool Output: `{msg.name}`",
                    "content": msg.content,
                    "tool_name": msg.name,
                })

            result = chunk

    except Exception as e:
        st.session_state.agent_error = f"Error during agent execution: {e}"
        update_log({"type": "error", "title": "❌ Execution Error",
                    "content": str(e)})
        return

    # ── 4. Extract the final result ──
    final_patch = None
    final_explanation = "No explanation provided."
    final_file_path = "Unknown"

    if result and "structured_response" in result and result["structured_response"]:
        resp = result["structured_response"]
        final_patch = resp.proposed_patch
        final_explanation = resp.explanation
        final_file_path = resp.file_path
    elif last_message_content:
        try:
            json_match = re.search(r"\{.*\}", last_message_content, re.DOTALL)
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
        clean_patch = final_patch.replace("\\n", "\n").replace('\\"', '"')
        st.session_state.agent_result = {
            "file_path": final_file_path,
            "explanation": final_explanation,
            "proposed_patch": clean_patch,
            "issue_number": issue_number,
            "model": model_name,
        }
        update_log({"type": "result", "title": "✅ Patch generated successfully",
                    "content": final_explanation})
    else:
        st.session_state.agent_error = "The agent did not produce a valid patch."
        update_log({
            "type": "error",
            "title": "❌ No patch produced",
            "content": last_message_content[:2000] if last_message_content else "Empty output",
        })


# ─────────────────────────── Execution Trigger ────────────────────────────
if start_btn:
    st.session_state.agent_running = True
    st.session_state.agent_result = None
    st.session_state.agent_error = None
    st.session_state.agent_steps = []

    with st.spinner("⏳ Agent running..."):
        run_agent(
            issue_number=int(issue_number),
            user_prompt=user_prompt,
            model_name=model_name,
            base_url=base_url,
            use_reasoning=use_reasoning,
            prompt_choice=prompt_choice,
            owner=owner,
            repo=repo,
        )
    st.session_state.agent_running = False
    st.rerun()


# ─────────────────────────── Render Steps (Final Display) ──────────────────
if st.session_state.agent_steps:
    st.divider()
    st.subheader("📜 Execution Log")

    for idx, step in enumerate(st.session_state.agent_steps):
        render_step(step, idx)

# ─────────────────────────── Error Display ────────────────────────────────
if st.session_state.agent_error:
    st.error(f"⚠️ {st.session_state.agent_error}")

# ─────────────────────────── Final Result Display ─────────────────────────
if st.session_state.agent_result:
    result = st.session_state.agent_result
    st.divider()
    st.subheader("🎯 Final Result")

    col_info, col_actions = st.columns([3, 1])
    with col_info:
        st.markdown(f"**Issue:** `#{result['issue_number']}`")
        st.markdown(f"**File:** `{result['file_path']}`")
        st.markdown(f"**Model:** `{result['model']}`")
    with col_actions:
        output_dir = ROOT_DIR / "agent_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"agent_output_issue_{result['issue_number']}.json"

        if st.button("💾 Save Patch to File", use_container_width=True):
            with open(output_file, "w") as f:
                json.dump(result, f, indent=4)
            st.success(f"Saved to `{output_file.name}`")

    # Explanation
    st.markdown("### 📝 Explanation")
    st.info(result["explanation"])

    # Patch with syntax highlighting
    st.markdown("### 🔨 Proposed Patch")
    st.code(result["proposed_patch"], language="python", line_numbers=True)

    # Download button
    st.download_button(
        label="📥 Download Patch (.py)",
        data=result["proposed_patch"],
        file_name=f"patch_issue_{result['issue_number']}.py",
        mime="text/x-python",
        use_container_width=True,
    )