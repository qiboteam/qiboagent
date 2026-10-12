#!/usr/bin/env python3
"""
LLM Autoscoring without RAG
"""

import logging
import os
from tqdm import tqdm
import sys
import re
import subprocess, requests
import json
import tempfile
from pathlib import Path
from typing import List

from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.rate_limiters import InMemoryRateLimiter

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an **expert quantum developer** specialized exclusively in the **Qibo quantum computing library**.
Your goal is to provide precise, actionable, and contextually grounded answers to the user's questions.

INSTRUCTIONS:
- If the question is not answerable from the context, respond with a short generic answer or "I don't know".
- Do NOT invent any functions, classes, or methods that do not exist in the Qibo library.
- Write only A SINGLE BLOCK CODE in Python if the answer requires code, do NOT write multiple code blocks or other languages. The single code block must be formatted using Markdown (```python ... ```).
- When providing code, include a brief explanation of what the code does and why it's the correct approach based on the context.
- Always import the necessary modules from Qibo, prefer to use Qibo built-in functions and methods.
- For simple or conversational questions (like "What is your name?", "How are you?", "What is Qibo?"), answer briefly and clearly even if not in the context. For example, "My name is QiboLLM."

EXAMPLE:
Question: Build a qibo circuit of 1 qubit, add an H gate to it, execute it and save the final state to the file `state.npy`.

Answer:
```python
import numpy as np
from qibo import Circuit, gates

c = Circuit(1)
c.add(gates.H(0))

state = c().state()

with open('state.npy', 'wb') as f:
    np.save(f, state)
```

Explanation:
You can append gates to a Circuit object through the method `add` and execute the circuit simply by calling it `c()`, then you can extract the final state by using the `state` method from the resulting object.
"""),
        ("human", "{question}")
    ])

#------------------------- UTILS AND AUTO-SCORING------------------------ #

def extract_code(answer: str) -> str:
    """Extract the first Python code block from the answer."""
    code_blocks = re.findall(r"```python(.*?)```", answer, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    code_blocks = re.findall(r"```(.*?)```", answer, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    return ""

def get_pylint_score(
    pyfile_path: str,
    disable: str = "missing-module-docstring,missing-final-newline",
    timeout: int = 30,
) -> float:
    """
    Run pylint on a Python file using the current Python environment (venv) 
    and return a normalized score (0.0..1.0). 
    Returns 0.0 if pylint fails or the score cannot be determined.
    """
    cmd = [
        sys.executable, "-m", "pylint", pyfile_path,
        f"--disable={disable}",
        "--score=y",
        "--output-format=text"
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        output = proc.stdout + "\n" + proc.stderr
        
        match = re.search(r"rated at\s+(-?\d+(?:\.\d+)?)/10", output, re.IGNORECASE)
        if match:
            try:
                rating = float(match.group(1))
                return max(0.0, min(1.0, rating / 10.0))
            except ValueError:
                pass

        for line in output.splitlines():
            if "rated" in line.lower():
                for n in re.findall(r"-?\d+(?:\.\d+)?", line):
                    try:
                        val = float(n)
                        if 0 <= val <= 10:
                            return val / 10.0
                    except ValueError:
                        continue

    except subprocess.TimeoutExpired:
        logger.warning("Pylint timeout for file: %s", pyfile_path)
    except FileNotFoundError:
        logger.warning("Pylint executable not found in current Python environment.")
    except Exception as e:
        logger.warning("Pylint execution failed for file %s: %s", pyfile_path, e)

    return 0.0

def save_json(data, path: str):
    """Write data to a JSON file atomically, so that a crash never leaves a truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def _normalize_and_extract(text: str) -> str:
    """
    Normalize text by:
    1. Extracting meaningful content (remove log prefixes, timestamps)
    2. Normalizing whitespace
    3. Standardizing numeric formats
    """
    if not text:
        return ""
    
    # Remove log prefixes
    log_patterns = [
        r'\[.*?\|.*?\|.*?\]:\s*',  # [Qibo x.x.x|INFO|timestamp]:
        r'\[INFO\]\s*',          
        r'\[WARNING\]\s*',          
        r'\[ERROR\]\s*',          
        r'\[DEBUG\]\s*',          
    ]
    for pattern in log_patterns:
        text = re.sub(pattern, '', text)

    # Remove timestamps and dates
    text = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', '', text)

    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text.strip())

    # Standardize complex numbers in arrays
    # 0.        +0.j -> 0.+0.j
    text = re.sub(r'(\d)\s+\+', r'\1+', text)
    text = re.sub(r'\+\s+(\d)', r'+\1', text)

    # Remove spaces around special characters in arrays
    text = re.sub(r'\s*\[\s*', '[', text)
    text = re.sub(r'\s*\]\s*', ']', text)
    text = re.sub(r'\s*j\s*', 'j', text)
    
    return text.strip()


_NUMBER = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")


def _tokenize(text: str) -> list:
    """Split text into numbers (as floats) and single non-space characters."""
    tokens, pos = [], 0
    for m in _NUMBER.finditer(text):
        tokens.extend(c for c in text[pos:m.start()] if not c.isspace())
        tokens.append(float(m.group()))
        pos = m.end()
    tokens.extend(c for c in text[pos:] if not c.isspace())
    return tokens


def _numeric_match(expected_norm: str, actual_norm: str, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """
    Check if the expected output appears in the actual output with the same text and numbers equal
    within tolerance (numpy.isclose defaults), so that signed zeros (-0.) and floating point noise
    (e.g. 1e-17 imaginary parts) in the printed result are not counted as errors.
    """
    exp = _tokenize(expected_norm)
    act = _tokenize(actual_norm)
    if not any(isinstance(t, float) for t in exp):
        return False
    n = len(exp)
    for start in range(len(act) - n + 1):
        for e, a in zip(exp, act[start:start + n]):
            if isinstance(e, float) != isinstance(a, float):
                break
            if isinstance(e, float):
                if abs(a - e) > atol + rtol * abs(e):
                    break
            elif a != e:
                break
        else:
            return True
    return False


def _output_matches(expected: str, actual: str) -> bool:
    """
    Check if the normalized expected output is contained in the normalized actual output: exact
    substring, or same text with numbers equal within tolerance. Empty strings never match.
    """
    expected_norm = _normalize_and_extract(expected)
    actual_norm = _normalize_and_extract(actual)
    if not expected_norm or not actual_norm:
        return False
    return expected_norm in actual_norm or _numeric_match(expected_norm, actual_norm)


def auto_scoring(results: list, golden_answers: dict, model_name: str, save: bool = False, scoring_path: str | None = None):
    """
    Score batch answers with the same strict criterion used by every pipeline (no-RAG, semantic, hybrid).
    An answer is correct only if its code block runs successfully (exit code 0, no timeout) and one of the
    expected outputs, after normalization, is an exact substring of the produced stdout + stderr.
    """
    path = (scoring_path or f"./scoring/scoring_noRAG_json/scoring_{model_name.replace('/', '_')}.json") if save else None
    scores = []
    for idx, result in enumerate(results, start=1):
        question = result["question"]
        golden_answer = golden_answers.get(question, {})
        score = {
            "model": model_name,
            "question_number": idx, 
            "question": question,
            "answer": result["answer"],
            "correctness": 0,
            "grounding": 0,
            "hallucination": 1,
            "pylint score": 0.0,
            "exit_code": None,
            "error": None
        }
        answer = result["answer"]
        raw_expected_output = golden_answer.get("expected_output", "")

        if isinstance(raw_expected_output, str):
            expected_outputs = [raw_expected_output.strip()] if raw_expected_output.strip() else []
        elif isinstance(raw_expected_output, list):
            expected_outputs = [eo.strip() for eo in raw_expected_output if isinstance(eo, str) and eo.strip()]
        else:
            expected_outputs = []

        code = extract_code(answer)

        stdout = ""
        stderr = ""
        exit_code = None

        if code:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            try:
                logger.debug(f"Executing code for question {idx}: {question[:50]}...")
                proc = subprocess.run(
                    [sys.executable, tmp_path], 
                    capture_output=True, 
                    text=True, 
                    timeout=30
                )
                stdout = proc.stdout or ""
                stderr = proc.stderr or ""
                exit_code = proc.returncode
                logger.debug(f"Code execution completed for question {idx}")
            except subprocess.TimeoutExpired:
                stderr = "Execution timeout (30s)"
                logger.warning(f"Code execution timeout for question {idx}")
            except Exception as e:
                stderr = f"Error executing code: {e}"
                logger.error(f"Code execution error for question {idx}: {e}")
            finally:
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass

        combined = (stdout + " " + stderr).strip()
        score["exit_code"] = exit_code

        if stderr and not any(k in stderr for k in ("DeprecationWarning", "FutureWarning", "UserWarning")):
            score["error"] = stderr

        # No code, timeouts and failed executions are never correct, even if a partial output matches
        if exit_code == 0 and expected_outputs:
            for eo in expected_outputs:
                if _output_matches(eo, combined):
                    score["correctness"] = 1
                    break

        if any(k in stderr for k in ("AttributeError", "NameError", "ImportError", "ModuleNotFoundError")):
            score["hallucination"] = 0

        # Pylint score
        if code:
            try:
                with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
                    tmp.write(code)
                    tmp_path = tmp.name
                score["pylint score"] = get_pylint_score(
                    tmp_path,
                    disable="missing-module-docstring,missing-final-newline"
                )
            finally:
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass

        sources = result.get("sources", [])
        expected_sources = golden_answer.get("expected_sources", [])
        if expected_sources:
            for src in expected_sources:
                if any(src in s for s in sources):
                    score["grounding"] += 1
            score["grounding"] /= len(expected_sources)

        scores.append(score)
        # Saved after every question, so partial results survive a crash
        if save:
            save_json(scores, path)

    if save:
        logger.info(f"Saved scoring results to {path}")

    return scores

def check_ollama_llm_availability(model_name: str, base_url: str) -> bool:
    """Check if an Ollama model is available; if not, attempt to pull it."""
    # Check if Ollama server is reachable
    try:
        # get list of available models, ollama api: https://docs.ollama.com/api
        response = requests.get(f"{base_url}/api/tags", timeout=5)
    except (requests.ConnectionError, requests.Timeout):
        logger.error(f"Cannot connect to Ollama at {base_url}. Is the server running?")
        return False

    # Check if model is already downloaded
    model_short = model_name.split(":")[0]
    models = response.json().get("models", [])
    available = [m["name"].split(":")[0] for m in models]
    if model_short in available:
        logger.info(f"Model '{model_name}' is available.")
        return True

    # Attempt to pull the model via CLI
    logger.info(f"Model '{model_name}' not found locally. Pulling...")
    result = subprocess.run(["ollama", "pull", model_name])
    if result.returncode != 0:
        logger.error(f"Invalid model name '{model_name}'. Check available models at https://ollama.com/library")
        return False

    logger.info(f"Model '{model_name}' pulled successfully.")
    return True

def initialize_llm(settings: dict):
    """Initialize LLM based on settings configuration."""
    if settings["llm"]["provider"] == "ollama":
        model_name = settings["llm"]["model_name"]
        base_url = settings["llm"]["base_url"]
        
        # Check if model is available, pull if necessary
        if not check_ollama_llm_availability(model_name, base_url):
            if not check_ollama_llm_availability(settings["llm"]["model_name"], settings["llm"]["base_url"]):
                raise RuntimeError(f"Model {model_name} is not available and could not be pulled.")
        
        return OllamaLLM(model=model_name, base_url=base_url, temperature=0.0)
    elif settings["llm"]["provider"] == "google_genai":
        # Spread the requests to stay within the Gemini API rate limit (requests per minute)
        rate_limiter = InMemoryRateLimiter(requests_per_second=float(os.environ.get("GEMINI_RPM", "8")) / 60)
        return ChatGoogleGenerativeAI(model=settings["llm"]["model_name"],
                                      google_api_key=settings["llm"]["api_key"],
                                      temperature=0.0,
                                      convert_system_message_to_human=True,
                                      rate_limiter=rate_limiter)
    else:
        raise ValueError(f"Unsupported LLM provider: {settings['llm']['provider']}")
    
def LLM_answer_questions(llm, questions: List[str], output_file: str) -> List[dict]:
    """Get answers from LLM for a list of questions using the global prompt template and save them to a JSON file."""
    results = []
    chain = prompt | llm | StrOutputParser()
    
    start_index = 0
    for idx, question in enumerate(tqdm(questions[start_index:], desc="Getting LLM answers", ncols=80), start=start_index):
        try:
            #logger.info("waiting 10 mins to avoid rate limits...")
            #time.sleep(600)
            answer = chain.invoke({"question": question})
            results.append({
                "question": question,
                "answer": answer
            })
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            results.append({
                "question": question,
                "answer": f"Error: {str(e)}"
            })
        # Saved after every question, so partial results survive a crash
        save_json(results, output_file)

    logger.info(f"Saved {len(results)} answers to {output_file}")
    return results

#------------------------- MAIN ------------------------ #

def main():
    # Load settings
    settings_file = "./settings_json/settings.json"
    if not Path(settings_file).exists():
        logger.error(f"Settings file not found: {settings_file}")
        sys.exit(1)
    
    with open(settings_file, "r", encoding="utf-8") as f:
        settings = json.load(f)

    questions_file = settings.get("questions_file", "./settings_json/questions.json")
    golden_file = settings.get("golden_file", "./settings_json/golden_answers.json")

    logger.info(f"Using LLM model: {settings['llm']['model_name']}")

    llm = initialize_llm(settings)

    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    with open(golden_file, "r", encoding="utf-8") as f:
        golden_answers = json.load(f)

    output_file = f"./answers/answers_noRAG_json/answers_{settings['llm']['model_name'].replace('/', '_')}.json"
    results = LLM_answer_questions(llm, questions, output_file)

    logger.info("Starting auto-scoring...")
    scores_file = f"./scoring/scoring_noRAG_json/scoring_{settings['llm']['model_name'].replace('/', '_')}.json"
    scores = auto_scoring(results, golden_answers, settings['llm']['model_name'], save=True, scoring_path=scores_file)
    
    total_questions = len(scores)
    correct_answers = sum(1 for score in scores if score["correctness"] == 1)
    accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Model: {settings['llm']['model_name']}")
    logger.info(f"Total questions: {total_questions}")
    logger.info(f"Correct answers: {correct_answers} ({accuracy:.2%})")
    logger.info(f"{'='*50}\n")


if __name__ == "__main__":
    main()


