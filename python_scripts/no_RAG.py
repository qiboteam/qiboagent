#!/usr/bin/env python3
"""
LLM Autoscoring without RAG
"""

import logging
from tqdm import tqdm
import os
import sys
import re
import subprocess
import json
import tempfile
from pathlib import Path
from typing import List

from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

def _normalize_text(s: str) -> str:
    """Normalize text by removing extra whitespace."""
    return re.sub(r"\s+", " ", s.strip())

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

def auto_scoring(results: list, golden_answers: dict, model_name: str):
    """Score batch answers: execute code, match expected output, detect hallucinations, compute pylint score."""
    scores = []
    for result in results:
        question = result["question"]
        golden_answer = golden_answers.get(question, {})
        score = {
            "model": model_name,
            "question": question,
            "answer": result["answer"], 
            "correctness": 0,
            "hallucination": 1,
            "pylint score": 0.0,
            "error": None
        }
        answer = result["answer"]
        raw_expected_output = golden_answer.get("expected_output", "")
        
        # Normalize expected_output into a list of non-empty strings
        if isinstance(raw_expected_output, str):
            expected_outputs = [raw_expected_output.strip()] if raw_expected_output.strip() else []
        elif isinstance(raw_expected_output, list):
            expected_outputs = [eo.strip() for eo in raw_expected_output if isinstance(eo, str) and eo.strip()]
        else:
            expected_outputs = []
        
        code = extract_code(answer)

        if code:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            try:
                proc = subprocess.run(
                    [sys.executable, tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                stdout = _normalize_text(proc.stdout or "")
                stderr = _normalize_text(proc.stderr or "")
                combined = (stdout + " " + stderr).strip()

                if stderr:
                    score["error"] = stderr

                # Check correctness
                if expected_outputs and any(_normalize_text(eo) in combined for eo in expected_outputs):
                    score["correctness"] = 1
                
                # Check hallucination
                if any(k in stderr for k in ("AttributeError", "NameError", "ImportError", "ModuleNotFoundError")):
                    score["hallucination"] = 0

                # Pylint score
                score["pylint score"] = get_pylint_score(
                    tmp_path,
                    disable="missing-module-docstring,missing-final-newline"
                )

            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout executing code for question: {question}")
                score["error"] = "Execution timeout (30s)"
            except Exception as e:
                logger.warning(f"Error executing code: {e}")
                score["error"] = str(e)
            finally:
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass
        
        scores.append(score)

    # Save scores to file
    Path("scoring_noRAG_json").mkdir(exist_ok=True)
    output_file = f"scoring/scoring_noRAG_json/scoring_{model_name.replace('/', '_')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Scoring results saved to {output_file}")
    return scores


def initialize_llm(settings: dict):
    """Initialize LLM based on settings configuration."""
    if settings["llm"]["provider"] == "ollama":
        return OllamaLLM(model=settings["llm"]["model_name"], base_url=["llm"]["base_url"], temperature=0.0)
    elif settings["llm"]["provider"] == "google_genai":
        return ChatGoogleGenerativeAI(model=settings["llm"]["model_name"],
                                      google_api_key=settings["llm"]["api_key"],
                                      temperature=0.0,
                                      convert_system_message_to_human=True)
    else:
        raise ValueError(f"Unsupported LLM provider: {settings['llm']['provider']}")
    
def LLM_answer_questions(llm, questions: List[str]) -> List[dict]:
    """Get answers from LLM for a list of questions using the global prompt template."""
    results = []
    chain = prompt | llm | StrOutputParser()
    
    for question in tqdm(questions, desc="Getting LLM answers", ncols=80):
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
    
    return results

#------------------------- MAIN ------------------------ #

def main():
    # Load settings
    settings_file = "../settings_json/settings.json"
    if not Path(settings_file).exists():
        logger.error(f"Settings file not found: {settings_file}")
        sys.exit(1)
    
    with open(settings_file, "r", encoding="utf-8") as f:
        settings = json.load(f)

    questions_file = "../settings_json/questions_2.json"
    golden_file = "../settings_json/golden_answers_2.json"

    logger.info(f"Using LLM model: {settings['llm']['model_name']}")

    llm = initialize_llm(settings)

    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    with open(golden_file, "r", encoding="utf-8") as f:
        golden_answers = json.load(f)

    print(f"no RAG run with {settings['llm']['model_name']}")

    results = LLM_answer_questions(llm, questions)

    logger.info("Starting auto-scoring...")
    scores = auto_scoring(results, golden_answers, settings['llm']['model_name'])
    
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


