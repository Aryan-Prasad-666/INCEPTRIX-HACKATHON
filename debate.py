from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from sarvamai import SarvamAI
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify
import os
import re
import json
import logging
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler

# ---------------- SETUP ----------------

load_dotenv()
app = Flask(__name__)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOGGING ----------------

logger = logging.getLogger("neurodialectic")
logger.setLevel(logging.INFO)

log_file_path = os.path.join(
    OUTPUT_DIR,
    f"neurodialectic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

file_handler = RotatingFileHandler(
    log_file_path,
    maxBytes=5 * 1024 * 1024,
    backupCount=2
)

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ---------------- API KEYS ----------------

groq_key = os.getenv("GROQ_KEERTHANA_KEY")
cohere_key = os.getenv("COHERE_API_KEY")
sarvam_key = os.getenv("SARVAM_API_KEY")

if not all([groq_key, cohere_key, sarvam_key]):
    raise ValueError("Missing API keys")

# ---------------- LIMITS ----------------

MAX_PROMPT_CHARS = 6000
MAX_MEMORY_CHARS = 1000
MAX_SUMMARY_CHARS = 1800

# ---------------- MODELS ----------------

generator_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=groq_key,
    temperature=0.6
)

critic_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_key,
    temperature=0.3
)

redteam_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_key,
    temperature=0.4
)

validator_llm = ChatGroq(
    model="qwen/qwen3-32b",
    api_key=groq_key,
    temperature=0.1
)

summarizer_llm = ChatCohere(
    api_key=cohere_key,
    temperature=0.3
)

# ---------------- MEMORY ----------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="neurodialectic_memory",
    embedding_function=embedding_model,
    persist_directory="./memory_db"
)

# ---------------- SARVAM ----------------

sarvam_client = SarvamAI(api_subscription_key=sarvam_key)

# ---------------- UTILITIES ----------------

def safe_invoke(llm, prompt, node_name="LLM", retries=3, base_delay=2):
    for attempt in range(retries):
        try:
            logger.info(f"{node_name} | Attempt {attempt+1}")
            response = llm.invoke(prompt[:MAX_PROMPT_CHARS]).content
            logger.info(f"{node_name} | Success")
            return response
        except Exception as e:
            logger.warning(f"{node_name} | Failed attempt {attempt+1} | {str(e)}")
            if attempt < retries - 1:
                sleep_time = base_delay * (2 ** attempt)
                time.sleep(sleep_time)
            else:
                raise


# ---------------- WORKFLOW ----------------

def create_workflow(query, max_iterations=5):

    class GraphState(TypedDict):
        query: str
        draft: Optional[str]
        critique: Optional[str]
        redteam: Optional[str]
        validation: Optional[str]
        confidence: float
        iteration: int
        final_answer: Optional[str]
        summary: Optional[str]
        generator_output: Optional[str]
        critic_output: Optional[str]
        redteam_output: Optional[str]
        validator_output: Optional[str]
        refinement_outputs: List[str]

    workflow = StateGraph(GraphState)

    # -------- GENERATOR --------
    def generator_node(state):

        memories = vector_store.similarity_search(state["query"], k=1)

        memory_context = ""
        for doc in memories:
            memory_context += doc.page_content[:500] + "\n"
        memory_context = memory_context[:MAX_MEMORY_CHARS]

        prompt = f"""
Answer clearly with structured reasoning.

Question:
{state['query']}

Avoid repeating past failure patterns:
{memory_context}
"""

        draft = safe_invoke(generator_llm, prompt, "GENERATOR")

        return {
            "draft": draft,
            "generator_output": draft,
            "iteration": 0,
            "refinement_outputs": []
        }

    # -------- CRITIC --------
    def critic_node(state):

        prompt = f"Critically analyze this answer:\n{state['draft']}"
        critique = safe_invoke(critic_llm, prompt, "CRITIC")

        return {
            "critique": critique,
            "critic_output": critique
        }

    # -------- RED TEAM --------
    def redteam_node(state):

        prompt = f"""
You are an adversarial red-team AI.

Attack the reasoning.
Identify hallucinations.
Detect hidden assumptions.
Provide counterexamples.
Flag logical gaps.
Estimate risk severity.

Question:
{state['query']}

Answer:
{state['draft']}

Critique:
{state['critique']}

Output format:
1. Hallucination Risk:
2. Logical Weaknesses:
3. Missing Assumptions:
4. Adversarial Counterexample:
5. Severity Score (0-1):
"""

        redteam_analysis = safe_invoke(redteam_llm, prompt, "REDTEAM")

        return {
            "redteam": redteam_analysis,
            "redteam_output": redteam_analysis
        }

    # -------- VALIDATOR --------
    def validator_node(state):

        prompt = f"""
Evaluate answer robustness.

Answer:
{state['draft']}

Critique:
{state['critique']}

Red Team Analysis:
{state['redteam']}

Format strictly:
Confidence: <0-1>
Reason: <brief>
"""

        response = safe_invoke(validator_llm, prompt, "VALIDATOR")

        match = re.search(r"Confidence:\s*([0-9.]+)", response)
        confidence = float(match.group(1)) if match else 0.5

        # Severity penalty
        severity_match = re.search(r"Severity Score.*?([0-9.]+)", state["redteam"])
        if severity_match:
            severity = float(severity_match.group(1))
            confidence = max(0, confidence - severity * 0.01)

        return {
            "validation": response,
            "confidence": confidence,
            "validator_output": response
        }

    # -------- CONTROLLER --------
    def controller(state):
        if state["confidence"] >= 0.85 or state["iteration"] >= max_iterations:
            return "finalize"
        return "refine"

    # -------- REFINE --------
    def refine_node(state):

        prompt = f"""
Improve the answer using:

Critique:
{state['critique']}

Red Team Findings:
{state['redteam']}

Previous Answer:
{state['draft']}

Fix logical gaps, weak assumptions, and adversarial vulnerabilities.
"""

        improved = safe_invoke(generator_llm, prompt, "REFINE")

        refinements = state.get("refinement_outputs", [])
        refinements.append(improved)

        return {
            "draft": improved,
            "iteration": state.get("iteration", 0) + 1,
            "refinement_outputs": refinements
        }

    # -------- FINALIZE --------
    def finalize_node(state):

        if state["confidence"] < 0.85:

            compressed = safe_invoke(
                summarizer_llm,
                f"Summarize failure in under 120 words:\n{state['critique']}",
                "FAILURE-SUMMARIZER"
            )[:500]

            vector_store.add_documents([
                Document(page_content=f"Failure Pattern: {compressed}")
            ])
            vector_store.persist()

        return {"final_answer": state["draft"]}

    # -------- SUMMARIZER --------
    def summarizer_node(state):

        prompt = """
Summarize reasoning (≤1700 characters).
Include conclusion, strengths, critiques, confidence.
"""

        summary = safe_invoke(summarizer_llm, prompt, "SUMMARIZER")
        summary = summary[:MAX_SUMMARY_CHARS]

        return {"summary": summary}

    # -------- GRAPH WIRING --------

    workflow.add_node("generator", generator_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("redteam_agent", redteam_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("refine", refine_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("summarizer", summarizer_node)

    workflow.set_entry_point("generator")

    workflow.add_edge("generator", "critic")
    workflow.add_edge("critic", "redteam_agent")
    workflow.add_edge("redteam_agent", "validator")

    workflow.add_conditional_edges(
        "validator",
        controller,
        {"refine": "refine", "finalize": "finalize"}
    )

    workflow.add_edge("refine", "critic")
    workflow.add_edge("finalize", "summarizer")
    workflow.add_edge("summarizer", END)

    return workflow


# ---------------- ROUTES ----------------

@app.route("/neurodialectic", methods=["GET", "POST"])
def neurodialectic():

    if request.method == "POST":

        query = request.form.get("query")
        max_iterations = int(request.form.get("max_iterations", 3))

        workflow = create_workflow(query, max_iterations)
        result = workflow.compile().invoke({"query": query})

        return render_template("neurodialectic.html", **result)

    return render_template("neurodialectic.html")


if __name__ == "__main__":
    app.run(debug=True)