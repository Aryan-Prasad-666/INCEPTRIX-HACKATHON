from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
import os
import re
import json
import logging
from datetime import datetime

# ---------------- SETUP ----------------

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

groq_key = os.getenv("GROQ_SCAR_KEY")
cohere_key = os.getenv("COHERE_API_KEY")

if not all([groq_key, cohere_key]):
    raise ValueError("Missing API keys")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# ---------------- WORKFLOW ----------------

def create_workflow(query, max_iterations=5):

    class GraphState(TypedDict):
        query: str
        draft: Optional[str]
        critique: Optional[str]
        validation: Optional[str]
        confidence: float
        iteration: int
        final_answer: Optional[str]
        summary: Optional[str]
        generator_output: Optional[str]
        critic_output: Optional[str]
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

        draft = generator_llm.invoke(prompt[:MAX_PROMPT_CHARS]).content

        return {
            "draft": draft,
            "generator_output": draft,
            "iteration": 0,
            "refinement_outputs": []
        }

    # -------- CRITIC --------
    def critic_node(state):

        prompt = f"""
Critically analyze the following answer.

Answer:
{state['draft']}
"""

        critique = critic_llm.invoke(prompt[:MAX_PROMPT_CHARS]).content

        return {
            "critique": critique,
            "critic_output": critique
        }

    # -------- VALIDATOR --------
    def validator_node(state):

        prompt = f"""
Evaluate answer quality.

Answer:
{state['draft']}

Critique:
{state['critique']}

Format strictly as:
Confidence: <0-1>
Reason: <brief>
"""

        response = validator_llm.invoke(prompt[:MAX_PROMPT_CHARS]).content

        match = re.search(r"Confidence:\s*([0-9.]+)", response)
        confidence = float(match.group(1)) if match else 0.5

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
Improve the answer using this critique.

Answer:
{state['draft']}

Critique:
{state['critique']}
"""

        improved = generator_llm.invoke(prompt[:MAX_PROMPT_CHARS]).content

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
            compressed = summarizer_llm.invoke(
                f"Summarize failure in under 120 words:\n{state['critique']}"
            ).content[:500]

            vector_store.add_documents([
                Document(page_content=f"Failure Pattern: {compressed}")
            ])
            vector_store.persist()

        return {"final_answer": state["draft"]}

    # -------- SUMMARIZER --------
    def summarizer_node(state):

        prompt = f"""
Summarize the full reasoning process.

Final Answer:
{state['final_answer']}

Critique:
{state['critique']}

Validation:
{state['validation']}

Include:
- Conclusion
- Strengths
- Weaknesses
- Confidence
"""

        summary = summarizer_llm.invoke(prompt).content
        summary = summary[:MAX_SUMMARY_CHARS]

        return {"summary": summary}

    # -------- GRAPH --------
    workflow.add_node("generator", generator_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("refine", refine_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("summarizer", summarizer_node)

    workflow.set_entry_point("generator")
    workflow.add_edge("generator", "critic")
    workflow.add_edge("critic", "validator")

    workflow.add_conditional_edges(
        "validator",
        controller,
        {"refine": "refine", "finalize": "finalize"}
    )

    workflow.add_edge("refine", "critic")
    workflow.add_edge("finalize", "summarizer")
    workflow.add_edge("summarizer", END)

    return workflow

# ---------------- TERMINAL EXECUTION ----------------

if __name__ == "__main__":

    print("\n===== NeuraDialectic Terminal Mode =====\n")

    query = input("Enter your question:\n> ")
    max_iterations_input = input("Max refinement iterations (default 5): ")

    try:
        max_iterations = int(max_iterations_input)
    except:
        max_iterations = 5

    workflow = create_workflow(query, max_iterations)
    result = workflow.compile().invoke({"query": query})

    print("\n===== FINAL ANSWER =====\n")
    print(result.get("final_answer"))

    print("\n===== SUMMARY =====\n")
    print(result.get("summary"))

    print("\n===== GENERATOR OUTPUT =====\n")
    print(result.get("generator_output"))

    print("\n===== CRITIC OUTPUT =====\n")
    print(result.get("critic_output"))

    print("\n===== VALIDATOR OUTPUT =====\n")
    print(result.get("validator_output"))

    print("\n===== REFINEMENT STEPS =====\n")
    for i, r in enumerate(result.get("refinement_outputs", []), 1):
        print(f"\n--- Refinement {i} ---\n")
        print(r)

    print("\n===== CONFIDENCE =====\n")
    print(result.get("confidence"))

    # Save run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(OUTPUT_DIR, f"neurodialectic_{timestamp}.json")

    run_data = {
        "metadata": {
            "timestamp": timestamp,
            "query": query
        },
        "outputs": result
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=4, ensure_ascii=False)

    print(f"\nRun saved to {file_path}")