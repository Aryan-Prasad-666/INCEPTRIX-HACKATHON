from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from sarvamai import SarvamAI
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify
import os
import re
import logging

# -------------------------------
# INITIAL SETUP
# -------------------------------

load_dotenv()
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

groq_key = os.getenv("GROQ_SCAR_KEY")
sarvam_key = os.getenv("SARVAM_API_KEY")

if not groq_key or not sarvam_key:
    raise ValueError("Missing API keys")

# -------------------------------
# LLM CONFIGURATION
# -------------------------------

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

# -------------------------------
# VECTOR MEMORY (FREE)
# -------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="neurodialectic_memory",
    embedding_function=embedding_model,
    persist_directory="./memory_db"
)

# -------------------------------
# SARVAM TRANSLATION
# -------------------------------

sarvam_client = SarvamAI(
    api_subscription_key=sarvam_key
)

def translate_text(text, target_lang):
    try:
        response = sarvam_client.text.translate(
            input=text,
            source_language_code="en-IN",
            target_language_code=target_lang
        )
        return response.translated_text
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text

# -------------------------------
# WORKFLOW CREATION
# -------------------------------

def create_neurodialectic_workflow(query, max_iterations=5):

    class GraphState(TypedDict):
        query: str
        draft: Optional[str]
        critique: Optional[str]
        validation: Optional[str]
        confidence: float
        history: str
        iteration: int
        final_answer: Optional[str]
        summary: Optional[str]

    workflow = StateGraph(GraphState)

    # ---------------- GENERATOR ----------------
    def generator_node(state):
        memories = vector_store.similarity_search(state["query"], k=2)
        memory_context = "\n".join([doc.page_content for doc in memories])

        prompt = f"""
        Answer clearly with structured reasoning.

        Question:
        {state['query']}

        Avoid repeating past failures:
        {memory_context}
        """

        draft = generator_llm.invoke(prompt).content

        return {
            "draft": draft,
            "history": state.get("history", "") + f"\n[Generator]\n{draft}"
        }

    # ---------------- CRITIC ----------------
    def critic_node(state):
        prompt = f"""
        Critically analyze this answer:

        {state['draft']}

        Identify logical flaws, unsupported claims,
        missing assumptions or weak reasoning.
        """

        critique = critic_llm.invoke(prompt).content

        return {
            "critique": critique,
            "history": state["history"] + f"\n[Critic]\n{critique}"
        }

    # ---------------- VALIDATOR ----------------
    def validator_node(state):
        prompt = f"""
        Evaluate answer quality considering critique.

        Answer:
        {state['draft']}

        Critique:
        {state['critique']}

        Format:
        Confidence: <0-1>
        Reason: <brief>
        """

        response = validator_llm.invoke(prompt).content
        match = re.search(r"Confidence:\s*([0-9.]+)", response)
        confidence = float(match.group(1)) if match else 0.5

        return {
            "validation": response,
            "confidence": confidence,
            "history": state["history"] + f"\n[Validator]\n{response}"
        }

    # ---------------- CONTROLLER ----------------
    def controller(state):
        if state["confidence"] >= 0.85 or state["iteration"] >= max_iterations:
            return "finalize"
        return "refine"

    # ---------------- REFINEMENT ----------------
    def refine_node(state):
        prompt = f"""
        Improve answer using critique.

        Original:
        {state['draft']}

        Critique:
        {state['critique']}
        """

        improved = generator_llm.invoke(prompt).content

        return {
            "draft": improved,
            "iteration": state.get("iteration", 0) + 1,
            "history": state["history"] + f"\n[Refined]\n{improved}"
        }

    # ---------------- FINALIZE ----------------
    def finalize_node(state):
        if state["confidence"] < 0.85:
            vector_store.add_documents([
                Document(
                    page_content=f"""
                    Query: {state['query']}
                    Weak Answer: {state['draft']}
                    Critique: {state['critique']}
                    """
                )
            ])
            vector_store.persist()

        return {"final_answer": state["draft"]}

    # ---------------- SUMMARIZER ----------------
    def summarizer_node(state):
        prompt = f"""
        Summarize the reasoning process clearly and concisely.

        Include:
        - Final conclusion
        - Key strengths
        - Major critiques
        - Confidence assessment

        Reasoning Trace:
        {state['history']}
        """

        summary = generator_llm.invoke(prompt).content

        return {"summary": summary}

    # ---------------- GRAPH STRUCTURE ----------------

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

# -------------------------------
# FLASK ROUTES
# -------------------------------

@app.route("/neurodialectic", methods=["GET", "POST"])
def neurodialectic():

    if request.method == "POST":
        query = request.form.get("query")
        max_iterations = int(request.form.get("max_iterations", 5))

        workflow = create_neurodialectic_workflow(query, max_iterations)
        app_workflow = workflow.compile()

        result = app_workflow.invoke({
            "query": query,
            "iteration": 0,
            "history": ""
        })

        return render_template(
            "neurodialectic.html",
            final_answer=result.get("final_answer"),
            reasoning_trace=result.get("history"),
            confidence=result.get("confidence"),
            summary=result.get("summary")
        )

    return render_template("neurodialectic.html")

@app.route("/translate_summary", methods=["POST"])
def translate_summary():
    data = request.json

    translated_summary = translate_text(
        data.get("summary", ""),
        data.get("language")
    )

    return jsonify({
        "translated_summary": translated_summary
    })

# -------------------------------

if __name__ == "__main__":
    app.run(debug=True)