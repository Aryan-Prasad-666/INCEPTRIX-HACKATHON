from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import re
import time
import requests

# ---------------- SETUP ----------------

load_dotenv()

groq_key = os.getenv("GROQ_KEERTHANA_KEY")
cohere_key = os.getenv("COHERE_API_KEY")
serper_key = os.getenv("SERPER_API_KEY")

if not groq_key or not cohere_key or not serper_key:
    raise ValueError("Missing API keys")

# ---------------- MODELS ----------------

generator_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=groq_key,
    temperature=0.6
)

verifier_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_key,
    temperature=0.2
)

logic_llm = ChatGroq(
    model="qwen/qwen3-32b",
    api_key=groq_key,
    temperature=0.1
)

confidence_llm = ChatCohere(
    api_key=cohere_key,
    temperature=0.2
)

# ---------------- SAFE INVOKE ----------------

def safe_invoke(llm, prompt, retries=3, base_delay=2):
    for attempt in range(retries):
        try:
            return llm.invoke(prompt).content
        except Exception:
            if attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt))
            else:
                raise

# ---------------- SERPER SEARCH ----------------

def search_web(query):
    url = "https://google.serper.dev/search"

    payload = {
        "q": query,
        "num": 5
    }

    headers = {
        "X-API-KEY": serper_key,
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()

    evidence_text = ""

    for result in data.get("organic", []):
        evidence_text += f"Title: {result.get('title','')}\n"
        evidence_text += f"Snippet: {result.get('snippet','')}\n"
        evidence_text += f"Link: {result.get('link','')}\n\n"

    return evidence_text if evidence_text else "No evidence found."

# ---------------- WORKFLOW ----------------

def create_workflow():

    class GraphState(TypedDict):
        query: str
        answer: Optional[str]
        claims: Optional[List[str]]
        fact_score: float
        logic_score: float
        confidence_score: float
        hallucination_risk: float
        final_report: Optional[str]

    workflow = StateGraph(GraphState)

    # -------- GENERATE --------
    def generate_node(state):
        prompt = f"Answer clearly:\n{state['query']}"
        answer = safe_invoke(generator_llm, prompt)
        return {"answer": answer}

    # -------- EXTRACT CLAIMS --------
    def extract_claims_node(state):
        prompt = f"""
Extract factual claims as separate bullet points.

Answer:
{state['answer']}
"""
        result = safe_invoke(verifier_llm, prompt)

        claims = []
        for line in result.split("\n"):
            line = line.strip("-• ").strip()
            if line:
                claims.append(line)

        return {"claims": claims}

    # -------- FACT CHECK CLAIM-BY-CLAIM --------
    def fact_check_node(state):

        claim_scores = []

        for claim in state["claims"]:

            evidence = search_web(claim)

            prompt = f"""
Check whether this claim is supported by the web evidence.

Claim:
{claim}

Evidence:
{evidence}

Return:
FactScore: <0-1>
"""

            result = safe_invoke(verifier_llm, prompt)

            match = re.search(r"FactScore:\s*([0-9.]+)", result)
            score = float(match.group(1)) if match else 0.5
            claim_scores.append(score)

        final_fact_score = sum(claim_scores) / len(claim_scores) if claim_scores else 0.5

        return {"fact_score": final_fact_score}

    # -------- LOGIC CHECK --------
    def logic_check_node(state):
        prompt = f"""
Check logical consistency of this answer.

Answer:
{state['answer']}

Return:
LogicScore: <0-1>
"""
        result = safe_invoke(logic_llm, prompt)
        match = re.search(r"LogicScore:\s*([0-9.]+)", result)
        score = float(match.group(1)) if match else 0.5
        return {"logic_score": score}

    # -------- CONFIDENCE --------
    def confidence_node(state):
        prompt = f"""
Estimate epistemic confidence of this answer.

Answer:
{state['answer']}

Return:
ConfidenceScore: <0-1>
"""
        result = safe_invoke(confidence_llm, prompt)
        match = re.search(r"ConfidenceScore:\s*([0-9.]+)", result)
        score = float(match.group(1)) if match else 0.5
        return {"confidence_score": score}

    # -------- RISK CALCULATION --------
    def risk_node(state):

        fact = state["fact_score"]
        logic = state["logic_score"]
        conf = state["confidence_score"]

        hallucination_risk = 1 - (0.5 * fact + 0.3 * logic + 0.2 * conf)

        report = f"""
Hallucination Detection Report
--------------------------------
Fact Score: {fact:.2f}
Logic Score: {logic:.2f}
Confidence Score: {conf:.2f}

Final Risk: {hallucination_risk:.2f}
"""

        return {
            "hallucination_risk": hallucination_risk,
            "final_report": report
        }

    # -------- GRAPH --------

    workflow.add_node("generate_node", generate_node)
    workflow.add_node("extract_claims_node", extract_claims_node)
    workflow.add_node("fact_check_node", fact_check_node)
    workflow.add_node("logic_check_node", logic_check_node)
    workflow.add_node("confidence_node", confidence_node)
    workflow.add_node("risk_node", risk_node)

    workflow.set_entry_point("generate_node")

    workflow.add_edge("generate_node", "extract_claims_node")
    workflow.add_edge("extract_claims_node", "fact_check_node")
    workflow.add_edge("fact_check_node", "logic_check_node")
    workflow.add_edge("logic_check_node", "confidence_node")
    workflow.add_edge("confidence_node", "risk_node")
    workflow.add_edge("risk_node", END)

    return workflow.compile()

# ---------------- FLASK ----------------

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("hallucination.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    try:
        workflow = create_workflow()
        result = workflow.invoke({"query": query})

        return jsonify({
            "answer": result.get("answer", ""),
            "claims": "\n".join(result.get("claims", [])),
            "fact_score": result.get("fact_score", 0.5),
            "logic_score": result.get("logic_score", 0.5),
            "confidence_score": result.get("confidence_score", 0.5),
            "hallucination_risk": result.get("hallucination_risk", 0.5),
            "final_report": result.get("final_report", "")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- CLI MODE ----------------

if __name__ == "__main__":
    import sys

    if "--cli" in sys.argv:
        query = input("Enter your question: ")
        workflow = create_workflow()
        result = workflow.invoke({"query": query})

        print("\nGenerated Answer:\n")
        print(result["answer"])
        print(result["final_report"])
    else:
        print("🚀 Starting Hallucination Detector with Web Grounding...")
        print("🌐 Open http://localhost:5000 in your browser")
        app.run(debug=True, port=5000)
