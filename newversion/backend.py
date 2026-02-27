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
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler

# ---------------- SETUP ----------------

load_dotenv()

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOGGING ----------------

logger = logging.getLogger("neurodialectic")
logger.setLevel(logging.INFO)

if not logger.handlers:
    log_file_path = os.path.join(
        OUTPUT_DIR,
        f"neurodialectic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=3
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# ---------------- LOG ACCUMULATOR ----------------

class LogAccumulator(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
        self.active = False

    def emit(self, record):
        if self.active:
            self.records.append({
                "timestamp": datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3],
                "level": record.levelname,
                "message": record.getMessage()
            })

    def start(self):
        self.records = []
        self.active = True

    def stop(self):
        self.active = False

    def get_logs(self):
        return list(self.records)


log_accumulator = LogAccumulator()
log_accumulator.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(log_accumulator)

# ---------------- EVENT STREAM CALLBACK ----------------

class NodeEventCallback:
    """Collects node completion events for SSE streaming."""

    def __init__(self):
        self.events = []

    def push(self, node_name, data):
        self.events.append({
            "node": node_name,
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "data": data
        })

    def get_events(self):
        return list(self.events)

    def clear(self):
        self.events = []


node_events = NodeEventCallback()

# ---------------- API KEYS ----------------

groq_key = os.getenv("GROQ_SCAR_KEY")
cohere_key = os.getenv("COHERE_API_KEY")

if not all([groq_key, cohere_key]):
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

# ---------------- RETRY LOGIC ----------------

def safe_invoke(llm, prompt, node_name="LLM", retries=3, base_delay=2):
    for attempt in range(retries):
        try:
            logger.info(f"{node_name} | Attempt {attempt + 1}/{retries}")
            start_time = time.time()
            response = llm.invoke(prompt[:MAX_PROMPT_CHARS]).content
            elapsed = round(time.time() - start_time, 2)
            logger.info(f"{node_name} | Success | {len(response)} chars | {elapsed}s")
            return response
        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            logger.warning(
                f"{node_name} | Attempt {attempt + 1}/{retries} failed after {elapsed}s | {type(e).__name__}: {str(e)[:200]}"
            )
            if attempt < retries - 1:
                sleep_time = base_delay * (2 ** attempt)
                logger.info(f"{node_name} | Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                logger.error(f"{node_name} | All {retries} retries exhausted")
                raise


# ---------------- CONFIDENCE PARSER ----------------

def parse_confidence(text: str) -> float:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = cleaned.strip()
    if not cleaned:
        cleaned = text

    patterns = [
        r"[Cc]onfidence[\s_]*(?:[Ss]core)?[\s:=]*\*?\*?([0-9]+(?:\.[0-9]+)?)\s*%",
        r"[Cc]onfidence[\s_]*(?:[Ss]core)?[\s:=]*\*?\*?([0-9]*\.?[0-9]+)\s*(?:/\s*1(?:\.0)?)?\*?\*?",
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            value = float(match.group(1))
            if value > 1.0:
                value = value / 100.0
            return max(0.0, min(1.0, value))

    decimals = re.findall(r"\b(0\.\d+|1\.0)\b", cleaned)
    if decimals:
        return float(decimals[0])

    percentages = re.findall(r"(\d{1,3})\s*%", cleaned)
    if percentages:
        value = float(percentages[0]) / 100.0
        return max(0.0, min(1.0, value))

    logger.warning(f"Confidence parse failed | First 200 chars: {cleaned[:200]}")
    return 0.5


def parse_severity(text: str) -> float:
    """Extract severity score from red team output."""
    patterns = [
        r"[Ss]everity[\s_]*[Ss]core[\s:=]*\*?\*?([0-9]*\.?[0-9]+)",
        r"[Ss]everity[\s:=]*\*?\*?([0-9]*\.?[0-9]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1))
            if value > 1.0:
                value = value / 10.0
            return max(0.0, min(1.0, value))
    return 0.0


# ---------------- WORKFLOW ----------------

def create_workflow(query, max_iterations=3):

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
        confidence_history: List[float]
        severity_history: List[float]

    workflow = StateGraph(GraphState)

    # -------- GENERATOR --------
    def generator_node(state):
        try:
            logger.info("=" * 50)
            logger.info(f"PIPELINE START | Query: {state['query'][:100]}")
            logger.info("=" * 50)

            memories = vector_store.similarity_search(state["query"], k=1)
            memory_context = ""
            for doc in memories:
                memory_context += doc.page_content[:500] + "\n"
            memory_context = memory_context.strip()[:MAX_MEMORY_CHARS]

            if memory_context:
                logger.info(f"MEMORY | Retrieved {len(memories)} relevant memories")
                prompt = f"""Answer clearly with structured reasoning.

Question:
{state['query']}

Avoid repeating these past failure patterns:
{memory_context}"""
            else:
                logger.info("MEMORY | No relevant memories found")
                prompt = f"""Answer clearly with structured reasoning.

Question:
{state['query']}"""

            draft = safe_invoke(generator_llm, prompt, "GENERATOR")
            logger.info("GENERATOR | Initial draft complete")

            node_events.push("generator", {
                "output": draft[:500],
                "status": "complete"
            })

            return {
                "draft": draft,
                "generator_output": draft,
                "iteration": 0,
                "refinement_outputs": [],
                "confidence_history": [],
                "severity_history": []
            }

        except Exception as e:
            logger.error(f"GENERATOR | Fatal error: {e}")
            fallback = f"I encountered an error generating a response: {str(e)}"
            node_events.push("generator", {"output": fallback, "status": "error"})
            return {
                "draft": fallback,
                "generator_output": fallback,
                "iteration": 0,
                "refinement_outputs": [],
                "confidence_history": [],
                "severity_history": []
            }

    # -------- CRITIC --------
    def critic_node(state):
        try:
            iteration = state.get("iteration", 0)
            logger.info(f"CRITIC | Starting analysis (iteration {iteration})")

            prompt = f"""You are a critical reviewer. Analyze the following answer for:
1. Factual accuracy
2. Logical consistency
3. Completeness
4. Clarity

Be specific about what is wrong and what could be improved.

Answer to critique:
{state['draft']}"""

            critique = safe_invoke(critic_llm, prompt, "CRITIC")
            logger.info("CRITIC | Analysis complete")

            node_events.push("critic", {
                "output": critique[:500],
                "status": "complete",
                "iteration": iteration
            })

            return {
                "critique": critique,
                "critic_output": critique
            }

        except Exception as e:
            logger.error(f"CRITIC | Fatal error: {e}")
            fallback = f"Critique unavailable due to error: {str(e)}"
            node_events.push("critic", {"output": fallback, "status": "error"})
            return {"critique": fallback, "critic_output": fallback}

    # -------- RED TEAM --------
    def redteam_node(state):
        try:
            iteration = state.get("iteration", 0)
            logger.info(f"REDTEAM | Starting adversarial analysis (iteration {iteration})")

            prompt = f"""You are an adversarial red-team AI. Your job is to attack the reasoning and find weaknesses.

Question:
{state['query'][:1500]}

Answer:
{state['draft'][:2000]}

Critique:
{state['critique'][:1500]}

Analyze and respond in EXACTLY this format:

1. Hallucination Risk: <identify any fabricated claims, unsupported facts, or made-up details>
2. Logical Weaknesses: <find contradictions, circular reasoning, or non-sequiturs>
3. Missing Assumptions: <what unstated assumptions does this answer rely on>
4. Adversarial Counterexample: <provide a specific scenario that breaks the answer>
5. Severity Score: <number from 0.0 to 1.0 where 0 is no issues and 1 is critically flawed>"""

            redteam_analysis = safe_invoke(redteam_llm, prompt, "REDTEAM")
            logger.info("REDTEAM | Adversarial analysis complete")

            severity = parse_severity(redteam_analysis)
            logger.info(f"REDTEAM | Severity score: {severity}")

            node_events.push("redteam", {
                "output": redteam_analysis[:500],
                "severity": severity,
                "status": "complete",
                "iteration": iteration
            })

            return {
                "redteam": redteam_analysis,
                "redteam_output": redteam_analysis
            }

        except Exception as e:
            logger.error(f"REDTEAM | Fatal error: {e}")
            fallback = f"Red team analysis unavailable: {str(e)}"
            node_events.push("redteam", {"output": fallback, "status": "error"})
            return {"redteam": fallback, "redteam_output": fallback}

    # -------- VALIDATOR --------
    def validator_node(state):
        try:
            logger.info("VALIDATOR | Starting confidence evaluation")

            prompt = f"""You are a strict answer quality evaluator.

Rate the answer below considering the critique AND the red team adversarial analysis.

ANSWER:
{state['draft'][:1500]}

CRITIQUE:
{state['critique'][:1500]}

RED TEAM ANALYSIS:
{state['redteam'][:1500]}

You MUST respond in EXACTLY this format and nothing else:

Confidence: <number between 0.0 and 1.0>
Reason: <one sentence explanation>

Do NOT include any other text, thinking, or preamble. Just those two lines."""

            response = safe_invoke(validator_llm, prompt, "VALIDATOR")

            confidence = parse_confidence(response)

            # If parse failed, retry once
            if confidence == 0.5 and "0.5" not in response[:100]:
                logger.warning("VALIDATOR | Confidence parse returned default, retrying...")
                retry_response = safe_invoke(validator_llm, prompt, "VALIDATOR-RETRY")
                retry_confidence = parse_confidence(retry_response)
                if retry_confidence != 0.5:
                    confidence = retry_confidence
                    response = retry_response

            # Apply red team severity penalty
            severity = parse_severity(state.get("redteam", ""))
            if severity > 0:
                original_confidence = confidence
                confidence = max(0.0, confidence - severity * 0.3)
                logger.info(f"VALIDATOR | Severity penalty applied: {original_confidence:.2f} - ({severity:.2f} × 0.3) = {confidence:.2f}")

            # Clean display response
            display_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            if not display_response:
                display_response = f"Confidence: {confidence}\nReason: Extracted from model reasoning"

            previous_conf_history = list(state.get("confidence_history", []))
            previous_conf_history.append(confidence)

            previous_sev_history = list(state.get("severity_history", []))
            previous_sev_history.append(severity)

            logger.info(f"VALIDATOR | Confidence: {confidence:.2f} | Severity: {severity:.2f} | History: {previous_conf_history}")

            node_events.push("validator", {
                "confidence": confidence,
                "severity": severity,
                "status": "complete"
            })

            return {
                "validation": display_response,
                "confidence": confidence,
                "validator_output": display_response,
                "confidence_history": previous_conf_history,
                "severity_history": previous_sev_history
            }

        except Exception as e:
            logger.error(f"VALIDATOR | Fatal error: {e}")
            node_events.push("validator", {"confidence": 0.5, "status": "error"})
            return {
                "validation": f"Validation error: {str(e)}",
                "confidence": 0.5,
                "validator_output": f"Validation error: {str(e)}",
                "confidence_history": list(state.get("confidence_history", [])) + [0.5],
                "severity_history": list(state.get("severity_history", [])) + [0.0]
            }

    # -------- CONTROLLER --------
    def controller(state):
        confidence = state.get("confidence", 0)
        iteration = state.get("iteration", 0)

        logger.info(f"CONTROLLER | Confidence: {confidence:.2f} | Iteration: {iteration}/{max_iterations}")

        if confidence >= 0.85:
            logger.info("CONTROLLER | → FINALIZE (confidence threshold met)")
            node_events.push("controller", {"decision": "finalize", "reason": "confidence_met"})
            return "finalize"
        if iteration >= max_iterations:
            logger.info("CONTROLLER | → FINALIZE (max iterations reached)")
            node_events.push("controller", {"decision": "finalize", "reason": "max_iterations"})
            return "finalize"

        logger.info("CONTROLLER | → REFINE (below threshold)")
        node_events.push("controller", {"decision": "refine", "reason": "below_threshold"})
        return "refine"

    # -------- REFINE --------
    def refine_node(state):
        try:
            current_iteration = state.get("iteration", 0) + 1
            logger.info(f"REFINE | Starting iteration {current_iteration}")

            prompt = f"""Improve the answer below using both the critique and red team adversarial findings.

Fix:
- Factual errors and hallucinations flagged by the red team
- Logical gaps and weak assumptions
- Missing information noted by the critic
- Adversarial vulnerabilities

Critique:
{state['critique'][:2000]}

Red Team Findings:
{state['redteam'][:2000]}

Previous Answer:
{state['draft'][:2000]}

Provide the improved answer:"""

            improved = safe_invoke(generator_llm, prompt, f"REFINE-{current_iteration}")

            previous = list(state.get("refinement_outputs", []))
            previous.append(improved)

            logger.info(f"REFINE | Iteration {current_iteration} complete")

            node_events.push("refine", {
                "iteration": current_iteration,
                "output": improved[:500],
                "status": "complete"
            })

            return {
                "draft": improved,
                "iteration": current_iteration,
                "refinement_outputs": previous
            }

        except Exception as e:
            logger.error(f"REFINE | Fatal error: {e}")
            node_events.push("refine", {"status": "error"})
            return {"iteration": state.get("iteration", 0) + 1}

    # -------- FINALIZE --------
    def finalize_node(state):
        try:
            confidence = state.get("confidence", 1.0)
            logger.info(f"FINALIZE | Final confidence: {confidence:.2f}")

            if confidence < 0.85:
                logger.info("FINALIZE | Storing failure pattern in memory")
                compressed = safe_invoke(
                    summarizer_llm,
                    f"Summarize this critique as a failure pattern in under 120 words:\n{state.get('critique', 'No critique')}",
                    "FAILURE-SUMMARIZER"
                )[:500]

                vector_store.add_documents([
                    Document(page_content=f"Failure Pattern: {compressed}")
                ])
                logger.info("FINALIZE | Failure pattern stored")
            else:
                logger.info("FINALIZE | High confidence — no failure pattern stored")

            node_events.push("finalize", {"confidence": confidence, "status": "complete"})

        except Exception as e:
            logger.warning(f"FINALIZE | Memory storage failed (non-fatal): {e}")

        return {"final_answer": state.get("draft", "No answer generated")}

    # -------- SUMMARIZER --------
    def summarizer_node(state):
        try:
            logger.info("SUMMARIZER | Generating process summary")

            prompt = f"""Summarize the full reasoning process concisely.

Final Answer:
{state.get('final_answer', 'N/A')[:1500]}

Critique:
{state.get('critique', 'N/A')[:800]}

Red Team Analysis:
{state.get('redteam', 'N/A')[:800]}

Validation:
{state.get('validation', 'N/A')[:500]}

Include:
- Conclusion
- Strengths
- Weaknesses
- Red team findings summary
- Final confidence: {state.get('confidence', 'N/A')}"""

            summary = safe_invoke(summarizer_llm, prompt, "SUMMARIZER")
            summary = summary[:MAX_SUMMARY_CHARS]

            logger.info("SUMMARIZER | Summary complete")
            logger.info("=" * 50)
            logger.info("PIPELINE COMPLETE")
            logger.info("=" * 50)

            node_events.push("summarizer", {"status": "complete"})

            return {"summary": summary}

        except Exception as e:
            logger.error(f"SUMMARIZER | Fatal error: {e}")
            return {"summary": f"Summary generation failed: {str(e)}"}

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

# ---------------- TERMINAL EXECUTION ----------------

if __name__ == "__main__":

    print("\n===== NeuraDialectic Terminal Mode =====\n")

    query = input("Enter your question:\n> ")
    max_iterations_input = input("Max refinement iterations (default 3): ")

    try:
        max_iterations = int(max_iterations_input)
    except:
        max_iterations = 3

    log_accumulator.start()
    node_events.clear()

    workflow = create_workflow(query, max_iterations)
    result = workflow.compile().invoke(
        {"query": query},
        config={"recursion_limit": 50}
    )

    log_accumulator.stop()

    print("\n===== FINAL ANSWER =====\n")
    print(result.get("final_answer"))

    print("\n===== CONFIDENCE =====\n")
    print(result.get("confidence"))

    print("\n===== RED TEAM OUTPUT =====\n")
    print(result.get("redteam_output"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(OUTPUT_DIR, f"neurodialectic_{timestamp}.json")

    run_data = {
        "metadata": {"timestamp": timestamp, "query": query},
        "outputs": result,
        "logs": log_accumulator.get_logs(),
        "events": node_events.get_events()
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=4, ensure_ascii=False)

    print(f"\nRun saved to {file_path}")