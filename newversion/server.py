from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import asyncio
import logging
import traceback
import json
import os

from backend import create_workflow, log_accumulator, node_events

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NeuraDialectic API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    max_iterations: Optional[int] = Field(default=3, ge=1, le=10)


class QueryResponse(BaseModel):
    final_answer: Optional[str] = None
    summary: Optional[str] = None
    generator_output: Optional[str] = None
    critic_output: Optional[str] = None
    redteam_output: Optional[str] = None
    validator_output: Optional[str] = None
    refinement_outputs: list = []
    confidence: float = 0.0
    iterations_used: int = 0
    confidence_history: list = []
    severity_history: list = []
    logs: list = []
    events: list = []


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "NeuraDialectic"}


@app.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    try:
        logger.info(f"Received query: {request.query[:100]}...")

        log_accumulator.start()
        node_events.clear()

        workflow = create_workflow(request.query, request.max_iterations)
        compiled = workflow.compile()

        result = await asyncio.to_thread(
            compiled.invoke,
            {"query": request.query},
            config={"recursion_limit": 50}
        )

        log_accumulator.stop()

        response = QueryResponse(
            final_answer=result.get("final_answer"),
            summary=result.get("summary"),
            generator_output=result.get("generator_output"),
            critic_output=result.get("critic_output"),
            redteam_output=result.get("redteam_output"),
            validator_output=result.get("validator_output"),
            refinement_outputs=result.get("refinement_outputs", []),
            confidence=result.get("confidence", 0.0),
            iterations_used=result.get("iteration", 0),
            confidence_history=result.get("confidence_history", []),
            severity_history=result.get("severity_history", []),
            logs=log_accumulator.get_logs(),
            events=node_events.get_events()
        )

        logger.info(f"Query completed. Confidence: {response.confidence}")
        return response

    except Exception as e:
        log_accumulator.stop()
        logger.error(f"Error processing query: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/query/stream")
async def run_query_stream(request: QueryRequest):
    """SSE endpoint — streams node completion events in real time."""

    async def event_generator():
        log_accumulator.start()
        node_events.clear()

        workflow = create_workflow(request.query, request.max_iterations)
        compiled = workflow.compile()

        last_event_idx = 0

        # Run the workflow in a background thread
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(
            None,
            lambda: compiled.invoke(
                {"query": request.query},
                config={"recursion_limit": 50}
            )
        )

        # Poll for new events while workflow runs
        while not task.done():
            await asyncio.sleep(0.5)
            current_events = node_events.get_events()
            while last_event_idx < len(current_events):
                event = current_events[last_event_idx]
                yield f"data: {json.dumps(event)}\n\n"
                last_event_idx += 1

        # Get final result
        result = task.result()
        log_accumulator.stop()

        # Flush remaining events
        current_events = node_events.get_events()
        while last_event_idx < len(current_events):
            event = current_events[last_event_idx]
            yield f"data: {json.dumps(event)}\n\n"
            last_event_idx += 1

        # Send final complete result
        final_data = {
            "node": "complete",
            "timestamp": current_events[-1]["timestamp"] if current_events else "",
            "data": {
                "final_answer": result.get("final_answer"),
                "summary": result.get("summary"),
                "generator_output": result.get("generator_output"),
                "critic_output": result.get("critic_output"),
                "redteam_output": result.get("redteam_output"),
                "validator_output": result.get("validator_output"),
                "refinement_outputs": result.get("refinement_outputs", []),
                "confidence": result.get("confidence", 0.0),
                "iterations_used": result.get("iteration", 0),
                "confidence_history": result.get("confidence_history", []),
                "severity_history": result.get("severity_history", []),
                "logs": log_accumulator.get_logs(),
                "events": node_events.get_events()
            }
        }
        yield f"data: {json.dumps(final_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/")
async def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(html_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)