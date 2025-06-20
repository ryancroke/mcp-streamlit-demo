"""
FastAPI backend for MCP Comparison Framework.
Supports dual-chat interface with separate baseline and enhanced MCP servers.
"""

import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from framework.orchestrator.simple_orchestrator import SimpleMCPOrchestrator


# Pydantic models for API
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    sql_query: str | None = None
    timestamp: str


class QueryRequest(BaseModel):
    message: str
    thread_id: str | None = None


class QueryResponse(BaseModel):
    response: str
    sql_query: str | None = None
    thread_id: str
    timestamp: str
    mcp_server_name: str


class ChatHistoryResponse(BaseModel):
    messages: list[ChatMessage]
    thread_id: str


class ConfigResponse(BaseModel):
    comparison_name: str
    comparison_description: str
    baseline: dict
    enhanced: dict


# Global orchestrator instances
baseline_orchestrator: SimpleMCPOrchestrator | None = None
enhanced_orchestrator: SimpleMCPOrchestrator | None = None

# Global configuration
comparison_config: dict | None = None

# In-memory chat history (in production, use a database)
chat_sessions: dict[str, list[ChatMessage]] = {}


def _load_comparison_config() -> dict:
    """Load comparison configuration from environment variable or default path."""
    config_path = os.getenv("MCP_COMPARISON_CONFIG", "configs/sqlite/comparison_config.json")
    print(f"📋 Loading comparison config from: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load comparison config: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    global baseline_orchestrator, enhanced_orchestrator, comparison_config
    print("🚀 Starting MCP Comparison Framework...")

    # Load comparison configuration
    comparison_config = _load_comparison_config()
    print(f"✅ Loaded comparison: {comparison_config['comparison_name']}")

    # Initialize baseline orchestrator
    print("🔄 Initializing Baseline MCP Orchestrator...")
    try:
        baseline_orchestrator = SimpleMCPOrchestrator(comparison_config["baseline"])
        await baseline_orchestrator.initialize()
        print("✅ Baseline MCP Orchestrator ready!")
    except Exception as e:
        print(f"❌ Baseline orchestrator failed: {e}")
        baseline_orchestrator = None

    # Initialize enhanced orchestrator
    print("🔄 Initializing Enhanced MCP Orchestrator...")
    try:
        enhanced_orchestrator = SimpleMCPOrchestrator(comparison_config["enhanced"])
        await enhanced_orchestrator.initialize()
        print("✅ Enhanced MCP Orchestrator ready!")
    except Exception as e:
        print(f"❌ Enhanced orchestrator failed: {e}")
        enhanced_orchestrator = None

    print("✅ MCP Comparison Framework ready!")

    yield

    # Shutdown
    print("🔄 Shutting down...")
    if baseline_orchestrator:
        await baseline_orchestrator.close()
    if enhanced_orchestrator:
        await enhanced_orchestrator.close()
    print("✅ Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="🎵 MCP Comparison Framework",
    description="Compare baseline vs enhanced MCP servers side-by-side",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="framework/dual_chat_ui"), name="static")


@app.get("/")
async def get_chat_interface():
    """Serve the main comparison chat interface."""
    return FileResponse("framework/dual_chat_ui/index.html")


@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    """Get UI configuration from the comparison config."""
    if not comparison_config:
        raise HTTPException(status_code=503, detail="Configuration not loaded")
    
    return ConfigResponse(
        comparison_name=comparison_config["comparison_name"],
        comparison_description=comparison_config["comparison_description"],
        baseline=comparison_config["baseline"]["ui"],
        enhanced=comparison_config["enhanced"]["ui"]
    )


@app.post("/api/baseline/query", response_model=QueryResponse)
async def process_baseline_query(request: QueryRequest):
    """Process query through baseline MCP orchestrator."""
    if not baseline_orchestrator:
        raise HTTPException(status_code=503, detail="Baseline orchestrator not available")

    return await _process_query(request, baseline_orchestrator, "baseline")


@app.post("/api/enhanced/query", response_model=QueryResponse)
async def process_enhanced_query(request: QueryRequest):
    """Process query through enhanced MCP orchestrator."""
    if not enhanced_orchestrator:
        raise HTTPException(status_code=503, detail="Enhanced orchestrator not available")

    return await _process_query(request, enhanced_orchestrator, "enhanced")


async def _process_query(
    request: QueryRequest, 
    orchestrator: SimpleMCPOrchestrator, 
    orchestrator_type: str
) -> QueryResponse:
    """Shared query processing logic."""
    # Generate thread ID if not provided
    thread_id = request.thread_id or str(uuid.uuid4())

    try:
        # Process query through orchestrator
        final_state = await orchestrator.run(
            user_query=request.message, 
            thread_id=thread_id
        )

        response_content = final_state.get("final_response", "Sorry, an error occurred.")
        sql_query = final_state.get("sql_query")
        mcp_server_name = final_state.get("mcp_server_name", f"{orchestrator_type}_mcp")
        timestamp = datetime.now().isoformat()

        # Store in chat history
        session_key = f"{orchestrator_type}_{thread_id}"
        if session_key not in chat_sessions:
            chat_sessions[session_key] = []

        # Add user message
        chat_sessions[session_key].append(
            ChatMessage(
                role="user", 
                content=request.message, 
                timestamp=timestamp
            )
        )

        # Add assistant response
        chat_sessions[session_key].append(
            ChatMessage(
                role="assistant",
                content=response_content,
                sql_query=sql_query,
                timestamp=timestamp,
            )
        )

        return QueryResponse(
            response=response_content,
            sql_query=sql_query,
            thread_id=thread_id,
            timestamp=timestamp,
            mcp_server_name=mcp_server_name,
        )

    except Exception as e:
        print(f"❌ {orchestrator_type.title()} query processing failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"{orchestrator_type.title()} query processing failed: {str(e)}"
        ) from e


@app.get("/api/baseline/history/{thread_id}", response_model=ChatHistoryResponse)
async def get_baseline_chat_history(thread_id: str):
    """Get baseline chat history for a thread."""
    session_key = f"baseline_{thread_id}"
    messages = chat_sessions.get(session_key, [])
    return ChatHistoryResponse(messages=messages, thread_id=thread_id)


@app.get("/api/enhanced/history/{thread_id}", response_model=ChatHistoryResponse)
async def get_enhanced_chat_history(thread_id: str):
    """Get enhanced chat history for a thread."""
    session_key = f"enhanced_{thread_id}"
    messages = chat_sessions.get(session_key, [])
    return ChatHistoryResponse(messages=messages, thread_id=thread_id)


@app.delete("/api/baseline/history/{thread_id}")
async def clear_baseline_chat_history(thread_id: str):
    """Clear baseline chat history for a thread."""
    session_key = f"baseline_{thread_id}"
    if session_key in chat_sessions:
        del chat_sessions[session_key]
    return {"message": "Baseline chat history cleared"}


@app.delete("/api/enhanced/history/{thread_id}")
async def clear_enhanced_chat_history(thread_id: str):
    """Clear enhanced chat history for a thread."""
    session_key = f"enhanced_{thread_id}"
    if session_key in chat_sessions:
        del chat_sessions[session_key]
    return {"message": "Enhanced chat history cleared"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint for both orchestrators."""
    baseline_healthy = False
    enhanced_healthy = False

    # Check baseline health
    if baseline_orchestrator and baseline_orchestrator.mcp_interface:
        try:
            baseline_healthy = await baseline_orchestrator.mcp_interface.health_check()
        except Exception as e:
            print(f"Baseline health check failed: {e}")

    # Check enhanced health
    if enhanced_orchestrator and enhanced_orchestrator.mcp_interface:
        try:
            enhanced_healthy = await enhanced_orchestrator.mcp_interface.health_check()
        except Exception as e:
            print(f"Enhanced health check failed: {e}")

    overall_status = "healthy" if (baseline_healthy and enhanced_healthy) else "degraded"

    return {
        "status": overall_status,
        "baseline_healthy": baseline_healthy,
        "enhanced_healthy": enhanced_healthy,
        "baseline_orchestrator": "ready" if baseline_orchestrator else "not_initialized",
        "enhanced_orchestrator": "ready" if enhanced_orchestrator else "not_initialized",
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    print("🎵 Starting MCP Comparison Framework")
    print("📍 Open: http://localhost:8001")

    uvicorn.run(
        "comparison_app:app", 
        host="0.0.0.0", 
        port=8001, 
        reload=True, 
        log_level="info"
    )