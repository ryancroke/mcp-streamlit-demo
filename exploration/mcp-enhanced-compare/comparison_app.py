"""
FastAPI backend for MCP Comparison Framework.
Supports dual-chat interface with separate baseline and enhanced MCP servers.
"""

import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from framework.orchestrator.simple_orchestrator import SimpleMCPOrchestrator
from framework.comparison_manager import ComparisonManager


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


class ComparisonListResponse(BaseModel):
    comparisons: list[dict]


class SwitchComparisonRequest(BaseModel):
    comparison_id: str


# Global comparison manager
comparison_manager: ComparisonManager | None = None

# In-memory chat history (in production, use a database)
chat_sessions: dict[str, list[ChatMessage]] = {}




@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    global comparison_manager
    print("🚀 Starting MCP Comparison Framework...")

    # Initialize comparison manager
    comparison_manager = ComparisonManager()
    await comparison_manager.load_all_comparisons()
    
    available_comparisons = comparison_manager.get_available_comparisons()
    print(f"✅ Loaded {len(available_comparisons)} comparison(s)")
    for comp in available_comparisons:
        print(f"  - {comp['name']} ({comp['id']})")
    
    if comparison_manager.active_comparison:
        print(f"✅ Active comparison: {comparison_manager.active_comparison}")
    else:
        print("⚠️ No active comparison set")

    print("✅ MCP Comparison Framework ready!")

    yield

    # Shutdown
    print("🔄 Shutting down...")
    if comparison_manager:
        await comparison_manager.cleanup()
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
    """Get UI configuration for the active comparison."""
    if not comparison_manager or not comparison_manager.active_comparison:
        raise HTTPException(status_code=503, detail="No active comparison")

    try:
        config = comparison_manager.get_current_comparison_config()
        return ConfigResponse(
            comparison_name=config["comparison_name"],
            comparison_description=config["comparison_description"],
            baseline=config["baseline"]["ui"],
            enhanced=config["enhanced"]["ui"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get config: {e}")


@app.get("/api/comparisons/list", response_model=ComparisonListResponse)
async def get_available_comparisons():
    """Get list of available comparisons."""
    if not comparison_manager:
        raise HTTPException(status_code=503, detail="Comparison manager not initialized")
    
    try:
        comparisons = comparison_manager.get_available_comparisons()
        return ComparisonListResponse(comparisons=comparisons)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get comparisons: {e}")


@app.get("/api/comparisons/current")
async def get_current_comparison():
    """Get current active comparison configuration."""
    if not comparison_manager or not comparison_manager.active_comparison:
        raise HTTPException(status_code=503, detail="No active comparison")
    
    try:
        config = comparison_manager.get_current_comparison_config()
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get current comparison: {e}")


@app.post("/api/comparisons/switch")
async def switch_comparison(request: SwitchComparisonRequest):
    """Switch to a different comparison."""
    if not comparison_manager:
        raise HTTPException(status_code=503, detail="Comparison manager not initialized")
    
    try:
        config = await comparison_manager.switch_comparison(request.comparison_id)
        return {
            "message": f"Switched to comparison: {request.comparison_id}",
            "config": config
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch comparison: {e}")


@app.post("/api/baseline/query", response_model=QueryResponse)
async def process_baseline_query(request: QueryRequest):
    """Process query through baseline MCP orchestrator."""
    if not comparison_manager or not comparison_manager.active_comparison:
        raise HTTPException(status_code=503, detail="No active comparison")
    
    try:
        orchestrators = comparison_manager.get_active_orchestrators()
        baseline_orchestrator = orchestrators.get("baseline")
        if not baseline_orchestrator:
            raise HTTPException(status_code=503, detail="Baseline orchestrator not available")
        
        return await _process_query(request, baseline_orchestrator, "baseline")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process baseline query: {e}")


@app.post("/api/enhanced/query", response_model=QueryResponse)
async def process_enhanced_query(request: QueryRequest):
    """Process query through enhanced MCP orchestrator."""
    if not comparison_manager or not comparison_manager.active_comparison:
        raise HTTPException(status_code=503, detail="No active comparison")
    
    try:
        orchestrators = comparison_manager.get_active_orchestrators()
        enhanced_orchestrator = orchestrators.get("enhanced")
        if not enhanced_orchestrator:
            raise HTTPException(status_code=503, detail="Enhanced orchestrator not available")
        
        return await _process_query(request, enhanced_orchestrator, "enhanced")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process enhanced query: {e}")


async def _process_query(
    request: QueryRequest, orchestrator: SimpleMCPOrchestrator, orchestrator_type: str
) -> QueryResponse:
    """Shared query processing logic."""
    # Generate thread ID if not provided
    thread_id = request.thread_id or str(uuid.uuid4())
    
    # Include comparison_id in session key for isolation
    comparison_id = comparison_manager.active_comparison if comparison_manager else "default"

    try:
        # Process query through orchestrator
        final_state = await orchestrator.run(
            user_query=request.message, thread_id=thread_id
        )

        response_content = final_state.get(
            "final_response", "Sorry, an error occurred."
        )
        sql_query = final_state.get("sql_query")
        mcp_server_name = final_state.get("mcp_server_name", f"{orchestrator_type}_mcp")
        timestamp = datetime.now().isoformat()

        # Store in chat history with comparison_id prefix
        session_key = f"{comparison_id}_{orchestrator_type}_{thread_id}"
        if session_key not in chat_sessions:
            chat_sessions[session_key] = []

        # Add user message
        chat_sessions[session_key].append(
            ChatMessage(role="user", content=request.message, timestamp=timestamp)
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
            detail=f"{orchestrator_type.title()} query processing failed: {e!s}",
        ) from e


@app.get("/api/baseline/history/{thread_id}", response_model=ChatHistoryResponse)
async def get_baseline_chat_history(thread_id: str):
    """Get baseline chat history for a thread."""
    comparison_id = comparison_manager.active_comparison if comparison_manager else "default"
    session_key = f"{comparison_id}_baseline_{thread_id}"
    messages = chat_sessions.get(session_key, [])
    return ChatHistoryResponse(messages=messages, thread_id=thread_id)


@app.get("/api/enhanced/history/{thread_id}", response_model=ChatHistoryResponse)
async def get_enhanced_chat_history(thread_id: str):
    """Get enhanced chat history for a thread."""
    comparison_id = comparison_manager.active_comparison if comparison_manager else "default"
    session_key = f"{comparison_id}_enhanced_{thread_id}"
    messages = chat_sessions.get(session_key, [])
    return ChatHistoryResponse(messages=messages, thread_id=thread_id)


@app.delete("/api/baseline/history/{thread_id}")
async def clear_baseline_chat_history(thread_id: str):
    """Clear baseline chat history for a thread."""
    comparison_id = comparison_manager.active_comparison if comparison_manager else "default"
    session_key = f"{comparison_id}_baseline_{thread_id}"
    chat_sessions.pop(session_key, None)
    return {"message": "Baseline chat history cleared"}


@app.delete("/api/enhanced/history/{thread_id}")
async def clear_enhanced_chat_history(thread_id: str):
    """Clear enhanced chat history for a thread."""
    comparison_id = comparison_manager.active_comparison if comparison_manager else "default"
    session_key = f"{comparison_id}_enhanced_{thread_id}"
    chat_sessions.pop(session_key, None)
    return {"message": "Enhanced chat history cleared"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint for active comparison orchestrators."""
    baseline_healthy = False
    enhanced_healthy = False
    
    if not comparison_manager or not comparison_manager.active_comparison:
        return {
            "status": "degraded",
            "baseline_healthy": False,
            "enhanced_healthy": False,
            "baseline_orchestrator": "no_active_comparison",
            "enhanced_orchestrator": "no_active_comparison",
            "timestamp": datetime.now().isoformat(),
        }
    
    try:
        orchestrators = comparison_manager.get_active_orchestrators()
        baseline_orchestrator = orchestrators.get("baseline")
        enhanced_orchestrator = orchestrators.get("enhanced")
        
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
    except Exception as e:
        print(f"Health check failed: {e}")

    overall_status = (
        "healthy" if (baseline_healthy and enhanced_healthy) else "degraded"
    )

    return {
        "status": overall_status,
        "baseline_healthy": baseline_healthy,
        "enhanced_healthy": enhanced_healthy,
        "baseline_orchestrator": "ready"
        if baseline_orchestrator
        else "not_initialized",
        "enhanced_orchestrator": "ready"
        if enhanced_orchestrator
        else "not_initialized",
        "active_comparison": comparison_manager.active_comparison if comparison_manager else None,
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    print("🎵 Starting MCP Comparison Framework")
    print("📍 Open: http://localhost:8001")

    # Check if using GitHub sources (disable reload to avoid infinite loops)
    config_path = os.getenv("MCP_COMPARISON_CONFIG", "configs/sqlite/comparison_config.json")
    use_reload = True
    try:
        with open(config_path) as f:
            config = json.load(f)
            # Disable reload if any source uses GitHub
            if (config.get("baseline", {}).get("source", {}).get("type") == "github" or 
                config.get("enhanced", {}).get("source", {}).get("type") == "github"):
                use_reload = False
                print("🔄 GitHub sources detected - disabling reload to prevent loops")
    except Exception:
        pass  # Use default reload=True if config can't be read

    uvicorn.run(
        "comparison_app:app", 
        host="0.0.0.0", 
        port=8001, 
        reload=use_reload,
        log_level="info"
    )
