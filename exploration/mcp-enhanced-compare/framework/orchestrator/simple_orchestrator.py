"""
Simple orchestrator with LangGraph memory for MCP comparison.
Keeps conversational context while using mcp-use pattern.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from ..handlers import MCPServerHandlerRegistry
from .mcp_interface import MCPInterface, create_mcp_interface_from_config


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    final_response: str
    mcp_server_name: str


class SimpleMCPOrchestrator:
    """Simple orchestrator with LangGraph memory and MCP agent."""

    def __init__(self, config: dict | str):
        if isinstance(config, str):
            # Legacy mode: config_path provided
            self.config_path = config
            self.config = self._load_config_from_file()
        else:
            # New mode: config dict provided
            self.config_path = None
            self.config = config
        self.mcp_interface: MCPInterface | None = None
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.checkpointer = InMemorySaver()
        self.graph = None
        self.handler_registry = MCPServerHandlerRegistry()

    def _load_config_from_file(self) -> dict:
        """Load MCP server configuration from file."""
        with open(self.config_path) as f:
            return json.load(f)

    async def initialize(self):
        """Initialize MCP interface and build simple graph."""
        # Handle GitHub source if present
        if "source" in self.config:
            updated_mcp_server_config = await self._setup_github_source(self.config)
            self.mcp_interface = await create_mcp_interface_from_config(
                updated_mcp_server_config
            )
        else:
            self.mcp_interface = await create_mcp_interface_from_config(
                self.config["mcp_server"]
            )

        self.graph = self._build_graph()

    async def _setup_github_source(self, config: dict) -> dict:
        """Clone GitHub repository and setup MCP server configuration using handler system."""
        source_config = config["source"]
        mcp_server_config = config["mcp_server"].copy()

        if source_config.get("type") != "github":
            return mcp_server_config

        repo = source_config["repo"]
        branch = source_config.get("branch", "main")
        subdirectory = source_config.get("subdirectory", "")
        install_dir = source_config["install_dir"]

        try:
            # Clean up existing directory if it exists
            if os.path.exists(install_dir):
                shutil.rmtree(install_dir)

            # Clone repository with retry logic
            self._clone_with_retry(repo, install_dir, branch)

            # Determine working directory
            work_dir = (
                Path(install_dir) / subdirectory if subdirectory else Path(install_dir)
            )

            # Auto-detect appropriate handler for this MCP server type
            handler = self.handler_registry.get_handler(work_dir, mcp_server_config)

            # Install dependencies using the appropriate handler
            handler.install_dependencies(work_dir)

            # Transform command configuration for local execution
            updated_config = handler.transform_command(mcp_server_config, work_dir)

            print(f"✅ GitHub source setup complete for {repo}")
            return updated_config

        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to setup GitHub source {repo}: {e.stderr if e.stderr else str(e)}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error setting up GitHub source {repo}: {e!s}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)


    def _build_graph(self):
        """Build simple LangGraph with memory."""
        graph = StateGraph(State)

        graph.add_node("contextualize", self._contextualize_query)
        graph.add_node("query_mcp", self._query_mcp)

        graph.set_entry_point("contextualize")
        graph.add_edge("contextualize", "query_mcp")
        graph.add_edge("query_mcp", END)

        return graph.compile(checkpointer=self.checkpointer)

    def _clone_with_retry(
        self,
        repo: str,
        install_dir: str,
        branch: str = "main",
        max_retries: int = 3,
        timeout: int = 300,
    ) -> None:
        """
        Clone a Git repository with retry logic for network resilience.

        Args:
            repo: Repository name (e.g., 'modelcontextprotocol/servers')
            install_dir: Local directory to clone into
            branch: Git branch to clone
            max_retries: Maximum number of retry attempts
            timeout: Timeout in seconds for each clone attempt

        Raises:
            subprocess.CalledProcessError: If all retry attempts fail
        """
        repo_url = f"https://github.com/{repo}.git"
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 Cloning {repo} (branch: {branch}) to {install_dir}")
                if attempt > 0:
                    print(f"   Attempt {attempt + 1}/{max_retries}")
                
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--branch",
                        branch,
                        repo_url,
                        install_dir,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                print(f"✅ Successfully cloned {repo}")
                return  # Success - exit retry loop
                
            except subprocess.TimeoutExpired:
                error_msg = f"Git clone timed out after {timeout} seconds"
                if attempt == max_retries - 1:  # Last attempt
                    print(f"❌ {error_msg} - all retry attempts exhausted")
                    raise subprocess.CalledProcessError(
                        1, "git clone", stderr=f"Timeout after {max_retries} attempts"
                    )
                else:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"⚠️  {error_msg} - retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    
            except subprocess.CalledProcessError as e:
                error_details = e.stderr if e.stderr else str(e)
                if "network" in error_details.lower() or "timeout" in error_details.lower() or "recv failure" in error_details.lower():
                    # Network-related error - retry
                    if attempt == max_retries - 1:  # Last attempt
                        print(f"❌ Network error persists after {max_retries} attempts: {error_details}")
                        raise
                    else:
                        wait_time = 2 ** attempt  # Exponential backoff
                        print(f"⚠️  Network error (attempt {attempt + 1}/{max_retries}): {error_details}")
                        print(f"   Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                else:
                    # Non-network error (repo not found, auth issues, etc.) - don't retry
                    print(f"❌ Repository error (not retrying): {error_details}")
                    raise

    async def _contextualize_query(self, state: State) -> State:
        """Rewrite the user's query using conversation history for context."""
        # The last message is the current user query
        user_query = state["messages"][-1].content

        # If it's the start of a conversation, no context needed
        if len(state["messages"]) <= 1:
            return {"user_query": user_query}

        # Use LLM to contextualize the query based on conversation history
        context_prompt = f"""Based on the chat history, rewrite the following user query to be a standalone question that includes necessary context.

History:
{state["messages"]}

User Query: {user_query}

Standalone Query:"""

        response = await self.llm.ainvoke(context_prompt)
        contextualized_query = response.content.strip()

        return {"user_query": contextualized_query}

    async def _query_mcp(self, state: State) -> State:
        """Query the MCP agent with contextualized query."""
        if not self.mcp_interface:
            raise RuntimeError("MCP interface not initialized")

        try:
            # Use the MCP agent with the contextualized query
            result = await self.mcp_interface.query(state["user_query"])

            return {
                "final_response": str(result),
                "mcp_server_name": self.mcp_interface.server_name,
            }

        except Exception as e:
            print(f"❌ MCP query failed: {e}")
            return {
                "final_response": f"Error: {e!s}",
                "mcp_server_name": self.mcp_interface.server_name
                if self.mcp_interface
                else "unknown",
            }

    async def run(self, user_query: str, thread_id: str) -> State:
        """Process query through LangGraph with memory."""
        if not self.graph:
            raise RuntimeError("Graph not initialized")

        inputs = {"messages": [("user", user_query)]}
        config = {"configurable": {"thread_id": thread_id}}

        try:
            final_state = await self.graph.ainvoke(inputs, config)
            return final_state
        except Exception as e:
            print(f"❌ Graph execution failed: {e}")
            raise

    async def close(self):
        """Clean up resources."""
        if self.mcp_interface:
            await self.mcp_interface.close()
