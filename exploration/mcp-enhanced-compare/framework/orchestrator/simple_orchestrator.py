"""
Simple orchestrator with LangGraph memory for MCP comparison.
Keeps conversational context while using mcp-use pattern.
"""

import json
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

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

    def _load_config_from_file(self) -> dict:
        """Load MCP server configuration from file."""
        with open(self.config_path, 'r') as f:
            return json.load(f)

    async def initialize(self):
        """Initialize MCP interface and build simple graph."""
        self.mcp_interface = await create_mcp_interface_from_config(self.config["mcp_server"])
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build simple LangGraph with memory."""
        graph = StateGraph(State)

        graph.add_node("contextualize", self._contextualize_query)
        graph.add_node("query_mcp", self._query_mcp)
        
        graph.set_entry_point("contextualize")
        graph.add_edge("contextualize", "query_mcp")
        graph.add_edge("query_mcp", END)

        return graph.compile(checkpointer=self.checkpointer)

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
                "mcp_server_name": self.mcp_interface.server_name
            }
            
        except Exception as e:
            print(f"❌ MCP query failed: {e}")
            return {
                "final_response": f"Error: {str(e)}",
                "mcp_server_name": self.mcp_interface.server_name if self.mcp_interface else "unknown"
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