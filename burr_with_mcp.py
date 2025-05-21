import sys
import asyncio
import os
import functools
import openai
from typing import List, Optional, Dict, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

from burr.core import Application, ApplicationBuilder, State, default, when, graph, expr
from burr.core.application import ApplicationContext
from burr.core.action import action
from burr.lifecycle import LifecycleAdapter
from burr.tracking import LocalTrackingClient

load_dotenv()

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

DESTINATIONS = {
    "search_internet": "internet_search",
    "search_github": "github_search",
    "search_atlassian": "atlassian_search",
    "general_ai_response": "general",
    "email_assistant": "email_assistant",
    "search_google_maps": "google_maps_search",
    "search_knowledge_base": "knowledge_base_search",
    "unknown": "unknown",
    "reset_mode": "general",
}

MCP_CONFIGS = {
    "github": {
        "command": "github-mcp-server/cmd/github-mcp-server/github-mcp-server",
        "args": ["stdio"],
        "working_directory": "github-mcp-server",
        "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_PAT}"},
    },
    "brave-search": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env": {"BRAVE_API_KEY": "${BRAVE_API_KEY}"},
    },
    "atlassian": {
        "command": "npx",
        "args": [
            "-y",
            "mcp-remote",
            "https://mcp.atlassian.com/v1/sse",
            "--jira-url",
            "https://haptiq.atlassian.net",
        ]
    },
    "google-maps": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-google-maps"],
      "env": {
        "GOOGLE_MAPS_API_KEY": "${GOOGLE_MAPS_API_KEY}"
      }
    },
    "chroma": {
        "command": "uvx",
        "args": [
            "chroma-mcp",
            "--client-type", 
            "persistent",
            "--data-dir", 
            "/Users/ryanhaptiq/Projects/mcp-streamlit-demo/chroma/chroma_db"
        ]
    }
}


def add_keys_to_config() -> dict:
    config = {}
    config["mcpServers"] = {}
    for key, server_config in MCP_CONFIGS.items():
        config["mcpServers"][key] = server_config.copy()
        if "env" in config["mcpServers"][key]:
            for env_var_key, env_var_placeholder in config["mcpServers"][key][
                "env"
            ].items():
                if env_var_placeholder.startswith(
                    "${"
                ) and env_var_placeholder.endswith("}"):
                    env_var_name = env_var_placeholder[2:-1]
                    config["mcpServers"][key]["env"][env_var_key] = os.getenv(
                        env_var_name
                    )

    print(config)
    return config


def truncate_history(
    history: List[Dict[str, str]], max_turns: int = 5
) -> List[Dict[str, str]]:
    if len(history) <= max_turns * 2:
        return history
    return history[-(max_turns * 2) :]


@functools.lru_cache
def _get_openai_client():
    openai_client = openai.Client()
    return openai_client


@action(reads=[], writes=["user_input"])
def get_user_input(state: State, user_input: str) -> State:
    return state.update(user_input=user_input)



@action(reads=["user_input", "chat_history", "current_mode", "email_state"], 
        writes=["destination"])
async def route_request(state: State, common_llm: ChatOpenAI) -> State:
    """
    Routes user input to the appropriate destination based on content analysis.
    """
    # Check if we're in the email assistant flow and awaiting data
    if state.get("current_mode") == "email_assistant" and state.get("email_state") == "awaiting_data":
        return state.update(destination="create_reply_email")
    
    # For all other cases, determine the appropriate destination
    return await _determine_destination(state, common_llm)

async def _determine_destination(state: State, llm: ChatOpenAI) -> State:
    """
    Helper function to determine the appropriate destination for user input.
    
    Args:
        state: Current application state
        llm: Language model for classification
        
    Returns:
        Updated state with destination set
    """
    prompt = _build_routing_prompt(state["user_input"])
    result = await llm.ainvoke(prompt)
    
    # Process and validate the model's response
    raw_destination = result.content.strip().lower()
    destination = _validate_destination(raw_destination)
    
    # Log the routing decision
    current_mode = state.get("current_mode", "general")
    print(f"Current Mode: {current_mode}, Determined Destination: {destination}")
    
    return state.update(destination=destination)


def _build_routing_prompt(user_input: str) -> str:
    """
    Constructs a clear prompt for the routing LLM.
    
    Args:
        user_input: The user's message
        
    Returns:
        A formatted prompt string
    """
    # Get valid destination options (excluding reset_mode)
    valid_options = [mode for mode in DESTINATIONS.keys() if mode != 'reset_mode']
    options_list = ', '.join(valid_options)
    
    # Multi-line prompt for better readability
    prompt = f"""You are a chatbot classifier. Analyze this user input: "{user_input}"

Your task is to classify this input into exactly ONE of these categories:
{options_list}

Respond with ONLY the category name, nothing else. For example:
- For search queries about the web: "search_internet"
- For questions about coding repositories: "search_github"
- For JIRA or Atlassian questions: "search_atlassian"
- For help with emails or email drafting: "email_assistant"
- For general knowledge questions: "general_ai_response"
- For location or map queries: "search_google_maps"
- For questions about the internal knowledge base. This is a ChromaDB collection and is about our users and their interactions with our product: "search_knowledge_base"
- For queries that don't fit any category: "unknown"

Classification:"""
    
    return prompt


def _validate_destination(raw_destination: str) -> str:
    """
    Validates and sanitizes the destination returned by the LLM.
    
    Args:
        raw_destination: The raw destination string from the LLM
        
    Returns:
        A validated destination string
    """
    # Check if the raw destination is a valid key in DESTINATIONS
    if raw_destination in DESTINATIONS:
        return raw_destination
    
    # Log invalid destinations for debugging
    print(f"Warning: Invalid destination '{raw_destination}', defaulting to 'unknown'")
    
    # Try to find a close match (optional)
    for valid_dest in DESTINATIONS.keys():
        if valid_dest in raw_destination:
            print(f"Found partial match: '{valid_dest}'")
            return valid_dest
    
    # Default fallback
    return "unknown"

@action(
    reads=["user_input"],
    writes=["internet_search_results"],
)
async def perform_internet_search(state: State, common_llm: ChatOpenAI) -> State:
    print(">>> Performing internet search...")
    query = state["user_input"]
    
    # Create a fresh MCP client for each request
    brave_mcp_config = {
        "mcpServers": {"brave-search": full_mcp_config["mcpServers"]["brave-search"]}
    }
    fresh_client = MCPClient.from_dict(brave_mcp_config)
    fresh_agent = MCPAgent(llm=common_llm, client=fresh_client, max_steps=5)
    
    # Send a simple query without conversation history
    result = await fresh_agent.run(f"Search for information about: {query}")

    return state.update(internet_search_results=result)
# @action(
#     reads=["user_input", "chat_history", "current_mode"],
#     writes=["internet_search_results"],
# )
# async def perform_internet_search(state: State, internet_agent: MCPAgent) -> State:
#     print(">>> Performing internet search...")
#     query = state["user_input"]
#     chat_history = state.get("chat_history", [])

#     truncated_history = truncate_history(chat_history)
#     history_string = "\n".join(
#         [f"{msg['role'].capitalize()}: {msg['content']}" for msg in truncated_history]
#     )

#     query_to_send = (
#         f"Conversation History:\n{history_string}\nLatest User Input: {query}\nTask:"
#     )

#     result = await internet_agent.run(query_to_send)

    return state.update(internet_search_results=result)


@action(
    reads=["user_input", "chat_history", "current_mode"],
    writes=["github_search_results"],
)
async def perform_github_search(state: State, common_llm: ChatOpenAI) -> State:
    print(">>> Searching GitHub...")
    query = state["user_input"]
    chat_history = state.get("chat_history", [])

    truncated_history = truncate_history(chat_history)
    history_string = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in truncated_history]
    )

    query_to_send = (
        f"Conversation History:\n{history_string}\nLatest User Input: {query}\nTask:"
    )
    
    github_mcp_config = {"mcpServers": {"github": full_mcp_config["mcpServers"]["github"]}}
    github_mcp_client = MCPClient.from_dict(github_mcp_config)
    github_mcp_agent = MCPAgent(
        llm=common_llm, client=github_mcp_client, verbose=True, max_steps=10
)

    result = await github_mcp_agent.run(query_to_send)

    return state.update(github_search_results=result)


@action(
    reads=["user_input", "chat_history", "current_mode"],
    writes=["atlassian_search_results"],
)
async def perform_atlassian_search(state: State, common_llm: ChatOpenAI) -> State:
    print(">>> Searching JIRA...")
    query = state["user_input"]
    chat_history = state.get("chat_history", [])

    truncated_history = truncate_history(chat_history)
    history_string = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in truncated_history]
    )

    query_to_send = (
        f"Conversation History:\n{history_string}\nLatest User Input: {query}\nTask:"
    )
    
    
    atlassian_mcp_config = {
        "mcpServers": {"atlassian": full_mcp_config["mcpServers"]["atlassian"]}
    }
    atlassian_mcp_client = MCPClient.from_dict(atlassian_mcp_config)
    atlassian_mcp_agent = MCPAgent(llm=common_llm, client=atlassian_mcp_client, max_steps=5)

    result = await atlassian_mcp_agent.run(query_to_send)

    return state.update(atlassian_search_results=result)


@action(reads=["user_input", "chat_history"], writes=["general_ai_response"])
async def generate_general_ai_response(state: State, common_llm: ChatOpenAI) -> State:
    print(">>> Generating general AI response...")
    chat_history = state.get("chat_history", [])

    truncated_history = truncate_history(chat_history)

    messages = (
        [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        + truncated_history
        + [{"role": "user", "content": state["user_input"]}]
    )

    result = await common_llm.ainvoke(messages)
    response_content = result.content

    return state.update(general_ai_response=response_content)


@action(reads=["user_input"], writes=["chat_history"])
def prompt_for_more(state: State) -> State:
    print(">>> Unsure how to handle, prompting for clarification...")
    response = "AI: I'm not sure how to help with that. Could you please rephrase or provide more detail?"
    print(response)

    updated_history = state.get("chat_history", []) + [
        {"role": "user", "content": state["user_input"]},
        {"role": "assistant", "content": response.replace("AI: ", "")},
    ]

    return state.update(chat_history=updated_history)

@action(
    reads=["user_input", "chat_history", "current_mode"],
    writes=["gmaps_results"],
)
async def perform_google_maps_search(state: State, common_llm: ChatOpenAI) -> State:
    print(">>> Performing Google Maps search...")
    query = state["user_input"]
    chat_history = state.get("chat_history", [])
    
    gmaps_mcp_config = {
    "mcpServers": {"google-maps": full_mcp_config["mcpServers"]["google-maps"]}
    }
    gmaps_mcp_client = MCPClient.from_dict(gmaps_mcp_config)
    gmaps_mcp_agent = MCPAgent(llm=common_llm, client=gmaps_mcp_client, max_steps=10)

    result = await gmaps_mcp_agent.run(query)
    return state.update(gmaps_results=result)

@action(
    reads=["user_input", "chat_history", "current_mode"],
    writes=["knowledge_base_results"],
)
async def search_knowledge_base(state: State, common_llm: ChatOpenAI) -> State:
    print(">>> Searching knowledge base...")
    query = state["user_input"]
    chat_history = state.get("chat_history", [])
    
    # Now proceed with the actual search
    truncated_history = truncate_history(chat_history)
    history_string = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in truncated_history]
    )

    query_to_send = (
        f"Task: The user is looking for information from our ChromaDB knowledge base. "
        f"There is a 'user_interactions' collection in ChromaDB. "
        f"Then use the the appropriate chromaDB tool to satisfy the user query: {query}"
    )

    print(f">>> Sending query to ChromaDB agent: {query_to_send}...")
    
    chroma_mcp_config = {
    "mcpServers": {"chroma": full_mcp_config["mcpServers"]["chroma"]}
}
    print(f">>> Chroma MCP config: {chroma_mcp_config}")
    chroma_mcp_client = MCPClient.from_dict(chroma_mcp_config)
    chroma_mcp_agent = MCPAgent(llm=common_llm, client=chroma_mcp_client, max_steps=10)
    
    result = await chroma_mcp_agent.run(query_to_send)
    print(f">>> ChromaDB Search Result: {result}")

    # Make sure we're not returning None or an empty result
    if not result or result.strip() == "":
        result = "Knowledge Base Search Results:\nI searched the knowledge base but couldn't find any information about your query. It's possible that this topic hasn't been discussed before or the knowledge base is still being populated."
    else:
        result = f"Knowledge Base Search Results:\n{result}"
    
    return state.update(knowledge_base_results=result)

@action(
    reads=[
        "user_input",
        "internet_search_results",
        "github_search_results",
        "general_ai_response",
        "chat_history",
        "destination",
        "atlassian_search_results",
        "knowledge_base_results",
        "gmaps_results",
    ],
    writes=["chat_history", "final_response", "current_mode"],
)
def generate_final_response(state: State) -> State:
    print(">>> Finalizing response...")
    
    final_response_content = "An issue occurred."
    destination = state["destination"]
    
    # Map destination to current_mode using DESTINATIONS
    new_current_mode = DESTINATIONS.get(destination, "general")
    
    print(f">>> Destination: {destination}")
    print(f">>> Mode: {new_current_mode}")
    
    # Results mapping - links modes to their result variables
    results_mapping = {
        "knowledge_base_search": ("knowledge_base_results", "Knowledge Base Results"),
        "internet_search": ("internet_search_results", "Internet Search Results"),
        "github_search": ("github_search_results", "GitHub Search Results"),
        "atlassian_search": ("atlassian_search_results", "JIRA Search Results"),
        "google_maps_search": ("gmaps_results", "Google Maps Search Results"),
        "general": ("general_ai_response", None)  # No prefix for general response
    }
    
    # Get the relevant result for the current mode
    if new_current_mode in results_mapping:
        result_key, prefix = results_mapping[new_current_mode]
        result_content = state.get(result_key)
        
        if result_content:
            if prefix:
                final_response_content = f"{prefix}:\n{result_content}"
            else:
                final_response_content = result_content
    
    # Build the chat history update
    user_message = {"role": "user", "content": state["user_input"]}
    assistant_message = {"role": "assistant", "content": final_response_content}
    updated_history = state.get("chat_history", []) + [user_message, assistant_message]
    
    print(f">>> Setting new current mode to: {new_current_mode}")
    
    return state.update(
        chat_history=updated_history,
        final_response=final_response_content,
        current_mode=new_current_mode,
    )


@action(
    reads=[
        "final_response",
        "current_mode",
        "user_input",
        "internet_search_results",
        "github_search_results",
        "general_ai_response",
        "atlassian_search_results",
        "knowledge_base_results",
        "gmaps_results",
    ],
    writes=[
        "user_input",
        "internet_search_results",
        "github_search_results",
        "general_ai_response",
        "atlassian_search_results",
        "knowledge_base_results",
        "gmaps_results",
    ],
)
def present_response(state: State) -> State:
    print(f"AI: {state['final_response']}")

    user_input_to_clear = None
    internet_results_to_clear = state.get("internet_search_results")
    github_results_to_clear = state.get("github_search_results")
    atlassian_results_to_clear = state.get("atlassian_search_results")
    general_ai_response_to_clear = state.get("general_ai_response")
    knowledge_base_results_to_clear = state.get("knowledge_base_results")
    gmaps_results_to_clear = state.get("gmaps_results")
    current_mode = state["current_mode"]

    if current_mode != "internet_search":
        internet_results_to_clear = None
    if current_mode != "github_search":
        github_results_to_clear = None
    if current_mode != "knowledge_base_search":
        knowledge_base_results_to_clear = None
    if current_mode != "google_maps_search":
        gmaps_results_to_clear = None
    if current_mode != "atlassian_search":
        atlassian_results_to_clear = None
    if current_mode != "general":
        general_ai_response_to_clear = None

    return state.update(
        user_input=user_input_to_clear,
        internet_search_results=internet_results_to_clear,
        github_search_results=github_results_to_clear,
        general_ai_response=general_ai_response_to_clear,
        atlassian_search_results=atlassian_results_to_clear,
        knowledge_base_results=knowledge_base_results_to_clear,
        gmaps_results=gmaps_results_to_clear,
    )

@action(reads=["user_input", "chat_history"], writes=["chat_history", "current_mode", "email_state"])
def get_email_data(state: State) -> State:
    """Prompts the user to provide the email and instructions in a single step"""
    
    response = "To help you draft an email reply, please provide both:\n1. The original email text\n2. Your instructions for how you want to respond\n\nYou can separate them with '---' or clearly label which is which."
    
    updated_history = state.get("chat_history", []) + [
        {"role": "user", "content": state["user_input"]},
        {"role": "assistant", "content": response},
    ]
    
    return state.update(
        chat_history=updated_history,
        current_mode="email_assistant",
        email_state="awaiting_data"
    )

@action(reads=["user_input", "chat_history", "email_state"], writes=["email_response", "chat_history", "email_state"])
async def create_reply_email(state: State, common_llm: ChatOpenAI) -> State:
    """Creates a reply email based on the provided data"""
    
    if state.get("email_state") != "awaiting_data":
        return state
    
    # Process the user input to separate email and instructions
    user_input = state["user_input"].strip()
    
    # Check if there's a separator
    if "---" in user_input:
        parts = user_input.split("---", 1)
        email_content = parts[0].strip()
        instructions = parts[1].strip()
    else:
        # Send the input to the LLM to intelligently extract email and instructions
        messages = [
            {"role": "system", "content": "You are a helpful assistant that extracts the original email and response instructions from user input."},
            {"role": "user", "content": f"Please identify the original email text and the instructions for how to respond within the following text. Return ONLY a JSON with two keys: 'email' and 'instructions'.\n\n{user_input}"}
        ]
        
        result = await common_llm.ainvoke(messages)
        try:
            # Try to parse as JSON
            import json
            parsed = json.loads(result.content)
            email_content = parsed.get("email", "")
            instructions = parsed.get("instructions", "")
        except:
            # Fallback: try to detect if there are clear sections
            if "email:" in user_input.lower() and "instructions:" in user_input.lower():
                # Find the indices
                email_start = user_input.lower().find("email:")
                instructions_start = user_input.lower().find("instructions:")
                
                if email_start < instructions_start:
                    email_content = user_input[email_start+6:instructions_start].strip()
                    instructions = user_input[instructions_start+12:].strip()
                else:
                    instructions = user_input[instructions_start+12:email_start].strip()
                    email_content = user_input[email_start+6:].strip()
            else:
                # Last resort: assume first half is email, second half is instructions
                half_point = len(user_input) // 2
                email_content = user_input[:half_point].strip()
                instructions = user_input[half_point:].strip()
    
    # Print debugging info
    print(f"Extracted Email: {email_content[:100]}...")
    print(f"Extracted Instructions: {instructions[:100]}...")
    
    # Generate email reply
    messages = [
        {"role": "system", "content": "You are an email assistant that crafts professional, contextually appropriate responses."},
        {"role": "user", "content": f"Original Email:\n{email_content}\n\nInstructions for Response:\n{instructions}\n\nPlease craft a response email based on these instructions."}
    ]
    
    result = await common_llm.ainvoke(messages)
    email_response = result.content
    
    # Build response to show user
    response = f"Here's the email response I've created based on your input:\n\n{email_response}"
    
    updated_history = state.get("chat_history", []) + [
        {"role": "user", "content": state["user_input"]},
        {"role": "assistant", "content": response},
    ]
    
    # Clear the email_state to prevent re-processing
    return state.update(
        email_response=email_response,
        chat_history=updated_history,
        email_state=None  # Important: clear the state after processing
    )
@action(reads=["email_response"], writes=["final_response", "current_mode"])
def finalize_email_response(state: State) -> State:
    """Finalizes the email response and returns to general mode"""
    
    return state.update(
        final_response=f"Here's the email response I've created based on your input:\n\n{state['email_response']}",
        current_mode="general",
        email_state=None 
    )
    
    
common_llm = ChatOpenAI(model="gpt-4o", temperature=0)
full_mcp_config = add_keys_to_config()

graph = (
    graph.GraphBuilder()
    .with_actions(
        get_user_input=get_user_input,
        route_request=route_request.bind(common_llm=common_llm),
        perform_internet_search=perform_internet_search.bind(
            common_llm=common_llm
        ),
        perform_github_search=perform_github_search.bind(common_llm=common_llm),
        perform_atlassian_search=perform_atlassian_search.bind(common_llm=common_llm),
        search_knowledge_base=search_knowledge_base.bind(common_llm=common_llm),
        generate_general_ai_response=generate_general_ai_response.bind(
            common_llm=common_llm
        ),
        prompt_for_more=prompt_for_more,
        generate_final_response=generate_final_response,
        present_response=present_response,
        
        # Google Maps actions
        perform_google_maps_search=perform_google_maps_search.bind(common_llm=common_llm),
        
        get_email_data=get_email_data,
        create_reply_email=create_reply_email.bind(common_llm=common_llm),
        finalize_email_response=finalize_email_response,
        
    )
    .with_transitions(
        # Main conversation flow
        ("get_user_input", "route_request"),
        
        # Routing based on destination
        ("route_request", "perform_internet_search", when(destination="search_internet")),
        ("route_request", "perform_github_search", when(destination="search_github")),
        ("route_request", "perform_atlassian_search", when(destination="search_atlassian")),
        ("route_request", "search_knowledge_base", when(destination="search_knowledge_base")),
        ("route_request", "generate_general_ai_response", when(destination="general_ai_response")),
        ("route_request", "perform_google_maps_search", when(destination="search_google_maps")),
        ("route_request", "generate_final_response", when(destination="reset_mode")),
        
        # Email Assistant simplified flow
        ("route_request", "get_email_data", when(destination="email_assistant")),
        ("get_email_data", "get_user_input"),
        ("route_request", "create_reply_email", when(destination="create_reply_email")),
        ("create_reply_email", "finalize_email_response"),
        ("finalize_email_response", "generate_final_response"),
        
        # Default fallback
        ("route_request", "prompt_for_more", default),
        
        # Search actions to final response
        ("perform_internet_search", "generate_final_response"),
        ("perform_github_search", "generate_final_response"),
        ("perform_atlassian_search", "generate_final_response"),
        ("search_knowledge_base", "generate_final_response"),
        ("generate_general_ai_response", "generate_final_response"),
        ("perform_google_maps_search", "generate_final_response"),
        # Final response and loop back
        ("generate_final_response", "present_response"),
        ("present_response", "get_user_input"),
        ("prompt_for_more", "get_user_input"),
    )
    .build()
)


def base_application(
    hooks: List[LifecycleAdapter],
    app_id: str,
    storage_dir: str,
    project_id: str,
):
    if hooks is None:
        hooks = []
    tracker = LocalTrackingClient(project=project_id, storage_dir=storage_dir)

    return (
        ApplicationBuilder()
        .with_graph(graph)
        .initialize_from(
            tracker,
            resume_at_next_action=False,
            default_state={"chat_history": [], "current_mode": "general"},
            default_entrypoint="get_user_input",
        )
        .with_hooks(*hooks)
        .with_tracker(tracker)
        .with_identifiers(app_id=app_id)
        .build()
    )


def application(
    app_id: Optional[str] = None,
    project_id: str = "burr_with_mcp",
    storage_dir: Optional[str] = "~/.burr",
    hooks: Optional[List[LifecycleAdapter]] = None,
) -> Application:
    return base_application(hooks, app_id, storage_dir, project_id=project_id)