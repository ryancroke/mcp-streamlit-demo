import sys
import asyncio
import os
import json
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

from burr.core import Application, ApplicationBuilder, State, default, when, expr, graph
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
    "unknown": "unknown",
    "reset_mode": "general"
}

MCP_CONFIGS = {
    "github": {
        "command": "github-mcp-server/cmd/github-mcp-server/github-mcp-server",
        "args": ["stdio"],
        "working_directory": "github-mcp-server",
        "env": {
            "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_PAT}"
        }
    },
    "brave-search": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env": {
            "BRAVE_API_KEY": "${BRAVE_API_KEY}"
        }
    },
    "atlassian": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://mcp.atlassian.com/v1/sse",
      "--jira-url", "https://haptiq.atlassian.net"]
    }
}

def add_keys_to_config() -> dict:
    keys_with_env_vars = [key for key in MCP_CONFIGS.keys() if "env" in MCP_CONFIGS[key]]
    other_keys = [key for key in MCP_CONFIGS.keys() if key not in keys_with_env_vars]
    
    config = {}
    config["mcpServers"] = {}
    for key in keys_with_env_vars:
        config["mcpServers"][key] = MCP_CONFIGS[key]
        if "env" in config["mcpServers"][key]:
            for env_var in config["mcpServers"][key]["env"]:
                config["mcpServers"][key]["env"][env_var] = os.getenv(env_var)
                
    for key in other_keys:
        config["mcpServers"][key] = MCP_CONFIGS[key]

    return config

# Simple function to truncate chat history
def truncate_history(history: List[Dict[str, str]], max_turns: int = 5) -> List[Dict[str, str]]:
    """Truncates chat history to the last max_turns."""
    if len(history) <= max_turns * 2: # Each turn is a user and assistant message
        return history
    return history[-(max_turns * 2):]


@action(reads=[], writes=["user_input"])
def get_user_input(state: State, user_input: str) -> State:
    return state.update(user_input=user_input)


@action(reads=["user_input", "chat_history", "current_mode"], writes=["destination"])
async def route_request(state: State) -> State:
    user_input = state["user_input"].lower()
    current_mode = state.get("current_mode", "general")
    chat_history = state.get('chat_history', []) # Read full history

    # Truncate history for the LLM call
    truncated_history = truncate_history(chat_history)
    history_string = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in truncated_history])
    
    prompt = (
        f"You are a chatbot. You've been prompted this: {state['user_input']}. "
        "Pay more attention to the user input but also consider the current mode and conversation history. "
        f"Your current interaction mode is: {current_mode}. "
        f"Conversation History:\n{history_string}\n"
        f"You have the capability of responding in the following modes: {', '.join([mode for mode in DESTINATIONS.keys() if mode != 'reset_mode'])}. "
        "Please respond with *only* a single word representing the mode that most accurately "
        "corresponds to the prompt. "
        "For instance, if the prompt is 'search the internet for the latest news on the stock market', "
        "the mode would be 'search_internet'. If the prompt is 'what is the capital of France', the mode would be 'general_ai_response'."
        "If the prompt is about GitHub, the mode would be 'search_github'. If the prompt is about Brave Search, the mode would be 'search_internet'."
        "If the prompt is about JIRA, the mode would be 'search_atlassian'."
        "If none of these modes apply, please respond with 'unknown'."
    )

    print(f"Prompt: {prompt}")


    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    result = await llm.ainvoke(prompt)

    content = result.content.strip().lower()
    destination = content if content in DESTINATIONS else "unknown"

    print(f"Current Mode: {current_mode}, Determined Destination: {destination}")
    print("Chat History (full): ", chat_history) 


    return state.update(destination=destination)


@action(reads=["user_input", "chat_history", "current_mode"], writes=["internet_search_results"])
async def perform_internet_search(state: State) -> State:
    print(">>> Performing internet search...")
    query = state["user_input"]
    chat_history = state.get("chat_history", []) 
    current_mode = state.get("current_mode", "general") 

    truncated_history = truncate_history(chat_history)
    history_string = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in truncated_history])

    query_to_send = f"Conversation History:\n{history_string}\nLatest User Input: {query}\nTask:"


    config = add_keys_to_config()

    client = MCPClient.from_dict(config)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agent = MCPAgent(llm=llm, client=client, max_steps=5)

    result = await agent.run(query_to_send) 

    return state.update(internet_search_results=result)


@action(reads=["user_input", "chat_history", "current_mode"], writes=["github_search_results"]) 
async def perform_github_search(state: State) -> State:
    print(">>> Searching GitHub...")
    query = state["user_input"]
    chat_history = state.get("chat_history", []) 
    current_mode = state.get("current_mode", "general") 

    truncated_history = truncate_history(chat_history)
    history_string = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in truncated_history])

    query_to_send = f"Conversation History:\n{history_string}\nLatest User Input: {query}\nTask:"


    config = add_keys_to_config()

    #TODO: Shouldnt we be subsetting the MCP server for each agaent since we are in that (Burr) state?
    client = MCPClient.from_dict(config)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agent = MCPAgent(llm=llm, client=client, max_steps=5)

    result = await agent.run(query_to_send) 

    return state.update(github_search_results=result)

@action(reads=["user_input", "chat_history", "current_mode"], writes=["atlassian_search_results"]) 
async def perform_atlassian_search(state: State) -> State:
    print(">>> Searching JIRA...")
    query = state["user_input"]
    chat_history = state.get("chat_history", []) 
    current_mode = state.get("current_mode", "general") 

    truncated_history = truncate_history(chat_history)
    history_string = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in truncated_history])

    query_to_send = f"Conversation History:\n{history_string}\nLatest User Input: {query}\nTask:"


    config = add_keys_to_config()

    client = MCPClient.from_dict(config)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agent = MCPAgent(llm=llm, client=client, max_steps=5)

    result = await agent.run(query_to_send) 

    return state.update(atlassian_search_results=result)

@action(reads=["user_input", "chat_history"], writes=["general_ai_response"])
async def generate_general_ai_response(state: State) -> State:
    print(">>> Generating general AI response...")
    chat_history = state.get("chat_history", []) 

    truncated_history = truncate_history(chat_history)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ] + truncated_history + [
        {"role": "user", "content": state["user_input"]}
    ]

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    result = await llm.ainvoke(messages)
    response_content = result.content

    return state.update(general_ai_response=response_content)


@action(reads=["user_input"], writes=["chat_history"])
def prompt_for_more(state: State) -> State:
    print(">>> Unsure how to handle, prompting for clarification...")
    response = "AI: I'm not sure how to help with that. Could you please rephrase or provide more detail?"
    print(response)

    updated_history = state.get("chat_history", []) + [
        {"role": "user", "content": state["user_input"]},
        {"role": "assistant", "content": response.replace("AI: ", "")}
    ]

    return state.update(chat_history=updated_history)


@action(
    reads=["user_input", "internet_search_results", "github_search_results", "general_ai_response", "chat_history", "destination"],
    writes=["chat_history", "final_response", "current_mode"]
)
def generate_final_response(state: State) -> State:
    print(">>> Finalizing response...")
    final_response_content = "An issue occurred."

    internet_results = state.get("internet_search_results")
    github_results = state.get("github_search_results")
    atlassian_results = state.get("atlassian_search_results")
    general_ai_resp = state.get("general_ai_response")
    destination = state["destination"]

    if internet_results:
        final_response_content = f"Internet Search Results:\n{internet_results}"
    elif github_results:
        final_response_content = f"GitHub Search Results:\n{github_results}"
    elif atlassian_results:
        final_response_content = f"JIRA Search Results:\n{atlassian_results}"
    elif general_ai_resp:
        final_response_content = general_ai_resp

    # Add the user input and the assistant response to the chat history
    user_message = {"role": "user", "content": state["user_input"]}
    assistant_message = {"role": "assistant", "content": final_response_content}

    updated_history = state.get("chat_history", []) + [
        user_message,
        assistant_message
    ]

    new_current_mode = "general"
    if destination == "search_internet":
        new_current_mode = "internet_search"
    elif destination == "search_github":
        new_current_mode = "github_search"
    elif destination == "search_atlassian":
        new_current_mode = "atlassian_search"
    elif destination == "general_ai_response":
        new_current_mode = "general"
    elif destination == "reset_mode":
         new_current_mode = "general"
    elif destination == "unknown": 
         new_current_mode = "general"


    return state.update(
        chat_history=updated_history,
        final_response=final_response_content,
        current_mode=new_current_mode
    )


@action(reads=["final_response", "current_mode", "user_input", "internet_search_results", "github_search_results", "general_ai_response", "atlassian_search_results"],
        writes=["user_input", "internet_search_results", "github_search_results", "general_ai_response", "atlassian_search_results"])
def present_response(state: State) -> State:
    print(f"AI: {state['final_response']}")

    user_input_to_clear = None
    internet_results_to_clear = state.get("internet_search_results")
    github_results_to_clear = state.get("github_search_results")
    atlassian_results_to_clear = state.get("atlassian_search_results")
    general_ai_response_to_clear = state.get("general_ai_response")
    current_mode = state["current_mode"]

    if current_mode != "internet_search":
        internet_results_to_clear = None
    if current_mode != "github_search":
        github_results_to_clear = None
    if current_mode != "atlassian_search":
        atlassian_results_to_clear = None
    if current_mode != "general":
        general_ai_response_to_clear = None


    return state.update(
        user_input=user_input_to_clear,
        internet_search_results=internet_results_to_clear,
        github_search_results=github_results_to_clear,
        general_ai_response=general_ai_response_to_clear,
        atlassian_search_results=atlassian_results_to_clear
    )

graph = (
    graph.GraphBuilder()
    .with_actions(
        get_user_input=get_user_input,
        route_request=route_request,
        perform_internet_search=perform_internet_search,
        perform_atlassian_search=perform_atlassian_search,
        perform_github_search=perform_github_search,
        generate_general_ai_response=generate_general_ai_response,
        prompt_for_more=prompt_for_more,
        generate_final_response=generate_final_response,
        present_response=present_response,
    )
    .with_transitions(
        ("get_user_input", "route_request"),
        ("route_request", "perform_internet_search", when(destination="search_internet")),
        ("route_request", "perform_github_search", when(destination="search_github")),
        ("route_request", "perform_atlassian_search", when(destination="search_atlassian")),
        ("route_request", "generate_general_ai_response", when(destination="general_ai_response")),
        ("route_request", "generate_final_response", when(destination="reset_mode")),
        ("route_request", "prompt_for_more", default),

        (
            ["perform_internet_search", "perform_github_search", "perform_atlassian_search", "generate_general_ai_response"],
            "generate_final_response"
        ),
        
        ("generate_final_response", "present_response"),
        ("present_response", "get_user_input"),
        ("prompt_for_more", "get_user_input")
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