import sys
import asyncio
import os
import functools
import openai
from typing import List, Optional, Dict, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

from burr.core import Application, ApplicationBuilder, State, default, when, graph
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


@action(reads=["user_input", "chat_history", "current_mode", "email_instructions_phase"], writes=["destination"])
async def route_request(state: State, common_llm: ChatOpenAI) -> State:
    # Check if we're in the email assistant flow
    if state.get("email_instructions_phase"):
        # Route based on the current phase of email assistant
        email_phase = state.get("email_instructions_phase")
        if email_phase == "collect_email":
            return state.update(destination="collect_email_data")
        elif email_phase == "collect_instructions":
            return state.update(destination="collect_email_data")
        elif email_phase == "collect_clarifications":
            return state.update(destination="collect_clarifications")
        elif email_phase == "formulate_draft":
            return state.update(destination="formulate_draft")
        elif email_phase == "feedback_or_finalize":
            return state.update(destination="process_feedback")
        
    # Regular routing logic
    current_mode = state.get("current_mode", "general")
    chat_history = state.get("chat_history", [])

    prompt = (
        f"You are a chatbot. You've been prompted this: {state['user_input']}. "
        f"You have the capability of responding in the following modes: {', '.join([mode for mode in DESTINATIONS.keys() if mode != 'reset_mode'])}. "
        "Please respond with *only* a single response representing the mode that most accurately, for example 'search_knowledge_base' or 'search_internet' or 'email_assistant' or 'general_ai_response' or 'unknown' or 'reset_mode'."
        "corresponds to the prompt. "
        "For instance, if the prompt is 'search the internet for the latest news on the stock market', "
        "the mode would be 'search_internet'. If the prompt is 'what is the capital of France', the mode would be 'general_ai_response'."
        "If the prompt is about GitHub, the mode would be 'search_github'. If the prompt is about Brave Search, the mode would be 'search_internet'."
        "If the prompt is about JIRA, the mode would be 'search_atlassian'."
        "If the prompt mentions wanting help with emails, writing emails, or creating email responses, "
        "the mode would be 'email_assistant'."
        "If the prompt asks a question about the knowledge base, or to look at ChromaDB, the mode would be 'search_knowledge_base'."
        "If none of these modes apply, please respond with 'unknown'."
    )

    print(f"Prompt: {prompt}")

    result = await common_llm.ainvoke(prompt)

    content = result.content.strip().lower()
    destination = content if content in DESTINATIONS else "unknown"

    print(f"Current Mode: {current_mode}, Determined Destination: {destination}")
    print("Chat History (full): ", chat_history)

    return state.update(destination=destination)


@action(
    reads=["user_input", "chat_history", "current_mode"],
    writes=["internet_search_results"],
)
async def perform_internet_search(state: State, internet_agent: MCPAgent) -> State:
    print(">>> Performing internet search...")
    query = state["user_input"]
    chat_history = state.get("chat_history", [])

    truncated_history = truncate_history(chat_history)
    history_string = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in truncated_history]
    )

    query_to_send = (
        f"Conversation History:\n{history_string}\nLatest User Input: {query}\nTask:"
    )

    result = await internet_agent.run(query_to_send)

    return state.update(internet_search_results=result)


@action(
    reads=["user_input", "chat_history", "current_mode"],
    writes=["github_search_results"],
)
async def perform_github_search(state: State, github_agent: MCPAgent) -> State:
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

    result = await github_agent.run(query_to_send)

    return state.update(github_search_results=result)


@action(
    reads=["user_input", "chat_history", "current_mode"],
    writes=["atlassian_search_results"],
)
async def perform_atlassian_search(state: State, atlassian_agent: MCPAgent) -> State:
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

    result = await atlassian_agent.run(query_to_send)

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
    writes=["knowledge_base_results"],
)
async def search_knowledge_base(state: State, chroma_agent: MCPAgent) -> State:
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
    result = await chroma_agent.run(query_to_send)
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
    ],
    writes=["chat_history", "final_response", "current_mode"],
)
def generate_final_response(state: State) -> State:
    print(">>> Finalizing response...")
    # Print current state values for debugging
    print(f">>> State contains knowledge_base_results: {'knowledge_base_results' in state}")
    if 'knowledge_base_results' in state:
        print(f">>> Knowledge base results value: {state['knowledge_base_results']}...")
    
    final_response_content = "An issue occurred."

    # Get all possible results
    internet_results = state.get("internet_search_results")
    github_results = state.get("github_search_results")
    atlassian_results = state.get("atlassian_search_results")
    knowledge_base_results = state.get("knowledge_base_results")
    general_ai_resp = state.get("general_ai_response")
    destination = state["destination"]

    print(f">>> Destination: {destination}")
    
    # Prioritize based on destination
    if destination == "search_knowledge_base" and knowledge_base_results:
        print(">>> Using knowledge base results for response")
        final_response_content = knowledge_base_results
    elif destination == "search_internet" and internet_results:
        final_response_content = f"Internet Search Results:\n{internet_results}"
    elif destination == "search_github" and github_results:
        final_response_content = f"GitHub Search Results:\n{github_results}"
    elif destination == "search_atlassian" and atlassian_results:
        final_response_content = f"JIRA Search Results:\n{atlassian_results}"
    elif general_ai_resp:
        final_response_content = general_ai_resp

    # Rest of the function remains the same...
    user_message = {"role": "user", "content": state["user_input"]}
    assistant_message = {"role": "assistant", "content": final_response_content}

    updated_history = state.get("chat_history", []) + [user_message, assistant_message]

    new_current_mode = "general"
    if destination == "search_internet":
        new_current_mode = "internet_search"
    elif destination == "search_github":
        new_current_mode = "github_search"
    elif destination == "search_atlassian":
        new_current_mode = "atlassian_search"
    elif destination == "search_knowledge_base":  # Add this condition
        new_current_mode = "knowledge_base_search"
    
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
    ],
    writes=[
        "user_input",
        "internet_search_results",
        "github_search_results",
        "general_ai_response",
        "atlassian_search_results",
        "knowledge_base_results", 
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
    current_mode = state["current_mode"]

    if current_mode != "internet_search":
        internet_results_to_clear = None
    if current_mode != "github_search":
        github_results_to_clear = None
    if current_mode != "knowledge_base_search":
        knowledge_base_results_to_clear = None
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
    )


# Email Assistant Actions

@action(reads=["user_input", "chat_history"], writes=["chat_history", "email_instructions_phase"])
def start_email_assistant(state: State) -> State:
    """Starts the email assistant process"""
    
    response = "To use the Email Assistant, I'll need two pieces of information:\n1. The email you want to respond to\n2. Instructions for how you want to respond\n\nPlease paste the email text first."
    
    updated_history = state.get("chat_history", []) + [
        {"role": "user", "content": state["user_input"]},
        {"role": "assistant", "content": response},
    ]
    
    return state.update(
        chat_history=updated_history,
        email_instructions_phase="collect_email"
    )


@action(reads=["user_input", "email_instructions_phase"], writes=["email_content", "email_instructions", "email_instructions_phase", "chat_history"])
def collect_email_data(state: State) -> State:
    """Collects email content and instructions based on the phase we're in"""
    
    # Get the current phase
    phase = state.get("email_instructions_phase", "collect_email")
    
    if phase == "collect_email":
        # Save the email content
        email_content = state["user_input"]
        response = "Thanks for the email content. Now, please provide your instructions for how you want to respond to this email."
        
        # Add response to chat history
        updated_history = state.get("chat_history", []) + [
            {"role": "user", "content": state["user_input"]},
            {"role": "assistant", "content": response},
        ]
        
        return state.update(
            chat_history=updated_history,
            email_content=email_content,
            email_instructions="",  # Initialize with empty string to satisfy the writes requirement
            email_instructions_phase="collect_instructions"
        )
    
    elif phase == "collect_instructions":
        # Save the instructions
        email_instructions = state["user_input"]
        response = "Thank you. I'm now going to help you draft a response based on these instructions."
        
        # Add response to chat history
        updated_history = state.get("chat_history", []) + [
            {"role": "user", "content": state["user_input"]},
            {"role": "assistant", "content": response},
        ]
        
        return state.update(
            chat_history=updated_history,
            email_instructions=email_instructions,
            email_instructions_phase="determine_clarifications"
        )
    
    return state


@action(reads=["email_content", "email_instructions"], writes=["clarification_questions", "email_instructions_phase", "chat_history"])
def determine_clarifications(state: State) -> State:
    """Determines if clarification is needed for the email response"""
    email_content = state["email_content"]
    email_instructions = state["email_instructions"]
    
    client = _get_openai_client()
    result = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a chatbot that has the task of generating responses to an email on behalf of a user. ",
            },
            {
                "role": "user",
                "content": (
                    f"The email you are to respond to is: {email_content}."
                    f"Your instructions are: {email_instructions}."
                    "Your first task is to ask any clarifying questions for the user"
                    " who is asking you to respond to this email. These clarifying questions are for the user, "
                    "*not* for the original sender of the email. Please "
                    "generate a list of at most 3 questions (and you really can do less -- 2, 1, or even none are OK! joined by newlines, included only if you feel that you could leverage "
                    "clarification (my time is valuable)."
                    "The questions, joined by newlines, must be the only text you return. If you do not need clarification, return an empty string."
                ),
            },
        ],
    )
    content = result.choices[0].message.content
    all_questions = content.split("\n") if content else []
    
    # Add response to chat history if there are questions
    if all_questions and all_questions[0].strip():
        questions_text = "\n".join(all_questions)
        response = f"I need a few clarifications before drafting the response:\n\n{questions_text}\n\nPlease provide answers to these questions."
        
        updated_history = state.get("chat_history", []) + [
            {"role": "assistant", "content": response},
        ]
        
        return state.update(
            chat_history=updated_history,
            clarification_questions=all_questions,
            email_instructions_phase="collect_clarifications"
        )
    else:
        # No questions needed, go straight to drafting
        return state.update(
            clarification_questions=[],
            email_instructions_phase="formulate_draft"
        )


@action(reads=["user_input", "clarification_questions"], writes=["clarification_answers", "email_instructions_phase", "chat_history"])
def collect_clarifications(state: State) -> State:
    """Collects answers to clarification questions"""
    
    # Get the user's answers
    clarification_answers = [state["user_input"]]  # Simple case - just use the whole input as one answer
    
    # Add to chat history
    updated_history = state.get("chat_history", []) + [
        {"role": "user", "content": state["user_input"]},
    ]
    
    return state.update(
        chat_history=updated_history,
        clarification_answers=clarification_answers,
        email_instructions_phase="formulate_draft"
    )


@action(reads=["email_content", "email_instructions", "clarification_questions", "clarification_answers"], 
        writes=["current_draft", "draft_history", "email_instructions_phase", "chat_history"])
def formulate_draft(state: State) -> State:
    """Formulates a draft email response"""
    
    email_content = state["email_content"]
    email_instructions = state["email_instructions"]
    
    # Format clarification Q&A if they exist
    clarification_answers_formatted_q_a = ""
    if state.get("clarification_questions") and state.get("clarification_answers"):
        clarification_answers_formatted_q_a = "\n".join(
            [
                f"Q: {q}\nA: {a}"
                for q, a in zip(
                    state["clarification_questions"], state.get("clarification_answers", [])
                )
            ]
        )
    
    client = _get_openai_client()
    
    instructions = [
        f"The email you are to respond to is: {email_content}.",
        f"Your instructions are: {email_instructions}.",
    ]
    
    if clarification_answers_formatted_q_a:
        instructions.append("You have already asked the following questions and received the following answers: ")
        instructions.append(clarification_answers_formatted_q_a)
    
    draft_history = state.get("draft_history", [])
    if draft_history:
        instructions.append("Your previous draft was: ")
        instructions.append(draft_history[-1])
        instructions.append(
            "you received the following feedback, please incorporate this into your response: "
        )
        instructions.append(state.get("feedback", ""))
    
    instructions.append("Please generate a draft response using all this information!")
    prompt = " ".join(instructions)

    result = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a chatbot that has the task of generating responses to an email. ",
            },
            {"role": "user", "content": prompt},
        ],
    )
    
    draft_content = result.choices[0].message.content
    
    # Add to chat history
    response = f"Here's my draft email response:\n\n{draft_content}\n\nWhat do you think? Would you like me to make any changes?"
    
    updated_history = state.get("chat_history", []) + [
        {"role": "assistant", "content": response},
    ]
    
    # Update draft history
    new_draft_history = draft_history + [draft_content] if draft_history else [draft_content]
    
    return state.update(
        chat_history=updated_history,
        current_draft=draft_content,
        draft_history=new_draft_history,
        email_instructions_phase="feedback_or_finalize"
    )


@action(reads=["user_input"], writes=["feedback", "email_instructions_phase", "chat_history", "current_mode"])
def process_feedback(state: State) -> State:
    """Processes feedback for the draft email"""
    
    user_input = state["user_input"].lower()
    
    # Check if the user wants to finalize or provide feedback
    if any(word in user_input for word in ["good", "great", "perfect", "finalize", "done", "looks good"]):
        # User is satisfied with the draft
        response = "I'm glad you're happy with the draft! The final email response is ready to send."
        
        updated_history = state.get("chat_history", []) + [
            {"role": "user", "content": state["user_input"]},
            {"role": "assistant", "content": response},
        ]
        
        return state.update(
            chat_history=updated_history,
            email_instructions_phase=None,  # Clear the email state
            current_mode="general"  # Return to general mode
        )
    else:
        # User is providing feedback
        response = "Thank you for your feedback. I'll revise the draft accordingly."
        
        updated_history = state.get("chat_history", []) + [
            {"role": "user", "content": state["user_input"]},
            {"role": "assistant", "content": response},
        ]
        
        return state.update(
            chat_history=updated_history,
            feedback=state["user_input"],
            email_instructions_phase="formulate_draft"  # Go back to drafting
        )


common_llm = ChatOpenAI(model="gpt-4o", temperature=0)
full_mcp_config = add_keys_to_config()

github_mcp_config = {"mcpServers": {"github": full_mcp_config["mcpServers"]["github"]}}
github_mcp_client = MCPClient.from_dict(github_mcp_config)
github_mcp_agent = MCPAgent(
    llm=common_llm, client=github_mcp_client, verbose=True, max_steps=10
)

brave_mcp_config = {
    "mcpServers": {"brave-search": full_mcp_config["mcpServers"]["brave-search"]}
}
brave_mcp_client = MCPClient.from_dict(brave_mcp_config)
brave_mcp_agent = MCPAgent(llm=common_llm, client=brave_mcp_client, max_steps=5)

atlassian_mcp_config = {
    "mcpServers": {"atlassian": full_mcp_config["mcpServers"]["atlassian"]}
}
atlassian_mcp_client = MCPClient.from_dict(atlassian_mcp_config)
atlassian_mcp_agent = MCPAgent(llm=common_llm, client=atlassian_mcp_client, max_steps=5)

chroma_mcp_config = {
    "mcpServers": {"chroma": full_mcp_config["mcpServers"]["chroma"]}
}
print(f">>> Chroma MCP config: {chroma_mcp_config}")
chroma_mcp_client = MCPClient.from_dict(chroma_mcp_config)
chroma_mcp_agent = MCPAgent(llm=common_llm, client=chroma_mcp_client, max_steps=10)


graph = (
    graph.GraphBuilder()
    .with_actions(
        get_user_input=get_user_input,
        route_request=route_request.bind(common_llm=common_llm),
        perform_internet_search=perform_internet_search.bind(
            internet_agent=brave_mcp_agent
        ),
        perform_github_search=perform_github_search.bind(github_agent=github_mcp_agent),
        perform_atlassian_search=perform_atlassian_search.bind(
            atlassian_agent=atlassian_mcp_agent
        ),
        search_knowledge_base=search_knowledge_base.bind(
            chroma_agent=chroma_mcp_agent
        ),
        generate_general_ai_response=generate_general_ai_response.bind(
            common_llm=common_llm
        ),
        prompt_for_more=prompt_for_more,
        generate_final_response=generate_final_response,
        present_response=present_response,
        
        # Email Assistant actions
        start_email_assistant=start_email_assistant,
        collect_email_data=collect_email_data,
        determine_clarifications=determine_clarifications,
        collect_clarifications=collect_clarifications,
        formulate_draft=formulate_draft,
        process_feedback=process_feedback,
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
        ("route_request", "generate_final_response", when(destination="reset_mode")),
        ("route_request", "start_email_assistant", when(destination="email_assistant")),
        ("route_request", "collect_email_data", when(destination="collect_email_data")),
        ("route_request", "collect_clarifications", when(destination="collect_clarifications")),
        ("route_request", "formulate_draft", when(destination="formulate_draft")),
        ("route_request", "process_feedback", when(destination="process_feedback")),
        ("route_request", "prompt_for_more", default),  # Default fallback last
        
        # Search actions to final response
        ("perform_internet_search", "generate_final_response"),
        ("perform_github_search", "generate_final_response"),
        ("perform_atlassian_search", "generate_final_response"),
        ("search_knowledge_base", "generate_final_response"),
        ("generate_general_ai_response", "generate_final_response"),
        
        # Email assistant flow
        ("start_email_assistant", "get_user_input"),
        ("collect_email_data", "get_user_input"),
        ("collect_email_data", "determine_clarifications", when(email_instructions_phase="determine_clarifications")),
        ("determine_clarifications", "get_user_input"),
        ("collect_clarifications", "formulate_draft"),
        ("formulate_draft", "get_user_input"),
        ("process_feedback", "get_user_input"),
        
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