import uuid
import asyncio

import burr_with_mcp as chatbot_application
import streamlit as st

import burr.core


def render_chat_message(chat_item: dict):
    content = chat_item["content"]
    role = chat_item.get("role", "assistant")

    with st.chat_message(role):
        st.write(content)


def initialize_burr_app() -> burr.core.Application:
    if "burr_app" not in st.session_state:
        st.session_state.burr_app = chatbot_application.application(
            app_id=f"chat:{str(uuid.uuid4())[0:6]}"
        )
    return st.session_state.burr_app


def show_server_status(burr_app):
    """Display server status in the sidebar with indicators."""
    
    # Get current mode from Burr app state
    current_mode = burr_app.state.get("current_mode", "general")
    email_phase = burr_app.state.get("email_instructions_phase")
    
    # Define all available servers
    servers = {
        "General AI": "general",
        "Internet Search": "internet_search",
        "GitHub": "github_search",
        "Atlassian": "atlassian_search",
        "Knowledge Base": "knowledge_base_search",
        "Google Maps": "google_maps_search"
    }
    
    # Create the sidebar
    st.sidebar.title("MCP Server Status")
    
    st.logo(
        "documentation/haptiq.png",
    )
    
    # List all servers with indicators
    for server_name, mode in servers.items():
        # Check if this server is active
        is_active = current_mode == mode and not email_phase
        
        # Create the indicator color
        indicator = "🟢" if is_active else "⚪"
        
        # Show the server with its indicator
        st.sidebar.markdown(f"{indicator} {server_name}")
    
    st.sidebar.title("Tools and Apps Status")
    
    if email_phase:
        st.sidebar.markdown(f"🟢 Email Assistant ({email_phase})")
    else:
        st.sidebar.markdown("⚪ Email Assistant")
        
    # Add a divider
    st.sidebar.divider()
    
    # Optionally show the current query being processed
    user_input = burr_app.state.get("user_input", "")
    if user_input:
        st.sidebar.caption("Current query:")
        st.sidebar.text(user_input)


async def main():
    st.title("Chatbot with Finite State Machine, MCPs, and Apps")
    burr_app = initialize_burr_app()
    
    # Show server status in sidebar
    show_server_status(burr_app)
    
    # Display chat history
    for chat_message in burr_app.state.get("chat_history", []):
        render_chat_message(chat_message)

    # Get user input
    prompt = st.chat_input("Ask me a question!", key="chat_input")

    if prompt:
        # Regular chatbot flow - the FSM handles all the logic now
        async for action, result, state in burr_app.aiterate(
            inputs={"user_input": prompt},
            halt_after=[
            "generate_final_response", 
            "prompt_for_more",
            "start_email_assistant",
            "collect_email_data", 
            "determine_clarifications",
            "formulate_draft",
            "process_feedback"
        ]
        ):
            # Debug print kept to see action execution
            print(f">>>> Executing Burr Action: {action.name}")
            pass

        print(">>> Burr aiterate loop finished")
        st.rerun()


if __name__ == "__main__":
    # This remains for Streamlit's async entry point handling
    asyncio.run(main())