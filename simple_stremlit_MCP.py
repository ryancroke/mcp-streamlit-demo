import uuid
import asyncio

import burr_with_mcp as chatbot_application
import streamlit as st

import burr.core

if "need_rerun" not in st.session_state:
    st.session_state.need_rerun = False
    
def render_chat_message(chat_item: dict):
    content = chat_item["content"]
    role = chat_item.get("role", "assistant")

    with st.chat_message(role):
        st.write(content)

def show_server_status(burr_app):
    """Display server status in the sidebar with indicators."""
    
    # Get current mode from Burr app state
    current_mode = burr_app.state.get("current_mode", "general")
    
    # Define all available servers
    servers = {
        "General AI": "general",
        "Internet Search": "internet_search",
        "GitHub": "github_search",
        "Atlassian": "atlassian_search"
    }
    
    # Create the sidebar
    st.sidebar.title("MCP Server Status")
    
    # List all servers with indicators
    for server_name, mode in servers.items():
        # Check if this server is active
        is_active = current_mode == mode
        
        # Create the indicator color
        indicator = "🟢" if is_active else "⚪"
        
        # Show the server with its indicator
        st.sidebar.markdown(f"{indicator} {server_name}")
        
    # Add a divider
    st.sidebar.divider()
    
    # Optionally show the current query being processed
    user_input = burr_app.state.get("user_input", "")
    if user_input:
        st.sidebar.caption("Current query:")
        st.sidebar.text(user_input)

def initialize_burr_app() -> burr.core.Application:
    # First, ensure the agents are created only once and stored in session state
    if "mcp_agents" not in st.session_state:
        # Import agents from the burr_with_mcp module
        import burr_with_mcp
        st.session_state.mcp_agents = {
            "github": burr_with_mcp.github_mcp_agent,
            "brave": burr_with_mcp.brave_mcp_agent,
            "atlassian": burr_with_mcp.atlassian_mcp_agent
        }
    
    # Now initialize the Burr app
    if "burr_app" not in st.session_state:
        st.session_state.burr_app = chatbot_application.application(
            app_id=f"chat:{str(uuid.uuid4())[0:6]}"
        )
    return st.session_state.burr_app


async def main():
    st.title("Chatbot example with Burr and MCP")
    burr_app = initialize_burr_app()
    
    show_server_status(burr_app)

    for chat_message in burr_app.state.get("chat_history", []):
        render_chat_message(chat_message)

    prompt = st.chat_input("Ask me a question!", key="chat_input")

    if prompt:
        try:
            async for action, result, state in burr_app.aiterate(
                inputs={"user_input": prompt},
                halt_after=["generate_final_response", "prompt_for_more"]
            ):
                print(f">>>> Executing Burr Action: {action.name}")
                pass

            print(">>> Burr aiterate loop finished")
            
            # Store that we need to rerun but don't do it yet
            st.session_state.need_rerun = True
            
        except Exception as e:
            st.error(f"Error: {e}")
    
    # After the main logic runs, check if we need to rerun
    if st.session_state.get("need_rerun", False):
        st.session_state.need_rerun = False
        st.rerun()
        print(">>> Triggered Streamlit rerun")


if __name__ == "__main__":
    # This remains for Streamlit's async entry point handling
    # It will run the async main function
    asyncio.run(main())