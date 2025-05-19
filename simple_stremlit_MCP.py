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


async def main():
    st.title("Chatbot example with Burr and MCP")
    burr_app = initialize_burr_app()

    for chat_message in burr_app.state.get("chat_history", []):
        render_chat_message(chat_message)

    prompt = st.chat_input("Ask me a question!", key="chat_input")

    if prompt:
        async for action, result, state in burr_app.aiterate(
            inputs={"user_input": prompt}, # CORRECTED input key
            halt_after=["generate_final_response", "prompt_for_more"]
        ):
            # Debug print kept to see action execution
            print(f">>>> Executing Burr Action: {action.name}")
            pass


        print(">>> Burr aiterate loop finished") # Debug print
        st.rerun() # CORRECTED rerun function
        print(">>> Triggered Streamlit rerun") # Debug print


if __name__ == "__main__":
    # This remains for Streamlit's async entry point handling
    # It will run the async main function
    asyncio.run(main())