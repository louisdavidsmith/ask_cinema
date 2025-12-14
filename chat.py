import streamlit as st
from src.cinema_expert import CinemaExpert
from src.models import CinemaExpertRequest
from src.config import get_config
from openai import OpenAI
from src.agent_tools import AgentTools


@st.cache_resource
def get_cinema_expert():
    config = get_config()
    client = OpenAI(api_key=config.open_ai_key)
    tools = AgentTools(config)
    return CinemaExpert(config, client, tools)


expert = get_cinema_expert()

st.title("Cinema Expert Chatbot")
st.write("Ask questions about movies, reviews, or cinema topics.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know about cinema?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Generating response..."):
        try:
            expert_request = CinemaExpertRequest(user_input=prompt)

            response = expert.invoke(expert_request)

            with st.chat_message("assistant"):
                st.markdown(response.generated_response)

            st.session_state.messages.append(
                {"role": "assistant", "content": response.generated_response}
            )

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            with st.chat_message("assistant"):
                st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )
