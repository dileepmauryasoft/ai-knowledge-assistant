from langchain_core.messages import HumanMessage, AIMessage
from app import chain
import streamlit as st

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def ask_question(query):
    chat_history = st.session_state.chat_history

    response = chain.invoke({
        "input": query,
        "chat_history": chat_history
    })

    # Save history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["answer"]))

    return response["answer"]