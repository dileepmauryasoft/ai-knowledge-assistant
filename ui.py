import streamlit as st
from memory import ask_question as ask

st.title("Advanced RAG Chatbot")
print("----CHAT HISTORY----")
for msg in st.session_state.chat_history:
    print(msg)
query = st.text_input("Ask:")

if query:
    answer = ask(query)
    st.write(answer)