import streamlit as st
from agents import compiled_graph

st.set_page_config(page_title="Legal Chatbot")
st.title("🧑‍⚖️ Legal Multi-Agent Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask a legal question…")
if st.button("Send") and query:
    # Run the langgraph workflow
    result = compiled_graph.invoke({"query": query})
    answer = result.get("summary", "Sorry, I couldn't summarize that.")
    st.session_state.history.append((query, answer))

for user_q, bot_a in st.session_state.history:
    st.markdown(f"**You:** {user_q}")
    st.markdown(f"**Bot:** {bot_a}")