import streamlit as st
import os
from rag_pipeline import build_retriever, build_rag_chain
from langchain_core.messages import HumanMessage, AIMessage


st.title("📄 PDF RAG Chatbot")

# Load API key from environment
# os.environ["GROQ_API_KEY"] = st.secrets["groq_key"]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    retriever = build_retriever("temp.pdf")
    st.session_state.rag_chain = build_rag_chain(retriever)
    st.success("PDF processed successfully!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

user_question = st.chat_input("Ask a question about the PDF")

if user_question and st.session_state.rag_chain:

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_question)

    result = st.session_state.rag_chain.invoke({
        "question": user_question,
        "chat_history": st.session_state.chat_history
    })

    # Save to memory
    st.session_state.chat_history.append(
        HumanMessage(content=user_question)
    )
    st.session_state.chat_history.append(
        AIMessage(content=result)
    )

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(result)


