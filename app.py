import streamlit as st
import os
from rag_pipeline import build_retriever, build_rag_chain
from langchain_core.messages import HumanMessage, AIMessage
import json
from datetime import datetime


def log_interaction(question, answer, docs):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "pages": [doc.metadata.get("page", "Unknown") for doc in docs]
    }

    with open("rag_logs.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")

st.title("📄 PDF RAG Chatbot")

if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

if st.button("Reset PDF"):
    st.session_state.rag_chain = None
    st.session_state.chat_history = []
    st.cache_resource.clear()
    st.rerun()


# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Build retriever only once per upload
@st.cache_resource
def load_retriever(pdf_path):
    return build_retriever(pdf_path)


if uploaded_file and st.session_state.rag_chain is None:
    with st.spinner("Processing PDF..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        retriever = load_retriever("temp.pdf")
        st.session_state.rag_chain = build_rag_chain(retriever)

    st.success("PDF processed successfully!")

# Display previous chat messages
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# User input
user_question = st.chat_input("Ask a question about the PDF")

if user_question and st.session_state.rag_chain is not None:
    
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate response with spinner
    try:
      with st.spinner("Thinking..."):
        response = st.session_state.rag_chain({
            "question": user_question,
            "chat_history": st.session_state.chat_history
        })
    except Exception:
      st.error("Something went wrong while generating the response.")
      st.stop()


    answer = response["answer"]
    docs = response["docs"]
    log_interaction(user_question, answer, docs)

    # Save memory
    st.session_state.chat_history.append(
        HumanMessage(content=user_question)
    )
    st.session_state.chat_history.append(
        AIMessage(content=answer)
    )

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)

        st.markdown("### 📚 Sources")
        shown_pages = set()

        for i, doc in enumerate(docs[:5], 1):  # Show top 5
            page = doc.metadata.get("page", "Unknown")

            # Avoid duplicate page display
            if page in shown_pages:
                continue

            shown_pages.add(page)

            preview = doc.page_content[:200].replace("\n", " ")

            st.markdown(
                f"**Page {page}** — {preview}..."
            )
