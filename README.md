📄 PDF RAG Chatbot (LangChain + Groq)

This project is an end-to-end Retrieval-Augmented Generation (RAG) chatbot that allows users to upload a PDF and ask questions about its content.

Instead of relying purely on a language model, the system retrieves relevant sections from the document and generates answers grounded in the uploaded text.

The goal of this project was to build a production-style GenAI application from scratch — including retrieval logic, vector storage, LLM integration, UI, and deployment readiness.

🚀 What This Project Does

Upload any PDF document

Ask natural language questions

Retrieve semantically relevant chunks using embeddings

Generate context-grounded responses using a hosted LLM

Maintain conversational memory within a session

🏗 How It Works

The uploaded PDF is split into overlapping chunks.

Each chunk is converted into vector embeddings using a lightweight sentence-transformer model.

The embeddings are stored in a Chroma vector database.

When a question is asked, the system retrieves the most relevant chunks.

The retrieved context is passed to the LLM (Groq LLaMA 3.1).

The model generates a response strictly grounded in the document.

This approach reduces hallucination and improves answer relevance compared to a standalone LLM.

🛠 Tech Stack

Python

LangChain

HuggingFace Embeddings (MiniLM)

ChromaDB

Groq LLM API

Streamlit

python-dotenv

🎯 Why I Built This

I wanted to move beyond notebook-style ML projects and build something closer to a real-world GenAI system.

Through this project I:

Designed a modular RAG pipeline

Tuned chunk size and overlap for long technical documents

Handled dependency conflicts and environment issues

Implemented conversational memory

Built and structured a deployable application

▶️ Run Locally
pip install -r requirements.txt
streamlit run app.py

🔮 Possible Improvements

Add reranking to improve retrieval precision

Implement persistent chat memory

Add evaluation metrics for grounding accuracy

Benchmark chunking strategies