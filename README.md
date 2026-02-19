📄 PDF RAG Chatbot
Retrieval-Augmented Generation System with Source Citation

An end-to-end Retrieval-Augmented Generation (RAG) application that allows users to upload PDFs and ask context-grounded questions. The system retrieves relevant document chunks using semantic search and generates answers backed by source citations.

🚀 Features

Upload any PDF document

Semantic search using embeddings

MMR-based top-k retrieval (k=8)

Chunking (800) with 150 overlap

Context-grounded LLM responses

Source citation with page references

Deployed on Streamlit Cloud

🧠 Architecture Overview

Pipeline Flow:

PDF ingestion & text extraction

Recursive chunking (size=800, overlap=150)

Embedding generation (HuggingFace)

Vector storage in ChromaDB

MMR retrieval (Top 8 relevant chunks)

Context injection into Groq LLM

Answer generation with cited sources

⚙️ Tech Stack

LLM: Groq

Embeddings: HuggingFace model

Vector Database: ChromaDB

Framework: LangChain

UI: Streamlit

Deployment: Streamlit Cloud

🎥 Demo: https://youtu.be/6Bh7FiNFnu0

Watch the Demo Video

📌 Key Engineering Decisions

Balanced chunk size to optimize precision vs context coherence

150 overlap to preserve semantic continuity

MMR retrieval to reduce redundancy in retrieved chunks

Source-grounded generation to reduce hallucinations

🔮 Future Improvements

Confidence scoring mechanism

Hybrid search (BM25 + Vector search)

Adaptive chunking based on document structure

Guardrail-based refusal mechanism

📂 Project Structure
├── app.py
├── utils/
├── vector_store/
├── requirements.txt
├── README.md


(Modify structure according to your actual repo.)

🧑‍💻 Author

Ankur Karmakar
ML / AI Engineer (Aspiring)
Open to off-campus opportunities
