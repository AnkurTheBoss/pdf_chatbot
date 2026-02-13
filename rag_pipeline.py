import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()




def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_KEY")
    )


def get_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         """You are a helpful assistant that answers questions using ONLY the provided document context.

Rules:
- Use only the information from the context.
- If the answer is not in the context, say: "I could not find this information in the document."
- Do not guess.
- Think step by step before answering.
- Provide a detailed explanation in 3-5 paragraphs."""
         ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human",
         """Context:
{context}

Question:
{question}"""
         )
    ])



def build_retriever(uploaded_pdf_path):
    
    loader = PyPDFLoader(uploaded_pdf_path)
    docs = loader.load()

    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250
    )
    chunks = splitter.split_documents(docs)

   
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

   
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}
    )



def build_rag_chain(retriever):

    llm = load_llm()
    prompt = get_prompt()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": lambda x: format_docs(
                retriever.invoke(x["question"])
            ),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
