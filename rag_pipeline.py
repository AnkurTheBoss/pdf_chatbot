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
from langchain_core.prompts import PromptTemplate


load_dotenv()




def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_KEY")
    )


def get_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         """You are an intelligent assistant answering questions using ONLY the provided document context.

Guidelines:
- Base your answer strictly on the retrieved context.
- If the answer is not in the context, say: "I could not find this information in the document."
- Be clear, structured and concise.
- Provide unnecessary repitition.
- If the question is a follow up, use chat history to understand intent.
- Do not mention the context explicitly in your answer."""
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
        chunk_size=800,
        chunk_overlap=150
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
        search_type="mmr",
        search_kwargs={"k": 8,
                       "fetch_k":20}
    )



def build_rag_chain(retriever):

    llm = load_llm()
    prompt = get_prompt()

    def rag_chain(input_data):

        question = input_data["question"]
        chat_history = input_data["chat_history"]

        # Combine last user question for follow-up handling
        if chat_history:
            for msg in reversed(chat_history):
                if msg.type == "human":
                    question = msg.content + " " + question
                    break

        # Retrieve documents
        docs = retriever.invoke(question)

        formatted_context = "\n\n".join(
            doc.page_content for doc in docs
        )

        # Generate answer
        answer = (
            prompt
            | llm
            | StrOutputParser()
        ).invoke({
            "context": formatted_context,
            "question": input_data["question"],
            "chat_history": chat_history
        })

        return {
            "answer": answer,
            "docs": docs
        }

    return rag_chain

         
        
    
 
