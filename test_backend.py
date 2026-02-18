from langchain_core.messages import HumanMessage, AIMessage
print('file is running')
from rag_pipeline import build_retriever, build_rag_chain
print('starting test')

pdf_path = 'dokumen.pub_practical-mlops-operationalizing-machine-learning-models-1nbsped-1098103017-9781098103019-n-4964661.pdf'

retriever = build_retriever(pdf_path)
print('retriever built')

rag_chain = build_rag_chain(retriever)
print('rag chain built')

chat_history = []

while True:
    question = input("\nAsk a question (or type 'exit'): ")

    if question.lower() == "exit":
        break

    result = rag_chain({
        "question": question,
        "chat_history": chat_history
    })

    answer = result["answer"]
    docs = result["docs"]

    print("\nAnswer:\n", answer)
    print("\nSources:")
    for i, doc in enumerate(docs[:3], 1):
        page = doc.metadata.get("page", "Unknown")
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"{i}. Page {page} — {preview}...")

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))