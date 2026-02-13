print('file is running')
from rag_pipeline import build_retriever,build_rag_chain
print('starting test')
pdf_path= 'dokumen.pub_practical-mlops-operationalizing-machine-learning-models-1nbsped-1098103017-9781098103019-n-4964661.pdf' 

retriever = build_retriever(pdf_path)
print('retriever built')
rag_chain = build_rag_chain(retriever)
print('rag chain built')

result = rag_chain.invoke({
    "question": "What is this document about?",
    "chat_history": []
})
print('result')
print(result)