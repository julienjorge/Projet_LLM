from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
#from langchain import hub
from dotenv import load_dotenv
import os
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
import weaviate
#from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from pathlib import Path
#from langchain_community.document_loaders import DirectoryLoader, TextLoader
from weaviate import connect_to_local






load_dotenv()

st.set_page_config(page_title="RAG MVP", page_icon="🧩 ")
st.title(" 🤖 RAG MVP – Local")


client = connect_to_local()



embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)




#weaviate.from_documents(splitted_docs, 
#    embeddings, 
#    client=client,
#    index_name="projet_LLM"
#)

vectorstore = WeaviateVectorStore(
    client=client,
    index_name="projet_LLM",
    text_key="text",
    embedding=embeddings
)




retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
prompt = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:
"""

prompt = ChatPromptTemplate(
    ("system", prompt)
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


llm = ChatMistralAI(model="mistral-small-latest")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()} 
    | prompt 
    | llm 
    | StrOutputParser()
)


question = st.text_input("Pose ta question")

if question:
    with st.spinner("Recherche de la réponse..."):
        answer = rag_chain.invoke(question)
        st.markdown("### 🧠 Réponse")
        st.write(answer)