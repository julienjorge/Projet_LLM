import os
import streamlit as st

# ===============================
# CONFIG
# ===============================

from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY manquante. Mets-la dans un fichier .env")

# ===============================
# LANGCHAIN IMPORTS
# ===============================
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="RAG MVP", page_icon="🤖")
st.title("🤖 RAG MVP – Local")

# ===============================
# LOAD VECTORSTORE
# ===============================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db_COMPLETE",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ===============================
# LLM
# ===============================
llm = ChatMistralAI(model="mistral-large-latest")

# ===============================
# PROMPT
# ===============================
prompt_template = """
You are an assistant for question-answering tasks.
Use the following context to answer the question.
If you don't know the answer, say you don't know.
Use three sentences maximum.

Question: {question}

Context:
{context}

Answer:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# ===============================
# FORMAT DOCS
# ===============================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ===============================
# RAG CHAIN
# ===============================
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ===============================
# UI INTERACTION
# ===============================
question = st.text_input("Pose ta question")

if question:
    with st.spinner("Recherche de la réponse..."):
        answer = rag_chain.invoke(question)
        st.markdown("### ✅ Réponse")
        st.write(answer)