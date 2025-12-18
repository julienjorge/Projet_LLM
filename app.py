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

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# ===============================
# LLM
# ===============================
llm = ChatMistralAI(model="mistral-large-latest")

# ===============================
# PROMPT
# ===============================

prompt = ChatPromptTemplate.from_template("""
You are an expert assistant.

Use ONLY the information provided in the context to answer the question.
If the answer is not in the context, say clearly that you don't know.

Your answer must:
- be detailed
- be structured
- include explanations
- use bullet points when relevant

Question:
{question}

Context:
{context}

Answer:
""")

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
#question = st.text_input("Pose ta question")

#if question:
    #with st.spinner("Recherche de la réponse..."):
        #answer = rag_chain.invoke(question)
        #st.markdown("### ✅ Réponse")
        #st.write(answer)

question = st.text_input("Pose ta question")

if question:
    # 1️⃣ récupérer les documents
    docs = retriever.invoke(question)

    # 2️⃣ afficher les sources
    st.subheader("📄 Documents utilisés")
    for i, doc in enumerate(docs):
        st.markdown(f"**Chunk {i+1}**")
        st.markdown(f"- **Source** : {doc.metadata.get('source', 'inconnu')}")
        st.markdown(f"- **Contenu** : {doc.page_content[:500]}...")
        st.markdown("---")

    # 3️⃣ générer la réponse
    answer = rag_chain.invoke(question)

    st.subheader("🤖 Réponse du chatbot")
    st.write(answer)

#score de similiraté

docs_and_scores = vectorstore.similarity_search_with_score(question, k=5)

for doc, score in docs_and_scores:
    st.write(f"Score: {score}")
    st.write(doc.page_content[:300])