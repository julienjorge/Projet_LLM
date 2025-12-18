import os
import streamlit as st
from dotenv import load_dotenv

# ===============================
# ENV
# ===============================
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY manquante dans le fichier .env")

# ===============================
# LANGCHAIN
# ===============================
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="NIH Clinical AI",
    page_icon="🧬",
    layout="wide"
)

# ===============================
# NEON UI (NOIR + BLEU ÉLECTRIQUE)
# ===============================
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #070B10;
    color: #EAFBFF;
}

h1, h2, h3 {
    color: #00BFFF;
    text-shadow:
        0 0 10px rgba(0,191,255,0.9),
        0 0 25px rgba(0,191,255,0.7),
        0 0 45px rgba(0,191,255,0.4);
}

input {
    background-color: #050B12 !important;
    color: #00BFFF !important;
    border: 2px solid #00BFFF !important;
    border-radius: 10px !important;
    box-shadow: 0 0 10px rgba(0,191,255,0.6);
}

.stButton>button {
    background: linear-gradient(90deg, #00BFFF, #007BFF);
    color: black;
    font-weight: 700;
    border-radius: 10px;
    box-shadow: 0 0 12px rgba(0,191,255,0.8);
}

section[data-testid="stSidebar"] {
    background-color: #050B12;
    border-right: 1px solid #00BFFF;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
col_logo, col_title = st.columns([1, 6])

with col_logo:
    st.image("ChatGPT Image 18 déc. 2025.png", width=120)

with col_title:
    st.markdown("""
    <h1>NIH Clinical AI</h1>
    <p style="color:#7DEBFF;font-size:1.1rem;">
        Advanced AI for NIH Clinical Trial Intelligence
    </p>
    """, unsafe_allow_html=True)

st.markdown("---")

# ===============================
# SIDEBAR OPTIONS
# ===============================
st.sidebar.markdown("## ⚙️ Options")

expert_mode = st.sidebar.toggle("Mode expert (voir sources & chunks)", value=True)
show_scores = st.sidebar.toggle("Afficher scores (debug)", value=False)
k_chunks = st.sidebar.slider("Nombre de chunks (k)", 2, 10, 6)

# ===============================
# VECTORSTORE
# ===============================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db_COMPLETE",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": k_chunks})

# ===============================
# LLM
# ===============================
llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.2
)

# ===============================
# PROMPTS
# ===============================
main_prompt = ChatPromptTemplate.from_template("""
You are an expert clinical research assistant.

Use ONLY the provided context.
If the answer is not explicitly stated, say: "I don't know."

Answer must be:
- detailed
- structured
- clinically precise
- supported by the context

Question:
{question}

Context:
{context}

Answer:
""")

summary_prompt = ChatPromptTemplate.from_template("""
Summarize the key information from the following sources
in a concise bullet-point format.

Sources:
{context}

Summary:
""")

# ===============================
# HELPERS
# ===============================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ===============================
# CHAINS
# ===============================
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | main_prompt
    | llm
    | StrOutputParser()
)

summary_chain = (
    {"context": RunnablePassthrough()}
    | summary_prompt
    | llm
    | StrOutputParser()
)

# ===============================
# UI – QUESTION
# ===============================
question = st.text_input(
    "💬 Ask a clinical question",
    placeholder="e.g. What was the primary objective of the Phase I trial?"
)

# ===============================
# EXECUTION
# ===============================
if question:
    with st.spinner("🔍 Analyzing NIH clinical documents..."):
        docs = retriever.invoke(question)
        answer = rag_chain.invoke(question)
        sources_summary = summary_chain.invoke(format_docs(docs))

    st.markdown("## 🤖 Response")
    st.write(answer)

    st.markdown("## 🧾 Source Summary")
    st.write(sources_summary)

    if expert_mode:
        st.markdown("## 📄 Retrieved Chunks")
        for i, doc in enumerate(docs):
            with st.expander(f"Chunk {i+1}"):
                st.markdown(f"**Source:** {doc.metadata.get('source', 'unknown')}")
                if "page" in doc.metadata:
                    st.markdown(f"**Page:** {doc.metadata['page']}")
                st.write(doc.page_content)

    if show_scores:
        st.markdown("## 📊 Similarity Scores")
        scored_docs = vectorstore.similarity_search_with_score(question, k=k_chunks)
        for doc, score in scored_docs:
            st.write(f"Score: {score:.4f}")
            st.write(doc.page_content[:300])
            st.markdown("---")
