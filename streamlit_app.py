import os
import time
import json
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

# --- 1. CONFIG & PERSISTANCE ---
load_dotenv()
ARCHIVE_FILE = "archives_oracle.json"

# --- SESSION STATE INIT ---
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []

# ✅ PARAMÈTRES UI (CRITIQUE)
if "k_val" not in st.session_state:
    st.session_state.k_val = 12
if "expert_overlay" not in st.session_state:
    st.session_state.expert_overlay = True
if "show_scores" not in st.session_state:
    st.session_state.show_scores = False


def save_to_archive(history):
    archive_data = []
    if os.path.exists(ARCHIVE_FILE):
        with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
            archive_data = json.load(f)

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "full_chat": history
    }
    archive_data.append(entry)

    with open(ARCHIVE_FILE, "w", encoding="utf-8") as f:
        json.dump(archive_data, f, indent=4, ensure_ascii=False)


st.set_page_config(page_title="THE CLINICAL ORACLE", page_icon="🧬", layout="wide")

# --- 2. STYLE ---
st.markdown("""<style> ... </style>""", unsafe_allow_html=True)

# --- 3. ENGINE LOADING ---
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

@st.cache_resource(show_spinner=False)
def load_oracle():
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = Chroma(persist_directory="chroma_db", embedding_function=emb)
    llm = ChatMistralAI(model="mistral-small-latest", temperature=0)
    return vs, llm

vectorstore, llm = load_oracle()

# --- 4. BOOT SEQUENCE ---
if not st.session_state.initialized:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("<br><br>", unsafe_allow_html=True)
        _, col_img, _ = st.columns([1, 2, 1])
        with col_img:
            st.image("logo.png", width=400)
            st.markdown("<div class='oracle-title'>THE CLINICAL ORACLE</div>", unsafe_allow_html=True)
            bar = st.progress(0)
            for i in range(101):
                time.sleep(0.006)
                bar.progress(i)
    st.session_state.initialized = True
    placeholder.empty()

# --- 5. SIDEBAR ---
with st.sidebar:
    st.image("logo.png", use_container_width=True)
    st.markdown("<h2 style='color:#0047AB; font-family:Orbitron; text-align:center;'>COMMAND CENTER</h2>", unsafe_allow_html=True)

    if st.button("🗑️ CLEAR CONVERSATION"):
        st.session_state.chat_history = []
        st.session_state.last_docs = []
        st.rerun()

    tabs = st.tabs(["SETTINGS", "ARCHIVES"])

    with tabs[0]:
        st.session_state.k_val = st.slider(
            "Scan Depth (Chunks)", 4, 30, st.session_state.k_val
        )
        st.session_state.expert_overlay = st.toggle(
            "Expert Data Overlay", value=st.session_state.expert_overlay
        )
        st.session_state.show_scores = st.toggle(
            "Show Similarity Scores", value=st.session_state.show_scores
        )

    with tabs[1]:
        if os.path.exists(ARCHIVE_FILE):
            with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
                history_files = json.load(f)
                for item in reversed(history_files[-5:]):
                    if st.button(f"📄 {item['timestamp']}", key=item['timestamp']):
                        st.session_state.chat_history = item['full_chat']
                        st.rerun()

    st.markdown("---")
    st.markdown("<div style='text-align:center; color:#0047AB; font-family:Orbitron; font-size:0.7rem;'>MEDICAL AGENT v3.0 ELITE</div>", unsafe_allow_html=True)

# --- 6. MAIN ---
st.markdown("<div class='oracle-title'>THE CLINICAL ORACLE</div>", unsafe_allow_html=True)
st.markdown("<div class='nih-subtitle'>NIH CLINICAL INTELLIGENCE SYSTEM</div>", unsafe_allow_html=True)

for entry in st.session_state.chat_history:
    st.markdown(f"**>> QUERY:** {entry['query']}")
    st.markdown(f"<div class='chat-entry'>{entry['response']}</div>", unsafe_allow_html=True)

with st.form(key='chat_form', clear_on_submit=True):
    query = st.text_input(">> INITIALIZE ORACLE QUERY :")
    submit_button = st.form_submit_button(label='SEND TO CORE')

if submit_button and query:
    with st.spinner("⚡ ORACLE ANALYZING..."):
        search_results = vectorstore.similarity_search_with_relevance_scores(
            query, k=st.session_state.k_val
        )

        docs = [res[0] for res in search_results]
        context = "\n\n".join([d.page_content for d in docs])

        prompt = ChatPromptTemplate.from_template(
            "Analyze carefully: {context}\n\nQuestion: {question}"
        )
        response = (prompt | llm | StrOutputParser()).invoke(
            {"context": context, "question": query}
        )

        st.session_state.chat_history.append({"query": query, "response": response})
        st.session_state.last_docs = search_results
        st.rerun()

# --- MODE EXPERT ---
if st.session_state.chat_history:
    st.markdown("---")

    if st.session_state.expert_overlay and st.session_state.last_docs:
        st.markdown("### 📁 RAW DATA CHUNKS (LAST SCAN)")
        for i, (doc, score) in enumerate(st.session_state.last_docs):
            score_text = f" | SCORE: {score:.4f}" if st.session_state.show_scores else ""
            with st.expander(
                f"SOURCE DATA {i+1} | {Path(doc.metadata.get('source','')).name}{score_text}"
            ):
                st.write(doc.page_content)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🚀 ARCHIVE FULL SESSION"):
            save_to_archive(st.session_state.chat_history)
            st.success("SESSION PERSISTED.")
    with c2:
        full_text = "\n\n".join(
            [f"Q: {e['query']}\nA: {e['response']}" for e in st.session_state.chat_history]
        )
        st.download_button("📄 DOWNLOAD FULL REPORT", full_text, file_name="full_report.txt")
