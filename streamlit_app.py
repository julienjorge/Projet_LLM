import os
import time
import json
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

def invalidate_results():
    st.session_state.last_docs = []

# --- 1. CONFIG & PERSISTANCE ---
load_dotenv()
ARCHIVE_FILE = "archives_oracle.json"

# Initialisation rigoureuse du session_state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []
# Ajout des clés pour les réglages persistants
if "k_val" not in st.session_state:
    st.session_state.k_val = 12

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

# --- 2. STYLE ULTRA-COBALT ÉLECTRIQUE ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=JetBrains+Mono:wght@400;700&display=swap');
    
    .stApp { background-color: #000000 !important; color: #FFFFFF !important; font-family: 'JetBrains Mono', monospace; }
    
    [data-testid="stSidebar"] { 
        background-color: #000000 !important; 
        border-right: 3px solid #0047AB !important; 
        box-shadow: 5px 0 25px rgba(0, 71, 171, 0.6);
    }
    
    .oracle-title { 
        font-family: 'Orbitron', sans-serif; 
        color: #0047AB; 
        text-shadow: 0 0 15px #0047AB, 0 0 30px #0000FF; 
        text-align: center; 
        font-size: 3.5rem; 
        font-weight: 900; 
        letter-spacing: 8px; 
        padding: 20px;
    }

    .nih-subtitle {
        color: #0047AB; text-align: center; font-family: 'Orbitron';
        letter-spacing: 4px; font-size: 0.9rem; margin-top: -20px; margin-bottom: 30px;
    }

    div[data-baseweb="input"] {
        border: 2px solid #0047AB !important; background-color: #000000 !important; border-radius: 5px !important;
    }
    
    .chat-entry {
        border-left: 2px solid #0047AB; padding-left: 15px; margin-bottom: 25px;
        background: rgba(0, 71, 171, 0.05);
    }

    .stMarkdown p, .stMarkdown li, .stMarkdown h3 { color: #FFFFFF !important; }

    .stProgress > div > div > div > div { background-color: #0047AB !important; box-shadow: 0 0 15px #0000FF; }
    
    .stButton>button { 
        background: #000000 !important; color: #0047AB !important; 
        border: 1px solid #0047AB !important; font-family: 'Orbitron', sans-serif; font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover { 
        border: 1px solid #FFFFFF !important; color: #FFFFFF !important; box-shadow: 0 0 15px #0047AB;
    }

    .stExpander { border: 1px solid #0047AB !important; background: rgba(0, 71, 171, 0.05) !important; }
</style>
""", unsafe_allow_html=True)

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

# --- 5. SIDEBAR (Command Center) ---
with st.sidebar:

    st.image("logo.png", use_container_width=True)
    st.markdown(
        "<h2 style='color:#0047AB; font-family:Orbitron; text-align:center;'>COMMAND CENTER</h2>",
        unsafe_allow_html=True
    )

    if st.button("🧹 CLEAR CONVERSATION"):
        st.session_state.chat_history = []
        st.session_state.last_docs = []
        st.rerun()

    tabs = st.tabs(["SETTINGS", "ARCHIVES"])

    with tabs[0]:
        st.slider(
            "Scan Depth (Chunks)",
            4, 30,
            st.session_state.k_val,
            key="k_val",
            on_change=invalidate_results
        )

        st.toggle(
            "Expert Data Overlay",
            key="expert_overlay",
            value=True,
            on_change=invalidate_results
        )

        st.toggle(
            "Show Similarity Scores",
            key="show_scores",
            value=False,
            on_change=invalidate_results
        )

        st.markdown(
            "<div style='text-align:center; color:#0047AB; font-family:Orbitron; font-size:0.7rem;'>"
            "MEDICAL AGENT v3.0 ELITE</div>",
            unsafe_allow_html=True
        )

    with tabs[1]:
        if os.path.exists(ARCHIVE_FILE):
            with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
                history_files = json.load(f)
                for item in reversed(history_files[-5:]):
                    if st.button(
                       f"🕘 {item['timestamp']}",
                       key=f"arch_{item['timestamp']}"
):
                      st.session_state.chat_history = item["full_chat"]
                      st.rerun()

    
st.markdown("---")
st.markdown("<div style='text-align:center; color:#0047AB; font-family:Orbitron; font-size:0.7rem;'>MEDICAL AGENT v3.0 ELITE</div>", unsafe_allow_html=True)

# --- 6. MAIN ---
st.markdown("<div class='oracle-title'>THE CLINICAL ORACLE</div>", unsafe_allow_html=True)
st.markdown("<div class='nih-subtitle'>NIH CLINICAL INTELLIGENCE SYSTEM</div>", unsafe_allow_html=True)

# Affichage de l'historique
for entry in st.session_state.chat_history:
    st.markdown(f"**>> QUERY:** {entry['query']}")
    st.markdown(f"<div class='chat-entry'>{entry['response']}</div>", unsafe_allow_html=True)

# Zone de saisie
with st.form(key='chat_form', clear_on_submit=True):
    query = st.text_input(">> INITIALIZE ORACLE QUERY :")
    submit_button = st.form_submit_button(label='SEND TO CORE')

if submit_button and query:
    with st.spinner("⚡ ORACLE ANALYZING..."):
        # On utilise k_val depuis le session_state
        search_results = vectorstore.similarity_search_with_relevance_scores(query, k=st.session_state.k_val)
        
        docs = [res[0] for res in search_results]
        context = "\n\n".join([d.page_content for d in docs])
        prompt = ChatPromptTemplate.from_template("Analyze carefully: {context}\n\nQuestion: {question}")
        response = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": query})
        
        st.session_state.chat_history.append({"query": query, "response": response})
        st.session_state.last_docs = search_results 
        st.rerun()

# Actions de fin et Mode Expert
if st.session_state.chat_history:
    st.markdown("---")
    
    if st.session_state.expert_overlay and st.session_state.last_docs:
        st.markdown("### 📁 RAW DATA CHUNKS (LAST SCAN)")
        for i, (doc, score) in enumerate(st.session_state.last_docs):
            score_text = f" | SCORE: {score:.4f}" if st.session_state.show_scores else ""
            source_path = doc.metadata.get('source','')
            with st.expander(f"SOURCE DATA {i+1} | {Path(source_path).name}{score_text}"):
                st.write(doc.page_content)
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🚀 ARCHIVE FULL SESSION"):
            save_to_archive(st.session_state.chat_history)
            st.success("SESSION PERSISTED.")
    with c2:
        full_text = "\n\n".join([f"Q: {e['query']}\nA: {e['response']}" for e in st.session_state.chat_history])
        st.download_button("📄 DOWNLOAD FULL REPORT", full_text, file_name="full_report.txt")