import streamlit as st
import subprocess
import os
import pandas as pd
from PyPDF2 import PdfReader
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="Gemma 3:1B Local LLM", page_icon="🤖", layout="wide")
st.markdown("""
    <style>
        body {background-color:#0d1117;color:#f5f5f5;}
        .stChatMessage {border-radius:12px;padding:10px 15px;margin-bottom:10px;max-width:85%;}
        .user {background-color:#238636;color:white;margin-left:auto;}
        .assistant {background-color:#30363d;color:#f5f5f5;margin-right:auto;}
        h1,h2,h3 {color:#58a6ff;}
    </style>
""", unsafe_allow_html=True)

st.title("💬 Gemma 3 1B — Private Local Chatbot")
st.caption("Runs entirely offline — powered by Ollama + Streamlit + FAISS 🔒")
st.markdown("---")

# ---------------- STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ---------------- SIDEBAR ----------------
st.sidebar.header("📁 Upload Files for Context")
uploaded_files = st.sidebar.file_uploader(
    "Upload TXT, PDF or CSV files", type=["txt", "pdf", "csv"], accept_multiple_files=True
)

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# ---------------- FILE TEXT EXTRACTION ----------------
def extract_text(file):
    text = ""
    ext = file.name.split(".")[-1].lower()
    if ext == "txt":
        text = file.read().decode("utf-8")
    elif ext == "pdf":
        reader = PdfReader(file)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    elif ext == "csv":
        df = pd.read_csv(file)
        text = df.to_string()
    return text

# ---------------- INDEX FILES INTO FAISS ----------------
if uploaded_files:
    st.sidebar.info("📚 Building local index...")
    docs = []
    for file in uploaded_files:
        text = extract_text(file)
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_text(text)
        docs.extend(chunks)
    embeddings = OllamaEmbeddings(model="gemma3:1b")
    st.session_state.vector_store = FAISS.from_texts(docs, embedding=embeddings)
    st.sidebar.success("✅ Files processed and indexed locally!")

# ---------------- DISPLAY CHAT ----------------
for msg in st.session_state.messages:
    cls = "user" if msg["role"] == "user" else "assistant"
    st.markdown(
        f'<div class="stChatMessage {cls}"><b>{msg["role"].capitalize()}:</b><br>{msg["content"]}</div>',
        unsafe_allow_html=True,
    )

# ---------------- CHAT FUNCTION ----------------
def query_local_llm(prompt, context=""):
    full_prompt = f"Use the following context to answer clearly:\n{context}\n\nUser: {prompt}\nAssistant:"
    result = subprocess.run(
        ["ollama", "run", "gemma3:1b", full_prompt],
        capture_output=True,
        text=True,
        timeout=120
    )
    return result.stdout.strip()

# ---------------- USER INPUT ----------------
prompt = st.chat_input("Ask something...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        context = ""
        if st.session_state.vector_store:
            docs = st.session_state.vector_store.similarity_search(prompt, k=3)
            context = "\n".join([d.page_content for d in docs])
        response = query_local_llm(prompt, context)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.experimental_rerun()

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("💡 All computation and data storage occur locally. Nothing is uploaded to the cloud.")
