import streamlit as st
import subprocess
import os
import pandas as pd
from PyPDF2 import PdfReader
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions

# ---- PAGE SETUP ----
st.set_page_config(page_title="Gemma 3:1B Local LLM", page_icon="🤖", layout="wide")

# ---- CUSTOM STYLING ----
st.markdown("""
    <style>
        body {
            background-color: #0d1117;
            color: #f5f5f5;
        }
        .main {
            background-color: #0d1117;
        }
        .stChatInputContainer {
            background-color: #161b22 !important;
            border-radius: 10px;
            padding: 8px;
        }
        .stChatMessage {
            border-radius: 12px;
            padding: 10px 15px;
            margin-bottom: 10px;
            max-width: 85%;
        }
        .user {
            background-color: #238636;
            color: white;
            margin-left: auto;
        }
        .assistant {
            background-color: #30363d;
            color: #f5f5f5;
            margin-right: auto;
        }
        .sidebar .sidebar-content {
            background-color: #161b22 !important;
        }
        h1, h2, h3, h4 {
            color: #58a6ff;
        }
        .stSpinner > div {
            color: #58a6ff;
        }
        .uploadBox {
            border: 1px dashed #58a6ff;
            border-radius: 10px;
            padding: 20px;
            background-color: #161b22;
        }
        .privacyNote {
            font-size: 0.9rem;
            color: #9ba3af;
        }
    </style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.title("🤖 Gemma 3:1B — Private Local LLM Chatbot")
st.markdown("**Built with Ollama + ChromaDB + Streamlit 1.39**")
st.markdown('<p class="privacyNote">🔒 100% offline — no data leaves your machine</p>', unsafe_allow_html=True)
st.markdown("---")

# ---- INITIALIZE STATE ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- SIDEBAR ----
st.sidebar.header("📂 Upload Your Documents")
st.sidebar.markdown('<div class="uploadBox">Upload multiple TXT, PDF, or CSV files for context understanding.</div>', unsafe_allow_html=True)
uploaded_files = st.sidebar.file_uploader("Upload Files", type=["txt", "pdf", "csv"], accept_multiple_files=True)

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# ---- TEXT EXTRACTION ----
def extract_text(file):
    text = ""
    file_ext = file.name.split(".")[-1].lower()
    if file_ext == "txt":
        text = file.read().decode("utf-8")
    elif file_ext == "pdf":
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    elif file_ext == "csv":
        df = pd.read_csv(file)
        text = df.to_string()
    return text

# ---- CHROMADB ----
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="local_files",
    embedding_function=embedding_functions.DefaultEmbeddingFunction()
)

if uploaded_files:
    st.sidebar.info("Indexing uploaded files...")
    for file in uploaded_files:
        text = extract_text(file)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            doc_id = f"{file.name}_{i}"
            collection.add(documents=[chunk], ids=[doc_id])
    st.sidebar.success("✅ Files indexed successfully!")

# ---- DISPLAY CHAT ----
for msg in st.session_state.messages:
    role_class = "user" if msg["role"] == "user" else "assistant"
    st.markdown(
        f'<div class="stChatMessage {role_class}"><b>{msg["role"].capitalize()}:</b><br>{msg["content"]}</div>',
        unsafe_allow_html=True
    )

# ---- CHAT INPUT ----
prompt = st.chat_input("💬 Ask me something...")

def query_local_llm(query, context):
    """Run Gemma locally using Ollama"""
    full_prompt = f"Use the following context to answer:\n{context}\n\nUser: {query}\nAssistant:"
    result = subprocess.run(
        ["ollama", "run", "gemma3:1b", full_prompt],
        capture_output=True,
        text=True,
        timeout=120
    )
    return result.stdout.strip()

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        results = collection.query(query_texts=[prompt], n_results=3)
        context = "\n".join(results["documents"][0]) if results["documents"] else ""
        response = query_local_llm(prompt, context)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.experimental_rerun()

# ---- FOOTER ----
st.markdown("---")
st.markdown('<p class="privacyNote">💡 All computation and storage happen locally. Your documents are never sent to the cloud.</p>', unsafe_allow_html=True)
