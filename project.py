import streamlit as st
import subprocess
import pandas as pd
import os
from PyPDF2 import PdfReader

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Gemma 3:1B Local LLM", page_icon="🤖", layout="wide")
st.title("💬 Gemma 3:1B — Local LLM Chatbot (Private & Offline)")

st.caption("Running locally with **Ollama** — No data leaves your machine 🔒")

# ---- INIT SESSION ----
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_text" not in st.session_state:
    st.session_state.uploaded_text = ""

# ---- SIDEBAR ----
st.sidebar.header("📁 Upload a File for Context")
uploaded_file = st.sidebar.file_uploader("Upload TXT, PDF, or CSV file", type=["txt", "pdf", "csv"])

if uploaded_file:
    text_data = ""
    file_ext = uploaded_file.name.split(".")[-1].lower()

    if file_ext == "txt":
        text_data = uploaded_file.read().decode("utf-8")
    elif file_ext == "pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text_data += page.extract_text() + "\n"
    elif file_ext == "csv":
        df = pd.read_csv(uploaded_file)
        text_data = df.to_string()

    st.session_state.uploaded_text = text_data[:10000]  # limit context length
    st.sidebar.success("✅ File uploaded successfully!")

    with st.expander("📄 Preview File Content"):
        st.text_area("Extracted Text", st.session_state.uploaded_text, height=250)

# ---- DISPLAY CHAT ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- USER INPUT ----
prompt = st.chat_input("Ask something...")

if prompt:
    # Combine prompt with uploaded context (if any)
    context = ""
    if st.session_state.uploaded_text:
        context = f"The following is context from a local file:\n{st.session_state.uploaded_text}\n\n"
    full_prompt = context + f"User: {prompt}\nAssistant:"

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Call local Gemma model through Ollama CLI
                result = subprocess.run(
                    ["ollama", "run", "gemma3:1b", full_prompt],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                response = result.stdout.strip()
            except Exception as e:
                response = f"⚠️ Error running model: {e}"

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
