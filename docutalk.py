__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA

# --- UI CONFIG ---
st.set_page_config(page_title="Docu Talk", page_icon="📄")
st.title("📄 Chat with your PDF")

# 1. API Setup (Best practice: use an environment variable or secret)
os.environ["GOOGLE_API_KEY"] = "AIzaSyAxw4sP09RlnENawkGDPJDQMwCNmtVAGGI"

# --- SIDEBAR: UPLOAD ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    process_button = st.button("Process PDF")

# --- SESSION STATE (Memory) ---
if "messages" not in st.session_state:
    st.session_state.messages = [] # Stores chat history

# --- STEP 2: PROCESSING THE PDF ---
if uploaded_file and process_button:
    with st.spinner("Analyzing document..."):
        # Save temp file
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Load & Split
        loader = PyPDFLoader("temp.pdf")
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        
        # Create Vector Store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vectorstore = Chroma.from_documents(chunks, embeddings)
        
        # Setup QA Chain
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
        )
        st.success("Ready to chat!")

# --- STEP 3: CHAT INTERFACE ---
# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask me something about the PDF..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI Response
    if "qa_chain" in st.session_state:
        with st.chat_message("assistant"):
            response = st.session_state.qa_chain.invoke(prompt)["result"]
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("Please upload and process a PDF first!")
