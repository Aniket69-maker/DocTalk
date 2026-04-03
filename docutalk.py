import os
import sys

# --- DATABASE FIX FOR STREAMLIT CLOUD ---
# Streamlit Cloud uses an older version of sqlite3; this keeps ChromaDB happy.
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass 

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA  # Changed from langchain_classic

# --- UI CONFIG ---
st.set_page_config(page_title="Docu Talk", page_icon="📄")
st.title("📄 Chat with your PDF")

# --- 1. API SETUP (SECURE) ---
# We use st.secrets so your key isn't public on GitHub
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please add your GOOGLE_API_KEY to Streamlit Secrets.")
    st.stop()

# --- SIDEBAR: UPLOAD ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    process_button = st.button("Process PDF")

# --- SESSION STATE (Memory) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

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
        # Note: Using 'text-embedding-004' for better stability
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        # Using a temporary directory for Chroma to avoid permission issues
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        # Setup QA Chain (Updated to 'gemini-1.5-flash' for reliability)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
        )
        st.success("Ready to chat!")

# --- STEP 3: CHAT INTERFACE ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me something about the PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "qa_chain" in st.session_state:
        with st.chat_message("assistant"):
            # Updated to .invoke() which is the modern LangChain standard
            result = st.session_state.qa_chain.invoke({"query": prompt})
            response = result["result"]
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please upload and process a PDF first!")
