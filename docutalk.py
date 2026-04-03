import os
import sys

# --- 1. SQLITE3 FIX FOR STREAMLIT CLOUD ---
# Streamlit Cloud's default sqlite3 is often too old for ChromaDB.
# This forces the app to use 'pysqlite3' instead.
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # If running locally and pysqlite3 isn't installed, it will fallback to standard sqlite3
    pass

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# --- 2. UI CONFIGURATION ---
st.set_page_config(page_title="Docu Talk", page_icon="📄", layout="centered")
st.title("📄 Chat with your PDF")
st.markdown("---")

# --- 3. SECURE API SETUP ---
# Pulls the key from Streamlit's Secret Manager (NOT hardcoded)
if "GOOGLE_API_KEY" in st.secrets:
    user_api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = user_api_key # Sets it for the system
else:
    st.error("Missing GOOGLE_API_KEY in Secrets!")
    st.stop()

# --- 4. SIDEBAR: FILE UPLOADER ---
with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    process_button = st.button("Process & Train AI")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- 5. INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 6. CORE RAG LOGIC: PDF PROCESSING ---
if uploaded_file and process_button:
    with st.spinner("Analyzing your document... this usually takes 10-20 seconds."):
        try:
            # Save the uploaded file temporarily
            with open("temp_doc.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Load and Split the PDF
            loader = PyPDFLoader("temp_doc.pdf")
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_documents(docs)
            
            # Use the most stable 2026 embedding model
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=user_api_key)
            
            # Create a vector store in memory (ephemeral for each session)
            vectorstore = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings
            )
            
            # Initialize the LLM (Gemini 1.5 Flash is best for speed/cost)
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, google_api_key=user_api_key)
            
            # Create the Retrieval Chain
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff", 
                retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
            )
            
            st.success("✅ Document processed! You can now ask questions below.")
            
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

# --- 7. THE CHAT INTERFACE ---
# Display historical messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input("What would you like to know about the document?"):
    # Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI Response
    if "qa_chain" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Modern LangChain .invoke() syntax
                    result = st.session_state.qa_chain.invoke({"query": prompt})
                    response = result["result"]
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {e}")
    else:
        st.warning("Please upload and 'Process' a PDF in the sidebar first!")
