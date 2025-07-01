import fitz  # PyMuPDF: For extracting text from PDF
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import streamlit as st
import os

# --- Load .env file ---
load_dotenv()

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_text(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return chunks

def create_vector_store(chunks):
    # Try to get key from Streamlit secrets first, fallback to environment variable
    openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")


    if not openai_api_key:
        raise ValueError("❌ Missing OPENAI_API_KEY. Set it in .env or .streamlit/secrets.toml.")

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = FAISS.from_texts(chunks, embeddings)
    return vectordb
