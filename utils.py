import fitz  # PyMuPDF for PDF text extraction
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import streamlit as st
import os

# --- Load environment variables ---
load_dotenv()

# --- Extract text from uploaded PDF ---
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"❌ Error extracting text from PDF: {e}")
        return ""

# --- Split text into smaller chunks ---
def split_text(text):
    try:
        splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"❌ Error splitting text: {e}")
        return []

# --- Create FAISS vector store (in-memory only) ---
def create_vector_store(chunks):
    try:
        openai_api_key = (
            st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
        )

        if not openai_api_key:
            raise ValueError("❌ Missing OPENAI_API_KEY. Set it in .env or .streamlit/secrets.toml.")

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectordb = FAISS.from_texts(chunks, embeddings)
        return vectordb
    except Exception as e:
        st.error(f"❌ Error creating vector store: {e}")
        return None
