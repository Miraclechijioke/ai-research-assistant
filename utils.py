import fitz  # PyMuPDF: For extracting text content from PDF documents
from langchain_community.embeddings import OpenAIEmbeddings  # For generating vector embeddings using OpenAI
from langchain_community.vectorstores import FAISS  # For storing and searching vectors using FAISS
from langchain.text_splitter import CharacterTextSplitter  # For splitting long text into smaller, manageable chunks
from dotenv import load_dotenv  # Loads environment variables from a .env file
import streamlit as st  # Streamlit: for web interface and handling secrets
import os  # For accessing environment variables as a fallback

# --- Load environment variables from .env file (if present) ---
load_dotenv()

# --- Function to extract text from an uploaded PDF file ---
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

# --- Function to split long text into smaller chunks for embedding ---
def split_text(text):
    try:
        splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"❌ Error splitting text: {e}")
        return []

# --- Function to create a FAISS vector store from text chunks ---
def create_vector_store(chunks, store_name="vector_store"):
    try:
        # API Key from Streamlit secrets or environment variable
        openai_api_key = (
            st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
        )

        if not openai_api_key:
            raise ValueError("❌ Missing OPENAI_API_KEY. Set it in .env or .streamlit/secrets.toml.")

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Create FAISS vector store
        vectordb = FAISS.from_texts(chunks, embeddings)

        # Ensure the store folder exists
        os.makedirs(store_name, exist_ok=True)

        # Save the vector store locally (similar to HuggingFace style persistence)
        faiss_path = os.path.join(store_name, "faiss_index")
        vectordb.save_local(faiss_path)

        st.success(f"✅ Vector store saved at: {faiss_path}")
        return vectordb
    except Exception as e:
        st.error(f"❌ Error creating vector store: {e}")
        return None

# --- Function to load an existing FAISS vector store ---
def load_vector_store(store_name="vector_store"):
    try:
        openai_api_key = (
            st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
        )

        if not openai_api_key:
            raise ValueError("❌ Missing OPENAI_API_KEY. Set it in .env or .streamlit/secrets.toml.")

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        faiss_path = os.path.join(store_name, "faiss_index")

        vectordb = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

        st.success(f"✅ Vector store loaded from: {faiss_path}")
        return vectordb
    except Exception as e:
        st.error(f"❌ Error loading vector store: {e}")
        return None
