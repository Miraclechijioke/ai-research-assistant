# --- Import required libraries ---
import fitz  # PyMuPDF: For extracting text from PDF
from langchain_openai import OpenAIEmbeddings  # ✅ NEW import for OpenAI embeddings
from langchain.vectorstores import FAISS  # Fast similarity search
from langchain.text_splitter import CharacterTextSplitter  # For splitting text into manageable chunks
from dotenv import load_dotenv  # Load API keys from .env
import os

# --- Load Environment Variables ---
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")  # Read the file from memory (Streamlit upload)
    text = ""
    for page in doc:
        text += page.get_text()  # Extract text from each page
    return text

def split_text(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return chunks

def create_vector_store(chunks):
    # ✅ Use Streamlit secrets
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    if not openai_key:
        raise ValueError("Missing OPENAI_API_KEY. Please set it in .env or Streamlit secrets.")
        
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)  # ✅ Explicitly pass API key
    vectordb = FAISS.from_texts(chunks, embeddings)
    return vectordb
