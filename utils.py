# --- Import required libraries ---

import fitz  # PyMuPDF: For extracting text from PDF
from langchain.embeddings import OpenAIEmbeddings # Turn text into vector embeddings
from langchain.vectorstores import FAISS # Fast similarity search
from langchain.text_splitter import CharacterTextSplitter #For splitting text into manageable chunks
from dotenv import load_dotenv # to load OpenAI API key from .env
import os

# --- Load Environment Variables (OpenAI Key) ---

load_dotenv()

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf") # Read the file from memory (Streamlit upload)
    text = ""
    for page in doc:
        text += page.get_text() # Extract text from each page
    return text

def split_text(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return chunks

def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings() # Uses OpenAI's embedding model by default
    vectordb = FAISS.from_texts(chunks, embeddings)
    return vectordb
