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
    # Open the PDF file in-memory using PyMuPDF
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    
    # Loop through each page and extract text content
    for page in doc:
        text += page.get_text()
    
    return text  # Return the full extracted text from all pages

# --- Function to split long text into smaller chunks for embedding ---
def split_text(text):
    # Create a splitter that splits text at newlines into chunks of 1000 characters
    # with an overlap of 200 characters for better context retention
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    
    # Split the full text into chunks
    chunks = splitter.split_text(text)
    
    return chunks  # Return the list of text chunks

# --- Function to create a FAISS vector store from text chunks ---
def create_vector_store(chunks):
    # Attempt to get the OpenAI API key from Streamlit secrets first
    # If not found, fall back to checking system environment variables
    openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

    # Raise an error if no API key is found
    if not openai_api_key:
        raise ValueError("❌ Missing OPENAI_API_KEY. Set it in .env or .streamlit/secrets.toml.")

    # Generate embeddings for the text chunks using OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Create a FAISS vector store from the embedded text chunks
    vectordb = FAISS.from_texts(chunks, embeddings)

    return vectordb  # Return the FAISS vector database for later querying
