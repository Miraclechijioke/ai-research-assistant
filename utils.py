import fitz  # PyMuPDF: Extract text content from PDF documents
from langchain_community.embeddings import OpenAIEmbeddings  # For generating vector embeddings
from langchain_community.vectorstores import FAISS  # For storing and searching vectors
from langchain.text_splitter import CharacterTextSplitter  # For splitting text into chunks
from dotenv import load_dotenv  # Loads environment variables from .env
import streamlit as st  # For Streamlit interface and secrets management
import os  # For file and environment operations
import logging  # For logging errors and actions

# --- Load environment variables ---
load_dotenv()

# --- Configure logging ---
logging.basicConfig(
    filename="app_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Extract text from a PDF file (page by page) ---
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        texts = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                texts.append({"page": page_num, "content": text})
        logging.info(f"Extracted text from {len(texts)} pages.")
        return texts
    except Exception as e:
        st.error(f"❌ Error extracting text from PDF: {e}")
        logging.error(f"Error extracting text from PDF: {e}")
        return []


# --- Split text into smaller chunks with page tracking ---
def split_text(texts):
    try:
        splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        chunks = []
        for item in texts:
            for chunk in splitter.split_text(item["content"]):
                chunks.append({"page": item["page"], "content": chunk})
        logging.info(f"Split document into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        st.error(f"❌ Error splitting text: {e}")
        logging.error(f"Error splitting text: {e}")
        return []


# --- Create FAISS vector store ---
def create_vector_store(chunks, store_name="vector_store", persist=False):
    """
    persist=True -> Saves the FAISS index locally (for trusted internal use only)
    persist=False -> Keeps it in memory (recommended for public Streamlit apps)
    """
    try:
        openai_api_key = (
            st.secrets["OPENAI_API_KEY"]
            if "OPENAI_API_KEY" in st.secrets
            else os.getenv("OPENAI_API_KEY")
        )
        if not openai_api_key:
            raise ValueError("❌ Missing OPENAI_API_KEY. Set it in .env or .streamlit/secrets.toml.")

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        texts = [chunk["content"] for chunk in chunks]
        metadatas = [{"page": chunk["page"]} for chunk in chunks]

        vectordb = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

        if persist:
            os.makedirs(store_name, exist_ok=True)
            faiss_path = os.path.join(store_name, "faiss_index")
            vectordb.save_local(faiss_path)
            logging.info(f"Vector store saved at: {faiss_path}")
            st.success(f"✅ Vector store saved at: {faiss_path}")
        else:
            logging.info("Vector store created in memory (not persisted).")

        return vectordb
    except Exception as e:
        st.error(f"❌ Error creating vector store: {e}")
        logging.error(f"Error creating vector store: {e}")
        return None


# --- Load an existing FAISS vector store (trusted use only) ---
def load_vector_store(store_name="vector_store"):
    try:
        openai_api_key = (
            st.secrets["OPENAI_API_KEY"]
            if "OPENAI_API_KEY" in st.secrets
            else os.getenv("OPENAI_API_KEY")
        )
        if not openai_api_key:
            raise ValueError("❌ Missing OPENAI_API_KEY. Set it in .env or .streamlit/secrets.toml.")

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        faiss_path = os.path.join(store_name, "faiss_index")

        vectordb = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=False)
        logging.info(f"Loaded FAISS vector store from: {faiss_path}")
        st.success(f"✅ Vector store loaded from: {faiss_path}")
        return vectordb
    except Exception as e:
        st.error(f"❌ Error loading vector store: {e}")
        logging.error(f"Error loading vector store: {e}")
        return None
