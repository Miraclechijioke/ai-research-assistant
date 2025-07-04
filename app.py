# === Import Necessary Libraries ===

import streamlit as st  # Web app UI framework
from utils import extract_text_from_pdf, split_text, create_vector_store  # Custom functions from utils.py
from langchain.llms import OpenAI  # LLM interface to interact with OpenAI's GPT
from langchain.chains.question_answering import load_qa_chain  # Pre-built QA pipeline
from dotenv import load_dotenv  # For loading environment variables from .env file
import os  # OS-level operations (e.g., accessing env vars)

# === Load Environment Variables ===

load_dotenv()  # Load API keys and config from .env file
openai_api_key = os.getenv("OPENAI_API_KEY")  # Access the OpenAI API key securely

# === Configure Streamlit Page ===

st.set_page_config(page_title="AI PDF Assistant", layout="centered")  # Set tab title and layout
st.title("📄 AI Top Research Assistant")  # Main page title

# === Upload PDF ===

uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")  # Upload widget (PDF only)

# === Process PDF if Uploaded ===

if uploaded_pdf:
    with st.spinner("Processing PDF..."):  # Show loading animation
        text = extract_text_from_pdf(uploaded_pdf)  # Extract raw text from the PDF
        chunks = split_text(text)  # Split the text into manageable chunks
        vectordb = create_vector_store(chunks)  # Create FAISS vector store for similarity search
        st.success(f"✅ PDF processed into {len(chunks)} chunks.")  # Notify task success

        # Determine k based on chunk count
        if len(chunks) <= 10:
            k = 3
        elif len(chunks) <= 30:
            k = 5
        else:
            k = 10

        # --- Auto Summary ---
    with st.spinner("Summarizing the document..."):
        summary_prompt = f"Summarize this document:\n\n{text[:3000]}"  # first 3000 chars
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        summary = llm.predict(summary_prompt)

    st.markdown("### 📌 Document Summary:")
    st.write(summary)

    st.info(f"Auto-selected top {k} chunks based on document size.")

    # === Accept Question from User ===
    query = st.text_input("Ask a question about the document:")  # Text input for user query

    if query:
        with st.spinner("Searching for answers..."):  # Show loading during processing
            docs = vectordb.similarity_search(query, k=k)  # Retrieve top "k" relevant chunks based on user selection for "k"
            llm = OpenAI(temperature=0)  # Initialize GPT model with deterministic output
            chain = load_qa_chain(llm, chain_type="stuff")  # Load a basic QA chain (stuff method)
            answer = chain.run(input_documents=docs, question=query)  # Generate answer

        # === Display Answer ===
        st.markdown("### 🧠 Answer:")
        st.write(answer)  # Show the answer on the page

    
