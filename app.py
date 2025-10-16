# === Import Necessary Libraries ===
import streamlit as st
from utils import extract_text_from_pdf, split_text, create_vector_store, load_vector_store
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os

# === Load Environment Variables ===
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# === Configure Streamlit Page ===
st.set_page_config(page_title="AI PDF Assistant", layout="centered")
st.title("📘 ResearchMate – Your AI-Powered PDF Assistant")
st.caption("Upload, explore, and ask intelligent questions about your research or policy documents.")

# === API Key Check ===
if not openai_api_key:
    st.error("❌ Missing OpenAI API key. Please set it in `.env` or Streamlit `secrets.toml`.")
    st.stop()

# === File Upload Section ===
uploaded_pdf = st.file_uploader("📤 Upload a PDF file", type="pdf")

# === Allow Deserialization Toggle (for trusted environments) ===
allow_dangerous = st.checkbox(
    "Allow unsafe deserialization (⚠️ only for your own trusted PDFs)", value=False
)

# === Process PDF if Uploaded ===
if uploaded_pdf:
    try:
        with st.spinner("🔍 Extracting and processing PDF..."):
            text = extract_text_from_pdf(uploaded_pdf)
            if not text.strip():
                st.warning("⚠️ The PDF appears to contain no extractable text.")
                st.stop()

            chunks = split_text(text)
            st.success(f"✅ PDF processed into {len(chunks)} text chunks.")

            # Try loading vector store (if exists)
            vectordb = load_vector_store()
            if not vectordb:
                vectordb = create_vector_store(chunks)

            # Adaptive top-k selection
            if len(chunks) <= 10:
                k = 3
            elif len(chunks) <= 30:
                k = 5
            else:
                k = 10

            st.info(f"Auto-selected top {k} chunks based on document size.")

        # === Document Summary ===
        with st.spinner("🧠 Summarizing document..."):
            summary_prompt = (
                "Provide a concise and structured summary of the following document:\n\n"
                f"{text[:5000]}"
            )
            llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
            summary = llm.predict(summary_prompt)

        st.markdown("### 📌 Document Summary")
        st.write(summary)

        # === Question Input ===
        query = st.text_input("💬 Ask a question about the document:")

        if query:
            with st.spinner("🤖 Searching for answers..."):
                docs = vectordb.similarity_search(query, k=k)

                # Build readable context info
                context_info = "\n".join(
                    [f"- Page {doc.metadata.get('page', '❓')}: {doc.page_content[:200]}..." for doc in docs]
                )

                # Load QA chain
                llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
                chain = load_qa_chain(llm, chain_type="stuff")
                answer = chain.run(input_documents=docs, question=query)

            # === Display Answer ===
            st.markdown("### 🧩 Answer")
            st.write(answer)

            with st.expander("🔎 View context chunks used"):
                st.text(context_info)

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
