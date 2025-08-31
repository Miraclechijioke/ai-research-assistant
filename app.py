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
st.title("📄 AI Top Research Assistant")

# === Upload PDF ===
uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

# === Process PDF if Uploaded ===
if uploaded_pdf:
    try:
        with st.spinner("Processing PDF..."):
            text = extract_text_from_pdf(uploaded_pdf)
            chunks = split_text(text)

            # Try loading vector store first (if already exists)
            vectordb = load_vector_store()
            if not vectordb:
                vectordb = create_vector_store(chunks)

            st.success(f"✅ PDF processed into {len(chunks)} chunks.")

            # Adaptive top-k selection
            if len(chunks) <= 10:
                k = 3
            elif len(chunks) <= 30:
                k = 5
            else:
                k = 10

        # --- Auto Summary ---
        with st.spinner("Summarizing the document..."):
            summary_prompt = f"Summarize this document:\n\n{text[:3000]}"  # Limit size
            llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
            summary = llm.predict(summary_prompt)

        st.markdown("### 📌 Document Summary:")
        st.write(summary)

        st.info(f"Auto-selected top {k} chunks based on document size.")

        # === Accept Question from User ===
        query = st.text_input("Ask a question about the document:")

        if query:
            with st.spinner("Searching for answers..."):
                docs = vectordb.similarity_search(query, k=k)

                # Show sources/pages in results
                context_info = "\n".join(
                    [f"- Page {doc.metadata.get('page', '❓')}: {doc.page_content[:200]}..." for doc in docs]
                )

                llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
                chain = load_qa_chain(llm, chain_type="stuff")
                answer = chain.run(input_documents=docs, question=query)

            # === Display Answer ===
            st.markdown("### 🧠 Answer:")
            st.write(answer)

            with st.expander("🔎 Context Chunks Used"):
                st.write(context_info)

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
