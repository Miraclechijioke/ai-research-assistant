# 🧠 AI Research Assistant

An intelligent PDF-based research assistant that summarizes academic papers and answers questions about them — built with **Streamlit**, **LangChain**, **FAISS**, and **OpenAI's GPT**.

<p align="center">
  <img src="https://img.shields.io/badge/Built%20With-LangChain-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/LLM-OpenAI-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/UI-Streamlit-orange?style=flat-square" />
</p>

---

## 📌 Features

- ✅ Upload a research paper (PDF)
- 🧠 AI generates a **summary** instantly
- 🔍 Ask context-aware questions and get **accurate answers**
- 🧬 Uses vector search with **OpenAI Embeddings + FAISS**
- ⚙️ Modular code design: easy to extend or deploy

---

## 📂 Folder Structure

```yaml
ai-research-assistant/
│
├── app.py # Streamlit app interface
├── utils.py # PDF parsing, chunking, vector store setup
├── requirements.txt # Project dependencies
├── .env # API key (excluded via .gitignore)
├── sample_papers/ # Example PDFs (optional)
├── vector_store/ # Temporary FAISS vector index

```
---
## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Miraclechijioke/ai-research-assistant.git
cd ai-research-assistant

```
### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your OpenAI API key
Create a .env file in the root:

```ini
OPENAI_API_KEY=your_openai_key_here
```
### 4. Run the app
bash
Copy
Edit
streamlit run app.py
🛠 Built With
Streamlit

LangChain

FAISS

OpenAI API

PyMuPDF

👨‍💻 Author
Miracle Chijioke Alozie
Associate Data Scientist & Machine Learning Engineer
LinkedIn • GitHub • Portfolio

⭐️ Star This Repo
If you find this useful, feel free to give it a ⭐️ to support the project!
