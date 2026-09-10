# ⚖️ CyberLaw Pakistan AI

A Retrieval-Augmented Generation (RAG) application
for Pakistani cyber-law questions.

The application automatically downloads the legal PDF
from Google Drive, extracts the text, creates embeddings,
stores the vectors in FAISS, retrieves relevant legal
sections, and uses Groq to generate an evidence-grounded
answer.

---

# Features

- Pakistani cyber-law RAG
- Google Drive PDF source
- Automatic PDF download
- Automatic embedding on first startup
- Local Sentence Transformer embeddings
- FAISS vector database
- Groq LLM
- Streamlit UI
- Chat interface
- Conversation history
- Source/page display
- Technical-level selection
- Response-size selection
- Language selection
- Answer-style selection
- Retrieval top-K control
- Rebuild embeddings button
- Legal disclaimer
- Hallucination-reduction prompt
- Cyber-safety restrictions

---

# Architecture

```text
Google Drive PDF
       |
       v
PDF Downloader
       |
       v
PyPDF
       |
       v
Text Extraction
       |
       v
Chunking
       |
       v
Sentence Transformers
       |
       v
Embeddings
       |
       v
FAISS
       |
       |
       +------------------+
       |                  |
       v                  |
User Question             |
       |                  |
       v                  |
Question Embedding        |
       |                  |
       v                  |
FAISS Similarity Search <+
       |
       v
Relevant Legal Chunks
       |
       v
Groq
       |
       v
Grounded Legal Answer
       |
       v
Page / Source Evidence
