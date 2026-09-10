import os
import re
import html
import urllib.parse
import urllib.request
import urllib.error

import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CyberLaw Pakistan AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "CyberLaw Pakistan AI"

# Your Google Drive PDF
GOOGLE_DRIVE_URL = (
    "https://drive.google.com/file/d/"
    "1d7xD2E1HrZBpkv16yHcqTb75C0MOzdmx/view?usp=sharing"
)

DATA_DIR = ".cyberlaw_data"
PDF_FILE = os.path.join(DATA_DIR, "pakistan_cyber_law.pdf")
INDEX_FILE = os.path.join(DATA_DIR, "cyberlaw.index")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.npy")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.npy")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .info-box {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 15px;
    }

    .source-box {
        padding: 12px;
        border-radius: 10px;
        border-left: 4px solid #888;
        margin-top: 8px;
        font-size: 14px;
    }

    .warning-box {
        padding: 15px;
        border-radius: 10px;
        background: rgba(255, 193, 7, 0.10);
        border: 1px solid rgba(255, 193, 7, 0.35);
    }

    .success-box {
        padding: 15px;
        border-radius: 10px;
        background: rgba(40, 167, 69, 0.10);
        border: 1px solid rgba(40, 167, 69, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GOOGLE DRIVE FUNCTIONS
# ============================================================

def extract_google_drive_file_id(url):
    """
    Extract a Google Drive file ID from common Drive URL formats.
    """

    if not url:
        return None

    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
        r"/open\?id=([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    # If user directly provides the ID
    if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", url.strip()):
        return url.strip()

    return None


def build_google_drive_download_url(file_id):
    return (
        "https://drive.google.com/uc?"
        + urllib.parse.urlencode(
            {
                "export": "download",
                "id": file_id,
            }
        )
    )


def download_google_drive_pdf(url, destination):
    """
    Download a publicly shared Google Drive PDF.

    Handles:
    - /file/d/.../view URLs
    - uc?export=download URLs
    - Google Drive confirmation pages
    """

    file_id = extract_google_drive_file_id(url)

    if not file_id:
        raise ValueError(
            "Could not extract the Google Drive file ID from the provided URL."
        )

    download_url = build_google_drive_download_url(file_id)

    opener = urllib.request.build_opener()
    opener.addheaders = [
        (
            "User-Agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131 Safari/537.36",
        )
    ]

    try:
        response = opener.open(download_url, timeout=60)
        content = response.read()

        content_type = response.headers.get("Content-Type", "").lower()

        # A real PDF normally starts with %PDF-
        if content.startswith(b"%PDF-") or "application/pdf" in content_type:
            with open(destination, "wb") as file:
                file.write(content)

            return destination

        # Google Drive may return an HTML confirmation page.
        text = content.decode("utf-8", errors="ignore")

        token_match = re.search(
            r'name="confirm"\s+value="([^"]+)"',
            text,
            re.IGNORECASE,
        )

        if not token_match:
            token_match = re.search(
                r"confirm=([0-9A-Za-z_-]+)",
                text,
                re.IGNORECASE,
            )

        if token_match:
            confirm_token = token_match.group(1)

            confirm_url = (
                "https://drive.usercontent.google.com/download?"
                + urllib.parse.urlencode(
                    {
                        "id": file_id,
                        "export": "download",
                        "confirm": confirm_token,
                    }
                )
            )

            response = opener.open(confirm_url, timeout=120)
            pdf_content = response.read()

            if not pdf_content.startswith(b"%PDF-"):
                raise ValueError(
                    "Google Drive returned a response that is not a PDF. "
                    "Make sure the file is publicly accessible."
                )

            with open(destination, "wb") as file:
                file.write(pdf_content)

            return destination

        # Another possible Google Drive download form
        confirm_url = (
            "https://drive.usercontent.google.com/download?"
            + urllib.parse.urlencode(
                {
                    "id": file_id,
                    "export": "download",
                }
            )
        )

        response = opener.open(confirm_url, timeout=120)
        pdf_content = response.read()

        if pdf_content.startswith(b"%PDF-"):
            with open(destination, "wb") as file:
                file.write(pdf_content)

            return destination

        raise ValueError(
            "Google Drive did not return a PDF. "
            "Check that the Drive file is shared as "
            "'Anyone with the link'."
        )

    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"Google Drive download failed with HTTP {error.code}."
        ) from error

    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not connect to Google Drive: {error.reason}"
        ) from error


# ============================================================
# PDF PROCESSING
# ============================================================

def extract_pdf_pages(pdf_path):
    """
    Extract text page-by-page from the PDF.
    """

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        text = re.sub(r"\s+", " ", text).strip()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


def create_chunks(
    pages,
    chunk_size=1000,
    overlap=150,
):
    """
    Split PDF text into overlapping chunks.

    Each chunk keeps its original PDF page number.
    """

    if overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    chunks = []
    metadata = []

    for page_data in pages:
        page_number = page_data["page"]
        text = page_data["text"]

        if not text:
            continue

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)
                metadata.append(
                    {
                        "page": page_number,
                    }
                )

            if end >= len(text):
                break

            start = end - overlap

    return chunks, metadata


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# BUILD / LOAD VECTOR DATABASE
# ============================================================

def database_exists():
    return (
        os.path.exists(PDF_FILE)
        and os.path.exists(INDEX_FILE)
        and os.path.exists(CHUNKS_FILE)
        and os.path.exists(METADATA_FILE)
    )


def save_database(chunks, metadata, embeddings):
    """
    Save FAISS index, chunks and metadata.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    index.add(embeddings.astype("float32"))

    faiss.write_index(index, INDEX_FILE)

    np.save(
        CHUNKS_FILE,
        np.array(chunks, dtype=object),
        allow_pickle=True,
    )

    np.save(
        METADATA_FILE,
        np.array(metadata, dtype=object),
        allow_pickle=True,
    )


def load_database():
    index = faiss.read_index(INDEX_FILE)

    chunks = np.load(
        CHUNKS_FILE,
        allow_pickle=True,
    ).tolist()

    metadata = np.load(
        METADATA_FILE,
        allow_pickle=True,
    ).tolist()

    return index, chunks, metadata


def build_database(force_rebuild=False):
    """
    Download PDF if necessary and create/load vector database.
    """

    if database_exists() and not force_rebuild:
        try:
            return load_database()
        except Exception:
            pass

    with st.status(
        "Preparing the CyberLaw knowledge base...",
        expanded=True,
    ) as status:

        st.write("Downloading the legal source PDF...")

        download_google_drive_pdf(
            GOOGLE_DRIVE_URL,
            PDF_FILE,
        )

        st.write("Extracting PDF text...")

        pages = extract_pdf_pages(PDF_FILE)

        if not pages:
            raise ValueError(
                "No readable text was extracted from the PDF. "
                "The PDF may be scanned/image-only."
            )

        st.write(
            f"Extracted text from {len(pages)} pages."
        )

        st.write("Creating text chunks...")

        chunks, metadata = create_chunks(pages)

        if not chunks:
            raise ValueError(
                "No text chunks were created from the PDF."
            )

        st.write(
            f"Created {len(chunks)} searchable chunks."
        )

        st.write("Creating embeddings...")

        model = load_embedding_model()

        embeddings = model.encode(
            chunks,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        embeddings = embeddings.astype("float32")

        st.write("Building FAISS vector index...")

        save_database(
            chunks,
            metadata,
            embeddings,
        )

        status.update(
            label="Knowledge base ready.",
            state="complete",
            expanded=False,
        )

    return load_database()


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(
    question,
    index,
    chunks,
    metadata,
    top_k=5,
):
    """
    Retrieve the most relevant legal document chunks.
    """

    model = load_embedding_model()

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks)),
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0 or idx >= len(chunks):
            continue

        results.append(
            {
                "text": chunks[idx],
                "page": metadata[idx]["page"],
                "score": float(score),
            }
        )

    return results


# ============================================================
# GROQ
# ============================================================

def get_groq_client(api_key):
    if not api_key:
        return None

    return Groq(api_key=api_key)


def get_max_tokens(response_size):
    mapping = {
        "Short": 700,
        "Medium": 1200,
        "Detailed": 2000,
        "Very detailed": 3000,
    }

    return mapping.get(response_size, 1200)


def build_system_prompt(
    technical_level,
    response_size,
    language,
    answer_style,
):
    return f"""
You are CyberLaw Pakistan AI, a retrieval-augmented legal information assistant.

Your task is to answer questions using ONLY the legal source material retrieved
from the configured Pakistani cyber-law PDF.

IMPORTANT LEGAL ACCURACY RULES:

1. Do not invent laws, sections, clauses, penalties, fines, procedures,
   authorities, dates or legal requirements.

2. Do not claim that a specific section exists unless the retrieved source
   supports it.

3. If the retrieved evidence is insufficient, clearly say that the provided
   source does not contain enough information to answer confidently.

4. Never fabricate citations.

5. Cite supporting evidence using:
   [Source 1, Page X]
   [Source 2, Page Y]

6. If multiple sources support an answer, cite the relevant sources.

7. Distinguish between:
   - what the provided legal document says
   - general explanation
   - practical guidance

8. Do not present yourself as a lawyer.

9. For urgent legal matters, serious criminal allegations, arrests,
   investigations or court proceedings, recommend consulting a qualified
   Pakistani lawyer or relevant authority.

10. Do not provide instructions that facilitate:
    - hacking
    - malware
    - credential theft
    - phishing
    - unauthorized access
    - evasion of law enforcement
    - cyber attacks
    - identity theft
    - fraud
    - abuse of computer systems

11. If the user asks for harmful cyber instructions, refuse the harmful
    operational part but provide a safe legal/security explanation when
    appropriate.

USER PREFERENCES:

Technical level: {technical_level}
Response size: {response_size}
Language: {language}
Answer style: {answer_style}

Answer entirely in the selected language.

Be precise, structured and easy to understand.
"""


def create_context(results):
    if not results:
        return "No relevant legal evidence was retrieved."

    context_parts = []

    for i, result in enumerate(results, start=1):
        context_parts.append(
            f"""
SOURCE {i}
PDF Page: {result["page"]}
Similarity Score: {result["score"]:.4f}

TEXT:
{result["text"]}
"""
        )

    return "\n".join(context_parts)


def build_conversation_history(messages):
    """
    Include a small amount of previous conversation so follow-up questions
    can work more naturally.
    """

    if not messages:
        return ""

    history = messages[-6:]

    lines = []

    for message in history:
        role = message.get("role", "")
        content = message.get("content", "")

        if role in ("user", "assistant"):
            lines.append(
                f"{role.upper()}: {content}"
            )

    return "\n".join(lines)


def ask_groq(
    question,
    context,
    history,
    api_key,
    model_name,
    technical_level,
    response_size,
    language,
    answer_style,
):
    client = get_groq_client(api_key)

    if client is None:
        raise ValueError(
            "GROQ_API_KEY is missing."
        )

    system_prompt = build_system_prompt(
        technical_level=technical_level,
        response_size=response_size,
        language=language,
        answer_style=answer_style,
    )

    user_prompt = f"""
RETRIEVED LEGAL EVIDENCE:
{context}

PREVIOUS CONVERSATION:
{history if history else "No previous conversation."}

CURRENT USER QUESTION:
{question}

INSTRUCTIONS:

Answer the current question based primarily on the retrieved legal evidence.

If the evidence does not support a specific legal claim, say so.

Do not fill missing legal information with assumptions.

Include source citations in this exact style where applicable:
[Source 1, Page X]

Give a useful answer rather than simply repeating the source.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.1,
        max_tokens=get_max_tokens(response_size),
    )

    return response.choices[0].message.content


# ============================================================
# COMPLAINT GENERATOR
# ============================================================

def generate_complaint(
    incident,
    context,
    api_key,
    model_name,
    language,
):
    client = get_groq_client(api_key)

    if client is None:
        raise ValueError(
            "GROQ_API_KEY is missing."
        )

    system_prompt = f"""
You are an assistant helping users prepare a factual cybercrime complaint
draft for Pakistan.

Language: {language}

Use the retrieved legal source only for legal references.

Do not invent:
- sections
- laws
- penalties
- authorities
- procedures
- dates
- evidence

Do not make accusations beyond the facts supplied by the user.

Clearly mark the result as a DRAFT that should be reviewed before submission.

Do not provide false legal certainty.
"""

    user_prompt = f"""
INCIDENT DETAILS:

{incident}

RELEVANT LEGAL SOURCE:

{context}

Create a professional complaint draft containing:

1. Subject
2. Complainant details placeholder
3. Incident description
4. Date/time placeholder if not supplied
5. Platform/device placeholder if relevant
6. Suspect information if supplied
7. Evidence list
8. Requested action
9. Declaration
10. Signature/date placeholders

Only mention legal provisions when they are supported by the supplied
retrieved source.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.1,
        max_tokens=2500,
    )

    return response.choices[0].message.content


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚖️ CyberLaw Pakistan AI")

    page = st.radio(
        "Navigation",
        [
            "AI Assistant",
            "File a Complaint",
            "About",
        ],
    )

    st.divider()

    st.markdown("### AI Settings")

    technical_level = st.selectbox(
        "Technical level",
        [
            "Beginner",
            "Intermediate",
            "Advanced",
            "Legal/Technical",
        ],
        index=0,
    )

    response_size = st.selectbox(
        "Response size",
        [
            "Short",
            "Medium",
            "Detailed",
            "Very detailed",
        ],
        index=1,
    )

    language = st.selectbox(
        "Response language",
        [
            "English",
            "Urdu",
            "Roman Urdu",
        ],
        index=0,
    )

    answer_style = st.selectbox(
        "Answer style",
        [
            "Clear explanation",
            "Step-by-step",
            "Bullet points",
            "Professional legal information",
        ],
        index=0,
    )

    top_k = st.slider(
        "Retrieved sources",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
    )

    show_sources = st.checkbox(
        "Show retrieved sources",
        value=True,
    )

    st.divider()

    st.markdown("### Groq")

    groq_api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Your Groq API key.",
    )

    groq_model = st.text_input(
        "Groq model",
        value=DEFAULT_GROQ_MODEL,
    )

    st.divider()

    st.markdown("### Knowledge Base")

    if database_exists():
        st.success("Knowledge base available.")
    else:
        st.warning("Knowledge base not built yet.")

    rebuild = st.button(
        "🔄 Rebuild knowledge base",
        use_container_width=True,
    )


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

try:
    if rebuild:
        index, chunks, metadata = build_database(
            force_rebuild=True
        )
    else:
        index, chunks, metadata = build_database(
            force_rebuild=False
        )

except Exception as error:
    st.error("Could not prepare the CyberLaw knowledge base.")

    st.code(
        str(error),
        language="text",
    )

    st.info(
        "Check that your Google Drive PDF is publicly accessible "
        "and that the file contains selectable text."
    )

    st.stop()


# ============================================================
# AI ASSISTANT
# ============================================================

if page == "AI Assistant":

    st.markdown(
        '<div class="main-title">⚖️ CyberLaw Pakistan AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Ask questions about the Pakistani cyber-law source document."
        "</div>",
        unsafe_allow_html=True,
    )

    st.info(
        "This application provides AI-assisted legal information based on "
        "the configured PDF. It is not a substitute for advice from a "
        "qualified lawyer."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.button(
        "Clear conversation",
        type="secondary",
    ):
        st.session_state.messages = []
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and show_sources
                and message.get("sources")
            ):
                with st.expander("Retrieved sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f"""
**Page {source["page"]}**  
Similarity: `{source["score"]:.4f}`

{source["text"]}
"""
                        )

    question = st.chat_input(
        "Ask a question about Pakistani cyber law..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            try:
                with st.spinner("Searching the legal knowledge base..."):

                    results = retrieve_documents(
                        question=question,
                        index=index,
                        chunks=chunks,
                        metadata=metadata,
                        top_k=top_k,
                    )

                context = create_context(results)

                history = build_conversation_history(
                    st.session_state.messages[:-1]
                )

                with st.spinner("Generating answer..."):

                    answer = ask_groq(
                        question=question,
                        context=context,
                        history=history,
                        api_key=groq_api_key,
                        model_name=groq_model,
                        technical_level=technical_level,
                        response_size=response_size,
                        language=language,
                        answer_style=answer_style,
                    )

                st.markdown(answer)

                if show_sources and results:

                    with st.expander(
                        "Retrieved sources",
                        expanded=False,
                    ):

                        for i, result in enumerate(
                            results,
                            start=1,
                        ):

                            st.markdown(
                                f"""
**Source {i} — Page {result["page"]}**

Similarity score: `{result["score"]:.4f}`

{result["text"]}
"""
                            )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": results,
                    }
                )

            except Exception as error:

                error_message = (
                    "I could not generate the answer.\n\n"
                    f"**Error:** `{error}`"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )


# ============================================================
# FILE A COMPLAINT
# ============================================================

elif page == "File a Complaint":

    st.markdown(
        '<div class="main-title">📝 Cybercrime Complaint Draft</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Describe the incident and generate a factual complaint draft."
        "</div>",
        unsafe_allow_html=True,
    )

    st.warning(
        "This creates a draft only. Review all details and legal references "
        "before submitting a complaint to any authority."
    )

    incident = st.text_area(
        "Describe the incident",
        height=250,
        placeholder=(
            "Example:\n"
            "Someone created a fake social media account using my name "
            "and uploaded my photos without permission. The account URL is..."
        ),
    )

    evidence = st.text_area(
        "Evidence you have",
        height=150,
        placeholder=(
            "Screenshots, profile URL, messages, transaction records, "
            "email headers, phone numbers, etc."
        ),
    )

    generate_button = st.button(
        "Generate complaint draft",
        type="primary",
        use_container_width=True,
    )

    if generate_button:

        if not groq_api_key:
            st.error(
                "Please enter your GROQ_API_KEY in the sidebar."
            )
            st.stop()

        if not incident.strip():
            st.error(
                "Please describe the incident first."
            )
            st.stop()

        complete_incident = f"""
INCIDENT:
{incident}

EVIDENCE:
{evidence if evidence.strip() else "No evidence details supplied."}
"""

        with st.spinner(
            "Searching the legal source..."
        ):

            results = retrieve_documents(
                question=incident,
                index=index,
                chunks=chunks,
                metadata=metadata,
                top_k=top_k,
            )

        context = create_context(results)

        with st.spinner(
            "Preparing complaint draft..."
        ):

            try:
                complaint = generate_complaint(
                    incident=complete_incident,
                    context=context,
                    api_key=groq_api_key,
                    model_name=groq_model,
                    language=language,
                )

                st.success(
                    "Complaint draft generated."
                )

                st.markdown("### Draft")

                st.markdown(complaint)

                st.download_button(
                    "Download complaint draft",
                    data=complaint,
                    file_name="cybercrime_complaint_draft.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

                if show_sources:

                    with st.expander(
                        "Legal sources used"
                    ):

                        for i, result in enumerate(
                            results,
                            start=1,
                        ):

                            st.markdown(
                                f"""
**Source {i} — Page {result["page"]}**

{result["text"]}
"""
                            )

            except Exception as error:

                st.error(
                    f"Could not generate the complaint: {error}"
                )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.markdown(
        '<div class="main-title">ℹ️ About</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
## CyberLaw Pakistan AI

CyberLaw Pakistan AI is a Retrieval-Augmented Generation (RAG)
application designed to answer questions using a Pakistani cyber-law
PDF as its knowledge source.

### How it works

1. Downloads the legal PDF from Google Drive.
2. Extracts text from the PDF.
3. Splits the text into searchable chunks.
4. Creates vector embeddings.
5. Stores the embeddings in FAISS.
6. Retrieves the most relevant legal passages.
7. Sends the retrieved evidence to Groq.
8. Generates an answer grounded in the retrieved source.

### Main technologies

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- FAISS
- Groq
- RAG

### Important limitation

The application can only provide reliable source-grounded answers
when the supplied PDF contains the relevant legal information.

It should not be treated as a lawyer, court, government authority,
or official legal-information service.

Always verify important legal matters against current official sources
and obtain professional legal advice when necessary.
        """
    )

    st.markdown("### Knowledge Base")

    st.write(
        f"PDF: `{os.path.basename(PDF_FILE)}`"
    )

    st.write(
        f"Indexed chunks: `{len(chunks)}`"
    )

    st.write(
        f"Embedding model: `{EMBEDDING_MODEL}`"
    )

    st.write(
        f"Groq model: `{groq_model}`"
    )

    st.markdown(
        """
### Source configuration

The application is configured to use the Google Drive PDF supplied
for this project.

Make sure the Google Drive file sharing setting allows access to
people with the link.
        """
    )
