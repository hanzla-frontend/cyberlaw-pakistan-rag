import os
import re
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
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CyberLaw Pakistan AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIGURATION
# ============================================================

GOOGLE_DRIVE_URL = (
    "https://drive.google.com/file/d/"
    "1d7xD2E1HrZBpkv16yHcqTb75C0MOzdmx/view?usp=sharing"
)

DATA_DIR = ".cyberlaw_data"
PDF_PATH = os.path.join(DATA_DIR, "cyber_law_source.pdf")
INDEX_PATH = os.path.join(DATA_DIR, "cyberlaw.index")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.npy")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_MODEL = "openai/gpt-oss-120b"

MODEL_OPTIONS = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "theme": "Dark",
    "accent_color": "#FF7A00",
    "background_color": "#0B0F14",
    "messages": [],
    "total_prompts": 0,
    "total_messages": 0,
    "total_tokens": 0,
    "selected_model": DEFAULT_MODEL,
    "knowledge_ready": False,
    "chunks": [],
    "index": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# THEME COLORS
# ============================================================

def get_theme_colors():
    """
    Returns a complete color system for Light/Dark mode.
    """

    if st.session_state.theme == "Dark":
        return {
            "bg": st.session_state.background_color,
            "surface": "#151A21",
            "surface_2": "#1C232D",
            "input": "#11161D",
            "border": "#303945",
            "text": "#F5F7FA",
            "secondary": "#B7C0CC",
            "placeholder": "#7F8A98",
            "accent": st.session_state.accent_color,
            "accent_hover": "#FF963D",
            "code": "#0D1117",
            "sidebar": "#10151C",
            "chat_user": "#1B2530",
            "chat_assistant": "#151A21",
            "success_bg": "#10261B",
            "warning_bg": "#2A2110",
            "error_bg": "#2A1515",
        }

    return {
        "bg": st.session_state.background_color,
        "surface": "#FFFFFF",
        "surface_2": "#F4F6F8",
        "input": "#FFFFFF",
        "border": "#D8DEE6",
        "text": "#111827",
        "secondary": "#4B5563",
        "placeholder": "#6B7280",
        "accent": st.session_state.accent_color,
        "accent_hover": "#E86F00",
        "code": "#F3F4F6",
        "sidebar": "#F8FAFC",
        "chat_user": "#F0F4F8",
        "chat_assistant": "#FFFFFF",
        "success_bg": "#ECFDF3",
        "warning_bg": "#FFFBEB",
        "error_bg": "#FEF2F2",
    }


COLORS = get_theme_colors()


# ============================================================
# CUSTOM CSS
# ============================================================

def apply_theme():

    c = COLORS

    st.markdown(
        f"""
        <style>

        /* ==================================================
           GLOBAL
           ================================================== */

        html,
        body,
        [data-testid="stAppViewContainer"],
        [data-testid="stApp"],
        .stApp {{
            background: {c["bg"]} !important;
            color: {c["text"]} !important;
        }}

        [data-testid="stAppViewContainer"] {{
            background: {c["bg"]} !important;
        }}

        .main {{
            background: {c["bg"]} !important;
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }}

        /* ==================================================
           ALL TEXT
           ================================================== */

        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        p,
        span,
        label,
        div {{
            color: {c["text"]};
        }}

        .stMarkdown,
        .stCaption,
        .stText,
        [data-testid="stMarkdownContainer"] {{
            color: {c["text"]} !important;
        }}

        .stCaption {{
            color: {c["secondary"]} !important;
        }}

        /* ==================================================
           SIDEBAR
           ================================================== */

        section[data-testid="stSidebar"] {{
            background: {c["sidebar"]} !important;
            border-right: 1px solid {c["border"]} !important;
        }}

        section[data-testid="stSidebar"] > div {{
            background: {c["sidebar"]} !important;
        }}

        section[data-testid="stSidebar"] * {{
            color: {c["text"]} !important;
        }}

        /* ==================================================
           HEADER
           ================================================== */

        .app-title {{
            font-size: 2.7rem;
            font-weight: 800;
            letter-spacing: -1px;
            margin-bottom: 0.2rem;
        }}

        .app-subtitle {{
            font-size: 1.05rem;
            color: {c["secondary"]} !important;
            margin-bottom: 1.5rem;
        }}

        .accent-line {{
            height: 4px;
            width: 90px;
            background: {c["accent"]};
            border-radius: 20px;
            margin: 8px 0 20px 0;
        }}

        /* ==================================================
           CARDS
           ================================================== */

        .custom-card {{
            background: {c["surface"]};
            border: 1px solid {c["border"]};
            border-radius: 18px;
            padding: 22px;
            margin-bottom: 18px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
        }}

        .custom-card h3 {{
            margin-top: 0;
        }}

        .stat-card {{
            background: {c["surface"]};
            border: 1px solid {c["border"]};
            border-radius: 16px;
            padding: 18px;
            text-align: center;
        }}

        .stat-number {{
            font-size: 1.8rem;
            font-weight: 800;
            color: {c["accent"]} !important;
        }}

        .stat-label {{
            color: {c["secondary"]} !important;
            font-size: 0.85rem;
        }}

        /* ==================================================
           TEXT INPUT
           ================================================== */

        div[data-testid="stTextInput"] input {{
            background: {c["input"]} !important;
            color: {c["text"]} !important;
            border: 1px solid {c["border"]} !important;
            border-radius: 10px !important;
        }}

        div[data-testid="stTextInput"] input:focus {{
            border-color: {c["accent"]} !important;
            box-shadow: 0 0 0 1px {c["accent"]} !important;
        }}

        div[data-testid="stTextInput"] input::placeholder {{
            color: {c["placeholder"]} !important;
            opacity: 1 !important;
        }}

        /* ==================================================
           TEXT AREA
           ================================================== */

        div[data-testid="stTextArea"] textarea {{
            background: {c["input"]} !important;
            color: {c["text"]} !important;
            border: 1px solid {c["border"]} !important;
            border-radius: 12px !important;
        }}

        div[data-testid="stTextArea"] textarea:focus {{
            border-color: {c["accent"]} !important;
            box-shadow: 0 0 0 1px {c["accent"]} !important;
        }}

        div[data-testid="stTextArea"] textarea::placeholder {{
            color: {c["placeholder"]} !important;
            opacity: 1 !important;
        }}

        /* ==================================================
           SELECTBOX / DROPDOWN
           ================================================== */

        div[data-baseweb="select"] > div {{
            background: {c["input"]} !important;
            color: {c["text"]} !important;
            border-color: {c["border"]} !important;
            border-radius: 10px !important;
        }}

        div[data-baseweb="select"] span {{
            color: {c["text"]} !important;
        }}

        div[data-baseweb="select"] svg {{
            fill: {c["text"]} !important;
        }}

        [role="listbox"] {{
            background: {c["surface"]} !important;
            border: 1px solid {c["border"]} !important;
        }}

        [role="option"] {{
            background: {c["surface"]} !important;
            color: {c["text"]} !important;
        }}

        [role="option"]:hover {{
            background: {c["surface_2"]} !important;
        }}

        /* ==================================================
           CHAT INPUT — CONTINUOUS + DARK/LIGHT TEXT
           ================================================== */

        div[data-testid="stChatInput"] {{
            background: transparent !important;
            padding: 0 !important;
        }}

        /* Continuous input + send button container */
        div[data-testid="stChatInput"] > div {{
            display: flex !important;
            align-items: center !important;
            gap: 0 !important;
            background: {c["input"]} !important;
            border: 1px solid {c["border"]} !important;
            border-radius: 14px !important;
            overflow: hidden !important;
            padding: 0 !important;
        }}

        /* Input text — automatically follows Dark/Light theme */
        div[data-testid="stChatInput"] textarea {{
            flex: 1 1 auto !important;
            background: transparent !important;
            color: {c["text"]} !important;
            -webkit-text-fill-color: {c["text"]} !important;
            caret-color: {c["text"]} !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            border-radius: 0 !important;
            margin: 0 !important;
        }}

        /* Placeholder */
        div[data-testid="stChatInput"] textarea::placeholder {{
            color: {c["placeholder"]} !important;
            -webkit-text-fill-color: {c["placeholder"]} !important;
            opacity: 1 !important;
        }}

        /* Focus entire input */
        div[data-testid="stChatInput"]:focus-within > div {{
            border-color: {c["accent"]} !important;
            box-shadow: 0 0 0 1px {c["accent"]} !important;
        }}

        /* Send button */
        div[data-testid="stChatInput"] button {{
            background: {c["accent"]} !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 0 !important;
            min-width: 52px !important;
            min-height: 52px !important;
            height: 100% !important;
            margin: 0 !important;
            align-self: stretch !important;
            flex: 0 0 52px !important;
        }}

        div[data-testid="stChatInput"] button:hover {{
            background: {c["accent_hover"]} !important;
            color: #FFFFFF !important;
        }}

        /* ==================================================
           BUTTONS
           ================================================== */

        .stButton > button {{
            background: {c["surface"]} !important;
            color: {c["text"]} !important;
            border: 1px solid {c["border"]} !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            min-height: 42px;
            transition: all 0.2s ease;
        }}

        .stButton > button:hover {{
            border-color: {c["accent"]} !important;
            color: {c["accent"]} !important;
        }}

        .stButton > button[kind="primary"] {{
            background: {c["accent"]} !important;
            color: #FFFFFF !important;
            border: none !important;
        }}

        .stButton > button[kind="primary"]:hover {{
            background: {c["accent_hover"]} !important;
            color: #FFFFFF !important;
        }}

        .stDownloadButton > button {{
            background: {c["accent"]} !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
        }}

        .stDownloadButton > button:hover {{
            background: {c["accent_hover"]} !important;
        }}

        /* ==================================================
           RADIO / CHECKBOX
           ================================================== */

        div[data-testid="stRadio"] label,
        div[data-testid="stCheckbox"] label {{
            color: {c["text"]} !important;
        }}

        div[data-testid="stRadio"] p,
        div[data-testid="stCheckbox"] p {{
            color: {c["text"]} !important;
        }}

        /* ==================================================
           SLIDER
           ================================================== */

        div[data-testid="stSlider"] {{
            color: {c["text"]} !important;
        }}

        div[data-testid="stSlider"] label {{
            color: {c["text"]} !important;
        }}

        /* ==================================================
           COLOR PICKER
           ================================================== */

        div[data-testid="stColorPicker"] label {{
            color: {c["text"]} !important;
        }}

        /* ==================================================
           METRICS
           ================================================== */

        div[data-testid="stMetric"] {{
            background: {c["surface"]} !important;
            border: 1px solid {c["border"]} !important;
            border-radius: 14px !important;
            padding: 12px !important;
        }}

        div[data-testid="stMetricLabel"] {{
            color: {c["secondary"]} !important;
        }}

        div[data-testid="stMetricValue"] {{
            color: {c["text"]} !important;
        }}

        /* ==================================================
           CHAT MESSAGES
           ================================================== */

        [data-testid="stChatMessage"] {{
            background: {c["chat_assistant"]} !important;
            border: 1px solid {c["border"]} !important;
            border-radius: 16px !important;
            margin-bottom: 12px !important;
        }}

        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li,
        [data-testid="stChatMessage"] span {{
            color: {c["text"]} !important;
        }}

        /* ==================================================
           EXPANDERS
           ================================================== */

        [data-testid="stExpander"] {{
            background: {c["surface"]} !important;
            border: 1px solid {c["border"]} !important;
            border-radius: 12px !important;
        }}

        [data-testid="stExpander"] summary {{
            color: {c["text"]} !important;
        }}

        /* ==================================================
           ALERTS
           ================================================== */

        [data-testid="stAlert"] {{
            border-radius: 12px !important;
        }}

        /* ==================================================
           CODE
           ================================================== */

        code {{
            background: {c["code"]} !important;
            color: {c["text"]} !important;
        }}

        pre {{
            background: {c["code"]} !important;
            border: 1px solid {c["border"]} !important;
            border-radius: 12px !important;
        }}

        /* ==================================================
           FILE UPLOADER
           ================================================== */

        [data-testid="stFileUploader"] {{
            background: {c["surface"]} !important;
            border: 1px solid {c["border"]} !important;
            border-radius: 12px !important;
        }}

        [data-testid="stFileUploader"] section {{
            background: {c["surface"]} !important;
        }}

        /* ==================================================
           TOGGLE SWITCH
           ================================================== */

        .theme-toggle-container {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: {c["surface"]};
            border: 1px solid {c["border"]};
            border-radius: 14px;
            padding: 10px 12px;
            margin: 6px 0 18px 0;
        }}

        .theme-toggle-title {{
            display: flex;
            align-items: center;
            gap: 9px;
            font-weight: 700;
            color: {c["text"]} !important;
        }}

        .theme-icon {{
            font-size: 20px;
        }}

        /* ==================================================
           DIVIDER
           ================================================== */

        hr {{
            border-color: {c["border"]} !important;
        }}

        /* ==================================================
           FOOTER
           ================================================== */

        .footer {{
            text-align: center;
            color: {c["secondary"]} !important;
            font-size: 0.82rem;
            padding: 25px 0 5px 0;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )


apply_theme()


# ============================================================
# GOOGLE DRIVE FUNCTIONS
# ============================================================

def extract_drive_file_id(url):
    """
    Extract Google Drive file ID from common sharing URL formats.
    """

    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def download_google_drive_file(url, destination):
    """
    Download a Google Drive file using its file ID.
    """

    file_id = extract_drive_file_id(url)

    if not file_id:
        raise ValueError("Could not extract Google Drive file ID.")

    download_url = (
        "https://drive.usercontent.google.com/download"
        f"?id={urllib.parse.quote(file_id)}&export=download&confirm=t"
    )

    request = urllib.request.Request(
        download_url,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()

        if not data:
            raise ValueError("Google Drive returned an empty file.")

        with open(destination, "wb") as f:
            f.write(data)

        return destination

    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Google Drive download failed: HTTP {e.code}"
        )

    except Exception as e:
        raise RuntimeError(
            f"Google Drive download failed: {str(e)}"
        )


# ============================================================
# PDF PROCESSING
# ============================================================

def extract_pdf_pages(pdf_path):
    """
    Extract text page-by-page.
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
    chunk_size=900,
    overlap=150,
):
    """
    Split pages into overlapping chunks while preserving page numbers.
    """

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]
        text = page_data["text"]

        start = 0

        while start < len(text):

            end = min(start + chunk_size, len(text))

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number,
                    }
                )

            if end >= len(text):
                break

            start = max(0, end - overlap)

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def create_faiss_index(chunks):
    """
    Create normalized FAISS inner-product index.
    """

    model = load_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index


def save_knowledge_base(index, chunks):

    faiss.write_index(
        index,
        INDEX_PATH,
    )

    np.save(
        CHUNKS_PATH,
        np.array(chunks, dtype=object),
        allow_pickle=True,
    )


def load_knowledge_base():

    if not os.path.exists(INDEX_PATH):
        return None, []

    if not os.path.exists(CHUNKS_PATH):
        return None, []

    try:

        index = faiss.read_index(
            INDEX_PATH
        )

        chunks_array = np.load(
            CHUNKS_PATH,
            allow_pickle=True,
        )

        chunks = chunks_array.tolist()

        return index, chunks

    except Exception:
        return None, []


def build_knowledge_base():

    with st.spinner(
        "Downloading and processing the cyber-law knowledge base..."
    ):

        if not os.path.exists(PDF_PATH):

            download_google_drive_file(
                GOOGLE_DRIVE_URL,
                PDF_PATH,
            )

        pages = extract_pdf_pages(
            PDF_PATH
        )

        if not pages:
            raise ValueError(
                "No readable text was extracted from the PDF."
            )

        chunks = create_chunks(
            pages
        )

        if not chunks:
            raise ValueError(
                "No chunks were created from the PDF."
            )

        index = create_faiss_index(
            chunks
        )

        save_knowledge_base(
            index,
            chunks
        )

        return index, chunks


# ============================================================
# KNOWLEDGE BASE INITIALIZATION
# ============================================================

if st.session_state.index is None:

    loaded_index, loaded_chunks = load_knowledge_base()

    if loaded_index is not None:

        st.session_state.index = loaded_index
        st.session_state.chunks = loaded_chunks
        st.session_state.knowledge_ready = True


# ============================================================
# GROQ
# ============================================================

def get_groq_api_key():

    try:

        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]

    except Exception:
        pass

    return os.getenv("GROQ_API_KEY")


def get_groq_client():

    api_key = get_groq_api_key()

    if not api_key:
        return None

    return Groq(
        api_key=api_key
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_context(
    query,
    top_k=5,
):

    if (
        st.session_state.index is None
        or not st.session_state.chunks
    ):
        return []

    model = load_embedding_model()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    scores, indices = st.session_state.index.search(
        query_embedding,
        min(
            top_k,
            len(st.session_state.chunks),
        ),
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):

        if idx < 0:
            continue

        chunk = st.session_state.chunks[idx]

        results.append(
            {
                "text": chunk["text"],
                "page": chunk["page"],
                "score": float(score),
            }
        )

    return results


# ============================================================
# PROMPT CONFIGURATION
# ============================================================

def get_response_tokens(size):

    mapping = {
        "Short": 700,
        "Medium": 1200,
        "Detailed": 2000,
        "Very detailed": 3000,
    }

    return mapping.get(
        size,
        1200,
    )


def build_system_prompt(
    technical_level,
    response_size,
    language,
    answer_style,
):

    language_instruction = {
        "English": "Respond in clear English.",
        "Urdu": "Respond in clear Urdu using Urdu script where appropriate.",
        "Roman Urdu": "Respond in simple Roman Urdu.",
    }.get(
        language,
        "Respond in clear English.",
    )

    style_instruction = {
        "Professional": "Use a professional legal-information style.",
        "Simple": "Explain concepts in simple language suitable for a beginner.",
        "Educational": "Teach the topic step-by-step with short explanations and examples.",
        "Direct": "Give a concise, direct answer and avoid unnecessary explanation.",
    }.get(
        answer_style,
        "Use a professional legal-information style.",
    )

    return f"""
You are CyberLaw Pakistan AI, a retrieval-augmented legal information assistant.

Your job is to answer questions using ONLY the retrieved knowledge-base context
provided to you.

IMPORTANT LEGAL SAFETY RULES:

1. Do not invent Pakistani laws.
2. Do not invent sections, clauses, penalties, procedures, authorities, dates,
   case citations, or legal terminology.
3. Do not claim that a law exists unless it is supported by the retrieved context.
4. If the knowledge base does not contain enough information, clearly say:
   "The available knowledge base does not provide enough information to answer this reliably."
5. Distinguish between information from the knowledge base and general explanation.
6. Do not provide instructions that facilitate cybercrime, unauthorized access,
   malware deployment, credential theft, evasion, exploitation, or other harmful activity.
7. For cyber incidents, provide safe defensive and lawful guidance.
8. Do not present the response as formal legal representation.
9. Encourage consultation with a qualified Pakistani lawyer or relevant official
   authority when the issue has significant legal consequences.
10. Never fabricate source references.

USER SETTINGS:

Technical level: {technical_level}
Response size: {response_size}
Language: {language}
Answer style: {answer_style}

{language_instruction}
{style_instruction}

SOURCE CITATION FORMAT:

When the answer is supported by the retrieved context, cite it as:

[Source 1, Page X]

Use the actual page number supplied with the context.

If multiple sources support a statement, cite each relevant source.

Keep citations close to the relevant claim.
"""


def build_context(results):

    if not results:
        return "No relevant knowledge-base context was retrieved."

    sections = []

    for i, result in enumerate(
        results,
        start=1,
    ):

        sections.append(
            f"""
SOURCE {i}
Page: {result["page"]}
Similarity: {result["score"]:.4f}

{result["text"]}
"""
        )

    return "\n".join(sections)


# ============================================================
# RAG ANSWER
# ============================================================

def ask_rag(
    question,
    technical_level,
    response_size,
    language,
    answer_style,
    top_k,
    model_name,
):

    client = get_groq_client()

    if client is None:
        return (
            "Groq API key is not configured.\n\n"
            "Add `GROQ_API_KEY` to Streamlit secrets or your environment variables."
        ), [], 0

    if not st.session_state.knowledge_ready:

        return (
            "The cyber-law knowledge base is not ready yet. "
            "Please build the knowledge base from the sidebar."
        ), [], 0

    results = retrieve_context(
        question,
        top_k=top_k,
    )

    context = build_context(
        results
    )

    system_prompt = build_system_prompt(
        technical_level,
        response_size,
        language,
        answer_style,
    )

    history = []

    for message in st.session_state.messages[-6:]:

        history.append(
            {
                "role": message["role"],
                "content": message["content"],
            }
        )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": f"""
QUESTION:

{question}

RETRIEVED KNOWLEDGE-BASE CONTEXT:

{context}

Answer the question based on the retrieved context.
Do not invent missing legal information.
""",
        }
    )

    try:

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.15,
            max_tokens=get_response_tokens(
                response_size
            ),
        )

        answer = (
            completion.choices[0].message.content
            if completion.choices
            else "No response was generated."
        )

        usage_tokens = 0

        if getattr(completion, "usage", None):

            usage_tokens = getattr(
                completion.usage,
                "total_tokens",
                0,
            ) or 0

        return (
            answer,
            results,
            usage_tokens,
        )

    except Exception as e:

        return (
            f"Unable to generate the answer.\n\nError: {str(e)}",
            results,
            0,
        )


# ============================================================
# COMPLAINT GENERATOR
# ============================================================

def generate_complaint(
    incident,
    language,
    model_name,
):

    client = get_groq_client()

    if client is None:
        return (
            "Groq API key is not configured.",
            0,
        )

    results = retrieve_context(
        incident,
        top_k=5,
    )

    context = build_context(
        results
    )

    prompt = f"""
Create a factual cybercrime complaint draft based on the incident below.

Use the supplied knowledge-base context for legal references.

Do NOT invent:
- laws
- sections
- penalties
- government procedures
- authorities
- case numbers
- legal claims

If something is unknown, use a placeholder or state that it needs verification.

Write the complaint in {language}.

Include:

1. Subject
2. Complainant information placeholders
3. Incident summary
4. Date/time placeholders
5. Evidence available
6. Requested action
7. Contact information placeholders
8. Disclaimer that the draft should be reviewed before submission

INCIDENT:

{incident}

KNOWLEDGE-BASE CONTEXT:

{context}
"""

    try:

        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You create factual, lawful cybercrime complaint drafts. "
                        "Never fabricate legal information."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.15,
            max_tokens=2200,
        )

        answer = completion.choices[0].message.content

        usage_tokens = 0

        if getattr(completion, "usage", None):

            usage_tokens = getattr(
                completion.usage,
                "total_tokens",
                0,
            ) or 0

        return answer, usage_tokens

    except Exception as e:

        return (
            f"Unable to generate complaint draft.\n\nError: {str(e)}",
            0,
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:25px;
            font-weight:800;
            margin-bottom:4px;
        ">
            ⚖️ CyberLaw Pakistan AI
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "RAG-powered Pakistani cyber-law information assistant"
    )

    st.divider()

    # --------------------------------------------------------
    # NAVIGATION
    # --------------------------------------------------------

    page = st.radio(
        "Navigation",
        [
            "AI Assistant",
            "File a Complaint",
            "About",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # --------------------------------------------------------
    # POLISHED THEME TOGGLE
    # --------------------------------------------------------

    current_dark = (
        st.session_state.theme == "Dark"
    )

    st.markdown(
        f"""
        <div class="theme-toggle-container">
            <div class="theme-toggle-title">
                <span class="theme-icon">
                    {"🌙" if current_dark else "☀️"}
                </span>
                <span>
                    {"Dark Mode" if current_dark else "Light Mode"}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    toggle_value = st.toggle(
        "Switch theme",
        value=current_dark,
        label_visibility="collapsed",
        key="theme_toggle",
    )

    new_theme = (
        "Dark"
        if toggle_value
        else "Light"
    )

    if new_theme != st.session_state.theme:

        st.session_state.theme = new_theme

        # Automatic background adjustment.
        if new_theme == "Dark":

            st.session_state.background_color = "#0B0F14"

        else:

            st.session_state.background_color = "#F5F7FA"

        st.rerun()

    # --------------------------------------------------------
    # COLOR SETTINGS
    # --------------------------------------------------------

    st.markdown("### Appearance")

    st.session_state.accent_color = st.color_picker(
        "Accent color",
        value=st.session_state.accent_color,
    )

    default_background = (
        "#0B0F14"
        if st.session_state.theme == "Dark"
        else "#F5F7FA"
    )

    if (
        "background_initialized" not in st.session_state
        or st.session_state.background_initialized
        != st.session_state.theme
    ):

        st.session_state.background_color = default_background

        st.session_state.background_initialized = (
            st.session_state.theme
        )

    st.session_state.background_color = st.color_picker(
        "Background color",
        value=st.session_state.background_color,
    )

    # --------------------------------------------------------
    # AI SETTINGS
    # --------------------------------------------------------

    st.divider()

    st.markdown("### AI Settings")

    technical_level = st.selectbox(
        "Technical level",
        [
            "Beginner",
            "Intermediate",
            "Advanced",
            "Legal / Technical",
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
        "Language",
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
            "Professional",
            "Simple",
            "Educational",
            "Direct",
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

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    st.divider()

    st.markdown("### Model")

    selected_model = st.selectbox(
        "Groq model",
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(
            st.session_state.selected_model
        )
        if st.session_state.selected_model
        in MODEL_OPTIONS
        else 0,
    )

    st.session_state.selected_model = (
        selected_model
    )

    # --------------------------------------------------------
    # USAGE
    # --------------------------------------------------------

    st.divider()

    st.markdown("### Usage")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Prompts",
            st.session_state.total_prompts,
        )

    with col2:
        st.metric(
            "Messages",
            st.session_state.total_messages,
        )

    st.metric(
        "Tokens",
        st.session_state.total_tokens,
    )

    # --------------------------------------------------------
    # KNOWLEDGE BASE
    # --------------------------------------------------------

    st.divider()

    st.markdown("### Knowledge Base")

    if st.session_state.knowledge_ready:

        st.success(
            f"Ready • {len(st.session_state.chunks)} chunks"
        )

    else:

        st.warning(
            "Knowledge base not built"
        )

    if st.button(
        "🔄 Build / Rebuild Knowledge Base",
        use_container_width=True,
    ):

        try:

            new_index, new_chunks = (
                build_knowledge_base()
            )

            st.session_state.index = new_index
            st.session_state.chunks = new_chunks
            st.session_state.knowledge_ready = True

            st.success(
                f"Knowledge base ready with {len(new_chunks)} chunks."
            )

        except Exception as e:

            st.error(
                f"Knowledge-base error: {str(e)}"
            )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="app-title">
        ⚖️ CyberLaw Pakistan AI
    </div>

    <div class="app-subtitle">
        Ask questions about Pakistani cyber law using a
        retrieval-augmented knowledge base.
    </div>

    <div class="accent-line"></div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# AI ASSISTANT PAGE
# ============================================================

if page == "AI Assistant":

    if not st.session_state.knowledge_ready:

        st.warning(
            "The knowledge base has not been built yet."
        )

        st.info(
            "Use **Build / Rebuild Knowledge Base** in the sidebar "
            "to download the supplied Google Drive PDF, extract its "
            "text, create embeddings, and build the FAISS index."
        )

    # --------------------------------------------------------
    # INTRO CARD
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="custom-card">

        <h3>Ask CyberLaw Pakistan AI</h3>

        <p>
        Ask questions about cyber-law concepts, legal terminology,
        cybercrime scenarios, reporting considerations, and other
        topics covered by the connected knowledge base.
        </p>

        <p>
        <strong>Important:</strong>
        This application provides information, not formal legal advice.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask a question about Pakistani cyber law..."
    )

    if question:

        question = question.strip()

        if not question:
            st.stop()

        # User message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        st.session_state.total_prompts += 1
        st.session_state.total_messages += 1

        with st.chat_message("user"):

            st.markdown(
                question
            )

        # Assistant
        with st.chat_message("assistant"):

            with st.spinner(
                "Searching the knowledge base and generating an answer..."
            ):

                answer, sources, token_count = ask_rag(
                    question=question,
                    technical_level=technical_level,
                    response_size=response_size,
                    language=language,
                    answer_style=answer_style,
                    top_k=top_k,
                    model_name=selected_model,
                )

            st.markdown(
                answer
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            st.session_state.total_messages += 1
            st.session_state.total_tokens += token_count

            if show_sources and sources:

                st.markdown(
                    "### Retrieved Sources"
                )

                for i, source in enumerate(
                    sources,
                    start=1,
                ):

                    with st.expander(
                        f"Source {i} • Page {source['page']} • Similarity {source['score']:.3f}"
                    ):

                        st.write(
                            source["text"]
                        )


# ============================================================
# COMPLAINT PAGE
# ============================================================

elif page == "File a Complaint":

    st.markdown(
        """
        <div class="custom-card">

        <h3>📝 Cybercrime Complaint Draft</h3>

        <p>
        Describe the incident and the AI will create a structured
        complaint draft using the available knowledge base.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    incident = st.text_area(
        "Describe the incident",
        height=250,
        placeholder=(
            "Example:\n"
            "My social media account was compromised and unauthorized "
            "messages were sent from my account..."
        ),
    )

    if st.button(
        "Generate Complaint Draft",
        type="primary",
        use_container_width=True,
    ):

        if not incident.strip():

            st.warning(
                "Please describe the incident first."
            )

        elif not st.session_state.knowledge_ready:

            st.warning(
                "Please build the knowledge base first."
            )

        else:

            with st.spinner(
                "Preparing complaint draft..."
            ):

                complaint, token_count = (
                    generate_complaint(
                        incident=incident,
                        language=language,
                        model_name=selected_model,
                    )
                )

            st.session_state.total_tokens += token_count

            st.markdown(
                "### Generated Draft"
            )

            st.markdown(
                complaint
            )

            st.download_button(
                "⬇️ Download Complaint Draft",
                data=complaint,
                file_name="cybercrime_complaint_draft.txt",
                mime="text/plain",
                use_container_width=True,
            )

            st.info(
                "Review all facts and legal references before submitting "
                "a complaint. The generated text is a draft and is not "
                "a substitute for professional legal advice."
            )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "About":

    st.markdown(
        """
        <div class="custom-card">

        <h2>About CyberLaw Pakistan AI</h2>

        <p>
        CyberLaw Pakistan AI is a Retrieval-Augmented Generation (RAG)
        application designed to answer questions using a connected
        cyber-law knowledge base.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="stat-card">

            <div class="stat-number">
                RAG
            </div>

            <div class="stat-label">
                Retrieval-Augmented Generation
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="stat-card">

            <div class="stat-number">
                FAISS
            </div>

            <div class="stat-label">
                Vector similarity search
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            """
            <div class="stat-card">

            <div class="stat-number">
                Groq
            </div>

            <div class="stat-label">
                LLM inference
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Technology Pipeline")

    st.markdown(
        """
        **1. Source PDF**

        The application downloads the configured Google Drive PDF.

        **2. Text Extraction**

        `pypdf` extracts readable text page-by-page.

        **3. Chunking**

        The document is divided into overlapping text chunks.

        **4. Embeddings**

        `all-MiniLM-L6-v2` converts chunks into vector embeddings.

        **5. Vector Search**

        FAISS retrieves the most relevant chunks for each question.

        **6. Generation**

        Groq generates an answer using the retrieved context.

        **7. Citation**

        Retrieved page numbers are included to make the answer traceable.
        """
    )

    st.markdown("### Privacy & API Key")

    st.info(
        "The Groq API key is not displayed in the user interface. "
        "Configure it using Streamlit secrets or the GROQ_API_KEY "
        "environment variable."
    )

    st.markdown("### Legal Disclaimer")

    st.warning(
        "This application is an AI information assistant. "
        "It does not provide formal legal advice, legal representation, "
        "or guaranteed legal conclusions. Always verify important legal "
        "matters with an appropriate qualified professional or official source."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        ⚖️ CyberLaw Pakistan AI • RAG + FAISS + Sentence Transformers + Groq
    </div>
    """,
    unsafe_allow_html=True,
)
