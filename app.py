```python
import os
import re
import html
import hashlib
from pathlib import Path

import requests
import faiss
import numpy as np
import streamlit as st

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# =========================================================
# APP CONFIG
# =========================================================

APP_TITLE = "CyberLaw Pakistan AI"

PDF_FILE_ID = "1d7xD2E1HrZBpkv16yHcqTb75C0MOzdmx"

PDF_URL = (
    f"https://drive.google.com/uc?export=download&id={PDF_FILE_ID}"
)

DATA_DIR = Path(".cyberlaw_data")
DATA_DIR.mkdir(exist_ok=True)

PDF_FILE = DATA_DIR / "pakistan_cyber_law.pdf"
INDEX_FILE = DATA_DIR / "cyberlaw.index"
CHUNKS_FILE = DATA_DIR / "chunks.npy"
META_FILE = DATA_DIR / "metadata.txt"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

if "page" not in st.session_state:
    st.session_state.page = "AI Assistant"


# =========================================================
# THEME
# =========================================================

def apply_theme(theme):

    if theme == "Dark":

        bg = "#080b12"
        card = "#111722"
        card2 = "#171e2b"
        text = "#f8fafc"
        muted = "#c5cedb"
        border = "#2b3545"
        accent = "#ff7a18"
        accent2 = "#ff9f43"
        input_bg = "#0d131d"

    else:

        bg = "#f5f7fb"
        card = "#ffffff"
        card2 = "#f0f3f8"
        text = "#111827"
        muted = "#596579"
        border = "#d9e0ea"
        accent = "#e85d04"
        accent2 = "#ff7a18"
        input_bg = "#ffffff"

    st.markdown(
        f"""
        <style>

        .stApp {{
            background: {bg};
        }}

        .main {{
            color: {text};
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}

        /* HERO */

        .hero {{
            background: {card};
            border: 1px solid {border};
            border-radius: 24px;
            padding: 32px;
            margin-bottom: 26px;
            animation: slideDown 0.55s ease;
            box-shadow: 0 15px 45px rgba(0, 0, 0, 0.08);
        }}

        .hero-title {{
            color: {text} !important;
            font-size: 42px;
            font-weight: 800;
            line-height: 1.2;
            letter-spacing: -1px;
            margin: 0;
        }}

        .hero-title span {{
            color: {accent} !important;
        }}

        .hero-subtitle {{
            color: {muted} !important;
            font-size: 16px;
            line-height: 1.7;
            margin-top: 12px;
            max-width: 850px;
        }}

        /* BADGES */

        .badge-container {{
            margin-top: 20px;
        }}

        .badge {{
            display: inline-block;
            color: {accent} !important;
            background: rgba(255, 122, 24, 0.12);
            border: 1px solid rgba(255, 122, 24, 0.30);
            padding: 6px 12px;
            border-radius: 50px;
            font-size: 12px;
            font-weight: 700;
            margin-right: 6px;
            margin-bottom: 5px;
        }}

        /* FEATURE CARDS */

        .feature-card {{
            background: {card};
            border: 1px solid {border};
            border-radius: 18px;
            padding: 20px;
            min-height: 145px;
            color: {text} !important;
            animation: slideUp 0.5s ease;
            transition: all 0.25s ease;
        }}

        .feature-card:hover {{
            transform: translateY(-4px);
            border-color: {accent};
        }}

        .feature-icon {{
            font-size: 28px;
            margin-bottom: 8px;
        }}

        .feature-title {{
            color: {text} !important;
            font-size: 17px;
            font-weight: 700;
        }}

        .feature-text {{
            color: {muted} !important;
            font-size: 13px;
            line-height: 1.6;
            margin-top: 6px;
        }}

        /* CARDS */

        .custom-card {{
            background: {card};
            border: 1px solid {border};
            border-radius: 20px;
            padding: 22px;
            color: {text} !important;
            animation: slideUp 0.5s ease;
        }}

        .custom-card h3,
        .custom-card h4,
        .custom-card p {{
            color: {text} !important;
        }}

        /* SIDEBAR */

        section[data-testid="stSidebar"] {{
            background: {card};
            border-right: 1px solid {border};
        }}

        section[data-testid="stSidebar"] * {{
            color: {text};
        }}

        /* CHAT */

        .user-message {{
            background: linear-gradient(
                135deg,
                {accent},
                {accent2}
            );
            color: #ffffff !important;
            padding: 15px 18px;
            border-radius: 18px 18px 5px 18px;
            margin: 12px 0 12px auto;
            max-width: 82%;
            animation: slideRight 0.35s ease;
        }}

        .user-message strong {{
            color: #ffffff !important;
        }}

        .assistant-message {{
            background: {card};
            border: 1px solid {border};
            color: {text} !important;
            padding: 18px;
            border-radius: 18px 18px 18px 5px;
            margin: 12px auto 12px 0;
            max-width: 92%;
            animation: slideLeft 0.35s ease;
        }}

        .assistant-message strong {{
            color: {accent} !important;
        }}

        /* SOURCES */

        .source-card {{
            background: {card2};
            border-left: 4px solid {accent};
            padding: 14px 16px;
            margin: 10px 0;
            border-radius: 10px;
            color: {text} !important;
        }}

        .source-title {{
            color: {text} !important;
            font-weight: 700;
        }}

        .source-page {{
            color: {muted} !important;
            font-size: 12px;
            margin-top: 3px;
        }}

        .source-text {{
            color: {text} !important;
            font-size: 13px;
            line-height: 1.6;
            margin-top: 10px;
        }}

        /* COMPLAINT */

        .complaint-header {{
            background:
                linear-gradient(
                    135deg,
                    rgba(255, 122, 24, 0.14),
                    {card}
                );
            border: 1px solid {border};
            border-radius: 22px;
            padding: 28px;
            margin-bottom: 25px;
            animation: slideDown 0.5s ease;
        }}

        .complaint-title {{
            color: {text} !important;
            font-size: 32px;
            font-weight: 800;
        }}

        .complaint-subtitle {{
            color: {muted} !important;
            font-size: 15px;
            line-height: 1.7;
            margin-top: 8px;
        }}

        /* INPUTS */

        .stTextInput input,
        .stTextArea textarea {{
            background: {input_bg} !important;
            color: {text} !important;
            border: 1px solid {border} !important;
            border-radius: 12px !important;
        }}

        .stTextInput input:focus,
        .stTextArea textarea:focus {{
            border-color: {accent} !important;
            box-shadow: 0 0 0 1px {accent} !important;
        }}

        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder {{
            color: {muted} !important;
            opacity: 0.8;
        }}

        /* BUTTONS */

        .stButton > button {{
            border-radius: 12px;
            border: 1px solid {border};
            min-height: 42px;
            transition: all 0.2s ease;
        }}

        .stButton > button:hover {{
            border-color: {accent};
            transform: translateY(-1px);
        }}

        /* SELECTBOX */

        div[data-baseweb="select"] > div {{
            background: {input_bg};
            border-color: {border};
        }}

        /* FOOTER */

        .footer {{
            text-align: center;
            color: {muted};
            padding: 35px 10px 10px;
            font-size: 12px;
        }}

        /* ANIMATIONS */

        @keyframes slideUp {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}

            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @keyframes slideDown {{
            from {{
                opacity: 0;
                transform: translateY(-20px);
            }}

            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @keyframes slideLeft {{
            from {{
                opacity: 0;
                transform: translateX(-20px);
            }}

            to {{
                opacity: 1;
                transform: translateX(0);
            }}
        }}

        @keyframes slideRight {{
            from {{
                opacity: 0;
                transform: translateX(20px);
            }}

            to {{
                opacity: 1;
                transform: translateX(0);
            }}
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )


apply_theme(st.session_state.theme)


# =========================================================
# GROQ KEY
# =========================================================

def get_groq_key():

    try:

        key = st.secrets.get("GROQ_API_KEY")

        if key:
            return key

    except Exception:
        pass

    return os.getenv("GROQ_API_KEY")


# =========================================================
# DOWNLOAD PDF FROM GOOGLE DRIVE
# =========================================================

def download_pdf():

    # Check cached PDF first
    if PDF_FILE.exists():

        try:

            if PDF_FILE.stat().st_size > 1000:

                with open(PDF_FILE, "rb") as file:

                    if file.read(5) == b"%PDF-":
                        return True

        except Exception:
            pass

    try:

        session = requests.Session()

        response = session.get(
            PDF_URL,
            timeout=60,
            allow_redirects=True
        )

        response.raise_for_status()

        content = response.content

        # Direct PDF
        if content[:5] == b"%PDF-":

            PDF_FILE.write_bytes(content)

            return True

        # Google Drive confirmation page
        html_content = content.decode(
            "utf-8",
            errors="ignore"
        )

        confirm_url = None

        patterns = [
            r'href="(/uc\?export=download[^"]+)"',
            r'href="(https://drive\.usercontent\.google\.com/download[^"]+)"',
            r'"downloadUrl":"([^"]+)"'
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                html_content
            )

            if match:

                confirm_url = match.group(1)

                confirm_url = (
                    confirm_url
                    .replace("&amp;", "&")
                    .replace("\\u003d", "=")
                    .replace("\\u0026", "&")
                )

                if confirm_url.startswith("/"):
                    confirm_url = (
                        "https://drive.google.com"
                        + confirm_url
                    )

                break

        if confirm_url:

            response = session.get(
                confirm_url,
                timeout=60,
                allow_redirects=True
            )

            response.raise_for_status()

            content = response.content

            if content[:5] == b"%PDF-":

                PDF_FILE.write_bytes(content)

                return True

        # Last fallback: try Drive alt=media endpoint
        fallback_url = (
            f"https://drive.google.com/uc"
            f"?export=download&id={PDF_FILE_ID}"
        )

        response = session.get(
            fallback_url,
            timeout=60,
            allow_redirects=True
        )

        response.raise_for_status()

        content = response.content

        if content[:5] == b"%PDF-":

            PDF_FILE.write_bytes(content)

            return True

        PDF_FILE.unlink(
            missing_ok=True
        )

        return False

    except Exception as error:

        PDF_FILE.unlink(
            missing_ok=True
        )

        st.error(
            f"PDF download error: {error}"
        )

        return False


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = text.replace(
        "\x00",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# EXTRACT PDF
# =========================================================

def extract_pdf():

    reader = PdfReader(
        str(PDF_FILE)
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:

            text = page.extract_text() or ""

        except Exception:

            text = ""

        text = clean_text(text)

        if text:

            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )

    return pages


# =========================================================
# CREATE LEGAL CHUNKS
# =========================================================

def create_chunks(
    pages,
    chunk_size=1000,
    overlap=180
):

    chunks = []

    for item in pages:

        text = item["text"]
        page = item["page"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end].strip()

            if len(chunk) > 80:

                chunks.append(
                    {
                        "text": chunk,
                        "page": page
                    }
                )

            next_start = end - overlap

            if next_start <= start:
                break

            start = next_start

    return chunks


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# =========================================================
# HASH
# =========================================================

def calculate_hash():

    sha = hashlib.sha256()

    with open(
        PDF_FILE,
        "rb"
    ) as file:

        while True:

            data = file.read(
                1024 * 1024
            )

            if not data:
                break

            sha.update(data)

    return sha.hexdigest()


# =========================================================
# BUILD VECTOR DATABASE
# =========================================================

def build_database(force=False):

    pdf_hash = calculate_hash()

    if (
        not force
        and INDEX_FILE.exists()
        and CHUNKS_FILE.exists()
        and META_FILE.exists()
    ):

        try:

            stored_hash = META_FILE.read_text(
                encoding="utf-8"
            ).strip()

            if stored_hash == pdf_hash:

                index = faiss.read_index(
                    str(INDEX_FILE)
                )

                chunks = np.load(
                    CHUNKS_FILE,
                    allow_pickle=True
                ).tolist()

                if index.ntotal == len(chunks):

                    return index, chunks

        except Exception:
            pass

    pages = extract_pdf()

    if not pages:

        raise ValueError(
            "No readable text was found in the PDF."
        )

    chunks = create_chunks(
        pages
    )

    if not chunks:

        raise ValueError(
            "Unable to create text chunks."
        )

    model = load_embedding_model()

    texts = [
        item["text"]
        for item in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    np.save(
        CHUNKS_FILE,
        np.array(
            chunks,
            dtype=object
        )
    )

    META_FILE.write_text(
        pdf_hash,
        encoding="utf-8"
    )

    return index, chunks


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve(
    question,
    index,
    chunks,
    top_k=5
):

    model = load_embedding_model()

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        min(
            top_k,
            len(chunks)
        )
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        results.append(
            {
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "score": float(score)
            }
        )

    return results


# =========================================================
# CONTEXT
# =========================================================

def create_context(results):

    context = []

    for i, item in enumerate(
        results,
        start=1
    ):

        context.append(
            f"""
[Source {i}, Page {item["page"]}]
{item["text"]}
"""
        )

    return "\n".join(context)


# =========================================================
# SYSTEM PROMPT
# =========================================================

def create_system_prompt(
    technical_level,
    response_size,
    language,
    answer_style
):

    return f"""
You are CyberLaw Pakistan AI.

You are a legal-information RAG assistant focused
on Pakistani cyber-law information.

USER PREFERENCES
----------------
Technical level: {technical_level}
Response size: {response_size}
Language: {language}
Answer style: {answer_style}

CORE RULE
---------
Use the retrieved legal document as your primary
knowledge source.

Do not treat your general model knowledge as evidence
for a specific Pakistani legal claim.

LEGAL ACCURACY
--------------
1. Never invent:

- sections
- subsections
- laws
- penalties
- authorities
- procedures
- legal claims

2. Mention section numbers only when supported by
the retrieved evidence.

3. If the evidence is insufficient, explicitly say:

"The available source does not provide enough
information to answer this confidently."

4. Separate your answer into appropriate parts such as:

- Legal provision
- Explanation
- Practical implication

5. Cite important claims using:

[Source X, Page Y]

6. Do not present the response as an official legal
opinion.

7. Recommend verification with a qualified lawyer
or relevant official authority when appropriate.

8. Do not claim that a law is currently in force unless
the retrieved material supports that conclusion.

CYBER SAFETY
------------
Do not provide operational instructions for:

- hacking
- credential theft
- malware
- unauthorized access
- cyber fraud
- bypassing security
- evading law enforcement

For harmful requests:

- refuse the operational instructions
- provide lawful defensive information
- provide relevant legal information when supported

COMPLAINTS
----------
If the user requests a cybercrime complaint draft:

Create a professional structured draft.

Use ONLY facts supplied by the user.

Use ONLY legal provisions supported by retrieved evidence.

Never fabricate facts.

Never fabricate section numbers.

Never fabricate authorities.

Start with:

AI-generated complaint draft — verify before submission.

Include where appropriate:

- Subject
- Complainant statement
- Incident details
- Platform
- Date/time
- Evidence
- Relevant legal provision
- Requested action
- Declaration
- Missing information

Never claim that the complaint has already been
submitted to an authority.

LANGUAGE
--------
Answer in the requested language.

If Urdu or Roman Urdu is selected, do not unnecessarily
switch to English.

RESPONSE STYLE
--------------
Follow the requested response size and answer style.

Be clear, accurate, and structured.
"""


# =========================================================
# MAX TOKENS
# =========================================================

def get_max_tokens(response_size):

    mapping = {
        "Short": 800,
        "Medium": 1400,
        "Detailed": 2200,
        "Very detailed": 3200
    }

    return mapping.get(
        response_size,
        1400
    )


# =========================================================
# GROQ
# =========================================================

def ask_groq(
    question,
    context,
    technical_level,
    response_size,
    language,
    answer_style,
    model_name
):

    api_key = get_groq_key()

    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    client = Groq(
        api_key=api_key
    )

    system_prompt = create_system_prompt(
        technical_level,
        response_size,
        language,
        answer_style
    )

    user_prompt = f"""
RETRIEVED LEGAL EVIDENCE
========================

{context}

USER QUESTION
=============

{question}

INSTRUCTIONS
============

Answer the user's question using the retrieved
legal evidence.

Cite important claims.

Do not invent unsupported information.

If the evidence is insufficient, say so clearly.
"""

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.1,
        max_tokens=get_max_tokens(
            response_size
        ),
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response.choices[0].message.content


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:26px;
            font-weight:800;
            margin-bottom:3px;
        ">
            ⚖️ CyberLaw
        </div>

        <div style="
            font-size:13px;
            opacity:0.7;
            margin-bottom:25px;
        ">
            Pakistan Legal AI
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Navigation")

    selected_page = st.radio(
        "Navigation",
        [
            "AI Assistant",
            "File a Complaint",
            "About"
        ],
        index=[
            "AI Assistant",
            "File a Complaint",
            "About"
        ].index(
            st.session_state.page
        ),
        label_visibility="collapsed"
    )

    if selected_page != st.session_state.page:

        st.session_state.page = selected_page

        st.rerun()

    st.divider()

    st.markdown("### Appearance")

    selected_theme = st.radio(
        "Theme",
        [
            "Dark",
            "Light"
        ],
        index=(
            0
            if st.session_state.theme == "Dark"
            else 1
        ),
        horizontal=True,
        label_visibility="collapsed"
    )

    if selected_theme != st.session_state.theme:

        st.session_state.theme = selected_theme

        st.rerun()

    st.divider()

    st.markdown("### AI Settings")

    technical_level = st.selectbox(
        "Technical level",
        [
            "Beginner",
            "Intermediate",
            "Advanced",
            "Legal / Professional"
        ]
    )

    response_size = st.selectbox(
        "Response size",
        [
            "Short",
            "Medium",
            "Detailed",
            "Very detailed"
        ],
        index=1
    )

    language = st.selectbox(
        "Language",
        [
            "English",
            "Urdu",
            "Roman Urdu"
        ]
    )

    answer_style = st.selectbox(
        "Answer style",
        [
            "Simple explanation",
            "Legal analysis",
            "Step-by-step",
            "Practical scenario"
        ]
    )

    top_k = st.slider(
        "Retrieved sources",
        min_value=3,
        max_value=10,
        value=5
    )

    model_name = st.text_input(
        "Groq model",
        value=DEFAULT_MODEL
    )

    show_sources = st.checkbox(
        "Show retrieved sources",
        value=True
    )

    st.divider()

    if st.button(
        "🔄 Rebuild knowledge base",
        use_container_width=True
    ):

        for file in [
            INDEX_FILE,
            CHUNKS_FILE,
            META_FILE
        ]:

            file.unlink(
                missing_ok=True
            )

        st.cache_resource.clear()

        st.success(
            "Knowledge base rebuilt on next load."
        )

        st.rerun()

    if st.button(
        "🗑️ Clear chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# PDF + DATABASE
# =========================================================

if not download_pdf():

    st.error(
        """
        ### PDF download failed

        Make sure the Google Drive file is configured as:

        **Anyone with the link → Viewer**

        The application cannot create the RAG knowledge
        base until the PDF is publicly accessible.
        """
    )

    st.info(
        f"""
        Google Drive file ID:

        `{PDF_FILE_ID}`
        """
    )

    st.stop()


try:

    index, chunks = build_database()

except Exception as error:

    st.error(
        f"Knowledge base error: {error}"
    )

    st.stop()


# =========================================================
# AI ASSISTANT
# =========================================================

if st.session_state.page == "AI Assistant":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                🇵🇰 <span>CyberLaw</span> Pakistan AI
            </div>

            <div class="hero-subtitle">
                Ask questions about Pakistani cyber law,
                retrieve relevant legal provisions, and
                understand them using source-based AI.
            </div>

            <div class="badge-container">

                <span class="badge">RAG Powered</span>

                <span class="badge">Pakistan Law</span>

                <span class="badge">Groq AI</span>

                <span class="badge">Source Based</span>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # FEATURES
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-icon">⚖️</div>

                <div class="feature-title">
                    Legal Analysis
                </div>

                <div class="feature-text">
                    Understand Pakistani cyber-law
                    information using the indexed source.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-icon">🔎</div>

                <div class="feature-title">
                    RAG Search
                </div>

                <div class="feature-text">
                    Search relevant legal passages
                    before generating an answer.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-icon">🌐</div>

                <div class="feature-title">
                    Multiple Languages
                </div>

                <div class="feature-text">
                    Get explanations in English,
                    Urdu or Roman Urdu.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-icon">📝</div>

                <div class="feature-title">
                    Complaint Draft
                </div>

                <div class="feature-text">
                    Generate a structured
                    AI-assisted complaint draft.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 💬 Ask CyberLaw Pakistan AI")

    # =====================================================
    # CHAT HISTORY
    # =====================================================

    for message in st.session_state.messages:

        if message["role"] == "user":

            safe_content = html.escape(
                message["content"]
            )

            safe_content = safe_content.replace(
                "\n",
                "<br>"
            )

            st.markdown(
                f"""
                <div class="user-message">
                    <strong>You</strong>
                    <br><br>
                    {safe_content}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="assistant-message">',
                unsafe_allow_html=True
            )

            st.markdown(
                "**⚖️ CyberLaw AI**"
            )

            st.markdown(
                message["content"]
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            if (
                show_sources
                and message.get("sources")
            ):

                with st.expander(
                    "📚 Retrieved legal sources"
                ):

                    for i, source in enumerate(
                        message["sources"],
                        start=1
                    ):

                        st.markdown(
                            f"""
                            <div class="source-card">

                                <div class="source-title">
                                    Source {i}
                                </div>

                                <div class="source-page">
                                    Page {source["page"]}
                                    · Similarity:
                                    {source["score"]:.3f}
                                </div>

                                <div class="source-text">
                                    {html.escape(source["text"])}
                                </div>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    # =====================================================
    # CHAT INPUT
    # =====================================================

    question = st.chat_input(
        "Ask a question about Pakistani cyber law..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.spinner(
            "🔎 Searching legal sources..."
        ):

            try:

                results = retrieve(
                    question,
                    index,
                    chunks,
                    top_k
                )

                context = create_context(
                    results
                )

                answer = ask_groq(
                    question,
                    context,
                    technical_level,
                    response_size,
                    language,
                    answer_style,
                    model_name
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": results
                    }
                )

            except Exception as error:

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"⚠️ Error: {error}"
                        ),
                        "sources": []
                    }
                )

        st.rerun()


# =========================================================
# COMPLAINT PAGE
# =========================================================

elif st.session_state.page == "File a Complaint":

    st.markdown(
        """
        <div class="complaint-header">

            <div class="complaint-title">
                📝 Cybercrime Complaint Assistant
            </div>

            <div class="complaint-subtitle">
                Describe your cyber incident and generate
                an AI-assisted structured complaint draft.
                Verify all information before official
                submission.
            </div>

            <div class="badge-container">

                <span class="badge">AI Draft</span>

                <span class="badge">Evidence Based</span>

                <span class="badge">Pakistan</span>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Incident Information")

    col1, col2 = st.columns(2)

    with col1:

        incident_type = st.selectbox(
            "Incident type",
            [
                "Select incident type",
                "Online harassment",
                "Cyberstalking",
                "Fake social media account",
                "Identity misuse",
                "Online fraud",
                "Threat / blackmail",
                "Unauthorized account access",
                "Data / privacy issue",
                "Other"
            ]
        )

        platform = st.text_input(
            "Platform / website",
            placeholder=(
                "e.g. Facebook, WhatsApp, Instagram"
            )
        )

        incident_date = st.text_input(
            "Date of incident",
            placeholder=(
                "e.g. 10 September 2026"
            )
        )

    with col2:

        suspect_known = st.selectbox(
            "Is the person involved known?",
            [
                "No",
                "Yes",
                "Partially"
            ]
        )

        evidence = st.text_area(
            "Evidence available",
            placeholder=(
                "Screenshots, URLs, messages, "
                "emails, transaction records, etc."
            ),
            height=120
        )

    incident_description = st.text_area(
        "Describe what happened",
        placeholder=(
            "Explain the incident in your own words. "
            "Include what happened, where it happened, "
            "who was involved, and relevant details."
        ),
        height=220
    )

    additional_info = st.text_area(
        "Additional information",
        placeholder=(
            "Add any other relevant facts..."
        ),
        height=120
    )

    generate_complaint = st.button(
        "⚖️ Generate Complaint Draft",
        type="primary",
        use_container_width=True
    )

    if generate_complaint:

        if not incident_description.strip():

            st.warning(
                "Please describe what happened first."
            )

        else:

            complaint_question = f"""
Create a structured AI-generated cybercrime
complaint draft.

Incident type:
{incident_type}

Platform:
{platform}

Date:
{incident_date}

Known suspect:
{suspect_known}

Evidence:
{evidence}

Incident description:
{incident_description}

Additional information:
{additional_info}

Requirements:

1. Start with:

AI-generated complaint draft — verify before submission.

2. Identify potentially relevant Pakistani cyber-law
provisions ONLY when supported by retrieved evidence.

3. Do not invent section numbers.

4. Do not invent authorities.

5. Do not invent facts.

6. Create a professional complaint structure.

7. Include:

- Subject
- Complainant statement
- Incident details
- Platform
- Date/time
- Evidence
- Relevant legal provision
- Requested action
- Declaration

8. Identify information that still needs to be
filled in by the complainant.

9. If the retrieved evidence is insufficient to identify
a legal provision, explicitly state that.
"""

            with st.spinner(
                "⚖️ Preparing complaint draft..."
            ):

                try:

                    results = retrieve(
                        complaint_question,
                        index,
                        chunks,
                        top_k
                    )

                    context = create_context(
                        results
                    )

                    complaint = ask_groq(
                        complaint_question,
                        context,
                        "Legal / Professional",
                        "Detailed",
                        language,
                        "Legal analysis",
                        model_name
                    )

                    st.success(
                        "Complaint draft generated."
                    )

                    st.markdown(
                        '<div class="custom-card">',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        complaint
                    )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )

                    with st.expander(
                        "📚 Legal sources used"
                    ):

                        for i, source in enumerate(
                            results,
                            start=1
                        ):

                            st.markdown(
                                f"""
                                **Source {i} — Page {source["page"]}**

                                {source["text"]}
                                """
                            )

                except Exception as error:

                    st.error(
                        f"Unable to generate complaint: {error}"
                    )

    st.info(
        """
        **Important:** This tool generates an
        AI-assisted draft. It is not an official
        legal complaint or legal opinion.

        Verify the facts, applicable legal provisions,
        and official submission procedure before
        submitting a complaint.
        """
    )


# =========================================================
# ABOUT
# =========================================================

elif st.session_state.page == "About":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                ⚖️ About
                <span>CyberLaw Pakistan AI</span>
            </div>

            <div class="hero-subtitle">
                An AI-powered Retrieval-Augmented Generation
                application for understanding Pakistani
                cyber-law information.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### How it works")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="custom-card">

                <h3>01 · Document</h3>

                <p>
                    The application downloads the configured
                    Pakistan cyber-law PDF.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="custom-card">

                <h3>02 · Retrieval</h3>

                <p>
                    The PDF is split into chunks and
                    converted into vector embeddings.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="custom-card">

                <h3>03 · Generation</h3>

                <p>
                    Relevant legal passages are provided
                    to Groq before generating an answer.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Technology")

    st.markdown(
        """
        - **Python**
        - **Streamlit**
        - **FAISS**
        - **Sentence Transformers**
        - **Groq**
        - **PyPDF**
        - **Requests**
        - **Retrieval-Augmented Generation**
        """
    )

    st.markdown("### Legal disclaimer")

    st.warning(
        """
        CyberLaw Pakistan AI provides informational
        assistance based on its indexed source material.

        It does not replace a qualified lawyer,
        official government guidance, or an official
        legal determination.

        Always verify current Pakistani law and official
        procedures before taking legal action.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

        ⚖️ CyberLaw Pakistan AI

        <br>

        AI-powered legal information assistant

        <br><br>

        Python · Streamlit · FAISS · Groq

    </div>
    """,
    unsafe_allow_html=True
)
```
