import os
import re
import hashlib
import urllib.request
from pathlib import Path

import faiss
import numpy as np
import streamlit as st

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# =========================================================
# CONFIGURATION
# =========================================================

APP_TITLE = "CyberLaw Pakistan AI"

PDF_URL = (
    "https://drive.google.com/uc?export=download"
    "&id=1d7xD2E1HrZBpkv16yHcqTb75C0MOzdmx"
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
# CUSTOM CSS
# =========================================================

def apply_theme(theme):

    if theme == "Dark":
        bg = "#09090f"
        card = "#11131c"
        card2 = "#171925"
        text = "#f5f7ff"
        muted = "#9ca3b5"
        border = "#292d3d"
        accent = "#ff7a18"
        accent2 = "#ff9f43"
        input_bg = "#10121a"

    else:
        bg = "#f5f7fb"
        card = "#ffffff"
        card2 = "#f0f2f7"
        text = "#171923"
        muted = "#687083"
        border = "#dfe3ec"
        accent = "#e85d04"
        accent2 = "#ff7a18"
        input_bg = "#ffffff"

    st.markdown(
        f"""
        <style>

        /* =========================
           GLOBAL
        ========================= */

        .stApp {{
            background:
                radial-gradient(
                    circle at 10% 10%,
                    rgba(255,122,24,0.08),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 90% 80%,
                    rgba(255,159,67,0.06),
                    transparent 30%
                ),
                {bg};
            color: {text};
        }}

        html, body, [class*="css"] {{
            font-family:
                Inter,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;
        }}

        /* =========================
           SIDEBAR
           ========================= */

        section[data-testid="stSidebar"] {{
            background: {card};
            border-right: 1px solid {border};
        }}

        section[data-testid="stSidebar"] > div {{
            padding-top: 1.5rem;
        }}

        /* =========================
           HEADER
           ========================= */

        .hero {{
            padding: 30px;
            border-radius: 24px;
            background:
                linear-gradient(
                    135deg,
                    {card},
                    {card2}
                );
            border: 1px solid {border};
            margin-bottom: 25px;
            animation: slideDown 0.6s ease;
            box-shadow:
                0 15px 45px rgba(0,0,0,0.08);
        }}

        .hero-title {{
            font-size: 42px;
            font-weight: 800;
            margin: 0;
            letter-spacing: -1px;
        }}

        .hero-title span {{
            color: {accent};
        }}

        .hero-subtitle {{
            margin-top: 10px;
            color: {muted};
            font-size: 16px;
            line-height: 1.6;
        }}

        /* =========================
           CARDS
           ========================= */

        .card {{
            background: {card};
            border: 1px solid {border};
            border-radius: 20px;
            padding: 22px;
            margin-bottom: 18px;
            animation: slideUp 0.5s ease;
            transition: all 0.25s ease;
        }}

        .card:hover {{
            transform: translateY(-2px);
            border-color: {accent};
            box-shadow:
                0 12px 30px rgba(0,0,0,0.08);
        }}

        .feature-card {{
            background:
                linear-gradient(
                    145deg,
                    {card},
                    {card2}
                );
            border: 1px solid {border};
            border-radius: 18px;
            padding: 20px;
            min-height: 130px;
            animation: slideUp 0.5s ease;
        }}

        .feature-icon {{
            font-size: 28px;
        }}

        .feature-title {{
            font-size: 17px;
            font-weight: 700;
            margin-top: 8px;
        }}

        .feature-text {{
            color: {muted};
            font-size: 13px;
            margin-top: 6px;
            line-height: 1.5;
        }}

        /* =========================
           CHAT
           ========================= */

        .user-message {{
            background:
                linear-gradient(
                    135deg,
                    {accent},
                    {accent2}
                );
            color: white;
            padding: 15px 18px;
            border-radius: 18px 18px 5px 18px;
            margin: 10px 0 10px auto;
            max-width: 80%;
            animation: slideRight 0.35s ease;
        }}

        .assistant-message {{
            background: {card};
            border: 1px solid {border};
            color: {text};
            padding: 18px;
            border-radius: 18px 18px 18px 5px;
            margin: 10px auto 10px 0;
            max-width: 90%;
            animation: slideLeft 0.35s ease;
        }}

        /* =========================
           SOURCE CARDS
           ========================= */

        .source-card {{
            background: {card2};
            border-left: 4px solid {accent};
            padding: 12px 15px;
            margin: 8px 0;
            border-radius: 10px;
            color: {text};
            font-size: 13px;
        }}

        .source-title {{
            font-weight: 700;
        }}

        .source-page {{
            color: {muted};
            font-size: 12px;
        }}

        /* =========================
           COMPLAINT
           ========================= */

        .complaint-header {{
            padding: 25px;
            border-radius: 20px;
            background:
                linear-gradient(
                    135deg,
                    rgba(255,122,24,0.14),
                    {card}
                );
            border: 1px solid {border};
            animation: slideDown 0.5s ease;
        }}

        .complaint-title {{
            font-size: 30px;
            font-weight: 800;
        }}

        .complaint-subtitle {{
            color: {muted};
            line-height: 1.6;
        }}

        /* =========================
           BADGES
           ========================= */

        .badge {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(255,122,24,0.12);
            color: {accent};
            border: 1px solid rgba(255,122,24,0.25);
            font-size: 12px;
            font-weight: 700;
            margin-right: 6px;
        }}

        /* =========================
           ANIMATIONS
           ========================= */

        @keyframes slideUp {{
            from {{
                opacity: 0;
                transform: translateY(25px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @keyframes slideDown {{
            from {{
                opacity: 0;
                transform: translateY(-25px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @keyframes slideLeft {{
            from {{
                opacity: 0;
                transform: translateX(-25px);
            }}
            to {{
                opacity: 1;
                transform: translateX(0);
            }}
        }}

        @keyframes slideRight {{
            from {{
                opacity: 0;
                transform: translateX(25px);
            }}
            to {{
                opacity: 1;
                transform: translateX(0);
            }}
        }}

        /* =========================
           BUTTONS
           ========================= */

        .stButton > button {{
            border-radius: 12px;
            border: 1px solid {border};
            transition: all 0.2s ease;
        }}

        .stButton > button:hover {{
            border-color: {accent};
            transform: translateY(-1px);
        }}

        /* =========================
           INPUT
           ========================= */

        .stTextInput input,
        .stTextArea textarea {{
            background: {input_bg} !important;
            color: {text} !important;
            border: 1px solid {border} !important;
            border-radius: 12px !important;
        }}

        /* =========================
           FOOTER
           ========================= */

        .footer {{
            text-align: center;
            color: {muted};
            padding: 30px;
            font-size: 12px;
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
# DOWNLOAD PDF
# =========================================================

def download_pdf():

    if PDF_FILE.exists() and PDF_FILE.stat().st_size > 1000:
        return True

    try:
        urllib.request.urlretrieve(
            PDF_URL,
            PDF_FILE
        )

        # Basic validation
        with open(PDF_FILE, "rb") as f:
            header = f.read(5)

        if header != b"%PDF-":
            PDF_FILE.unlink(missing_ok=True)
            return False

        return True

    except Exception:
        PDF_FILE.unlink(missing_ok=True)
        return False


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_pdf():

    reader = PdfReader(str(PDF_FILE))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        text = clean_text(text)

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


# =========================================================
# CHUNKING
# =========================================================

def create_chunks(
    pages,
    chunk_size=900,
    overlap=150
):

    chunks = []

    for item in pages:

        text = item["text"]
        page = item["page"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            if len(chunk_text.strip()) > 80:

                chunks.append(
                    {
                        "text": chunk_text.strip(),
                        "page": page,
                    }
                )

            start += chunk_size - overlap

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

    with open(PDF_FILE, "rb") as f:

        while True:

            data = f.read(1024 * 1024)

            if not data:
                break

            sha.update(data)

    return sha.hexdigest()


# =========================================================
# BUILD DATABASE
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

                return index, chunks

        except Exception:
            pass

    pages = extract_pdf()

    chunks = create_chunks(pages)

    if not chunks:
        raise ValueError(
            "No readable text was found in the PDF."
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
        show_progress_bar=False,
    )

    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    np.save(
        CHUNKS_FILE,
        np.array(chunks, dtype=object)
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
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        item = chunks[idx]

        results.append(
            {
                "text": item["text"],
                "page": item["page"],
                "score": float(score),
            }
        )

    return results


# =========================================================
# CONTEXT
# =========================================================

def create_context(results):

    context_parts = []

    for i, item in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"""
[Source {i}, Page {item["page"]}]
{item["text"]}
"""
        )

    return "\n".join(
        context_parts
    )


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
You are CyberLaw Pakistan AI, a legal-information
RAG assistant focused on Pakistani cyber law.

Your answers must be grounded in the retrieved
document provided by the application.

USER PREFERENCES
----------------
Technical level: {technical_level}
Response size: {response_size}
Language: {language}
Answer style: {answer_style}

IMPORTANT LEGAL RULES
--------------------
1. Do not invent laws, sections, penalties,
   authorities, procedures, or legal claims.

2. Only mention a section number when supported
   by the retrieved evidence.

3. If the retrieved document does not contain
   enough information, clearly say so.

4. Distinguish between:
   - Legal provision
   - Explanation
   - Practical implication

5. Cite relevant evidence using:
   [Source X, Page Y]

6. Do not claim that an AI response is an official
   legal opinion.

7. For legal uncertainty, recommend consulting
   a qualified Pakistani lawyer or the relevant
   official authority.

CYBER SAFETY
------------
Do not provide instructions for:
- hacking accounts
- malware creation
- credential theft
- unauthorized access
- cyber fraud
- bypassing security
- evading law enforcement

If a user asks for harmful cyber instructions,
refuse the operational instructions and instead
provide lawful defensive or legal information.

COMPLAINTS
----------
If the user asks for help preparing a cybercrime
complaint, create a structured complaint draft
based only on the facts supplied by the user and
legal provisions supported by the retrieved evidence.

Clearly label it:
"AI-generated complaint draft — verify before submission."

Do not fabricate names, dates, evidence, sections,
or authorities.
"""


# =========================================================
# GROQ REQUEST
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
        answer_style,
    )

    user_prompt = f"""
Retrieved legal evidence:

{context}

User question:

{question}

Answer using the retrieved evidence.
Cite important claims.
"""

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.1,
        max_tokens=2500,
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
    )

    return response.choices[0].message.content


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:24px;
            font-weight:800;
            margin-bottom:5px;
        ">
        ⚖️ CyberLaw
        </div>

        <div style="
            color:#9ca3b5;
            font-size:13px;
            margin-bottom:25px;
        ">
        Pakistan Legal AI
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Navigation")

    page = st.radio(
        "Select page",
        [
            "AI Assistant",
            "File a Complaint",
            "About",
        ],
        label_visibility="collapsed",
    )

    st.session_state.page = page

    st.divider()

    st.markdown("### Appearance")

    theme = st.radio(
        "Theme",
        ["Dark", "Light"],
        index=(
            0
            if st.session_state.theme == "Dark"
            else 1
        ),
        horizontal=True,
        label_visibility="collapsed",
    )

    if theme != st.session_state.theme:

        st.session_state.theme = theme

        st.rerun()

    st.divider()

    st.markdown("### AI Settings")

    technical_level = st.selectbox(
        "Technical level",
        [
            "Beginner",
            "Intermediate",
            "Advanced",
            "Legal / Professional",
        ],
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
    )

    answer_style = st.selectbox(
        "Answer style",
        [
            "Simple explanation",
            "Legal analysis",
            "Step-by-step",
            "Practical scenario",
        ],
    )

    top_k = st.slider(
        "Retrieved sources",
        3,
        10,
        5,
    )

    model_name = st.text_input(
        "Groq model",
        value=DEFAULT_MODEL,
    )

    show_sources = st.checkbox(
        "Show retrieved sources",
        value=True,
    )

    st.divider()

    if st.button(
        "🔄 Rebuild knowledge base",
        use_container_width=True,
    ):

        if INDEX_FILE.exists():
            INDEX_FILE.unlink()

        if CHUNKS_FILE.exists():
            CHUNKS_FILE.unlink()

        if META_FILE.exists():
            META_FILE.unlink()

        st.cache_resource.clear()

        st.success(
            "Knowledge base marked for rebuild."
        )

        st.rerun()

    if st.button(
        "🗑️ Clear chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

if not download_pdf():

    st.error(
        """
        Unable to download the Pakistan cyber-law PDF.

        Check that the Google Drive file is shared as:

        **Anyone with the link → Viewer**
        """
    )

    st.stop()


try:

    index, chunks = build_database()

except Exception as e:

    st.error(
        f"Knowledge base error: {e}"
    )

    st.stop()


# =========================================================
# AI ASSISTANT PAGE
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
                understand them in simple language.
            </div>

            <br>

            <span class="badge">RAG Powered</span>
            <span class="badge">Pakistan Law</span>
            <span class="badge">Groq AI</span>
            <span class="badge">Source Based</span>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature cards

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">⚖️</div>
                <div class="feature-title">
                    Legal Analysis
                </div>
                <div class="feature-text">
                    Understand cyber-law provisions
                    from the indexed legal document.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🔎</div>
                <div class="feature-title">
                    RAG Search
                </div>
                <div class="feature-text">
                    Retrieve relevant pages before
                    generating an answer.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🌐</div>
                <div class="feature-title">
                    Multiple Languages
                </div>
                <div class="feature-text">
                    Ask and receive explanations in
                    English, Urdu or Roman Urdu.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📝</div>
                <div class="feature-title">
                    Complaint Draft
                </div>
                <div class="feature-text">
                    Convert incident details into a
                    structured complaint draft.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 💬 Ask CyberLaw Pakistan AI")

    # Chat history

    for message in st.session_state.messages:

        if message["role"] == "user":

            st.markdown(
                f"""
                <div class="user-message">
                    <strong>You</strong><br>
                    {message["content"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                f"""
                <div class="assistant-message">
                    <strong>⚖️ CyberLaw AI</strong><br><br>
                    {message["content"]}
                </div>
                """,
                unsafe_allow_html=True,
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
                        start=1,
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

                            <br>

                            {source["text"]}

                            </div>
                            """,
                            unsafe_allow_html=True,
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

        with st.spinner(
            "🔎 Searching legal sources and generating answer..."
        ):

            try:

                results = retrieve(
                    question,
                    index,
                    chunks,
                    top_k,
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
                    model_name,
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": results,
                    }
                )

            except Exception as e:

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content":
                            f"⚠️ Unable to process the request: {e}",
                        "sources": [],
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
                a structured AI-assisted complaint draft.
                The draft should be reviewed and verified
                before official submission.
            </div>

            <br>

            <span class="badge">AI Draft</span>
            <span class="badge">Evidence Based</span>
            <span class="badge">Pakistan</span>

        </div>
        """,
        unsafe_allow_html=True,
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
                "Data/privacy issue",
                "Other",
            ],
        )

        platform = st.text_input(
            "Platform / website",
            placeholder="e.g. Facebook, WhatsApp, Instagram"
        )

        incident_date = st.text_input(
            "Date of incident",
            placeholder="e.g. 10 September 2026"
        )

    with col2:

        suspect_known = st.selectbox(
            "Do you know the person involved?",
            [
                "No",
                "Yes",
                "Partially",
            ],
        )

        evidence = st.text_area(
            "Evidence available",
            placeholder=(
                "Screenshots, URLs, messages, "
                "emails, transaction records, etc."
            ),
            height=120,
        )

    incident_description = st.text_area(
        "Describe what happened",
        placeholder=(
            "Explain the incident in your own words. "
            "Include what happened, who was involved, "
            "where it happened and how it affected you."
        ),
        height=220,
    )

    st.markdown("### Optional Information")

    additional_info = st.text_area(
        "Additional information",
        placeholder=(
            "Any other relevant facts..."
        ),
        height=120,
    )

    generate_complaint = st.button(
        "⚖️ Generate Complaint Draft",
        type="primary",
        use_container_width=True,
    )

    if generate_complaint:

        if not incident_description.strip():

            st.warning(
                "Please describe what happened first."
            )

        else:

            complaint_question = f"""
Create a structured AI-generated cybercrime
complaint draft based on the following incident.

Incident type:
{incident_type}

Platform:
{platform}

Date:
{incident_date}

Suspect information:
{suspect_known}

Evidence:
{evidence}

Incident description:
{incident_description}

Additional information:
{additional_info}

Requirements:

1. Clearly label this as:
AI-generated complaint draft — verify before submission.

2. Identify potentially relevant Pakistani cyber-law
provisions ONLY if supported by the retrieved evidence.

3. Do not invent section numbers.

4. Do not invent authorities.

5. Do not invent facts.

6. Structure the complaint professionally.

7. Include:
- Subject
- Complainant statement
- Incident details
- Evidence
- Relevant legal provision, if supported
- Requested action
- Declaration

8. Explain which information the complainant still
needs to fill in.
"""

            with st.spinner(
                "Preparing complaint draft..."
            ):

                try:

                    results = retrieve(
                        complaint_question,
                        index,
                        chunks,
                        top_k,
                    )

                    context = create_context(
                        results
                    )

                    complaint = ask_groq(
                        complaint_question,
                        context,
                        "Legal / Professional",
                        "Detailed",
                        "English",
                        "Legal analysis",
                        model_name,
                    )

                    st.success(
                        "Complaint draft generated."
                    )

                    st.markdown(
                        '<div class="card">',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        complaint
                    )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    with st.expander(
                        "📚 Legal sources used"
                    ):

                        for i, source in enumerate(
                            results,
                            start=1,
                        ):

                            st.markdown(
                                f"""
                                **Source {i} — Page {source["page"]}**

                                {source["text"]}
                                """
                            )

                except Exception as e:

                    st.error(
                        f"Unable to generate complaint: {e}"
                    )

    st.info(
        """
        **Important:** This feature creates an AI-assisted
        draft. It is not an official legal complaint or
        legal opinion. Verify the facts, legal provisions,
        and submission procedure with the relevant authority
        or a qualified Pakistani lawyer.
        """
    )


# =========================================================
# ABOUT PAGE
# =========================================================

elif st.session_state.page == "About":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                ⚖️ About <span>CyberLaw Pakistan AI</span>
            </div>

            <div class="hero-subtitle">
                An AI-powered retrieval-augmented generation
                application for understanding Pakistani
                cyber-law information.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### How it works")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="card">

            ### 01 · Document

            The application downloads the configured
            Pakistan cyber-law PDF.

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="card">

            ### 02 · Retrieval

            The PDF is split into chunks and converted
            into vector embeddings for semantic search.

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            """
            <div class="card">

            ### 03 · Generation

            Relevant legal passages are provided to
            Groq before generating the response.

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Technology")

    st.markdown(
        """
        - **Streamlit** — User interface
        - **Python** — Application backend
        - **FAISS** — Vector similarity search
        - **Sentence Transformers** — Embeddings
        - **Groq** — LLM generation
        - **PyPDF** — PDF extraction
        - **RAG** — Retrieval-Augmented Generation
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
        Built with Python · Streamlit · FAISS · Groq

    </div>
    """,
    unsafe_allow_html=True,
)
