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
# CONFIG
# ============================================================

APP_NAME = "CyberLaw Pakistan AI"

GOOGLE_DRIVE_URL = (
    "https://drive.google.com/file/d/"
    "1d7xD2E1HrZBpkv16yHcqTb75C0MOzdmx/view?usp=sharing"
)

DATA_DIR = ".cyberlaw_data"

PDF_FILE = os.path.join(
    DATA_DIR,
    "pakistan_cyber_law.pdf"
)

INDEX_FILE = os.path.join(
    DATA_DIR,
    "cyberlaw.index"
)

CHUNKS_FILE = os.path.join(
    DATA_DIR,
    "chunks.npy"
)

METADATA_FILE = os.path.join(
    DATA_DIR,
    "metadata.npy"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

DEFAULT_GROQ_MODEL = (
    "openai/gpt-oss-120b"
)

os.makedirs(
    DATA_DIR,
    exist_ok=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "prompt_count" not in st.session_state:
    st.session_state.prompt_count = 0

if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

if "accent_color" not in st.session_state:
    st.session_state.accent_color = "#FF7A00"

if "background_color" not in st.session_state:
    st.session_state.background_color = "#0E1117"


# ============================================================
# SECRET / API KEY
# ============================================================

def get_groq_api_key():
    """
    API key is NEVER displayed in the UI.

    Priority:
    1. Streamlit secrets
    2. Environment variable
    """

    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    return os.getenv("GROQ_API_KEY")


GROQ_API_KEY = get_groq_api_key()


# ============================================================
# THEME
# ============================================================

def apply_theme():
    theme = st.session_state.theme
    accent = st.session_state.accent_color
    background = st.session_state.background_color

    if theme == "Light":
        text_color = "#111111"
        secondary_text = "#555555"
        card_background = "#FFFFFF"
        border_color = "#DDDDDD"
        input_background = "#F7F7F7"
    else:
        text_color = "#F5F5F5"
        secondary_text = "#AAAAAA"
        card_background = "#161B22"
        border_color = "#30363D"
        input_background = "#0D1117"

    st.markdown(
        f"""
        <style>

        /* =========================
           MAIN APPLICATION
        ========================= */

        .stApp {{
            background: {background};
            color: {text_color};
        }}

        .main {{
            background: {background};
        }}

        /* =========================
           TEXT
        ========================= */

        h1, h2, h3, h4, h5, h6 {{
            color: {text_color} !important;
        }}

        p, li, label {{
            color: {text_color};
        }}

        /* =========================
           TITLE
        ========================= */

        .main-title {{
            font-size: 42px;
            font-weight: 800;
            color: {accent};
            margin-bottom: 5px;
        }}

        .subtitle {{
            font-size: 17px;
            color: {secondary_text};
            margin-bottom: 25px;
        }}

        /* =========================
           CARDS
        ========================= */

        .custom-card {{
            background: {card_background};
            border: 1px solid {border_color};
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 15px;
        }}

        /* =========================
           METRICS
        ========================= */

        .metric-card {{
            background: {card_background};
            border: 1px solid {border_color};
            border-radius: 14px;
            padding: 15px;
            text-align: center;
        }}

        .metric-number {{
            font-size: 28px;
            font-weight: 800;
            color: {accent};
        }}

        .metric-label {{
            color: {secondary_text};
            font-size: 13px;
        }}

        /* =========================
           SOURCE
        ========================= */

        .source-box {{
            background: {card_background};
            border-left: 4px solid {accent};
            border-radius: 10px;
            padding: 14px;
            margin-top: 8px;
        }}

        /* =========================
           BUTTONS
        ========================= */

        .stButton > button {{
            border-radius: 10px;
            border: 1px solid {accent};
        }}

        /* =========================
           INPUTS
        ========================= */

        textarea,
        input {{
            background-color: {input_background} !important;
            color: {text_color} !important;
        }}

        /* =========================
           SIDEBAR
        ========================= */

        section[data-testid="stSidebar"] {{
            background: {card_background};
        }}

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            color: {accent} !important;
        }}

        /* =========================
           CHAT
        ========================= */

        [data-testid="stChatMessage"] {{
            border-radius: 12px;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )


apply_theme()


# ============================================================
# GOOGLE DRIVE
# ============================================================

def extract_google_drive_file_id(url):

    if not url:
        return None

    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
        r"/open\?id=([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            url
        )

        if match:
            return match.group(1)

    if re.fullmatch(
        r"[a-zA-Z0-9_-]{20,}",
        url.strip()
    ):
        return url.strip()

    return None


def build_drive_download_url(file_id):

    return (
        "https://drive.google.com/uc?"
        + urllib.parse.urlencode(
            {
                "export": "download",
                "id": file_id,
            }
        )
    )


def download_google_drive_pdf(
    url,
    destination
):

    file_id = extract_google_drive_file_id(url)

    if not file_id:
        raise ValueError(
            "Invalid Google Drive URL."
        )

    download_url = (
        build_drive_download_url(
            file_id
        )
    )

    opener = urllib.request.build_opener()

    opener.addheaders = [
        (
            "User-Agent",
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/131 Safari/537.36"
        )
    ]

    try:

        response = opener.open(
            download_url,
            timeout=60
        )

        content = response.read()

        content_type = (
            response.headers
            .get(
                "Content-Type",
                ""
            )
            .lower()
        )

        if (
            content.startswith(b"%PDF-")
            or "application/pdf"
            in content_type
        ):

            with open(
                destination,
                "wb"
            ) as file:

                file.write(content)

            return destination

        text = content.decode(
            "utf-8",
            errors="ignore"
        )

        token_match = re.search(
            r'name="confirm"\s+value="([^"]+)"',
            text,
            re.IGNORECASE
        )

        if not token_match:

            token_match = re.search(
                r"confirm=([0-9A-Za-z_-]+)",
                text,
                re.IGNORECASE
            )

        if token_match:

            token = token_match.group(1)

            confirm_url = (
                "https://drive.usercontent.google.com/download?"
                + urllib.parse.urlencode(
                    {
                        "id": file_id,
                        "export": "download",
                        "confirm": token,
                    }
                )
            )

            response = opener.open(
                confirm_url,
                timeout=120
            )

            pdf_content = response.read()

            if not pdf_content.startswith(
                b"%PDF-"
            ):
                raise ValueError(
                    "Google Drive did not return a valid PDF."
                )

            with open(
                destination,
                "wb"
            ) as file:

                file.write(pdf_content)

            return destination

        raise ValueError(
            "Google Drive returned a non-PDF response. "
            "Make sure the file is shared as "
            "'Anyone with the link'."
        )

    except urllib.error.HTTPError as error:

        raise RuntimeError(
            f"Google Drive HTTP error: {error.code}"
        ) from error

    except urllib.error.URLError as error:

        raise RuntimeError(
            f"Google Drive connection error: {error.reason}"
        ) from error


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_pages(pdf_path):

    reader = PdfReader(
        pdf_path
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:

            text = (
                page.extract_text()
                or ""
            )

        except Exception:

            text = ""

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        if text:

            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    pages,
    chunk_size=1000,
    overlap=150
):

    if overlap >= chunk_size:
        raise ValueError(
            "Overlap must be smaller than chunk size."
        )

    chunks = []
    metadata = []

    for page_data in pages:

        page_number = page_data["page"]
        text = page_data["text"]

        start = 0

        while start < len(text):

            end = min(
                start + chunk_size,
                len(text)
            )

            chunk = text[
                start:end
            ].strip()

            if chunk:

                chunks.append(
                    chunk
                )

                metadata.append(
                    {
                        "page": page_number
                    }
                )

            if end >= len(text):
                break

            start = end - overlap

    return chunks, metadata


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# DATABASE
# ============================================================

def database_exists():

    return all(
        os.path.exists(path)
        for path in [
            PDF_FILE,
            INDEX_FILE,
            CHUNKS_FILE,
            METADATA_FILE,
        ]
    )


def save_database(
    chunks,
    metadata,
    embeddings
):

    faiss.normalize_L2(
        embeddings
    )

    dimension = (
        embeddings.shape[1]
    )

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings.astype(
            "float32"
        )
    )

    faiss.write_index(
        index,
        INDEX_FILE
    )

    np.save(
        CHUNKS_FILE,
        np.array(
            chunks,
            dtype=object
        ),
        allow_pickle=True
    )

    np.save(
        METADATA_FILE,
        np.array(
            metadata,
            dtype=object
        ),
        allow_pickle=True
    )


def load_database():

    index = faiss.read_index(
        INDEX_FILE
    )

    chunks = np.load(
        CHUNKS_FILE,
        allow_pickle=True
    ).tolist()

    metadata = np.load(
        METADATA_FILE,
        allow_pickle=True
    ).tolist()

    return (
        index,
        chunks,
        metadata
    )


def build_database(
    force_rebuild=False
):

    if (
        database_exists()
        and not force_rebuild
    ):

        try:

            return load_database()

        except Exception:

            pass

    with st.status(
        "Preparing knowledge base...",
        expanded=True
    ) as status:

        st.write(
            "Downloading legal PDF..."
        )

        download_google_drive_pdf(
            GOOGLE_DRIVE_URL,
            PDF_FILE
        )

        st.write(
            "Extracting PDF text..."
        )

        pages = extract_pdf_pages(
            PDF_FILE
        )

        if not pages:

            raise ValueError(
                "No readable text was found in the PDF."
            )

        st.write(
            f"Extracted {len(pages)} pages."
        )

        st.write(
            "Creating chunks..."
        )

        chunks, metadata = create_chunks(
            pages
        )

        st.write(
            f"Created {len(chunks)} chunks."
        )

        st.write(
            "Creating embeddings..."
        )

        model = load_embedding_model()

        embeddings = model.encode(
            chunks,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        embeddings = embeddings.astype(
            "float32"
        )

        st.write(
            "Building FAISS index..."
        )

        save_database(
            chunks,
            metadata,
            embeddings
        )

        status.update(
            label="Knowledge base ready.",
            state="complete",
            expanded=False
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
    top_k=5
):

    model = load_embedding_model()

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True
    ).astype(
        "float32"
    )

    faiss.normalize_L2(
        query_embedding
    )

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

        if (
            idx < 0
            or idx >= len(chunks)
        ):
            continue

        results.append(
            {
                "text": chunks[idx],
                "page": metadata[idx]["page"],
                "score": float(score),
            }
        )

    return results


def create_context(results):

    if not results:

        return (
            "No relevant legal evidence was retrieved."
        )

    parts = []

    for i, result in enumerate(
        results,
        start=1
    ):

        parts.append(
            f"""
SOURCE {i}
PAGE: {result["page"]}
SIMILARITY: {result["score"]:.4f}

{result["text"]}
"""
        )

    return "\n".join(parts)


# ============================================================
# GROQ
# ============================================================

def get_groq_client():

    if not GROQ_API_KEY:
        return None

    return Groq(
        api_key=GROQ_API_KEY
    )


def get_max_tokens(
    response_size
):

    values = {
        "Short": 700,
        "Medium": 1200,
        "Detailed": 2000,
        "Very detailed": 3000,
    }

    return values.get(
        response_size,
        1200
    )


def build_system_prompt(
    technical_level,
    response_size,
    language,
    answer_style
):

    return f"""
You are CyberLaw Pakistan AI.

You are a retrieval-augmented legal information assistant.

Use ONLY the retrieved legal evidence supplied in the user prompt
for specific legal claims.

LEGAL ACCURACY RULES:

1. Never invent laws.
2. Never invent sections.
3. Never invent penalties.
4. Never invent fines.
5. Never invent legal procedures.
6. Never fabricate citations.
7. If evidence is insufficient, clearly say so.
8. Do not pretend to be a lawyer.
9. Important legal matters should be verified with a qualified
   Pakistani lawyer or relevant official authority.
10. Do not provide operational instructions for hacking, malware,
    phishing, credential theft, unauthorized access, cyber attacks,
    fraud or other harmful cyber activity.
11. You may provide safe cybersecurity and legal information.

CITATIONS:

Use:
[Source 1, Page X]
[Source 2, Page Y]

Only cite sources that actually support the statement.

USER SETTINGS:

Technical level: {technical_level}
Response size: {response_size}
Language: {language}
Answer style: {answer_style}

Respond entirely in the selected language.

Be precise, clear and structured.
"""


def build_history():

    if not st.session_state.messages:
        return "No previous conversation."

    history = []

    for message in st.session_state.messages[-6:]:

        role = message.get(
            "role",
            ""
        )

        content = message.get(
            "content",
            ""
        )

        if role in [
            "user",
            "assistant"
        ]:

            history.append(
                f"{role.upper()}: {content}"
            )

    return "\n".join(
        history
    )


def ask_groq(
    question,
    context,
    history,
    model_name,
    technical_level,
    response_size,
    language,
    answer_style
):

    client = get_groq_client()

    if client is None:

        raise ValueError(
            "GROQ_API_KEY is not configured. "
            "Add it to Streamlit Secrets."
        )

    system_prompt = build_system_prompt(
        technical_level,
        response_size,
        language,
        answer_style
    )

    user_prompt = f"""
RETRIEVED LEGAL EVIDENCE:

{context}

PREVIOUS CONVERSATION:

{history}

CURRENT QUESTION:

{question}

Answer the current question using the retrieved evidence.

If the source does not provide enough information,
say that clearly.

Do not guess missing legal information.

Use citations such as:
[Source 1, Page 5]
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
        max_tokens=get_max_tokens(
            response_size
        ),
    )

    usage = getattr(
        response,
        "usage",
        None
    )

    if usage:

        total_tokens = getattr(
            usage,
            "total_tokens",
            0
        )

        st.session_state.total_tokens += (
            total_tokens or 0
        )

    return (
        response.choices[0]
        .message
        .content
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# ⚖️ CyberLaw AI"
    )

    st.caption(
        "Pakistan Cyber Law RAG Assistant"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "AI Assistant",
            "File a Complaint",
            "About",
        ]
    )

    st.divider()

    # --------------------------------------------------------
    # THEME
    # --------------------------------------------------------

    st.markdown(
        "### 🎨 Appearance"
    )

    theme = st.radio(
        "Theme",
        [
            "Dark",
            "Light",
        ],
        index=(
            0
            if st.session_state.theme == "Dark"
            else 1
        ),
        horizontal=True
    )

    if theme != st.session_state.theme:

        st.session_state.theme = theme

        if theme == "Dark":

            st.session_state.background_color = (
                "#0E1117"
            )

        else:

            st.session_state.background_color = (
                "#F5F7FA"
            )

        st.rerun()

    st.session_state.accent_color = st.color_picker(
        "Accent color",
        st.session_state.accent_color
    )

    st.session_state.background_color = st.color_picker(
        "Background color",
        st.session_state.background_color
    )

    apply_theme()

    st.divider()

    # --------------------------------------------------------
    # AI SETTINGS
    # --------------------------------------------------------

    st.markdown(
        "### ⚙️ AI Settings"
    )

    technical_level = st.selectbox(
        "Technical level",
        [
            "Beginner",
            "Intermediate",
            "Advanced",
            "Legal/Technical",
        ]
    )

    response_size = st.selectbox(
        "Response size",
        [
            "Short",
            "Medium",
            "Detailed",
            "Very detailed",
        ],
        index=1
    )

    language = st.selectbox(
        "Language",
        [
            "English",
            "Urdu",
            "Roman Urdu",
        ]
    )

    answer_style = st.selectbox(
        "Answer style",
        [
            "Clear explanation",
            "Step-by-step",
            "Bullet points",
            "Professional legal information",
        ]
    )

    top_k = st.slider(
        "Sources to retrieve",
        min_value=2,
        max_value=10,
        value=5
    )

    show_sources = st.checkbox(
        "Show retrieved sources",
        value=True
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    st.markdown(
        "### 🤖 Model"
    )

    groq_model = st.selectbox(
        "Groq model",
        [
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ],
        index=0
    )

    st.divider()

    # --------------------------------------------------------
    # USAGE
    # --------------------------------------------------------

    st.markdown(
        "### 📊 Usage"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Prompts",
            st.session_state.prompt_count
        )

    with col2:

        st.metric(
            "Messages",
            len(
                st.session_state.messages
            )
        )

    st.metric(
        "Tokens",
        f"{st.session_state.total_tokens:,}"
    )

    st.divider()

    st.markdown(
        "### 📚 Knowledge Base"
    )

    if database_exists():

        st.success(
            "Knowledge base ready"
        )

    else:

        st.warning(
            "Knowledge base not ready"
        )

    rebuild = st.button(
        "🔄 Rebuild knowledge base",
        use_container_width=True
    )


# ============================================================
# DATABASE
# ============================================================

try:

    if rebuild:

        index, chunks, metadata = (
            build_database(
                force_rebuild=True
            )
        )

    else:

        index, chunks, metadata = (
            build_database()
        )

except Exception as error:

    st.error(
        "Could not prepare the knowledge base."
    )

    st.code(
        str(error),
        language="text"
    )

    st.info(
        "Make sure your Google Drive PDF is publicly "
        "accessible and contains readable text."
    )

    st.stop()


# ============================================================
# KNOWLEDGE BASE METRIC
# ============================================================

with st.sidebar:

    st.metric(
        "Indexed chunks",
        len(chunks)
    )


# ============================================================
# AI ASSISTANT
# ============================================================

if page == "AI Assistant":

    st.markdown(
        '<div class="main-title">'
        '⚖️ CyberLaw Pakistan AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Ask questions about Pakistani cyber law '
        'using the configured legal knowledge base.'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "This application provides AI-assisted legal "
        "information based on the configured source PDF. "
        "It is not a substitute for professional legal advice."
    )

    if st.button(
        "🗑️ Clear conversation"
    ):

        st.session_state.messages = []
        st.session_state.prompt_count = 0
        st.session_state.total_tokens = 0

        st.rerun()

    # --------------------------------------------------------
    # PREVIOUS MESSAGES
    # --------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if (
                message["role"] == "assistant"
                and show_sources
                and message.get("sources")
            ):

                with st.expander(
                    "📚 Retrieved sources"
                ):

                    for i, source in enumerate(
                        message["sources"],
                        start=1
                    ):

                        st.markdown(
                            f"""
**Source {i} — Page {source["page"]}**

Similarity: `{source["score"]:.4f}`

{source["text"]}
"""
                        )

    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask a question about Pakistani cyber law..."
    )

    if question:

        if not GROQ_API_KEY:

            st.error(
                "GROQ_API_KEY is not configured. "
                "Add GROQ_API_KEY to Streamlit Secrets."
            )

            st.stop()

        st.session_state.prompt_count += 1

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):

            st.markdown(
                question
            )

        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    "Searching legal sources..."
                ):

                    results = retrieve_documents(
                        question,
                        index,
                        chunks,
                        metadata,
                        top_k
                    )

                context = create_context(
                    results
                )

                history = build_history()

                with st.spinner(
                    "Generating answer..."
                ):

                    answer = ask_groq(
                        question=question,
                        context=context,
                        history=history,
                        model_name=groq_model,
                        technical_level=technical_level,
                        response_size=response_size,
                        language=language,
                        answer_style=answer_style
                    )

                st.markdown(
                    answer
                )

                if (
                    show_sources
                    and results
                ):

                    with st.expander(
                        "📚 Retrieved sources"
                    ):

                        for i, result in enumerate(
                            results,
                            start=1
                        ):

                            st.markdown(
                                f"""
**Source {i} — Page {result["page"]}**

Similarity: `{result["score"]:.4f}`

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
                    "Unable to generate the answer.\n\n"
                    f"Error: `{error}`"
                )

                st.error(
                    error_message
                )

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
        '<div class="main-title">'
        '📝 Cybercrime Complaint'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Generate a factual complaint draft using the legal knowledge base.'
        '</div>',
        unsafe_allow_html=True
    )

    st.warning(
        "This generates a draft only. Review it carefully before "
        "submitting it to any authority."
    )

    incident = st.text_area(
        "Describe the incident",
        height=250,
        placeholder=(
            "Describe what happened, when it happened, "
            "which platform was involved, and any other relevant facts."
        )
    )

    evidence = st.text_area(
        "Evidence available",
        height=150,
        placeholder=(
            "Screenshots, URLs, messages, transaction records, "
            "emails, phone numbers, etc."
        )
    )

    if st.button(
        "📝 Generate complaint draft",
        type="primary",
        use_container_width=True
    ):

        if not GROQ_API_KEY:

            st.error(
                "GROQ_API_KEY is not configured."
            )

            st.stop()

        if not incident.strip():

            st.error(
                "Please describe the incident."
            )

            st.stop()

        complaint_query = (
            incident
            + "\n"
            + evidence
        )

        with st.spinner(
            "Searching legal sources..."
        ):

            results = retrieve_documents(
                complaint_query,
                index,
                chunks,
                metadata,
                top_k
            )

        context = create_context(
            results
        )

        client = get_groq_client()

        prompt = f"""
Create a professional cybercrime complaint draft.

Language: {language}

INCIDENT:
{incident}

EVIDENCE:
{evidence}

LEGAL SOURCE:
{context}

Rules:

- Do not invent legal sections.
- Do not invent penalties.
- Do not invent authorities.
- Do not make unsupported accusations.
- Use placeholders where information is missing.
- Mark the document as DRAFT.
- Only mention legal provisions supported by the source.

Structure:

1. Subject
2. Complainant information
3. Incident details
4. Suspect information
5. Evidence
6. Requested action
7. Declaration
8. Date and signature
"""

        try:

            response = client.chat.completions.create(
                model=groq_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You create factual cybercrime "
                            "complaint drafts."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.1,
                max_tokens=2500
            )

            complaint = (
                response.choices[0]
                .message
                .content
            )

            st.success(
                "Complaint draft generated."
            )

            st.markdown(
                "### 📄 Draft"
            )

            st.markdown(
                complaint
            )

            st.download_button(
                "⬇️ Download draft",
                data=complaint,
                file_name=(
                    "cybercrime_complaint_draft.txt"
                ),
                mime="text/plain",
                use_container_width=True
            )

            if show_sources:

                with st.expander(
                    "📚 Legal sources"
                ):

                    for i, result in enumerate(
                        results,
                        start=1
                    ):

                        st.markdown(
                            f"""
**Source {i} — Page {result["page"]}**

{result["text"]}
"""
                        )

        except Exception as error:

            st.error(
                f"Could not generate complaint: {error}"
            )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.markdown(
        '<div class="main-title">'
        'ℹ️ About CyberLaw Pakistan AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
## What is this application?

CyberLaw Pakistan AI is a Retrieval-Augmented Generation (RAG)
application for asking questions about the configured Pakistani
cyber-law PDF.

### RAG pipeline

**PDF → Text Extraction → Chunking → Embeddings → FAISS → Retrieval → Groq → Answer**

### Technologies

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- FAISS
- Groq
- RAG

### Security

The Groq API key is **not displayed in the application UI**.

For Streamlit Cloud, configure it using Streamlit Secrets:

`GROQ_API_KEY`

### Legal limitation

This application is an AI information tool.

It does not replace:

- a lawyer
- a court
- a government authority
- official legal advice

Important legal information should always be verified against
current official sources and, where appropriate, with a qualified
Pakistani lawyer.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Indexed chunks",
            len(chunks)
        )

    with col2:

        st.metric(
            "Prompts",
            st.session_state.prompt_count
        )

    with col3:

        st.metric(
            "Tokens",
            f"{st.session_state.total_tokens:,}"
        )
