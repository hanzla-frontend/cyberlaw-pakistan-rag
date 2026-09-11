import os
import re
import urllib.parse
from pathlib import Path

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
GOOGLE_DRIVE_URL = (
    "https://drive.google.com/file/d/"
    "1d7xD2E1HrZBpkv16yHcqTb75C0MOzdmx/view?usp=sharing"
)

DATA_DIR = Path(".cyberlaw_data")
PDF_PATH = DATA_DIR / "cyber_law_source.pdf"
INDEX_PATH = DATA_DIR / "cyberlaw.index"
CHUNKS_PATH = DATA_DIR / "chunks.npy"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MODEL_OPTIONS = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
DEFAULT_MODEL = MODEL_OPTIONS[0]

DATA_DIR.mkdir(parents=True, exist_ok=True)

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
# THEME
# ============================================================
def theme_colors():
    if st.session_state.theme == "Dark":
        return {
            "bg": st.session_state.background_color,
            "surface": "#151A21",
            "surface2": "#1C232D",
            "input": "#11161D",
            "border": "#303945",
            "text": "#F5F7FA",
            "secondary": "#B7C0CC",
            "placeholder": "#7F8A98",
            "accent": st.session_state.accent_color,
            "hover": "#FF963D",
            "sidebar": "#10151C",
            "code": "#0D1117",
        }
    return {
        "bg": st.session_state.background_color,
        "surface": "#FFFFFF",
        "surface2": "#F4F6F8",
        "input": "#FFFFFF",
        "border": "#D8DEE6",
        "text": "#111827",
        "secondary": "#4B5563",
        "placeholder": "#6B7280",
        "accent": st.session_state.accent_color,
        "hover": "#E86F00",
        "sidebar": "#F8FAFC",
        "code": "#F3F4F6",
    }

def apply_theme():
    c = theme_colors()
    st.markdown(
        f"""
        <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"],
        .stApp, .main {{
            background: {c["bg"]} !important;
            color: {c["text"]} !important;
        }}
        .block-container {{
            max-width: 1400px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}
        h1,h2,h3,h4,h5,h6,p,span,label,div,
        [data-testid="stMarkdownContainer"] {{
            color: {c["text"]};
        }}
        .stCaption {{
            color: {c["secondary"]} !important;
        }}
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div {{
            background: {c["sidebar"]} !important;
        }}
        section[data-testid="stSidebar"] {{
            border-right: 1px solid {c["border"]} !important;
        }}
        section[data-testid="stSidebar"] * {{
            color: {c["text"]} !important;
        }}
        .app-title {{
            font-size: 2.7rem;
            font-weight: 800;
            letter-spacing: -1px;
            margin-bottom: .2rem;
        }}
        .app-subtitle {{
            font-size: 1.05rem;
            color: {c["secondary"]} !important;
            margin-bottom: 1.2rem;
        }}
        .accent-line {{
            height: 4px;
            width: 90px;
            background: {c["accent"]};
            border-radius: 20px;
            margin: 8px 0 20px;
        }}
        .custom-card {{
            background: {c["surface"]};
            border: 1px solid {c["border"]};
            border-radius: 18px;
            padding: 22px;
            margin-bottom: 18px;
            box-shadow: 0 8px 30px rgba(0,0,0,.08);
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
            font-size: .85rem;
        }}
        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea {{
            background: {c["input"]} !important;
            color: {c["text"]} !important;
            border: 1px solid {c["border"]} !important;
            border-radius: 10px !important;
        }}
        div[data-testid="stTextInput"] input::placeholder,
        div[data-testid="stTextArea"] textarea::placeholder,
        div[data-testid="stChatInput"] textarea::placeholder {{
            color: {c["placeholder"]} !important;
            opacity: 1 !important;
        }}
        div[data-baseweb="select"] > div {{
            background: {c["input"]} !important;
            color: {c["text"]} !important;
            border-color: {c["border"]} !important;
        }}
        div[data-baseweb="select"] span {{
            color: {c["text"]} !important;
        }}
        [role="listbox"], [role="option"] {{
            background: {c["surface"]} !important;
            color: {c["text"]} !important;
        }}
        .stButton > button {{
            background: {c["surface"]} !important;
            color: {c["text"]} !important;
            border: 1px solid {c["border"]} !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
        }}
        .stButton > button:hover {{
            border-color: {c["accent"]} !important;
            color: {c["accent"]} !important;
        }}
        .stDownloadButton > button,
        .stButton > button[kind="primary"] {{
            background: {c["accent"]} !important;
            color: #fff !important;
            border: none !important;
        }}
        div[data-testid="stChatInput"] > div {{
            background: {c["input"]} !important;
            border: 1px solid {c["border"]} !important;
            border-radius: 14px !important;
            overflow: hidden !important;
        }}
        div[data-testid="stChatInput"] textarea {{
            background: transparent !important;
            color: {c["text"]} !important;
            -webkit-text-fill-color: {c["text"]} !important;
            border: none !important;
            box-shadow: none !important;
        }}
        div[data-testid="stChatInput"] button {{
            background: {c["accent"]} !important;
            color: #fff !important;
            border: none !important;
            min-width: 52px !important;
            min-height: 52px !important;
        }}
        [data-testid="stChatMessage"] {{
            background: {c["surface"]} !important;
            border: 1px solid {c["border"]} !important;
            border-radius: 16px !important;
            margin-bottom: 12px !important;
        }}
        [data-testid="stMetric"] {{
            background: {c["surface"]} !important;
            border: 1px solid {c["border"]} !important;
            border-radius: 14px !important;
        }}
        code, pre {{
            background: {c["code"]} !important;
        }}
        .footer {{
            text-align: center;
            color: {c["secondary"]} !important;
            font-size: .82rem;
            padding: 25px 0 5px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_theme()

# ============================================================
# GOOGLE DRIVE
# ============================================================
def extract_drive_file_id(url: str):
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def download_google_drive_file(url: str, destination: Path):
    """Download a public Google Drive file with gdown."""
    file_id = extract_drive_file_id(url)
    if not file_id:
        raise ValueError("Invalid Google Drive URL: file ID could not be found.")

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is not installed. Add gdown to requirements.txt and redeploy."
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(".download")

    if temp_path.exists():
        temp_path.unlink()

    try:
        result = gdown.download(
            id=file_id,
            output=str(temp_path),
            quiet=True,
            fuzzy=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Google Drive download failed: {exc}") from exc

    if not result or not temp_path.exists():
        raise RuntimeError(
            "Google Drive did not return a file. "
            "Set the PDF sharing permission to 'Anyone with the link → Viewer'."
        )

    data = temp_path.read_bytes()
    if not data.startswith(b"%PDF"):
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Google Drive returned something other than a PDF. "
            "Make sure the file is shared publicly as Viewer."
        )

    temp_path.replace(destination)
    return destination

# ============================================================
# PDF + RAG
# ============================================================
def extract_pdf_pages(pdf_path: Path):
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"Could not open the PDF: {exc}") from exc

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append({"page": page_number, "text": text})
    return pages

def create_chunks(pages, chunk_size=900, overlap=150):
    chunks = []
    for page_data in pages:
        text = page_data["text"]
        page = page_data["page"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({"text": chunk, "page": page})
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def create_faiss_index(chunks):
    if not chunks:
        raise ValueError("No text chunks were created.")

    model = load_embedding_model()
    texts = [x["text"] for x in chunks]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def save_knowledge_base(index, chunks):
    faiss.write_index(index, str(INDEX_PATH))
    np.save(
        str(CHUNKS_PATH),
        np.array(chunks, dtype=object),
        allow_pickle=True,
    )

def load_knowledge_base():
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        return None, []
    try:
        index = faiss.read_index(str(INDEX_PATH))
        chunks = np.load(
            str(CHUNKS_PATH),
            allow_pickle=True,
        ).tolist()
        if not chunks or index.ntotal != len(chunks):
            return None, []
        return index, chunks
    except Exception:
        return None, []

def build_knowledge_base(force_download=False):
    with st.spinner("Downloading and processing the cyber-law PDF..."):
        if force_download:
            PDF_PATH.unlink(missing_ok=True)

        if not PDF_PATH.exists():
            download_google_drive_file(GOOGLE_DRIVE_URL, PDF_PATH)

        pages = extract_pdf_pages(PDF_PATH)
        if not pages:
            raise ValueError(
                "The PDF contains no extractable text. "
                "If it is scanned, OCR is required."
            )

        chunks = create_chunks(pages)
        index = create_faiss_index(chunks)
        save_knowledge_base(index, chunks)
        return index, chunks

# ============================================================
# GROQ
# ============================================================
def get_groq_api_key():
    try:
        value = st.secrets.get("GROQ_API_KEY")
        if value:
            return value
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY")

@st.cache_resource(show_spinner=False)
def get_groq_client(api_key):
    return Groq(api_key=api_key) if api_key else None

def retrieve_context(query, top_k=5):
    index = st.session_state.index
    chunks = st.session_state.chunks

    if index is None or not chunks:
        return []

    model = load_embedding_model()
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = np.asarray(query_embedding, dtype="float32")

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks)),
    )

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append({
            "text": chunks[idx]["text"],
            "page": chunks[idx]["page"],
            "score": float(score),
        })
    return results

def build_context(results):
    if not results:
        return "No relevant knowledge-base context was retrieved."

    return "\n\n".join(
        f"SOURCE {i}\nPage: {r['page']}\nSimilarity: {r['score']:.4f}\n\n{r['text']}"
        for i, r in enumerate(results, 1)
    )

def response_tokens(size):
    return {
        "Short": 700,
        "Medium": 1200,
        "Detailed": 2000,
        "Very detailed": 3000,
    }.get(size, 1200)

def build_system_prompt(technical_level, response_size, language, answer_style):
    language_instruction = {
        "English": "Respond in clear English.",
        "Urdu": "Respond in clear Urdu using Urdu script where appropriate.",
        "Roman Urdu": "Respond in simple Roman Urdu.",
    }[language]

    style_instruction = {
        "Professional": "Use a professional legal-information style.",
        "Simple": "Explain concepts in simple language suitable for a beginner.",
        "Educational": "Teach the topic step-by-step with short explanations and examples.",
        "Direct": "Give a concise, direct answer and avoid unnecessary explanation.",
    }[answer_style]

    return f"""
You are CyberLaw Pakistan AI, a retrieval-augmented legal information assistant.

Answer using ONLY the retrieved knowledge-base context supplied with the user question.

Legal safety rules:
1. Do not invent Pakistani laws, sections, clauses, penalties, procedures, authorities,
   dates, case citations, or legal terminology.
2. If the context is insufficient, say:
   "The available knowledge base does not provide enough information to answer this reliably."
3. Never fabricate source references.
4. Do not provide instructions that facilitate cybercrime, unauthorized access, malware,
   credential theft, exploitation, evasion, or other harmful activity.
5. Give safe, lawful, defensive guidance for cyber incidents.
6. Do not present the answer as formal legal representation.
7. For important legal matters, recommend verification with a qualified Pakistani lawyer
   or relevant official authority.

Settings:
Technical level: {technical_level}
Response size: {response_size}
Language: {language}
Answer style: {answer_style}

{language_instruction}
{style_instruction}

When supported by context, cite claims using:
[Source 1, Page X]
Use the actual page number from the retrieved context.
"""

def ask_rag(question, technical_level, response_size, language, answer_style, top_k, model_name):
    api_key = get_groq_api_key()
    client = get_groq_client(api_key)

    if client is None:
        return (
            "Groq API key is not configured. Add `GROQ_API_KEY` to Streamlit Secrets.",
            [],
            0,
        )

    if not st.session_state.knowledge_ready:
        return "Please build the knowledge base first.", [], 0

    results = retrieve_context(question, top_k)
    context = build_context(results)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-6:]
    ]

    messages = [
        {
            "role": "system",
            "content": build_system_prompt(
                technical_level, response_size, language, answer_style
            ),
        },
        *history,
        {
            "role": "user",
            "content": (
                f"QUESTION:\n{question}\n\n"
                f"RETRIEVED KNOWLEDGE-BASE CONTEXT:\n{context}\n\n"
                "Answer strictly from the context."
            ),
        },
    ]

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.15,
            max_tokens=response_tokens(response_size),
        )
        answer = (
            completion.choices[0].message.content
            if completion.choices
            else "No response was generated."
        )
        usage = getattr(completion, "usage", None)
        tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0
        return answer, results, tokens
    except Exception as exc:
        return f"Unable to generate the answer.\n\nError: {exc}", results, 0

def generate_complaint(incident, language, model_name):
    api_key = get_groq_api_key()
    client = get_groq_client(api_key)
    if client is None:
        return "Groq API key is not configured.", 0

    results = retrieve_context(incident, 5)
    context = build_context(results)

    prompt = f"""
Create a factual cybercrime complaint draft in {language}.

Do not invent laws, sections, penalties, procedures, authorities, case numbers,
or legal claims. Use placeholders when information is missing.

Include:
1. Subject
2. Complainant information placeholders
3. Incident summary
4. Date/time placeholders
5. Evidence available
6. Requested action
7. Contact placeholders
8. A note that the draft should be reviewed before submission

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
                    "content": "Create factual, lawful cybercrime complaint drafts. Never fabricate legal information.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.15,
            max_tokens=2200,
        )
        answer = completion.choices[0].message.content
        usage = getattr(completion, "usage", None)
        tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0
        return answer, tokens
    except Exception as exc:
        return f"Unable to generate complaint draft.\n\nError: {exc}", 0

# ============================================================
# LOAD EXISTING KB
# ============================================================
if st.session_state.index is None:
    loaded_index, loaded_chunks = load_knowledge_base()
    if loaded_index is not None:
        st.session_state.index = loaded_index
        st.session_state.chunks = loaded_chunks
        st.session_state.knowledge_ready = True

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        '<div style="font-size:25px;font-weight:800;">⚖️ CyberLaw Pakistan AI</div>',
        unsafe_allow_html=True,
    )
    st.caption("RAG-powered Pakistani cyber-law information assistant")
    st.divider()

    page = st.radio(
        "Navigation",
        ["AI Assistant", "File a Complaint", "About"],
        label_visibility="collapsed",
    )

    st.divider()

    current_dark = st.session_state.theme == "Dark"
    toggle = st.toggle(
        "Dark Mode",
        value=current_dark,
        key="theme_toggle",
    )
    new_theme = "Dark" if toggle else "Light"

    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.session_state.background_color = (
            "#0B0F14" if new_theme == "Dark" else "#F5F7FA"
        )
        st.rerun()

    st.markdown("### Appearance")
    st.session_state.accent_color = st.color_picker(
        "Accent color",
        st.session_state.accent_color,
    )
    st.session_state.background_color = st.color_picker(
        "Background color",
        st.session_state.background_color,
    )

    st.divider()
    st.markdown("### AI Settings")

    technical_level = st.selectbox(
        "Technical level",
        ["Beginner", "Intermediate", "Advanced", "Legal / Technical"],
    )
    response_size = st.selectbox(
        "Response size",
        ["Short", "Medium", "Detailed", "Very detailed"],
        index=1,
    )
    language = st.selectbox(
        "Language",
        ["English", "Urdu", "Roman Urdu"],
    )
    answer_style = st.selectbox(
        "Answer style",
        ["Professional", "Simple", "Educational", "Direct"],
    )
    top_k = st.slider("Retrieved sources", 2, 10, 5)
    show_sources = st.checkbox("Show retrieved sources", True)

    st.divider()
    st.markdown("### Model")
    selected_model = st.selectbox(
        "Groq model",
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(st.session_state.selected_model)
        if st.session_state.selected_model in MODEL_OPTIONS else 0,
    )
    st.session_state.selected_model = selected_model

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prompts", st.session_state.total_prompts)
    with col2:
        st.metric("Messages", st.session_state.total_messages)
    st.metric("Tokens", st.session_state.total_tokens)

    st.divider()
    st.markdown("### Knowledge Base")
    if st.session_state.knowledge_ready:
        st.success(f"Ready • {len(st.session_state.chunks)} chunks")
    else:
        st.warning("Knowledge base not built")

    if st.button("🔄 Build / Rebuild Knowledge Base", use_container_width=True):
        try:
            idx, chunks = build_knowledge_base(force_download=True)
            st.session_state.index = idx
            st.session_state.chunks = chunks
            st.session_state.knowledge_ready = True
            st.success(f"Knowledge base ready with {len(chunks)} chunks.")
        except Exception as exc:
            st.session_state.knowledge_ready = False
            st.error(f"Knowledge-base error: {exc}")

# Re-apply CSS after sidebar controls because colors may have changed.
apply_theme()

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="app-title">⚖️ CyberLaw Pakistan AI</div>
    <div class="app-subtitle">
        Ask questions about Pakistani cyber law using a retrieval-augmented knowledge base.
    </div>
    <div class="accent-line"></div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# AI ASSISTANT
# ============================================================
if page == "AI Assistant":
    if not st.session_state.knowledge_ready:
        st.warning("The knowledge base has not been built yet.")
        st.info(
            "Use **Build / Rebuild Knowledge Base** in the sidebar. "
            "The app will download the configured Google Drive PDF, extract text, "
            "create embeddings, and build the FAISS index."
        )

    st.markdown(
        """
        <div class="custom-card">
            <h3>Ask CyberLaw Pakistan AI</h3>
            <p>
                Ask questions about cyber-law concepts, legal terminology,
                cybercrime scenarios, reporting considerations, and topics
                covered by the connected knowledge base.
            </p>
            <p><strong>Important:</strong> This application provides information,
            not formal legal advice.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about Pakistani cyber law...")
    if question:
        question = question.strip()
        if question:
            st.session_state.messages.append(
                {"role": "user", "content": question}
            )
            st.session_state.total_prompts += 1
            st.session_state.total_messages += 1

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching the knowledge base and generating an answer..."):
                    answer, sources, token_count = ask_rag(
                        question,
                        technical_level,
                        response_size,
                        language,
                        answer_style,
                        top_k,
                        selected_model,
                    )
                st.markdown(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                st.session_state.total_messages += 1
                st.session_state.total_tokens += token_count

                if show_sources and sources:
                    st.markdown("### Retrieved Sources")
                    for i, source in enumerate(sources, 1):
                        with st.expander(
                            f"Source {i} • Page {source['page']} • Similarity {source['score']:.3f}"
                        ):
                            st.write(source["text"])

# ============================================================
# COMPLAINT
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

    if st.button("Generate Complaint Draft", type="primary", use_container_width=True):
        if not incident.strip():
            st.warning("Please describe the incident first.")
        elif not st.session_state.knowledge_ready:
            st.warning("Please build the knowledge base first.")
        else:
            with st.spinner("Preparing complaint draft..."):
                complaint, token_count = generate_complaint(
                    incident,
                    language,
                    selected_model,
                )
            st.session_state.total_tokens += token_count

            st.markdown("### Generated Draft")
            st.markdown(complaint)
            st.download_button(
                "⬇️ Download Complaint Draft",
                complaint,
                "cybercrime_complaint_draft.txt",
                "text/plain",
                use_container_width=True,
            )
            st.info(
                "Review all facts and legal references before submission. "
                "The generated text is a draft and is not a substitute for professional legal advice."
            )

# ============================================================
# ABOUT
# ============================================================
else:
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

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="stat-card"><div class="stat-number">RAG</div>'
            '<div class="stat-label">Retrieval-Augmented Generation</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="stat-card"><div class="stat-number">FAISS</div>'
            '<div class="stat-label">Vector similarity search</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="stat-card"><div class="stat-number">Groq</div>'
            '<div class="stat-label">LLM inference</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Setup")
    st.info(
        "1. Set the Google Drive PDF to **Anyone with the link → Viewer**. "
        "2. Add `GROQ_API_KEY` to Streamlit Secrets. "
        "3. Deploy and press **Build / Rebuild Knowledge Base** once."
    )

    st.markdown("### Browser / WebSocket note")
    st.info(
        "Messages such as `WebSocket onerror`, `Page entered Back-Forward Cache`, "
        "or `/api/v2/user/details 404` can come from the Streamlit Cloud/browser "
        "runtime rather than this Python application. Do not add an API route for "
        "`/api/v2/user/details`. If the app remains usable, the DevTools 404 can "
        "be ignored; if the page disconnects, refresh the app and avoid browser "
        "Back/Forward navigation while a request is running."
    )

    st.markdown("### Legal Disclaimer")
    st.warning(
        "This application is an AI information assistant. It does not provide "
        "formal legal advice, legal representation, or guaranteed legal conclusions. "
        "Verify important legal matters with an appropriate qualified professional "
        "or official source."
    )

st.markdown(
    '<div class="footer">⚖️ CyberLaw Pakistan AI • RAG + FAISS + Sentence Transformers + Groq</div>',
    unsafe_allow_html=True,
)
