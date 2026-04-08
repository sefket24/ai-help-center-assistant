import streamlit as st
import openai
import os
import re
from pathlib import Path

st.set_page_config(
    page_title="AI Help Center Assistant",
    page_icon="🤖",
    layout="wide"
)

SAMPLE_DIR = Path(__file__).parent / "sample_articles"
MIN_RELEVANCE_SCORE = 0.15

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #f4f3ef; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1140px;
    }

    /* ── Nav bar ── */
    .nav-bar {
        background: #111;
        border-radius: 12px;
        padding: 1rem 1.75rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.25rem;
    }
    .nav-brand { display: flex; align-items: center; gap: 0.75rem; }
    .nav-brand h1 {
        color: #f4f3ef; margin: 0;
        font-size: 1.1rem; font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
    }
    .nav-badge {
        background: #c8f55a; color: #111;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.6rem; font-weight: 700;
        padding: 2px 7px; border-radius: 3px;
        letter-spacing: 1.5px; text-transform: uppercase;
    }
    .nav-meta { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #666; }

    /* ── Portfolio banner ── */
    .portfolio-banner {
        background: #fffdf0;
        border: 1px solid #f0e68c;
        border-left: 4px solid #c8f55a;
        border-radius: 8px;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }
    .portfolio-banner-text { font-size: 0.85rem; color: #444; line-height: 1.5; }
    .portfolio-banner-text strong { color: #111; font-weight: 600; }
    .portfolio-links { display: flex; gap: 0.6rem; white-space: nowrap; }
    .portfolio-link {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem; font-weight: 600;
        padding: 4px 10px; border-radius: 5px;
        text-decoration: none;
        border: 1px solid #d4d0c8; color: #333; background: #fff;
    }
    .portfolio-link:hover { background: #111; color: #c8f55a; border-color: #111; }

    /* ── Chat area ── */
    .stChatMessage {
        background: #fff;
        border: 1px solid #e0ddd6;
        border-radius: 12px;
        padding: 0.25rem 0.5rem;
        margin-bottom: 0.5rem;
    }

    /* ── Streamlit overrides ── */
    .stButton > button {
        background: #111; color: #f4f3ef;
        border: none; border-radius: 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem; font-weight: 600;
        padding: 0.6rem 1.5rem; width: 100%;
        letter-spacing: 0.5px; transition: all 0.15s;
    }
    .stButton > button:hover { background: #333; color: #c8f55a; }

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 8px !important;
        border: 1px solid #d4d0c8 !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.9rem !important;
        background: #ffffff !important;
        color: #111111 !important;
    }
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder { color: #aaa !important; }
    .stTextInput > label, .stTextArea > label {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.88rem; color: #444;
    }
</style>
""", unsafe_allow_html=True)

# ── Nav bar ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav-bar">
    <div class="nav-brand">
        <span style="font-size:1.3rem">🤖</span>
        <h1>AI Help Center Assistant</h1>
        <span class="nav-badge">AI-Powered</span>
    </div>
    <div class="nav-meta">Support Ops · Internal Tool</div>
</div>
""", unsafe_allow_html=True)

# ── Portfolio banner ──────────────────────────────────────────────────────────
st.markdown("""
<div class="portfolio-banner">
    <div class="portfolio-banner-text">
        👋 <strong>Hey, hiring team!</strong> This is a portfolio project by <strong>Sef Nouri</strong> —
        a working AI app that simulates how SaaS support teams can deflect tickets with instant,
        doc-grounded answers. Upload your own help articles, ask any support question, and the AI
        responds using only your content — with a built-in "contact support" fallback when it's unsure.
        <strong>Feel free to upload your own docs and test it live.</strong>
    </div>
    <div class="portfolio-links">
        <a class="portfolio-link" href="https://github.com/sefket24/ai-help-center-assistant" target="_blank">⌥ GitHub</a>
        <a class="portfolio-link" href="https://www.linkedin.com/in/sefketnouri" target="_blank">in LinkedIn</a>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Core logic ────────────────────────────────────────────────────────────────

def get_client() -> openai.OpenAI | None:
    api_key = (
        st.session_state.get("api_key")
        or st.secrets.get("OPENAI_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    )
    if not api_key:
        return None
    return openai.OpenAI(api_key=api_key)


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 2) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current, size = [], [], 0
    for sentence in sentences:
        if size + len(sentence) > chunk_size and current:
            chunks.append(" ".join(current))
            current = current[-overlap:]
            size = sum(len(s) for s in current)
        current.append(sentence)
        size += len(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


def score_chunk(chunk: str, query: str) -> float:
    terms = set(re.findall(r"\w+", query.lower()))
    chunk_lower = chunk.lower()
    hits = sum(1 for t in terms if t in chunk_lower)
    return hits / max(len(terms), 1)


def find_relevant_chunks(
    docs: dict[str, str],
    query: str,
    top_k: int = 4,
) -> list[tuple[str, str, float]]:
    results = []
    for name, content in docs.items():
        for chunk in chunk_text(content):
            score = score_chunk(chunk, query)
            results.append((name, chunk, score))
    results.sort(key=lambda x: x[2], reverse=True)
    return [(n, c, s) for n, c, s in results[:top_k] if s >= MIN_RELEVANCE_SCORE]


def get_ai_response(client: openai.OpenAI, query: str, chunks: list) -> str:
    if not chunks:
        return (
            "I wasn't able to find relevant information in our help articles for that question. "
            "Please **contact support** and our team will be happy to help."
        )

    context = "\n\n---\n\n".join(
        f"[Source: {name}]\n{chunk}" for name, chunk, _ in chunks
    )

    system = (
        "You are a helpful support assistant for a software product. "
        "Answer the user's question using ONLY the help article excerpts provided. "
        "Be concise, friendly, and accurate. "
        "If the answer is not clearly covered by the articles, respond with exactly: "
        '"I don\'t have enough information to answer that accurately. Please **contact support** for assistance." '
        "Never make up information or draw on outside knowledge."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"HELP ARTICLE EXCERPTS:\n{context}\n\nUSER QUESTION: {query}"},
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content


def load_sample_articles() -> dict[str, str]:
    articles = {}
    if SAMPLE_DIR.exists():
        for ext in ("*.md", "*.txt"):
            for f in sorted(SAMPLE_DIR.glob(ext)):
                articles[f.name] = f.read_text(encoding="utf-8")
    return articles


# ── Session state init ────────────────────────────────────────────────────────

if "docs" not in st.session_state:
    st.session_state.docs = load_sample_articles()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = os.environ.get("OPENAI_API_KEY", "")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    if not st.session_state.api_key:
        entered = st.text_input("OpenAI API Key", type="password", key="key_input")
        if entered:
            st.session_state.api_key = entered
            st.rerun()
    else:
        st.success("API key loaded ✓")
        if st.button("Remove API key"):
            st.session_state.api_key = ""
            st.rerun()

    st.divider()
    st.header("📚 Help Articles")

    uploaded = st.file_uploader(
        "Upload .txt or .md articles",
        type=["txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded:
        for f in uploaded:
            st.session_state.docs[f.name] = f.read().decode("utf-8")
        st.success(f"Uploaded {len(uploaded)} file(s)")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load samples", use_container_width=True):
            st.session_state.docs = load_sample_articles()
            st.rerun()
    with col2:
        if st.button("Clear all", use_container_width=True):
            st.session_state.docs = {}
            st.rerun()

    if st.session_state.docs:
        st.subheader("Loaded articles")
        for name in st.session_state.docs:
            st.markdown(f"- 📄 {name}")
    else:
        st.info("No articles loaded.")

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main UI ───────────────────────────────────────────────────────────────────

client = get_client()

if not client:
    st.warning("⚠️ Enter your OpenAI API key in the sidebar to get started.")
    st.stop()

if not st.session_state.docs:
    st.info("📂 No articles loaded. Upload files or click **Load samples** in the sidebar.")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Ask a question about our product…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching help articles…"):
            chunks = find_relevant_chunks(st.session_state.docs, prompt)
            answer = get_ai_response(client, prompt, chunks)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
