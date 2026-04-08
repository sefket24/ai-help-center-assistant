import streamlit as st
import anthropic
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


def get_client() -> anthropic.Anthropic | None:
    api_key = (
        st.session_state.get("api_key")
        or st.secrets.get("ANTHROPIC_API_KEY", "")
        or os.environ.get("ANTHROPIC_API_KEY", "")
    )
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


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


def get_ai_response(client: anthropic.Anthropic, query: str, chunks: list) -> str:
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

    prompt = f"""HELP ARTICLE EXCERPTS:
{context}

USER QUESTION: {query}"""

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def load_sample_articles() -> dict[str, str]:
    articles = {}
    if SAMPLE_DIR.exists():
        for ext in ("*.md", "*.txt"):
            for f in sorted(SAMPLE_DIR.glob(ext)):
                articles[f.name] = f.read_text(encoding="utf-8")
    return articles


# ── Session state init ──────────────────────────────────────────────────────

if "docs" not in st.session_state:
    st.session_state.docs = load_sample_articles()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    if not st.session_state.api_key:
        entered = st.text_input("Anthropic API Key", type="password", key="key_input")
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

# ── Main UI ─────────────────────────────────────────────────────────────────

st.title("🤖 AI Help Center Assistant")
st.markdown(
    "Ask a question and I'll answer using our help articles. "
    "If I can't find the answer, I'll direct you to support."
)

client = get_client()

if not client:
    st.warning("⚠️ Enter your Anthropic API key in the sidebar to get started.")
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
