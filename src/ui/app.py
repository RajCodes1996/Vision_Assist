"""
app.py
------
Lumen Reader — Streamlit Accessibility UI
WCAG AA compliant. Voice input + TTS output + citation display.

Run with:
    streamlit run src/ui/app.py
"""

import sys
import os

# ── Path setup ────────────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src_dir      = os.path.join(project_root, "src")
os.chdir(project_root)
for p in [project_root, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# ── Page config (must be first Streamlit call) ────────────────────
st.set_page_config(
    page_title="Lumen Reader",
    page_icon  ="🔍",
    layout     ="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — WCAG AA, high contrast, accessible ──────────────
st.markdown("""
<style>
/* ── Base Premium Light Theme ── */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

:root {
    --primary: #4f46e5;      /* Indigo 600 */
    --primary-dk: #4338ca;   /* Indigo 700 */
    --surface: rgba(255, 255, 255, 0.85);
    --surface2: rgba(249, 250, 251, 0.9);
    --border: rgba(0, 0, 0, 0.08);
    --text: #111827;         /* Gray 900 */
    --text-muted: #4b5563;   /* Gray 600 */
    --accent: #0ea5e9;       /* Light Blue 500 */
    --glass-bg: rgba(255, 255, 255, 0.7);
    --glass-border: rgba(255, 255, 255, 0.5);
}

html, body, [class*="css"] {
    font-family: 'Outfit', system-ui, sans-serif;
    color: var(--text);
}

/* Base App Background with a subtle animated gradient for Light Mode */
.stApp {
    background: linear-gradient(-45deg, #f3f4f6, #e5e7eb, #d1d5db, #f9fafb);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(243, 244, 246, 0.7) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* ── Main area ── */
.main .block-container {
    padding: 3rem 2.5rem;
    max-width: 1000px;
}

/* ── Header with Glassmorphism ── */
.lumen-header {
    background: var(--glass-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 20px;
    padding: 3rem 2.5rem;
    margin-bottom: 3rem;
    border: 1px solid var(--glass-border);
    text-align: center;
    box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.08);
    transition: transform 0.3s ease;
}
.lumen-header:hover {
    transform: translateY(-5px);
}
.lumen-header h1 {
    font-size: 3.2rem;
    font-weight: 700;
    background: -webkit-linear-gradient(45deg, #4f46e5, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
    letter-spacing: -1px;
}
.lumen-header p {
    color: var(--text-muted);
    font-size: 1.2rem;
    margin: 0;
    font-weight: 400;
}

/* ── Answer card ── */
.answer-card {
    background: var(--surface);
    backdrop-filter: blur(12px);
    border-left: 4px solid var(--accent);
    border-radius: 16px;
    padding: 2rem;
    margin: 1.5rem 0;
    font-size: 1.15rem;
    line-height: 1.8;
    color: var(--text);
    box-shadow: 0 8px 30px rgba(0,0,0,0.05);
    animation: fadeIn 0.5s ease-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ── Source card ── */
.source-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.95rem;
    transition: all 0.2s ease;
    box-shadow: 0 2px 10px rgba(0,0,0,0.02);
}
.source-card:hover {
    background: #f8fafc;
    transform: translateX(5px);
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}
.source-badge {
    background: linear-gradient(90deg, #4f46e5, #6366f1);
    color: white;
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 10px;
}
.score-badge {
    background: #f1f5f9;
    color: var(--text-muted);
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.8rem;
    margin-right: 10px;
}

/* ── Status pills ── */
.pill-success {
    background: #dcfce7;
    color: #166534;
    border: 1px solid #bbf7d0;
    border-radius: 20px;
    padding: 4px 16px;
    font-size: 0.85rem;
    font-weight: 600;
}
.pill-info {
    background: #dbeafe;
    color: #1e40af;
    border: 1px solid #bfdbfe;
    border-radius: 20px;
    padding: 4px 16px;
    font-size: 0.85rem;
    font-weight: 600;
}

/* ── Chat history ── */
.chat-user {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
    border-radius: 18px 18px 4px 18px;
    padding: 1rem 1.5rem;
    margin: 1rem 0 1rem 15%;
    color: white;
    font-size: 1.05rem;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.2);
}
.chat-assistant {
    background: var(--surface);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: 18px 18px 18px 4px;
    padding: 1rem 1.5rem;
    margin: 1rem 15% 1rem 0;
    color: var(--text);
    font-size: 1.05rem;
    border-left: 4px solid var(--accent);
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(45deg, #4f46e5, #0ea5e9);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 1.6rem;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(14, 165, 233, 0.4);
    color: white;
}
.stButton > button:focus {
    color: white;
}

/* ── Inline Speak button for chat responses ── */
div[data-testid="column"] button:has(div:contains("🔊")),
div[data-testid="stHorizontalBlock"] button {
    padding: 0.5rem 1rem;
}

/* ── Input ── */
.stTextInput > div > input,
.stTextArea > div > textarea {
    background: #ffffff;
    color: var(--text);
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    font-size: 1.05rem;
    padding: 0.8rem 1.2rem;
    transition: all 0.3s ease;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
}
.stTextInput > div > input:focus,
.stTextArea > div > textarea:focus {
    background: #ffffff;
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.2);
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #94a3b8;
    border-radius: 16px;
    padding: 2rem;
    background: rgba(248, 250, 252, 0.8);
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    background: #f1f5f9;
    border-color: var(--accent);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    height: 3.5rem;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 4px 4px 0px 0px;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-muted);
}
.stTabs [aria-selected="true"] {
    color: var(--primary) !important;
}

/* ── Divider ── */
hr { border-color: var(--border); margin: 2rem 0; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "reader"          : None,
        "loaded_docs"     : [],
        "chat_history"    : [],   # [{role, content, sources, label}]
        "tts_enabled"     : True,
        "last_answer"     : "",
        "last_sources"    : [],
        "pipeline_ready"  : False,
        "processing"      : False,
        "speech_input_query": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ════════════════════════════════════════════════════════════
#  PIPELINE LOADER
# ════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_pipeline(tts_enabled: bool = True):
    """Load the full Lumen pipeline once and cache it."""
    from generation.answer import LumenReader
    return LumenReader(in_memory=True, tts_enabled=tts_enabled)


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    # TTS toggle
    tts_on = st.toggle(
        "🔊 Read answers aloud (TTS)",
        value=st.session_state.tts_enabled,
        help="Speaks answers using pyttsx3 text-to-speech"
    )
    st.session_state.tts_enabled = tts_on

    st.divider()

    # Pipeline init
    st.markdown("### 🚀 Pipeline")
    if not st.session_state.pipeline_ready:
        if st.button("Initialise Lumen Reader", use_container_width=True):
            with st.spinner("Loading models... (~30s first time)"):
                try:
                    st.session_state.reader         = load_pipeline(tts_on)
                    st.session_state.pipeline_ready = True
                    st.success("Pipeline ready!")
                except Exception as e:
                    st.error(f"Init failed: {e}")
    else:
        st.markdown('<span class="pill-success">● Pipeline ready</span>',
                    unsafe_allow_html=True)

    st.divider()

    # Loaded documents
    st.markdown("### 📄 Loaded Documents")
    if st.session_state.loaded_docs:
        for doc in st.session_state.loaded_docs:
            st.markdown(
                f'<span class="pill-info">📄 {doc["filename"]}</span>'
                f'<br><small style="color:#9fa8da">'
                f'{doc["chunks"]} chunks · {doc["pages"]} pages</small>',
                unsafe_allow_html=True,
            )
            st.markdown("")
    else:
        st.caption("No documents loaded yet.")

    st.divider()

    # Memory controls
    st.markdown("### 🧠 Memory")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.reader:
                st.session_state.reader.reset_memory()
            st.rerun()
    with col2:
        if st.button("Reset All", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.loaded_docs  = []
            if st.session_state.reader:
                st.session_state.reader.reset_memory()
                st.session_state.reader.reset_store()
            st.rerun()

    st.divider()
    st.caption("Lumen Reader v1.0 · Built for accessibility")


# ════════════════════════════════════════════════════════════
#  MAIN UI
# ════════════════════════════════════════════════════════════

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="lumen-header">
    <h1>🔍 Lumen Reader</h1>
    <p>Label-Aware RAG Document Assistant for Visually Impaired Users</p>
</div>
""", unsafe_allow_html=True)

# ── Gate — pipeline must be ready ────────────────────────────
if not st.session_state.pipeline_ready:
    st.info("👈 Click **Initialise Lumen Reader** in the sidebar to begin.")
    st.stop()

reader = st.session_state.reader

# ════════════════════════════════════════════════════════════
#  TAB LAYOUT
# ════════════════════════════════════════════════════════════

tab_chat, tab_upload, tab_about = st.tabs([
    "💬 Ask Questions",
    "📂 Upload Documents",
    "ℹ️ About",
])


# ── TAB 1: CHAT ───────────────────────────────────────────────
with tab_chat:

    # Chat history display
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation")
        for idx, turn in enumerate(st.session_state.chat_history):
            if turn["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">🧑 {turn["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                col_bubble, col_btn = st.columns([10, 1.2])
                with col_bubble:
                    st.markdown(
                        f'<div class="chat-assistant">🤖 {turn["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                    if st.button("🔊 Read", key=f"speak_turn_{idx}", help="Read this response aloud", use_container_width=True):
                        if reader and reader.tts:
                            with st.spinner("Reading response aloud..."):
                                reader.tts.speak_answer(
                                    turn["content"],
                                    turn.get("sources", []),
                                )

                # Show sources inline
                if turn.get("sources"):
                    with st.expander(
                        f"📎 {len(turn['sources'])} source(s) · "
                        f"Topic: {turn.get('label','—')}",
                        expanded=False,
                    ):
                        for i, s in enumerate(turn["sources"], 1):
                            st.markdown(
                                f'<div class="source-card">'
                                f'<span class="source-badge">Source {i}</span>'
                                f'<span class="score-badge">Score: {s["score"]}</span>'
                                f'<span class="score-badge">Page {s["page"]}</span>'
                                f'<span class="source-badge">{s["label"]}</span>'
                                f'<br><small style="color:#9fa8da;margin-top:6px;'
                                f'display:block">{s["filename"]}</small>'
                                f'<p style="margin:8px 0 0 0;font-size:0.88rem;'
                                f'color:#cfd8dc">{s["preview"][:160]}...</p>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
        st.divider()

    # No docs loaded warning
    if not st.session_state.loaded_docs:
        st.warning(
            "⚠️ No documents loaded. "
            "Go to **Upload Documents** tab to load a PDF first."
        )

    # ── Question input ────────────────────────────────────────
    st.markdown("### ❓ Ask a Question")
    default_q = st.session_state.pop("speech_input_query", "")
    question = st.text_input(
        label      ="Your question",
        value      =default_q,
        placeholder="e.g. What is AI diligence? What are the key risks?",
        label_visibility="collapsed",
    )

    col_ask, col_tts, col_voice = st.columns([3, 1, 1])

    with col_ask:
        ask_clicked = st.button(
            "🔍 Ask",
            use_container_width=True,
            disabled=not bool(question.strip()),
        )

    with col_tts:
        if st.button("🔊 Replay", use_container_width=True):
            if st.session_state.last_answer and reader.tts:
                with st.spinner("Speaking..."):
                    reader.tts.speak_answer(
                        st.session_state.last_answer,
                        st.session_state.last_sources,
                    )

    voice_triggered_text = None
    with col_voice:
        if st.button("🎤 Voice", use_container_width=True):
            with st.spinner("Listening for 6 seconds... speak now"):
                try:
                    from accessibility.stt import STTEngine
                    stt  = STTEngine(model_size="base")
                    text = stt.listen_for_query()
                    if text:
                        st.success(f"Heard: {text}")
                        voice_triggered_text = text
                    else:
                        st.warning("No speech detected. Try again.")
                except Exception as e:
                    st.error(f"Voice input error: {e}")

    # ── Process question ──────────────────────────────────────
    effective_question = voice_triggered_text if voice_triggered_text else (question if ask_clicked else None)
    if effective_question and effective_question.strip():
        with st.spinner("Searching document and generating answer..."):
            try:
                result = reader.ask(
                    effective_question,
                    speak=st.session_state.tts_enabled,
                )

                # Store in chat history
                st.session_state.chat_history.append({
                    "role"   : "user",
                    "content": effective_question,
                })
                st.session_state.chat_history.append({
                    "role"   : "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "label"  : result["detected_label"],
                })

                # Cache for replay
                st.session_state.last_answer  = result["answer"]
                st.session_state.last_sources = result["sources"]

                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

    # Save audio button
    if st.session_state.last_answer:
        st.divider()
        if st.button("💾 Save Last Answer as MP3", use_container_width=False):
            out_path = "data/last_answer.mp3"
            try:
                reader.tts.save_audio(st.session_state.last_answer, out_path)
                with open(out_path, "rb") as f:
                    st.download_button(
                        label    ="⬇️ Download MP3",
                        data     =f.read(),
                        file_name="lumen_answer.mp3",
                        mime     ="audio/mpeg",
                    )
            except Exception as e:
                st.error(f"Could not save audio: {e}")


# ── TAB 2: UPLOAD ─────────────────────────────────────────────
with tab_upload:
    st.markdown("### 📂 Upload a Document")
    st.caption(
        "Supported formats: PDF, DOCX, TXT, Markdown, HTML · "
        "Max 50MB per file"
    )

    uploaded = st.file_uploader(
        label      ="Drop your document here",
        type       =["pdf", "docx", "txt", "md", "html"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded:
        # Save to data/uploads/
        save_dir  = Path("data/uploads")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / uploaded.name

        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.success(f"File saved: {uploaded.name} ({uploaded.size // 1024} KB)")

        if st.button("⚙️ Process & Load into Lumen", use_container_width=True):
            with st.spinner(
                f"Processing {uploaded.name}... "
                "embedding chunks into vector store"
            ):
                try:
                    info = reader.load_document(str(save_path))

                    # Store doc info
                    st.session_state.loaded_docs.append({
                        "filename": info["filename"],
                        "pages"   : info["pages"],
                        "chunks"  : info["chunks"],
                        "labels"  : info["labels_found"],
                    })

                    st.success(
                        f"✅ **{info['filename']}** loaded successfully!\n\n"
                        f"- 📝 **{info.get('text_chunks', 0)}** text chunks\n"
                        f"- 🖼️ **{info.get('image_chunks', 0)}** image captions\n"
                        f"- 📄 **{info['pages']}** pages\n"
                        f"- Topics detected: "
                        f"{', '.join(str(l) for l in info['labels_found'])}"
                    )
                    st.balloons()

                except Exception as e:
                    st.error(f"Failed to load document: {e}")
                    st.exception(e)

    # Currently loaded docs table
    if st.session_state.loaded_docs:
        st.divider()
        st.markdown("### 📋 Loaded Documents")
        for doc in st.session_state.loaded_docs:
            col_a, col_b, col_c = st.columns([3, 1, 1])
            with col_a:
                st.markdown(f"**📄 {doc['filename']}**")
            with col_b:
                st.markdown(
                    f'<span class="pill-info">{doc["chunks"]} chunks</span>',
                    unsafe_allow_html=True,
                )
            with col_c:
                st.markdown(
                    f'<span class="pill-success">{doc["pages"]} pages</span>',
                    unsafe_allow_html=True,
                )


# ── TAB 3: ABOUT ──────────────────────────────────────────────
with tab_about:
    st.markdown("### 🔍 About Lumen Reader")
    st.markdown("""
Lumen Reader is a **Label-Aware RAG Document Assistant** built specifically
for visually impaired users. It combines cutting-edge AI retrieval with
full text-to-speech accessibility.

---

#### 🏗️ Architecture

| Layer | Technology |
|-------|-----------|
| Document ingestion | PyMuPDF, semantic chunking |
| Label classification | Scikit-learn TF-IDF + Logistic Regression |
| Vector database | Qdrant (cosine similarity) |
| Embeddings | BAAI/bge-base-en-v1.5 (768-dim) |
| Retrieval graph | LangGraph StateGraph + corrective RAG |
| LLM generation | Groq (llama3-8b-8192) |
| Text-to-Speech | pyttsx3 (offline) / gTTS (online) |
| Speech-to-Text | OpenAI Whisper (offline) |
| Frontend | Streamlit + WCAG AA CSS |

---

#### ✨ Key Features
- **Label-aware retrieval** — queries routed to the right document section
- **Corrective RAG** — rewrites weak queries and retries automatically
- **Multi-turn memory** — remembers previous questions in the conversation
- **TTS playback** — every answer read aloud automatically
- **Voice input** — ask questions by speaking (Whisper STT)
- **Citations** — every answer shows page, section, and confidence score
- **MP3 export** — save any answer as an audio file

---

#### 🚀 How to use
1. Click **Initialise Lumen Reader** in the sidebar
2. Go to **Upload Documents** and upload a PDF
3. Click **Process & Load into Lumen**
4. Go to **Ask Questions** and type or speak your question
5. The answer is displayed and read aloud automatically
    """)


# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style="text-align:center;color:#9fa8da;font-size:0.82rem;padding:1rem 0">
    Lumen Reader · Built for accessibility ·
    Label-Aware RAG · LangGraph · Qdrant · pyttsx3
</div>
""", unsafe_allow_html=True)