import streamlit as st
import requests
import json
from pathlib import Path

# Streamlit Page Configuration
st.set_page_config(
    page_title="CodeSage | AI Codebase Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://127.0.0.1:8000"

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-card {
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .source-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 0.7rem 1rem;
        border-radius: 4px;
        margin-top: 0.5rem;
        font-family: monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def check_backend_health():
    """Checks if FastAPI backend is running and healthy."""
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if res.status_code == 200:
            return True, res.json()
        return False, None
    except Exception:
        return False, None


# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "scanned_repo" not in st.session_state:
    st.session_state.scanned_repo = None
if "scan_stats" not in st.session_state:
    st.session_state.scan_stats = None


# --- SIDEBAR: REPOSITORY INGESTION & STATUS ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/code.png", width=64)
    st.title("CodeSage Control")

    backend_ok, health_info = check_backend_health()
    if backend_ok:
        st.success("🟢 FastAPI Backend: Online (Port 8000)")
    else:
        st.warning("🟠 Backend Offline: Start with `uvicorn app.main:app`")

    st.markdown("---")
    st.subheader("📦 Ingest Repository")

    repo_url = st.text_input(
        "GitHub Repository URL",
        value="https://github.com/Payal-03/CodeSage",
        placeholder="https://github.com/owner/repo"
    )

    if st.button("🚀 Scan & Index Codebase", use_container_width=True, type="primary"):
        if not repo_url.strip():
            st.error("Please enter a valid GitHub repository URL.")
        else:
            with st.spinner("Cloning, parsing AST symbols, and computing vector embeddings..."):
                try:
                    payload = {"github_url": repo_url.strip()}
                    res = requests.post(f"{API_BASE_URL}/repository/scan", json=payload, timeout=300)

                    if res.status_code == 200:
                        data = res.json()
                        st.session_state.scanned_repo = repo_url
                        st.session_state.scan_stats = data
                        st.success(f"Indexed {data.get('total_files', 0)} files ({data.get('total_chunks', 0)} chunks)!")
                    else:
                        err_msg = res.json().get("detail", res.text)
                        st.error(f"Scan failed: {err_msg}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {str(e)}")

    if st.session_state.scan_stats:
        stats = st.session_state.scan_stats
        st.markdown("---")
        st.markdown("### 📊 Index Summary")
        col1, col2 = st.columns(2)
        col1.metric("Files", stats.get("total_files", 0))
        col2.metric("Chunks", stats.get("total_chunks", 0))

        with st.expander("Discovered Files"):
            for f in stats.get("files", []):
                st.write(f"- `{f.get('file_path')}` ({f.get('total_lines')} lines)")

    st.markdown("---")
    st.caption("CodeSage v0.1.0 | Fast RAG with Tree-Sitter, ChromaDB & BM25")


# --- MAIN VIEW: TABS FOR Q&A AND BUG INVESTIGATION ---
st.markdown('<div class="main-header">🔮 CodeSage AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Codebase Intelligence, Architecture Walkthroughs & Bug Investigation</div>', unsafe_allow_html=True)

tab_qa, tab_bug, tab_arch = st.tabs(["💬 Codebase Chat & Q&A", "🐛 Bug Investigation", "🏛️ Architecture & Pipeline"])

# --- TAB 1: CODEBASE CHAT & Q&A ---
with tab_qa:
    st.markdown("Ask deep questions about application logic, dependencies, routing, or functions in your indexed codebase.")

    # Preset quick-prompt pills
    preset_questions = [
        "Where is the health check API implemented?",
        "Where is the Ollama endpoint URL defined?",
        "How does the code parser handle decorators and async functions?",
        "Where is the hybrid search combining BM25 and vector store implemented?"
    ]

    selected_preset = st.selectbox(
        "💡 Quick Questions (or type custom question below):",
        ["Select a suggested question..."] + preset_questions
    )

    user_query = st.chat_input("Ask a question about the indexed codebase...")
    if selected_preset != "Select a suggested question...":
        user_query = selected_preset

    # Display Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("🔍 View Grounded Code Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"**`{src.get('file_path')}`** (Lines {src.get('start_line')}-{src.get('end_line')})")
                        st.caption(f"Symbol: `{src.get('name')}` | Type: `{src.get('type')}` | RRF Score: `{src.get('rrf_score')}`")

    # Handle New User Message
    if user_query:
        # Add user query to history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Searching repository and formulating grounded answer..."):
                try:
                    payload = {"question": user_query}
                    res = requests.post(f"{API_BASE_URL}/repository/ask", json=payload, timeout=120)

                    if res.status_code == 200:
                        data = res.json()
                        answer_text = data.get("answer", "No response generated.")
                        sources = data.get("sources", [])

                        st.markdown(answer_text)

                        if sources:
                            with st.expander("🔍 View Grounded Code Sources"):
                                for src in sources:
                                    st.markdown(f"**`{src.get('file_path')}`** (Lines {src.get('start_line')}-{src.get('end_line')})")
                                    st.caption(f"Symbol: `{src.get('name')}` | Type: `{src.get('type')}` | RRF Score: `{src.get('rrf_score')}`")

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer_text,
                            "sources": sources
                        })
                    else:
                        err = f"API Error ({res.status_code}): {res.text}"
                        st.error(err)
                        st.session_state.chat_history.append({"role": "assistant", "content": err})
                except Exception as e:
                    err_msg = f"Failed to connect to backend: {str(e)}"
                    st.error(err_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": err_msg})


# --- TAB 2: BUG INVESTIGATION ---
with tab_bug:
    st.subheader("Investigate Codebase Bugs & Stack Traces")
    st.markdown("Describe the issue or paste a stack trace. CodeSage will pinpoint affected files and generate a targeted, read-only fix.")

    bug_desc = st.text_area(
        "Bug Description",
        placeholder="e.g. ValueError when cloning repository because destination directory already exists.",
        height=100
    )

    error_trace = st.text_area(
        "Error Log / Stack Trace (Optional)",
        placeholder="e.g. Traceback (most recent call last):\n  File 'github_loader.py', line 33, in clone_github_repo\nValueError: Repository already downloaded.",
        height=120
    )

    if st.button("🔎 Run Bug Investigation", type="primary", use_container_width=True):
        if not bug_desc.strip():
            st.error("Please enter a bug description.")
        else:
            with st.spinner("Locating relevant code, analyzing cause, and drafting fix..."):
                try:
                    payload = {
                        "bug_description": bug_desc.strip(),
                        "error_trace": error_trace.strip() if error_trace else None
                    }
                    res = requests.post(f"{API_BASE_URL}/repository/investigate-bug", json=payload, timeout=180)

                    if res.status_code == 200:
                        report = res.json()
                        st.success("Investigation complete!")

                        # Affected files pill list
                        affected = report.get("affected_files", [])
                        if affected:
                            st.markdown("### 🎯 Affected Files Identified:")
                            for af in affected:
                                st.markdown(f"- 📁 `{af}`")

                        # Diagnosis markdown
                        st.markdown("---")
                        st.markdown(report.get("diagnosis", "No diagnosis returned."))

                        # Source snippets
                        sources = report.get("sources", [])
                        if sources:
                            with st.expander("🔍 Inspected Code Evidence"):
                                for src in sources:
                                    st.markdown(f"**`{src.get('file_path')}`** (Lines {src.get('start_line')}-{src.get('end_line')}) - `{src.get('name')}`")
                    else:
                        st.error(f"Investigation failed ({res.status_code}): {res.text}")
                except Exception as e:
                    st.error(f"Error communicating with backend: {str(e)}")


# --- TAB 3: ARCHITECTURE & PIPELINE ---
with tab_arch:
    st.subheader("System Architecture & Pipeline Walkthrough")
    st.markdown("""
    CodeSage is designed around a multi-stage ingestion, indexing, and retrieval pipeline:

    #### 1. Ingestion & Tree-Sitter AST Parsing
    - **`github_loader.py`**: Securely clones public HTTPS GitHub repositories with `depth=1`.
    - **`file_loader.py`**: Walks the directory tree with directory-level pruning to ignore `venv`, `node_modules`, and `.git`.
    - **`code_parser.py`**: Tree-Sitter AST parser extracts module imports, variables, async/sync functions with decorators, classes, and methods.

    #### 2. Hybrid Retrieval with Reciprocal Rank Fusion (RRF)
    - **Dense Semantic Search**: `SentenceTransformer("all-MiniLM-L6-v2")` generates dense vector representations stored in ChromaDB.
    - **Sparse Keyword Search**: BM25 Okapi tokenizer splits identifiers on snake_case and camelCase.
    - **Reciprocal Rank Fusion (RRF)**:
    """)
    st.latex(r"RRF\_Score(d) = \sum_{m \in \{dense, sparse\}} \frac{1}{60 + rank_m(d)}")
    st.markdown("""
    #### 3. Grounded QA & Bug Investigation
    - Prompt enforces strict grounding on retrieved code chunks.
    - Prohibits hallucinated line numbers or symbols.
    - Directly queries Ollama running local `qwen2.5:3b`.
    """)

