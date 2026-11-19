# 🔮 CodeSage: AI-Powered Codebase Intelligence & Debugging Assistant

CodeSage is an end-to-end codebase intelligence and debugging assistant. It ingests public GitHub repositories or local source code, extracts high-fidelity Abstract Syntax Tree (AST) symbols, indexes them with hybrid semantic and lexical retrieval, and enables developers to ask questions and investigate bugs using a locally hosted LLM.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["1. Ingestion & AST Parsing"]
        GH[GitHub / Local Repo] --> FL[File Loader\n(os.walk Directory Pruning)]
        FL --> TS[Tree-Sitter Parser\n(Functions, Decorators, Classes, Methods, Imports, Variables)]
        TS --> CK[Smart Chunker\n(Grouped Imports & Line Context)]
    end

    subgraph Storage ["2. Hybrid Storage & Indexing"]
        CK --> ST[Sentence Transformers\nall-MiniLM-L6-v2]
        ST --> CDB[(ChromaDB\nVector Store)]
        CK --> BM[BM25 Okapi\nIdentifier Tokenizer]
    end

    subgraph Retrieval ["3. Hybrid Retrieval & RRF"]
        Query[Developer Query / Bug Report] --> VR[Dense Vector Search]
        Query --> LR[Sparse Lexical Search]
        CDB --> VR
        BM --> LR
        VR & LR --> RRF[Reciprocal Rank Fusion\nScore = Sum 1 / (60 + Rank)]
    end

    subgraph Generation ["4. Grounded Reasoning"]
        RRF --> CTX[Grounded Context Assembler]
        CTX --> Prompt[Structured Prompt & Safeguards]
        Prompt --> OLLAMA[Ollama: Qwen 2.5 3B]
        OLLAMA --> UI[Streamlit UI / FastAPI REST Endpoints]
    end
```

---

## 🚀 Key Features

1. **Precision Tree-Sitter AST Code Parsing**:
   - Parses Python files into rich structural symbols without regular expression fragility.
   - Preserves **decorators** (`@app.get`, `@dataclass`, `@property`), **async functions**, **classes**, **methods** with qualified names (`ClassName.method_name`), **module-level variables** (including tuple/list unpacking), and **imports**.
   - Graceful fallback for non-code files (e.g. Markdown, JSON) into logical line blocks.

2. **Hybrid Search with Reciprocal Rank Fusion (RRF)**:
   - Combines **Dense Vector Retrieval** (Sentence Transformers `all-MiniLM-L6-v2` in ChromaDB) and **Sparse Lexical Retrieval** (BM25 Okapi with snake_case and camelCase token splitting).
   - Fuses ranked candidate lists via RRF formula:
     $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{60 + \text{rank}_m(d)}$$
   - Captures both conceptual similarity (e.g. *"Where is the database client configured?"*) and exact identifier keywords (e.g. *"scan_repository"*).

3. **Grounded Codebase Q&A**:
   - Answers questions strictly using retrieved code chunks.
   - Outputs file names, symbol names, and exact 1-indexed line numbers.
   - Guards against hallucinations: returns explicit fallback when context is missing.

4. **Interactive Bug Investigation & Root Cause Analysis**:
   - Takes a bug description and optional stack trace.
   - Locates candidate failure points in the repository.
   - Produces a structured diagnosis: root cause, affected files & lines, code evidence, and read-only code fix suggestions.

5. **Polished Streamlit Interface & FastAPI Backend**:
   - Complete interactive web interface with chat history, source inspection drawers, index metrics, and bug investigation mode.

---

## 🛠️ Technology Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Backend Framework** | FastAPI + Uvicorn | RESTful API endpoints for scanning, Q&A, and bug investigation |
| **Language Model** | Ollama (`qwen2.5:3b`) | Local inference for code comprehension and fix generation |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) | 384-dimensional dense semantic vector representations |
| **Vector Database** | ChromaDB | Persistent local vector store with cosine/L2 search |
| **Lexical Search** | `rank-bm25` (BM25Okapi) | Exact keyword matching with code-aware tokenization |
| **Parser** | Tree-Sitter (`tree-sitter-language-pack`) | Fast, concrete syntax tree generation for source code |
| **Git Ingestion** | GitPython | Shallow cloning (`depth=1`) and local repository management |
| **Frontend** | Streamlit | Responsive dashboard and conversational UI |

---

## 📦 Setup & Installation

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.11)
- **Ollama**: Download from [ollama.ai](https://ollama.ai/)
- Pull the local Qwen model:
  ```bash
  ollama pull qwen2.5:3b
  ```

### 2. Environment Setup
```bash
# Clone or navigate into CodeSage
cd CodeSage

# Activate virtual environment
backend\venv\Scripts\activate   # Windows
# or: source backend/venv/bin/activate  # macOS / Linux

# Install dependencies (if setting up fresh)
pip install -r backend/requirements.txt
```

---

## 🏃 Running the Application

### Option A: Start the FastAPI Backend
```bash
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Option B: Start the Streamlit Frontend
In a separate terminal window:
```bash
$env:PYTHONPATH="backend"
streamlit run frontend/streamlit_app.py
```
Open your browser at [http://localhost:8501](http://localhost:8501).

---

## 🧪 Test Suite & Evaluation Benchmark

CodeSage includes a test suite covering unit parsing, chunking, bug analysis, and retrieval quality metrics (Hit@k, MRR):

```bash
# Run all automated tests
$env:PYTHONPATH="backend"
backend\venv\Scripts\python.exe -m unittest discover -s backend/tests -p "test_*.py"
```

### Retrieval Evaluation Benchmark Results
Evaluated on a representative query suite against the CodeSage codebase:
- **Hit@1 Accuracy**: `14.3%`
- **Hit@3 Accuracy**: `85.7%`
- **Hit@5 Accuracy**: `85.7%`
- **Mean Reciprocal Rank (MRR)**: `0.452`

---

## 💡 Technical Interview Cheat Sheet (For Students)

When explaining CodeSage in technical interviews, highlight these engineering decisions:

1. **Why AST Parsing over Naive Chunking (e.g. 500-character split)?**
   - Naive text slicing breaks functions in half, separates decorators from endpoints, and cuts off variable assignments.
   - Tree-Sitter parses concrete syntax trees, allowing chunks to be aligned with logical language boundaries (whole functions, methods with class context, grouped imports).

2. **Why Hybrid Search (BM25 + Dense Vectors) with RRF?**
   - Dense embeddings capture semantic similarity (e.g. *"retrieve files"* $\leftrightarrow$ *"file_loader"*) but often fail on arbitrary code identifiers (e.g. searching for a specific variable name like `OLLAMA_URL`).
   - BM25 excels at keyword precision but misses semantic phrasing.
   - Reciprocal Rank Fusion combines the ranks without requiring complex score normalization.

3. **How Hallucinations Are Mitigated:**
   - Strict grounding instructions prompt the LLM to answer *only* from the code context provided in the prompt.
   - If the hybrid search finds no confident context, the system states that information is unavailable instead of fabricating fictitious files or functions.

