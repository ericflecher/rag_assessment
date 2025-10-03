# Project Backlog

## Future Enhancements

### Feature: Multiple Questions Support

**Priority:** Medium
**Effort:** Small (~15 minutes)
**Category:** User Experience

**Description:**

Currently, the application accepts only one question per run and exits after providing an answer. Add support for continuous querying within a single session.

**Implementation:**

Wrap lines 87-114 in `src/rag_assessment_partial.py` with a loop:

```python
while True:
    query = input("\nEnter your question (or 'quit' to exit): ")
    if query.lower() in ['quit', 'exit', 'q']:
        break

    # Embed the query
    response = client.embeddings.create(
        input=[query],
        model=EMBED_MODEL
    )
    query_emb = np.array(response.data[0].embedding)

    # Retrieve top-k chunks
    sims = np.array([cosine_sim(query_emb, chunk_emb) for chunk_emb in chunk_embeddings])
    top_indices = np.argsort(sims)[-TOP_K:][::-1]
    top_chunks = [chunks[i] for i in top_indices]
    top_files = [sources[i] for i in top_indices]

    # Generate answer
    context = "\n\n".join([f"From {sources[i]}:\n{chunks[i]}" for i in top_indices])
    prompt = (
        f"Answer the following question using the provided context. "
        f"Cite at least two of the file names in your answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer (cite sources):"
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    answer = response.choices[0].message.content.strip()

    # Output JSON
    output = {
        "answer": answer,
        "sources": list(sorted(set(top_files)))[:2]
    }
    print(json.dumps(output, indent=2))
```

**Benefits:**

- Better user experience - no need to restart for each question
- Faster subsequent queries (embeddings already computed)
- More natural interaction flow

**Testing:**

```bash
$ python src/rag_assessment_partial.py
Embedding: 100%|████████████████████| 15/15 [00:03<00:00,  4.2it/s]

Enter your question (or 'quit' to exit): How do I reset my password?
{
  "answer": "...",
  "sources": ["faq_auth.md", "faq_sso.md"]
}

Enter your question (or 'quit' to exit): What is the PTO policy?
{
  "answer": "...",
  "sources": ["faq_employee.md"]
}

Enter your question (or 'quit' to exit): quit
$
```

---

## Production Enhancements

### Feature: Embedding Cache

**Priority:** High
**Effort:** Medium (~1-2 hours)
**Category:** Performance

**Description:**

Cache document embeddings to avoid re-computing on every run.

**Implementation Options:**

1. **File-based cache** (simplest):
   ```python
   import pickle

   CACHE_FILE = "embeddings_cache.pkl"

   if os.path.exists(CACHE_FILE):
       with open(CACHE_FILE, 'rb') as f:
           chunk_embeddings = pickle.load(f)
   else:
       chunk_embeddings = embed_texts(chunks)
       with open(CACHE_FILE, 'wb') as f:
           pickle.dump(chunk_embeddings, f)
   ```

2. **Hash-based cache** (invalidates on content change):
   - Hash FAQ directory contents
   - Store embeddings with hash key
   - Regenerate if hash changes

**Benefits:**

- Instant startup after first run
- Reduced API costs
- Better user experience

---

### Feature: Batch Embedding API Calls

**Priority:** Medium
**Effort:** Small (~30 minutes)
**Category:** Performance, Cost Optimization

**Description:**

Batch embed API calls to reduce latency and cost. OpenAI API supports up to 2048 inputs per request.

**Implementation:**

```python
def embed_texts_batch(texts, batch_size=100):
    """
    Embed texts in batches for better performance.
    OpenAI supports up to 2048 inputs per request.
    """
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            input=batch,
            model=EMBED_MODEL
        )
        batch_embeddings = [np.array(item.embedding) for item in response.data]
        embeddings.extend(batch_embeddings)
    return embeddings
```

**Benefits:**

- 10-100x faster embedding for large document sets
- Reduced API calls = lower costs
- Same accuracy

**Note:** This aligns with the production scalability suggestion to batch API calls (up to 2048 inputs per request)

---

### Feature: Semantic Chunking with Overlap

**Priority:** Medium
**Effort:** Medium (~1 hour)
**Category:** Accuracy

**Description:**

Replace character-based chunking with semantic chunking that respects sentence/paragraph boundaries. Add chunk overlap (e.g., 50 characters) to maintain context across chunk boundaries.

**Implementation:**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text_semantic(text, size=200, overlap=50):
    """
    Split text using semantic boundaries (sentences/paragraphs).
    Adds overlap to maintain context across chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)
```

**Benefits:**

- Better context preservation
- Improved retrieval accuracy
- More coherent chunks
- Overlap prevents information loss at boundaries

**Dependencies:**

```bash
pip install langchain
```

**Note:** This aligns with the production scalability suggestion to use semantic chunking with overlap

---

### Feature: Vector Database Integration

**Priority:** Low (for demo), High (for production)
**Effort:** Large (~4-8 hours)
**Category:** Scalability

**Description:**

Replace in-memory storage with a vector database.

**Options:**

1. **Qdrant** (easiest to self-host)
2. **Pinecone** (managed service)
3. **Weaviate** (open-source, feature-rich)
4. **ChromaDB** (lightweight, embedded)

**Example (ChromaDB):**

```python
import chromadb

client_db = chromadb.Client()
collection = client_db.create_collection("faqs")

# Add documents
collection.add(
    documents=chunks,
    embeddings=chunk_embeddings,
    metadatas=[{"source": src} for src in sources],
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

# Query
results = collection.query(
    query_embeddings=[query_emb],
    n_results=TOP_K
)
```

**Benefits:**

- Scales to millions of documents
- Persistent storage
- Advanced filtering and metadata support
- Built-in similarity search optimization

---

### Feature: Re-ranking after Retrieval

**Priority:** Medium
**Effort:** Medium (~2 hours)
**Category:** Accuracy

**Description:**

Add a re-ranking step after initial retrieval to improve relevance. This two-stage approach (retrieve candidates, then re-rank) improves answer quality.

**Implementation:**

Use a cross-encoder model to re-rank retrieved chunks:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# After initial retrieval
pairs = [[query, chunks[i]] for i in top_indices]
scores = reranker.predict(pairs)
reranked_indices = [top_indices[i] for i in np.argsort(scores)[::-1]]
```

**Benefits:**

- Higher precision
- Better handling of ambiguous queries
- ~10-20% accuracy improvement

**Note:** This aligns with the production scalability suggestion to add re-ranking step after initial retrieval

---

### Feature: Error Handling & Validation with Retry Logic

**Priority:** High
**Effort:** Small (~1 hour)
**Category:** Robustness

**Description:**

Add comprehensive error handling for API failures, missing files, empty queries, etc. Implement retry logic for API failures and fallback responses when no relevant chunks are found.

**Implementation:**

```python
import sys
from openai import OpenAIError

def main():
    try:
        # Validate API key
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY not set in environment", file=sys.stderr)
            sys.exit(1)

        # Validate FAQ directory
        if not os.path.exists(FAQ_DIR):
            print(f"Error: FAQ directory '{FAQ_DIR}' not found", file=sys.stderr)
            sys.exit(1)

        chunks, sources = load_and_chunk_faqs(FAQ_DIR)

        if not chunks:
            print(f"Error: No .md files found in '{FAQ_DIR}'", file=sys.stderr)
            sys.exit(1)

        # Embed with retry logic
        chunk_embeddings = embed_texts_with_retry(chunks)

        # Query loop with validation
        while True:
            query = input("\nEnter your question (or 'quit' to exit): ").strip()

            if not query:
                print("Please enter a question.")
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                break

            # Process query with error handling
            try:
                # ... embedding and generation
                pass
            except OpenAIError as e:
                print(f"API Error: {e}", file=sys.stderr)
                continue

    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

def embed_texts_with_retry(texts, max_retries=3):
    """Embed texts with exponential backoff retry."""
    import time

    embeddings = []
    for text in tqdm(texts, desc="Embedding"):
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    input=[text],
                    model=EMBED_MODEL
                )
                embedding = np.array(response.data[0].embedding)
                embeddings.append(embedding)
                break
            except OpenAIError as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                print(f"Retry {attempt+1}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)

    return embeddings
```

**Benefits:**

- Graceful degradation
- Better user feedback
- Production-ready reliability

**Note:** This aligns with production scalability suggestions for retry logic, fallback responses, and input validation

---

### Feature: Query Expansion & Rewriting

**Priority:** Medium
**Effort:** Medium (~2 hours)
**Category:** Accuracy

**Description:**

Expand user queries with synonyms/related terms to improve recall. Implement query expansion/rewriting to handle different phrasings of the same question.

**Implementation:**

```python
def expand_query(query):
    """Use LLM to generate query variations."""
    prompt = f"Generate 2-3 alternative phrasings of this question:\n{query}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    alternatives = response.choices[0].message.content.strip().split('\n')
    return [query] + alternatives

# Then embed all variations and aggregate results
```

**Benefits:**

- Better handling of ambiguous queries
- Improved recall
- More robust to query phrasing

**Note:** This aligns with the production scalability suggestion to implement query expansion/rewriting

---

### Feature: Logging & Observability with Metrics

**Priority:** High (for production)
**Effort:** Medium (~2 hours)
**Category:** Monitoring

**Description:**

Add structured logging for debugging and analysis. Log query-answer pairs for evaluation and track latency metrics (embedding, retrieval, generation). Monitor API costs and rate limits.

**Implementation:**

```python
import logging
from datetime import datetime
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_queries.log'),
        logging.StreamHandler()
    ]
)

# In query loop
start_time = time.time()
logging.info(f"Query: {query}")

# Track embedding time
embed_start = time.time()
# ... embedding code
embed_time = time.time() - embed_start
logging.info(f"Embedding latency: {embed_time:.3f}s")

# Track retrieval time
retrieval_start = time.time()
# ... retrieval code
retrieval_time = time.time() - retrieval_start
logging.info(f"Retrieval latency: {retrieval_time:.3f}s")

# Track generation time
gen_start = time.time()
# ... generation code
gen_time = time.time() - gen_start
logging.info(f"Generation latency: {gen_time:.3f}s")

logging.info(f"Top chunks: {top_indices}")
logging.info(f"Similarity scores: {sims[top_indices]}")
logging.info(f"Sources: {top_files}")
logging.info(f"Answer length: {len(answer)}")
logging.info(f"Total latency: {time.time() - start_time:.3f}s")
```

**Benefits:**

- Debugging support
- Query analytics
- Performance monitoring
- Quality assessment
- Cost tracking
- Latency analysis

**Note:** This aligns with production scalability suggestions for logging query-answer pairs and tracking latency metrics

---

---

### Feature: Chat Logging with Similarity Scores

**Priority:** Medium
**Effort:** Small (~30 minutes)
**Category:** Observability, Debugging

**Description:**

Save chat interactions to JSON files in `logs/chats/` directory for analysis, debugging, and quality assessment. Each chat session will be logged with the user question, generated answer, sources, and similarity scores for transparency.

**Implementation:**

**1. Create logging function:**

```python
import json
import os
from datetime import datetime

def save_chat_log(query, answer, sources, top_indices, sims, chunks):
    """
    Save chat interaction to logs/chats/ directory.

    Args:
        query: User question
        answer: Generated answer
        sources: List of source files
        top_indices: Indices of retrieved chunks
        sims: Similarity scores array
        chunks: All text chunks
    """
    # Create logs directory if it doesn't exist
    log_dir = "logs/chats"
    os.makedirs(log_dir, exist_ok=True)

    # Generate timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{log_dir}/chat_{timestamp}.json"

    # Build log data with similarity scores
    retrieved_chunks = []
    for idx, i in enumerate(top_indices):
        retrieved_chunks.append({
            "rank": idx + 1,
            "source": sources[i],
            "similarity_score": float(sims[i]),
            "chunk_text": chunks[i]
        })

    # Get unique sources with their best similarity scores
    source_scores = {}
    for i in top_indices:
        source = sources[i]
        score = float(sims[i])
        if source not in source_scores or score > source_scores[source]:
            source_scores[source] = score

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "sources": {
            "files": list(set([sources[i] for i in top_indices])),
            "scores": source_scores
        },
        "retrieval": {
            "top_k": len(top_indices),
            "chunks": retrieved_chunks
        },
        "metadata": {
            "embed_model": "text-embedding-ada-002",
            "llm_model": "gpt-3.5-turbo",
            "chunk_size": 200
        }
    }

    # Write to file
    with open(filename, 'w') as f:
        json.dump(log_data, f, indent=2)

    return filename
```

**2. Update main() to call logging:**

```python
def main():
    # ... existing code ...

    # --- 5. Generate Answer ---
    # ... existing answer generation ...

    # --- 6. Output JSON ---
    unique_sources = []
    for source in top_files:
        if source not in unique_sources:
            unique_sources.append(source)

    output = {
        "answer": answer,
        "sources": unique_sources
    }
    print(json.dumps(output, indent=2))

    # --- 7. Save Chat Log ---
    log_file = save_chat_log(query, answer, sources, top_indices, sims, chunks)
    print(f"\n💾 Chat saved to: {log_file}", file=sys.stderr)
```

**3. Update `.gitignore`:**

```gitignore
# Chat logs
logs/
*.log
```

**Example Output File:** `logs/chats/chat_20250103_143022.json`

```json
{
  "timestamp": "2025-01-03T14:30:22.123456",
  "query": "What is the PTO policy?",
  "answer": "The PTO policy at TechFlow Solutions offers unlimited PTO with a minimum requirement of 15 days per year...",
  "sources": {
    "files": [
      "faq_employee.md",
      "faq_sso.md"
    ],
    "scores": {
      "faq_employee.md": 0.8482,
      "faq_sso.md": 0.7194
    }
  },
  "retrieval": {
    "top_k": 4,
    "chunks": [
      {
        "rank": 1,
        "source": "faq_employee.md",
        "similarity_score": 0.8482,
        "chunk_text": "# TechFlow Solutions Employee Handbook\n\n## What is our unlimited PTO policy?\nTechFlow offers unlimited PTO with a minimum requirement of 15 days per year..."
      },
      {
        "rank": 2,
        "source": "faq_employee.md",
        "similarity_score": 0.8135,
        "chunk_text": "eeding 2 consecutive weeks. PTO requests require 2-week notice except for emergencies..."
      },
      {
        "rank": 3,
        "source": "faq_employee.md",
        "similarity_score": 0.7332,
        "chunk_text": "r 1, then 22.5% each subsequent year. Equity grants are reviewed annually in March..."
      },
      {
        "rank": 4,
        "source": "faq_sso.md",
        "similarity_score": 0.7194,
        "chunk_text": "# SSO FAQ\n\nQ: How do I enable SSO?\nA: Contact your admin to enable SSO for your account."
      }
    ]
  },
  "metadata": {
    "embed_model": "text-embedding-ada-002",
    "llm_model": "gpt-3.5-turbo",
    "chunk_size": 200
  }
}
```

**Benefits:**

- ✅ **Debugging**: See exactly which chunks were retrieved and their scores
- ✅ **Quality Assessment**: Review if low-relevance sources are being included
- ✅ **Analytics**: Track query patterns and performance over time
- ✅ **Reproducibility**: Full context of each interaction saved
- ✅ **Transparency**: Understand why certain answers were generated
- ✅ **Training Data**: Potential use for fine-tuning or evaluation

**Usage:**

After each chat interaction, a timestamped JSON file is automatically created:

```bash
python src/rag_assessment_partial.py
# Enter question: What is the PTO policy?
# ... answer displayed ...
# 💾 Chat saved to: logs/chats/chat_20250103_143022.json
```

**Analysis Scripts (Future Enhancement):**

```bash
# Count total chats
ls logs/chats/ | wc -l

# Find low-similarity retrievals
jq '.retrieval.chunks[] | select(.similarity_score < 0.75)' logs/chats/*.json

# Average similarity by source file
jq '.sources.scores' logs/chats/*.json | jq -s 'add'
```

**Configuration Option:**

Add to config section to enable/disable logging:

```python
# --- Config ---
FAQ_DIR = "faqs"
EMBED_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo"
CHUNK_SIZE = 200
TOP_K = 4
ENABLE_LOGGING = True  # Set to False to disable chat logging
```

**Privacy Considerations:**

- Logs contain full query and answer text
- Ensure logs/ directory is in .gitignore
- Consider adding log rotation or auto-cleanup for production
- Could add flag to disable logging for sensitive queries

**Testing:**

```bash
# Run a few queries
python src/rag_assessment_partial.py
# Enter: What is the PTO policy?

python src/rag_assessment_partial.py
# Enter: How do I reset my password?

# Verify logs were created
ls -lh logs/chats/

# View a log file
cat logs/chats/chat_*.json | jq '.'
```

**Note:** This feature provides the observability needed to understand and debug the "why is faq_sso.md in sources?" type questions by showing exact similarity scores for each retrieved chunk.

---

### Feature: UV Package Manager Integration

**Priority:** Low
**Effort:** Small (~30 minutes)
**Category:** Developer Experience, Dependency Management

**Description:**

Transition from traditional pip/venv to UV for faster, simpler dependency management. UV is a modern Python package manager written in Rust that provides significantly faster installation and better dependency resolution.

**Benefits:**

- ✅ **10-100x faster** than pip for package installation
- ✅ **Simpler commands** - single tool for environment and dependencies
- ✅ **Better dependency resolution** - handles conflicts more intelligently
- ✅ **Automatic virtual environment management** - no manual venv creation
- ✅ **Lock file support** - reproducible builds with `uv.lock`
- ✅ **Drop-in replacement** - compatible with existing pip workflows

**Implementation:**

**1. Install UV:**
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via Homebrew
brew install uv

# Or via pip (ironically)
pip install uv
```

**2. Create `pyproject.toml`:**
```toml
[project]
name = "rag-assessment"
version = "0.1.0"
description = "RAG FAQ Question Answering System"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "openai>=1.0.0",
    "numpy>=1.24.0",
    "tqdm>=4.65.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]
local = [
    "sentence-transformers>=2.2.0",
    "ollama>=0.1.0",
]
production = [
    "langchain>=0.1.0",
    "chromadb>=0.4.0",
    "sentence-transformers>=2.2.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**3. Update Setup Instructions in README:**

**Old (pip/venv):**
```bash
python3 -m venv venv
source venv/bin/activate
pip install openai numpy tqdm python-dotenv
```

**New (UV):**
```bash
# Install dependencies and create venv in one command
uv sync

# Or specify extra groups
uv sync --extra local  # Install with local model support
uv sync --extra dev    # Install with dev tools
```

**4. Running the Application:**

**Old:**
```bash
source venv/bin/activate
python src/rag_assessment_partial.py
```

**New:**
```bash
# UV automatically activates the venv
uv run python src/rag_assessment_partial.py

# Or with extra dependencies
uv run --extra local python src/rag_assessment_partial.py
```

**5. Adding Dependencies:**

**Old:**
```bash
pip install new-package
pip freeze > requirements.txt
```

**New:**
```bash
uv add new-package
# Automatically updates pyproject.toml and uv.lock
```

**6. Create `requirements.txt` for Compatibility:**

For users without UV, generate requirements.txt:
```bash
uv pip compile pyproject.toml -o requirements.txt
```

**Migration Steps:**

1. Install UV
2. Create `pyproject.toml` with current dependencies
3. Run `uv sync` to create lock file
4. Update `.gitignore` to include `uv.lock` (or commit it for reproducibility)
5. Update README.md with new setup instructions
6. Keep `requirements.txt` for backwards compatibility

**Backwards Compatibility:**

Maintain both workflows:

```markdown
## Setup

### Option 1: UV (Recommended)
```bash
uv sync
uv run python src/rag_assessment_partial.py
```

### Option 2: Traditional pip
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/rag_assessment_partial.py
```
```

**Performance Comparison:**

| Task | pip | UV | Speedup |
|------|-----|----|----|
| Install deps (cold) | ~45s | ~2s | 22x |
| Install deps (cached) | ~15s | ~0.3s | 50x |
| Dependency resolution | ~10s | ~0.1s | 100x |
| Virtual env creation | ~3s | Automatic | N/A |

**Project Structure Update:**

```
rag_assessment/
├── pyproject.toml           # NEW: Project metadata and dependencies
├── uv.lock                  # NEW: Lock file for reproducibility
├── requirements.txt         # Keep for compatibility
├── .env.template
├── .gitignore
├── README.md
├── faqs/
├── src/
│   └── rag_assessment_partial.py
└── planning/
```

**Update `.gitignore`:**

```gitignore
# UV
.venv/
uv.lock  # Optional: commit for reproducibility, ignore for flexibility

# OR keep uv.lock in git for reproducible builds
# (remove uv.lock line from .gitignore)
```

**Example Commands:**

```bash
# Initial setup
uv sync

# Run with different dependency sets
uv run python src/rag_assessment_partial.py                    # Base deps
uv run --extra local python src/rag_assessment_partial.py      # With local models
uv run --extra production python src/rag_assessment_partial.py # With prod features

# Development
uv add --dev pytest        # Add dev dependency
uv add numpy --upgrade     # Upgrade specific package
uv pip list                # List installed packages

# Generate requirements.txt for compatibility
uv pip compile pyproject.toml -o requirements.txt
```

**Tradeoffs:**

- ✅ **Much faster** - significant time savings during development
- ✅ **Simpler workflow** - fewer commands to remember
- ✅ **Better tooling** - modern approach to Python packaging
- ⚠️ **New dependency** - users need to install UV first
- ⚠️ **Less familiar** - not as widely known as pip (yet)
- ⚠️ **Ecosystem adoption** - still newer, though growing rapidly

**Recommendation:**

Implement UV as the recommended approach while maintaining pip compatibility for maximum accessibility during the interview demo.

**Note:** UV is developed by Astral (creators of Ruff), showing strong backing and future support.

---

### Feature: Local Model Support (No API Keys Required)

**Priority:** Medium (Bonus points for interview)
**Effort:** Medium (~3-4 hours)
**Category:** Cost Optimization, Privacy, Offline Support

**Description:**

Replace OpenAI API calls with local models to eliminate API key dependency, reduce costs, and enable offline operation. This addresses the bonus points criteria: "If it can run with local models (and doesn't require API keys)".

**Implementation:**

Replace OpenAI embeddings and chat completions with local alternatives:

**Option 1: Using Sentence Transformers + Ollama**

```python
from sentence_transformers import SentenceTransformer
import ollama
import numpy as np

# --- Config ---
EMBED_MODEL_LOCAL = "all-MiniLM-L6-v2"  # 384-dim embeddings, fast
LLM_MODEL_LOCAL = "llama3.2"  # or "mistral", "phi3", etc.

# Initialize local models
embedding_model = SentenceTransformer(EMBED_MODEL_LOCAL)

def embed_texts_local(texts):
    """
    Embed texts using local sentence-transformers model.
    No API calls required.
    """
    embeddings = []
    for text in tqdm(texts, desc="Embedding (local)"):
        embedding = embedding_model.encode(text, convert_to_numpy=True)
        embeddings.append(embedding)
    return embeddings

def generate_answer_local(prompt):
    """
    Generate answer using local Ollama model.
    No API calls required.
    """
    response = ollama.chat(
        model=LLM_MODEL_LOCAL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}
    )
    return response['message']['content'].strip()

# In main():
# Replace: chunk_embeddings = embed_texts(chunks)
chunk_embeddings = embed_texts_local(chunks)

# Replace: query_emb = np.array(response.data[0].embedding)
query_emb = embedding_model.encode(query, convert_to_numpy=True)

# Replace: answer = response.choices[0].message.content.strip()
answer = generate_answer_local(prompt)
```

**Option 2: Using Hugging Face Transformers (Fully Offline)**

```python
from transformers import AutoTokenizer, AutoModel, pipeline
import torch

# Load models once at startup
embed_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embed_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# For text generation (smaller model for faster inference)
llm_pipeline = pipeline(
    "text-generation",
    model="microsoft/Phi-3-mini-4k-instruct",
    device="cpu",  # or "cuda" if GPU available
    max_new_tokens=512
)

def embed_text_hf(text):
    """Embed using Hugging Face model."""
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    # Mean pooling
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

def generate_answer_hf(prompt):
    """Generate answer using Hugging Face pipeline."""
    result = llm_pipeline(prompt, max_new_tokens=512, temperature=0.2, do_sample=True)
    return result[0]['generated_text'].replace(prompt, '').strip()
```

**Setup Instructions:**

**For Ollama (Recommended for ease of use):**
```bash
# 1. Install Ollama
# macOS:
brew install ollama

# 2. Pull a model
ollama pull llama3.2  # or mistral, phi3, etc.

# 3. Install Python dependencies
pip install sentence-transformers ollama
```

**For Hugging Face (Fully offline):**
```bash
pip install transformers torch sentence-transformers
```

**Model Comparison:**

| Model Type | Option | Size | Speed | Quality | Offline |
|------------|--------|------|-------|---------|---------|
| Embeddings | OpenAI ada-002 | N/A (API) | Fast | Excellent | ❌ |
| Embeddings | all-MiniLM-L6-v2 | 80MB | Very Fast | Good | ✅ |
| Embeddings | all-mpnet-base-v2 | 420MB | Medium | Excellent | ✅ |
| LLM | GPT-3.5-turbo | N/A (API) | Fast | Excellent | ❌ |
| LLM | Llama 3.2 (3B) | 2GB | Medium | Very Good | ✅ |
| LLM | Phi-3 Mini | 2.4GB | Fast | Good | ✅ |
| LLM | Mistral 7B | 4GB | Slow (CPU) | Excellent | ✅ |

**Benefits:**

- ✅ **No API keys required** (addresses bonus points)
- ✅ **Zero API costs** - completely free after initial download
- ✅ **Privacy** - data never leaves your machine
- ✅ **Offline operation** - works without internet
- ✅ **Deterministic** - same results every time
- ✅ **Lower latency** - no network calls (if using GPU)

**Tradeoffs:**

- ⚠️ **Initial setup** - requires downloading models (2-4GB)
- ⚠️ **Hardware requirements** - GPU recommended for LLMs (but works on CPU)
- ⚠️ **Slower inference** - especially for LLMs on CPU
- ⚠️ **Slightly lower quality** - local models may not match GPT-3.5-turbo

**Configuration Toggle (Best of Both Worlds):**

```python
import os

USE_LOCAL_MODELS = os.getenv("USE_LOCAL_MODELS", "false").lower() == "true"

if USE_LOCAL_MODELS:
    from sentence_transformers import SentenceTransformer
    import ollama
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Using local models (no API key required)")
else:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("Using OpenAI API")

def embed_texts(texts):
    if USE_LOCAL_MODELS:
        return embed_texts_local(texts)
    else:
        return embed_texts_openai(texts)

# Similar pattern for answer generation
```

**Usage:**
```bash
# Run with OpenAI (default)
python src/rag_assessment_partial.py

# Run with local models (no API key)
USE_LOCAL_MODELS=true python src/rag_assessment_partial.py
```

**Testing:**

Verify that answers are comparable between OpenAI and local models:

```bash
# Test with OpenAI
echo "What is the PTO policy?" | python src/rag_assessment_partial.py

# Test with local models
echo "What is the PTO policy?" | USE_LOCAL_MODELS=true python src/rag_assessment_partial.py

# Compare results
```

**Recommended Models for This Use Case:**

1. **Embeddings:** `all-MiniLM-L6-v2` - Fast, small, good quality
2. **LLM (CPU):** `phi3:mini` - Best balance of speed/quality on CPU
3. **LLM (GPU):** `llama3.2:3b` - Better quality, still fast on GPU

**Note:** This feature directly addresses the bonus points criterion: "If it can run with local models (and doesn't require API keys)"

---

## Production Scalability Summary

The features above align with key production scalability suggestions:

### Embedding Optimization
- ✅ **Batch API calls** (up to 2048 inputs per request) - See "Batch Embedding API Calls"
- ✅ **Cache embeddings with TTL** - See "Embedding Cache with TTL"
- 🔄 **Consider newer models** (text-embedding-3-small/large) - Future consideration

### Chunking Strategy
- ✅ **Use semantic chunking** (sentence/paragraph boundaries) - See "Semantic Chunking with Overlap"
- ✅ **Add chunk overlap** (e.g., 50 characters) to maintain context - See "Semantic Chunking with Overlap"
- ✅ **Consider RecursiveCharacterTextSplitter** from LangChain - Implemented in semantic chunking feature

### Vector Database Integration
- ✅ **Replace in-memory storage** with Pinecone, Weaviate, or Qdrant - See "Vector Database Integration"
- ✅ **Persistent storage** of embeddings - Covered in vector database feature
- ✅ **Efficient similarity search** at scale - Covered in vector database feature

### Answer Quality
- ✅ **Add re-ranking step** after initial retrieval - See "Re-ranking after Retrieval"
- ✅ **Implement query expansion/rewriting** - See "Query Expansion & Rewriting"
- 🔄 **Use prompt templates with few-shot examples** - Future consideration

### Observability
- ✅ **Log query-answer pairs** for evaluation - See "Logging & Observability with Metrics"
- ✅ **Track latency metrics** (embedding, retrieval, generation) - See "Logging & Observability with Metrics"
- ✅ **Monitor API costs and rate limits** - Covered in logging feature

### Error Handling
- ✅ **Retry logic** for API failures - See "Error Handling & Validation with Retry Logic"
- ✅ **Fallback responses** when no relevant chunks found - See "Error Handling & Validation with Retry Logic"
- ✅ **Input validation and sanitization** - See "Error Handling & Validation with Retry Logic"

---

## Dependencies to Add

### Current (Required for Base Implementation)
```
openai
numpy
tqdm
python-dotenv
```

### Future (as features are implemented)
```
langchain                  # For semantic chunking
chromadb                   # For vector database (lightweight option)
pinecone-client           # For Pinecone vector database (managed service)
qdrant-client             # For Qdrant vector database (self-hosted)
sentence-transformers     # For re-ranking AND local embeddings
ollama                     # For local LLM (easiest setup)
transformers               # For Hugging Face models (fully offline)
torch                      # Required for transformers
```

### For Local Models (Bonus Points)
```
# Option 1: Ollama (Recommended)
sentence-transformers     # For local embeddings
ollama                     # For local LLM inference

# Option 2: Fully Offline (Hugging Face)
transformers               # For both embeddings and LLM
torch                      # Required for transformers
sentence-transformers     # Alternative for embeddings
```
