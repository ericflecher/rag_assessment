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

**Quick Summary:**

- 10-100x faster than pip
- Single tool for environment and dependencies
- Lock file support for reproducibility
- Maintains pip compatibility

**See:** Full implementation details in original backlog section

### Feature: Multi-Model Support

**Priority:** High
**Effort:** Medium (~2-3 hours)
**Category:** Flexibility, Cost Optimization

**Description:**

Add support for switching between different embedding and LLM models (OpenAI, Anthropic, local models via Ollama). Allow runtime model selection and support multiple providers simultaneously.

**Implementation:**

```python
# --- Config ---
FAQ_DIR = "faqs"
CHUNK_SIZE = 200
TOP_K = 4

# Model configurations
MODELS = {
    "openai": {
        "embed_model": "text-embedding-ada-002",
        "llm_model": "gpt-3.5-turbo",
        "provider": "openai"
    },
    "openai-advanced": {
        "embed_model": "text-embedding-3-large",
        "llm_model": "gpt-4-turbo",
        "provider": "openai"
    },
    "anthropic": {
        "embed_model": "voyage-2",  # Via Voyage AI
        "llm_model": "claude-3-5-sonnet-20241022",
        "provider": "anthropic"
    },
    "local": {
        "embed_model": "all-MiniLM-L6-v2",
        "llm_model": "llama3.2:3b",
        "provider": "ollama"
    }
}

# Select active model configuration
ACTIVE_CONFIG = "openai"  # Change to switch models

def get_embedding_client(provider):
    """Return appropriate embedding client based on provider."""
    if provider == "openai":
        from openai import OpenAI
        return OpenAI()
    elif provider == "anthropic":
        # Use Voyage AI for embeddings with Anthropic
        import voyageai
        return voyageai.Client()
    elif provider == "ollama":
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODELS[ACTIVE_CONFIG]["embed_model"])
    else:
        raise ValueError(f"Unknown provider: {provider}")

def get_llm_client(provider):
    """Return appropriate LLM client based on provider."""
    if provider == "openai":
        from openai import OpenAI
        return OpenAI()
    elif provider == "anthropic":
        import anthropic
        return anthropic.Anthropic()
    elif provider == "ollama":
        import ollama
        return ollama.Client()
    else:
        raise ValueError(f"Unknown provider: {provider}")

def embed_texts(texts, provider, model_name):
    """Embed texts using specified provider."""
    embeddings = []

    if provider == "openai":
        client = get_embedding_client(provider)
        for text in tqdm(texts, desc="Embedding"):
            response = client.embeddings.create(
                input=[text],
                model=model_name
            )
            embeddings.append(np.array(response.data[0].embedding))

    elif provider == "anthropic":
        # Use Voyage AI for embeddings
        client = get_embedding_client(provider)
        for text in tqdm(texts, desc="Embedding"):
            result = client.embed([text], model=model_name)
            embeddings.append(np.array(result.embeddings[0]))

    elif provider == "ollama":
        # Use SentenceTransformers for local embeddings
        model = get_embedding_client(provider)
        embeddings = [np.array(emb) for emb in model.encode(texts, show_progress_bar=True)]

    return embeddings

def generate_answer(query, context, provider, model_name):
    """Generate answer using specified LLM provider."""
    prompt = (
        f"Answer the following question using the provided context. "
        f"Cite at least two of the file names in your answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer (cite sources):"
    )

    if provider == "openai":
        client = get_llm_client(provider)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    elif provider == "anthropic":
        client = get_llm_client(provider)
        message = client.messages.create(
            model=model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    elif provider == "ollama":
        client = get_llm_client(provider)
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
```

**CLI Interface (Optional Enhancement):**

```python
import argparse

parser = argparse.ArgumentParser(description="RAG Q&A with multiple model support")
parser.add_argument(
    "--model",
    choices=list(MODELS.keys()),
    default="openai",
    help="Model configuration to use"
)
args = parser.parse_args()

ACTIVE_CONFIG = args.model
```

**Usage:**

```bash
# Use OpenAI (default)
python src/rag_assessment_partial.py

# Use Claude with Anthropic
python src/rag_assessment_partial.py --model anthropic

# Use local Ollama models (no API costs)
python src/rag_assessment_partial.py --model local
```

**Benefits:**

- **Cost Optimization**: Choose cheaper models for development, better models for production
- **Privacy**: Use local models for sensitive data
- **Performance Testing**: Compare model quality across providers
- **Vendor Flexibility**: Avoid lock-in to single provider
- **Offline Support**: Local models work without internet

**Dependencies:**

```bash
# OpenAI (already installed)
pip install openai

# Anthropic + Voyage AI embeddings
pip install anthropic voyageai

# Local models (Ollama)
pip install ollama sentence-transformers

# Or install all at once
pip install openai anthropic voyageai ollama sentence-transformers
```

**Environment Variables:**

```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
VOYAGE_API_KEY=pa-...
```

**Model Comparison Matrix:**

| Provider | Embedding Model | LLM Model | Cost/1K | Speed | Quality | Offline |
|----------|----------------|-----------|---------|-------|---------|---------|
| OpenAI | ada-002 | gpt-3.5-turbo | $0.0002 | Fast | Good | ❌ |
| OpenAI | embedding-3-large | gpt-4-turbo | $0.0003 | Medium | Excellent | ❌ |
| Anthropic | voyage-2 | claude-3.5-sonnet | $0.0001 | Fast | Excellent | ❌ |
| Ollama | all-MiniLM-L6-v2 | llama3.2:3b | $0 | Very Fast | Good | ✅ |

**Testing:**

```bash
# Test each provider
for model in openai anthropic local; do
    echo "Testing $model..."
    python src/rag_assessment_partial.py --model $model <<< "What is the PTO policy?"
done

# Compare results
diff <(python src/rag_assessment_partial.py --model openai <<< "question") \
     <(python src/rag_assessment_partial.py --model anthropic <<< "question")
```

**Note:** This enables the flexibility to switch between cloud and local models, supporting the full range from development (local) to production (OpenAI/Anthropic).

---

### Feature: Multiple Document Format Support

**Priority:** Medium
**Effort:** Medium (~2-3 hours)
**Category:** Flexibility, Data Ingestion

**Description:**

Extend the document loader to support multiple file formats beyond Markdown (.md), including PDF, Word documents (.docx), plain text (.txt), HTML, and more. This allows the RAG system to ingest knowledge from diverse document sources.

**Implementation:**

```python
import os
from pathlib import Path
from typing import List, Tuple

# Add document parsing libraries
from pypdf import PdfReader
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
import chardet

def load_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    reader = PdfReader(file_path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text())
    return "\n\n".join(text)

def load_docx(file_path: str) -> str:
    """Extract text from Word document."""
    doc = DocxDocument(file_path)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n\n".join(paragraphs)

def load_html(file_path: str) -> str:
    """Extract text from HTML file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text(separator="\n", strip=True)

def load_txt(file_path: str) -> str:
    """Load plain text file with encoding detection."""
    # Detect encoding
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding'] or 'utf-8'

    # Read with detected encoding
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()

def load_markdown(file_path: str) -> str:
    """Load markdown file (existing implementation)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Document loader registry
DOCUMENT_LOADERS = {
    '.md': load_markdown,
    '.markdown': load_markdown,
    '.txt': load_txt,
    '.text': load_txt,
    '.pdf': load_pdf,
    '.docx': load_docx,
    '.doc': load_docx,  # Note: .doc requires python-docx or additional tools
    '.html': load_html,
    '.htm': load_html,
}

def load_and_chunk_documents(directory: str, chunk_size: int = 200) -> Tuple[List[str], List[str]]:
    """
    Load documents from directory supporting multiple formats.

    Args:
        directory: Path to directory containing documents
        chunk_size: Size of text chunks in characters

    Returns:
        Tuple of (chunks, sources) where sources contain filename and format
    """
    chunks = []
    sources = []

    # Scan directory for supported file types
    supported_extensions = list(DOCUMENT_LOADERS.keys())

    for file_path in Path(directory).rglob('*'):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        if ext not in supported_extensions:
            continue

        try:
            # Load document using appropriate loader
            loader = DOCUMENT_LOADERS[ext]
            text = loader(str(file_path))

            # Chunk the text
            file_chunks = chunk_text(text, chunk_size)

            # Track source with file type
            source_name = f"{file_path.name}"
            chunks.extend(file_chunks)
            sources.extend([source_name] * len(file_chunks))

            print(f"✓ Loaded {file_path.name} ({ext}) - {len(file_chunks)} chunks")

        except Exception as e:
            print(f"✗ Error loading {file_path.name}: {e}")
            continue

    return chunks, sources

def chunk_text(text: str, size: int = 200) -> List[str]:
    """Split text into chunks (existing implementation)."""
    return [text[i:i+size] for i in range(0, len(text), size)]
```

**Updated Main Function:**

```python
def main():
    # Load environment variables
    load_dotenv()

    # --- 1. Load and chunk documents (now supports multiple formats) ---
    chunks, sources = load_and_chunk_documents(FAQ_DIR, CHUNK_SIZE)

    if not chunks:
        print(f"Error: No supported documents found in '{FAQ_DIR}'", file=sys.stderr)
        print(f"Supported formats: {', '.join(DOCUMENT_LOADERS.keys())}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoaded {len(chunks)} chunks from {len(set(sources))} documents")
    print(f"Document types: {', '.join(set(Path(s).suffix for s in sources))}")

    # ... rest of implementation
```

**Configuration:**

```python
# --- Config ---
FAQ_DIR = "faqs"  # Can now contain .md, .pdf, .docx, .txt, .html files
CHUNK_SIZE = 200
TOP_K = 4

# Optional: Filter by document types
ALLOWED_FORMATS = ['.md', '.pdf', '.docx', '.txt']  # None = allow all
```

**Dependencies:**

```bash
# Core dependencies
pip install pypdf python-docx beautifulsoup4 chardet lxml

# Optional: For better PDF extraction
pip install pdfplumber  # Alternative to pypdf with better table support

# Optional: For .doc (legacy Word) support
pip install pywin32  # Windows only, or use LibreOffice
```

**Requirements.txt Update:**

```txt
openai>=1.0.0
numpy>=1.24.0
tqdm>=4.65.0
python-dotenv>=1.0.0

# Document parsing
pypdf>=3.17.0
python-docx>=1.1.0
beautifulsoup4>=4.12.0
chardet>=5.2.0
lxml>=4.9.0
```

**Directory Structure:**

```
faqs/
├── faq_auth.md          # Markdown (existing)
├── faq_sso.md           # Markdown (existing)
├── faq_employee.md      # Markdown (existing)
├── benefits.pdf         # PDF (new)
├── handbook.docx        # Word (new)
├── policies.html        # HTML (new)
└── guidelines.txt       # Plain text (new)
```

**Benefits:**

- **Versatile Ingestion**: Accept documents in common business formats
- **Legacy Support**: Work with existing PDF/Word documentation
- **Web Content**: Ingest HTML documentation or scraped web pages
- **Gradual Migration**: No need to convert everything to Markdown
- **Enterprise Ready**: Support formats used in corporate environments

**Advanced Features (Optional):**

```python
# 1. Document metadata extraction
def load_pdf_with_metadata(file_path: str) -> Tuple[str, dict]:
    """Extract text and metadata from PDF."""
    reader = PdfReader(file_path)
    text = "\n\n".join([page.extract_text() for page in reader.pages])
    metadata = {
        'author': reader.metadata.get('/Author', 'Unknown'),
        'title': reader.metadata.get('/Title', Path(file_path).stem),
        'pages': len(reader.pages),
        'created': reader.metadata.get('/CreationDate', None)
    }
    return text, metadata

# 2. Table extraction from PDFs
import pdfplumber

def load_pdf_with_tables(file_path: str) -> str:
    """Extract text and tables from PDF."""
    with pdfplumber.open(file_path) as pdf:
        text_parts = []
        for page in pdf.pages:
            # Extract text
            text_parts.append(page.extract_text())

            # Extract tables as markdown
            tables = page.extract_tables()
            for table in tables:
                table_md = "\n".join([" | ".join(row) for row in table])
                text_parts.append(f"\n\n{table_md}\n\n")

        return "\n\n".join(text_parts)

# 3. Image OCR from PDFs (requires tesseract)
from pdf2image import convert_from_path
import pytesseract

def load_pdf_with_ocr(file_path: str) -> str:
    """Extract text from scanned PDFs using OCR."""
    images = convert_from_path(file_path)
    text = []
    for image in images:
        text.append(pytesseract.image_to_string(image))
    return "\n\n".join(text)
```

**Error Handling:**

```python
def load_and_chunk_documents(directory: str, chunk_size: int = 200) -> Tuple[List[str], List[str]]:
    """Load documents with comprehensive error handling."""
    chunks = []
    sources = []
    errors = []

    for file_path in Path(directory).rglob('*'):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        if ext not in DOCUMENT_LOADERS:
            continue

        try:
            loader = DOCUMENT_LOADERS[ext]
            text = loader(str(file_path))

            # Validate extracted text
            if not text or len(text.strip()) < 10:
                errors.append(f"{file_path.name}: Empty or too short")
                continue

            file_chunks = chunk_text(text, chunk_size)
            source_name = f"{file_path.name}"
            chunks.extend(file_chunks)
            sources.extend([source_name] * len(file_chunks))

            print(f"✓ {file_path.name} ({ext}) - {len(file_chunks)} chunks")

        except Exception as e:
            errors.append(f"{file_path.name}: {str(e)}")
            continue

    # Report errors
    if errors:
        print(f"\n⚠️  Errors loading {len(errors)} document(s):")
        for error in errors:
            print(f"  - {error}")

    return chunks, sources
```

**Testing:**

```bash
# Create test documents directory
mkdir -p test_docs

# Add various document types
echo "Test content" > test_docs/test.txt
# Add PDF, Word files manually

# Run with test directory
FAQ_DIR=test_docs python src/rag_assessment_partial.py

# Expected output:
# ✓ Loaded test.txt (.txt) - 1 chunks
# ✓ Loaded benefits.pdf (.pdf) - 45 chunks
# ✓ Loaded handbook.docx (.docx) - 32 chunks
#
# Loaded 78 chunks from 3 documents
# Document types: .txt, .pdf, .docx
```

**Performance Considerations:**

- PDF extraction can be slow for large documents (use caching)
- Word documents (.docx) are fast to parse
- HTML parsing depends on document complexity
- Consider pre-processing large PDFs offline

**Note:** This feature makes the RAG system production-ready for enterprise environments where documentation exists in multiple formats. Combined with the embedding cache feature, it provides efficient multi-format document ingestion.

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
