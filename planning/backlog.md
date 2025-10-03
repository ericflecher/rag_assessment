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
