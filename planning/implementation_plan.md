# RAG Implementation Plan - Detailed Code Blueprints

## Acceptance Criteria Summary (from GUIDE.pdf)

### Core Requirements

1. ✅ Ingest FAQ documents from directory (already implemented)
2. ✅ Accept natural language questions via CLI (already implemented)
3. ⚠️ Retrieve relevant content using vector search (TODO)
4. ⚠️ Generate answers using LLM based on retrieved content (TODO)

### Technical Specifications (MUST adhere to)

- Use OpenAI's `text-embedding-ada-002` for embeddings ✅
- Use `gpt-3.5-turbo` for answer generation ✅
- Cosine similarity for relevance scoring ✅ (function exists)
- Chunk size: ~200 characters ✅
- Retrieve TOP_K = 4 chunks ✅
- **Ensure answers cite at least 2 source files**
- Output format: JSON with `answer` and `sources` fields

### Evaluation Criteria

- **Accuracy**: Must work and return correct results
- **Approach**: Clean design decisions
- **Practicality**: Lightweight, avoid over-engineering

---

## Current Code Analysis

### Completed

- ✅ Imports and dependencies
- ✅ Configuration constants
- ✅ `chunk_text()` - splits text into 200-char chunks
- ✅ `cosine_sim()` - computes similarity between vectors
- ✅ `load_and_chunk_faqs()` - reads and chunks FAQ files
- ✅ Main flow structure with comments

### TODOs Identified

1. **Line 74**: `embed_texts()` function - needs API call implementation
2. **Line 84**: Embed chunks call
3. **Line 89**: Embed query using OpenAI API
4. **Line 93**: Compute similarities and retrieve top-k
5. **Line 107**: Generate answer using chat completions

---

## Implementation Blueprints

### Blueprint 1: `embed_texts()` Function (Line 67-76)

**Location:** `src/rag_assessment_partial.py:67-76`

**Current Code:**
```python
def embed_texts(texts):
    """
    Given a list of texts, return a list of their embeddings as numpy arrays.
    Use OpenAI's embedding API.
    """
    embeddings = []
    for text in tqdm(texts, desc="Embedding"):
    # TODO: Call the OpenAI embedding API and append the result as a numpy array
    pass
    return embeddings
```

**Implementation Blueprint:**
```python
def embed_texts(texts):
    """
    Given a list of texts, return a list of their embeddings as numpy arrays.
    Use OpenAI's embedding API.
    """
    embeddings = []
    for text in tqdm(texts, desc="Embedding"):
        # Call OpenAI embedding API for each text chunk
        response = client.embeddings.create(
            input=[text],
            model=EMBED_MODEL  # "text-embedding-ada-002"
        )
        # Extract embedding and convert to numpy array
        embedding = np.array(response.data[0].embedding)
        embeddings.append(embedding)
    return embeddings
```

**Key Points:**

- Uses `client.embeddings.create()` per GUIDE specification
- Model: `text-embedding-ada-002` (from config)
- Wraps each text in a list `[text]` as required by API
- Extracts embedding from `response.data[0].embedding`
- Converts to numpy array for vector operations
- Progress bar via `tqdm` already in place

---

### Blueprint 2: Chunk Embedding (Line 82-84)

**Location:** `src/rag_assessment_partial.py:82-84`

**Current Code:**
```python
# --- 2. Embed Chunks ---
# TODO: Embed the chunks using embed_texts
chunk_embeddings = []  # Replace with actual embeddings
```

**Implementation Blueprint:**
```python
# --- 2. Embed Chunks ---
chunk_embeddings = embed_texts(chunks)
```

**Key Points:**

- Simply call the `embed_texts()` function with all chunks
- Returns list of numpy arrays (one embedding per chunk)
- This happens once at startup before query loop

---

### Blueprint 3: Query Embedding (Line 86-89)

**Location:** `src/rag_assessment_partial.py:86-89`

**Current Code:**
```python
# --- 3. Query Loop ---
query = input("Enter your question: ")
# TODO: Embed the query using client.embeddings.create
query_emb = None  # Replace with actual query embedding
```

**Implementation Blueprint:**
```python
# --- 3. Query Loop ---
query = input("Enter your question: ")
# Embed the query using the same embedding model
response = client.embeddings.create(
    input=[query],
    model=EMBED_MODEL  # "text-embedding-ada-002"
)
query_emb = np.array(response.data[0].embedding)
```

**Key Points:**

- Direct API call (not using `embed_texts()` to avoid tqdm for single query)
- Same model as chunks for compatibility
- Converts to numpy array for similarity computation
- Single embedding operation per query

---

### Blueprint 4: Similarity Computation & Top-K Retrieval (Line 91-95)

**Location:** `src/rag_assessment_partial.py:91-95`

**Current Code:**
```python
# --- 4. Retrieve Top-k ---
# TODO: Compute similarities and get top-k indices
top_indices = np.argsort(sims)[-TOP_K:][::-1]
top_chunks = [chunks[i] for i in top_indices]
top_files = [sources[i] for i in top_indices]
```

**Implementation Blueprint:**
```python
# --- 4. Retrieve Top-k ---
# Compute cosine similarity between query and all chunk embeddings
sims = np.array([cosine_sim(query_emb, chunk_emb) for chunk_emb in chunk_embeddings])

# Get indices of top-k most similar chunks
# np.argsort gives ascending order, so [-TOP_K:] gets last K (highest), [::-1] reverses
top_indices = np.argsort(sims)[-TOP_K:][::-1]
top_chunks = [chunks[i] for i in top_indices]
top_files = [sources[i] for i in top_indices]
```

**Key Points:**

- Computes cosine similarity using existing `cosine_sim()` function
- Creates array of similarities (one per chunk)
- `np.argsort(sims)` returns indices sorted by similarity (ascending)
- `[-TOP_K:]` gets the last K indices (highest similarities)
- `[::-1]` reverses to get descending order (most similar first)
- Extracts corresponding chunks and source filenames

**Alternative (more efficient for production):**
```python
# For production, could use np.argpartition for O(n) instead of O(n log n)
sims = np.array([cosine_sim(query_emb, chunk_emb) for chunk_emb in chunk_embeddings])
top_indices = np.argpartition(sims, -TOP_K)[-TOP_K:]
top_indices = top_indices[np.argsort(sims[top_indices])][::-1]  # Sort top-k only
```

---

### Blueprint 5: Answer Generation (Line 97-107)

**Location:** `src/rag_assessment_partial.py:97-107`

**Current Code:**
```python
# --- 5. Generate Answer ---
context = "\n\n".join([f"From {sources[i]}:\n{chunks[i]}" for i in top_indices])
prompt = (
    f"Answer the following question using the provided context. "
    f"Cite at least two of the file names in your answer.\n\n"
    f"Context:\n{context}\n\n"
    f"Question: {query}\n\n"
    f"Answer (cite sources):"
)
# TODO: Generate answer using client.chat.completions.create
answer = ""  # Replace with actual answer
```

**Implementation Blueprint:**
```python
# --- 5. Generate Answer ---
context = "\n\n".join([f"From {sources[i]}:\n{chunks[i]}" for i in top_indices])
prompt = (
    f"Answer the following question using the provided context. "
    f"Cite at least two of the file names in your answer.\n\n"
    f"Context:\n{context}\n\n"
    f"Question: {query}\n\n"
    f"Answer (cite sources):"
)

# Generate answer using GPT-3.5-turbo
response = client.chat.completions.create(
    model=LLM_MODEL,  # "gpt-3.5-turbo"
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2  # Low temperature for more consistent, factual answers
)
answer = response.choices[0].message.content.strip()
```

**Key Points:**

- Context already built with source attribution (format: "From filename:\nchunk")
- Prompt instructs LLM to cite at least 2 filenames (per requirements)
- Uses `gpt-3.5-turbo` model (from config)
- Temperature 0.2 for factual, deterministic responses
- Extracts answer from first choice and strips whitespace

**Prompt Engineering Notes:**

- Context includes source filenames for each chunk
- Explicit instruction to cite sources ensures compliance
- Question clearly stated at end
- Format aligns with GUIDE.pdf example

---

## Summary: Complete Implementation Plan

### Implementation Order

1. **Fix `embed_texts()` function** (Line 67-76)
   - Add OpenAI embedding API call inside loop
   - Convert responses to numpy arrays

2. **Enable chunk embedding** (Line 84)
   - Call `embed_texts(chunks)`

3. **Add query embedding** (Line 89)
   - Add direct API call for single query

4. **Implement similarity search** (Line 92-93)
   - Compute similarities using existing `cosine_sim()`
   - Variable `sims` must be created before line 93

5. **Add answer generation** (Line 107)
   - Call `client.chat.completions.create()`
   - Extract answer from response

### 5 TODO Items to Complete

| # | Location | Task | Complexity |
|---|----------|------|------------|
| 1 | Line 74 | Implement `embed_texts()` API call | Simple |
| 2 | Line 84 | Call `embed_texts(chunks)` | Trivial |
| 3 | Line 89 | Embed query with API call | Simple |
| 4 | Line 92 | Compute similarities & create `sims` variable | Simple |
| 5 | Line 107 | Generate answer with chat API | Simple |

### Validation Checklist (per GUIDE.pdf)

✅ **Technical Specifications:**

- Uses `text-embedding-ada-002` for embeddings
- Uses `gpt-3.5-turbo` for generation
- Implements cosine similarity
- Chunks at ~200 characters
- Retrieves top 4 chunks
- Cites at least 2 source files
- Outputs JSON format

✅ **Evaluation Criteria:**

- **Accuracy**: Implementation follows API specs exactly
- **Approach**: Simple, direct implementation using provided skeleton
- **Practicality**: Minimal changes, leverages existing structure

### Expected Output Format

```json
{
  "answer": "Generated answer citing faq_employee.md and faq_sso.md...",
  "sources": ["faq_employee.md", "faq_sso.md"]
}
```

### Dependencies Required

Already specified in code:

```bash
pip install openai numpy tqdm python-dotenv
```

### Environment Setup

`.env` file:

```
OPENAI_API_KEY=your-key-here
```

---

## Design Tradeoffs & Decisions

### 1. Character-based chunking (200 chars)

- ✅ Simple, predictable
- ⚠️ May split sentences/words awkwardly
- **Alternative:** Could use sentence-based chunking with max size

### 2. Individual API calls in `embed_texts()`

- ✅ Progress tracking with tqdm
- ⚠️ Slower than batch API calls
- **Production optimization:** Use batching for efficiency

### 3. In-memory storage

- ✅ Simple for small FAQ corpus
- ⚠️ Not scalable to large document sets
- **Production:** Use vector database (Pinecone, Weaviate, Qdrant)

### 4. No caching

- ✅ Keeps code simple
- ⚠️ Re-embeds chunks on every run
- **Production:** Cache embeddings to file/database

### 5. Simple cosine similarity

- ✅ Proven for semantic search
- ✅ Easy to understand
- Already implemented correctly

---

## Testing Strategy

After implementation, test with:

### Query 1: "How do I reset my password?"

- **Expected:** Should retrieve from `faq_auth.md`

### Query 2: "What is the PTO policy?"

- **Expected:** Should retrieve from `faq_employee.md`

### Query 3: "How does equity vesting work?"

- **Expected:** Should retrieve from `faq_employee.md`

### Query 4: "Tell me about SSO and authentication"

- **Expected:** Should cite both `faq_sso.md` AND `faq_auth.md`

---

## Production Scalability Suggestions

While this implementation is optimized for the interview requirements (2-3 hours, lightweight), here are considerations for production:

### Vector Database Integration

- Replace in-memory storage with Pinecone, Weaviate, or Qdrant
- Persistent storage of embeddings
- Efficient similarity search at scale

### Chunking Strategy

- Use semantic chunking (sentence/paragraph boundaries)
- Add chunk overlap (e.g., 50 characters) to maintain context
- Consider RecursiveCharacterTextSplitter from LangChain

### Embedding Optimization

- Batch API calls (up to 2048 inputs per request)
- Cache embeddings with TTL
- Consider newer models (text-embedding-3-small/large)

### Answer Quality

- Add re-ranking step after initial retrieval
- Implement query expansion/rewriting
- Use prompt templates with few-shot examples

### Observability

- Log query-answer pairs for evaluation
- Track latency metrics (embedding, retrieval, generation)
- Monitor API costs and rate limits

### Error Handling

- Retry logic for API failures
- Fallback responses when no relevant chunks found
- Input validation and sanitization
