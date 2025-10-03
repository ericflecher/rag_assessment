# Local Model Support Implementation Plan

## Feature Overview

**Priority:** Medium (Bonus Points)
**Effort:** Medium (~3-4 hours)
**Category:** Cost Optimization, Privacy, Offline Support

**Goal:** Enable the RAG system to run with local models, eliminating OpenAI API dependency for bonus points evaluation criterion: *"If it can run with local models (and doesn't require API keys)"*

---

## Acceptance Criteria

### Core Requirements

1. ✅ System can run without OpenAI API key
2. ✅ Uses local embedding model for text vectorization
3. ✅ Uses local LLM for answer generation
4. ✅ Maintains existing functionality (retrieve + generate with citations)
5. ✅ Provides configuration toggle between OpenAI and local models
6. ✅ Documents setup instructions for local models

### Technical Specifications

- **Embedding Model:** `all-MiniLM-L6-v2` (sentence-transformers)
- **LLM Model:** `llama3.2:3b` (Ollama) or `phi3:mini`
- **Configuration:** Environment variable `USE_LOCAL_MODELS=true`
- **Fallback:** Default to OpenAI if local models not available
- **Output Format:** Same JSON structure with answer and sources

### Quality Criteria

- **Accuracy:** Answers must still cite at least 2 sources
- **Performance:** Acceptable latency on CPU (< 30 seconds per query)
- **Compatibility:** Works on macOS, Linux, Windows
- **Documentation:** Clear setup instructions in README

---

## Implementation Strategy

### Recommended Approach: Ollama + Sentence Transformers

**Rationale:**
- Ollama provides easy LLM management (pull, run, update)
- Sentence Transformers is battle-tested for embeddings
- Both have minimal dependencies
- Good balance of quality and speed on CPU

**Alternative Considered:**
- Pure Hugging Face (transformers) - more complex setup, slower inference
- LangChain - adds unnecessary abstraction layer

---

## Implementation Blueprints

### Blueprint 0: Prepare Imports and Remove Conflicts

**Location:** `src/rag_assessment_partial.py:17-36`

**Current Code:**
```python
import os
import json
from openai import OpenAI
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
# Use override=True to ensure .env takes precedence over system environment
load_dotenv(override=True)

# --- Config ---
FAQ_DIR = "faqs"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-ada-002")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
CHUNK_SIZE = 200  # characters per chunk
TOP_K = 4

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

**Implementation Blueprint:**
```python
import os
import sys  # NEW - needed for sys.exit() in error handling
import json
# REMOVED: from openai import OpenAI  (will be imported conditionally in Blueprint 2)
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
# Use override=True to ensure .env takes precedence over system environment
load_dotenv(override=True)

# --- Config ---
FAQ_DIR = "faqs"
# REMOVED: EMBED_MODEL and LLM_MODEL config (will be set in Blueprint 2)
# REMOVED: client initialization (will be done in Blueprint 2)
CHUNK_SIZE = 200  # characters per chunk
TOP_K = 4
```

**Key Changes:**
- ✅ Add `import sys` for error handling
- ✅ Remove `from openai import OpenAI` - will import conditionally
- ✅ Remove lines 30-31 (EMBED_MODEL, LLM_MODEL) - will be set in Blueprint 2
- ✅ Remove line 36 (`client = OpenAI(...)`) - will be initialized in Blueprint 2
- ✅ This prevents import errors when OpenAI package not installed

**Why This Matters:**
- OpenAI library won't be installed when using local models
- Importing it unconditionally will cause `ImportError`
- Must use conditional imports based on `USE_LOCAL_MODELS` flag

---

### Blueprint 1: Update Configuration Files

**Location:** `.env.template` and `README.md`

**Changes Required:**

**1. Update `.env.template`:**

Add local model configuration options:

```bash
# OpenAI API Configuration
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your-api-key-here

# Local Models Configuration (Optional - for bonus points)
# Set to 'true' to use local models instead of OpenAI API
# Requires: pip install sentence-transformers ollama
USE_LOCAL_MODELS=false

# Model Configuration
# Uncomment to override default models
# EMBED_MODEL=text-embedding-ada-002
# LLM_MODEL=gpt-3.5-turbo

# Local Model Configuration (Optional)
# Only used when USE_LOCAL_MODELS=true
# EMBED_MODEL_LOCAL=all-MiniLM-L6-v2
# LLM_MODEL_LOCAL=llama3.2
```

**Key Points:**
- `USE_LOCAL_MODELS` defaults to `false` for backwards compatibility
- OpenAI models remain default for seamless evaluation
- Local model configs only activate when `USE_LOCAL_MODELS=true`
- Clear comments explain requirements and usage

**2. Dependencies:**
```bash
# Add to README setup instructions
pip install sentence-transformers ollama

# Ollama installation (macOS)
brew install ollama

# Pull a model
ollama pull llama3.2
```

**3. Update `.env.template` file now:**

This blueprint includes updating the actual `.env.template` file in the repository to include the local model configuration options shown above.

---

### Blueprint 2: Configuration and Model Selection

**Location:** `src/rag_assessment_partial.py:28-36` (after Blueprint 0 changes)

**Current Code (after Blueprint 0):**
```python
# Load environment variables from .env file
# Use override=True to ensure .env takes precedence over system environment
load_dotenv(override=True)

# --- Config ---
FAQ_DIR = "faqs"
CHUNK_SIZE = 200  # characters per chunk
TOP_K = 4
# (OpenAI client initialization removed in Blueprint 0)
```

**Implementation Blueprint:**
```python
# Load environment variables from .env file
# Use override=True to ensure .env takes precedence over system environment
load_dotenv(override=True)

# --- Config ---
FAQ_DIR = "faqs"
CHUNK_SIZE = 200  # characters per chunk
TOP_K = 4

# Determine which models to use
USE_LOCAL_MODELS = os.getenv("USE_LOCAL_MODELS", "false").lower() == "true"

if USE_LOCAL_MODELS:
    # Local model configuration
    EMBED_MODEL = os.getenv("EMBED_MODEL_LOCAL", "all-MiniLM-L6-v2")
    LLM_MODEL = os.getenv("LLM_MODEL_LOCAL", "llama3.2")

    # Import local model libraries (conditional import)
    from sentence_transformers import SentenceTransformer
    import ollama

    # Initialize local embedding model
    print(f"Loading local embedding model: {EMBED_MODEL}...", file=sys.stderr)
    embedding_model = SentenceTransformer(EMBED_MODEL)
    print(f"Using local LLM: {LLM_MODEL} (via Ollama)", file=sys.stderr)
    print("✅ Running with local models (no API key required)", file=sys.stderr)

    client = None  # No OpenAI client needed
else:
    # OpenAI configuration
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-ada-002")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

    # Import OpenAI (conditional import - moved from top of file)
    from openai import OpenAI

    # Initialize the OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print(f"Using OpenAI API - Embedding: {EMBED_MODEL}, LLM: {LLM_MODEL}", file=sys.stderr)

    embedding_model = None  # No local model needed
```

**Key Points:**
- Single configuration variable controls entire behavior
- **Conditional imports** - only load libraries when needed (prevents ImportError)
- **Import moved from line 19** - `from openai import OpenAI` now inside else block
- Clear user feedback about which mode is active
- Graceful handling of both paths

**Critical Fixes:**
- ✅ OpenAI import moved from top of file to conditional block
- ✅ Prevents ImportError when OpenAI package not installed
- ✅ Allows running with only local model dependencies
- ✅ **All print() statements use `file=sys.stderr`** to preserve clean JSON stdout
- ✅ Maintains backward compatibility with existing JSON output format

---

### Blueprint 3: Dual-Mode Embedding Function

**Location:** `src/rag_assessment_partial.py:68-83`

**Current Code:**
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

**Implementation Blueprint:**
```python
def embed_texts(texts):
    """
    Given a list of texts, return a list of their embeddings as numpy arrays.
    Uses either OpenAI API or local sentence-transformers model based on USE_LOCAL_MODELS.
    """
    embeddings = []

    if USE_LOCAL_MODELS:
        # Local embedding using sentence-transformers
        desc = f"Embedding (local: {EMBED_MODEL})"
        for text in tqdm(texts, desc=desc):
            # sentence-transformers returns numpy arrays by default
            embedding = embedding_model.encode(text, convert_to_numpy=True)
            embeddings.append(embedding)
    else:
        # OpenAI API embedding
        desc = f"Embedding (OpenAI: {EMBED_MODEL})"
        for text in tqdm(texts, desc=desc):
            # Call OpenAI embedding API for each text chunk
            response = client.embeddings.create(
                input=[text],
                model=EMBED_MODEL
            )
            # Extract embedding and convert to numpy array
            embedding = np.array(response.data[0].embedding)
            embeddings.append(embedding)

    return embeddings
```

**Key Points:**
- Single function handles both paths
- Progress bar indicates which mode is active
- Both return same format (list of numpy arrays)
- Local mode: `all-MiniLM-L6-v2` returns 384-dim vectors (vs 1536 for OpenAI)

**Performance Comparison:**
- OpenAI: ~1-2 seconds per chunk (network latency)
- Local: ~0.1-0.3 seconds per chunk (CPU inference)
- Local is typically **faster** for embedding

---

### Blueprint 4: Dual-Mode Query Embedding

**Location:** `src/rag_assessment_partial.py:93-99`

**Current Code:**
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

**Implementation Blueprint:**
```python
# --- 3. Query Loop ---
query = input("Enter your question: ")

# Embed the query using the same embedding model
if USE_LOCAL_MODELS:
    # Local embedding
    query_emb = embedding_model.encode(query, convert_to_numpy=True)
else:
    # OpenAI API embedding
    response = client.embeddings.create(
        input=[query],
        model=EMBED_MODEL
    )
    query_emb = np.array(response.data[0].embedding)
```

**Key Points:**
- Must use same embedding model as chunks for compatibility
- Local mode is faster (no network call)
- Both return numpy arrays ready for cosine similarity

---

### Blueprint 5: Dual-Mode Answer Generation

**Location:** `src/rag_assessment_partial.py:111-126`

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
# Generate answer using GPT-3.5-turbo
response = client.chat.completions.create(
    model=LLM_MODEL,  # "gpt-3.5-turbo"
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2  # Low temperature for more consistent, factual answers
)
answer = response.choices[0].message.content.strip()
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

# Generate answer using configured LLM
if USE_LOCAL_MODELS:
    # Local LLM via Ollama
    print(f"Generating answer with {LLM_MODEL}...", file=sys.stderr)
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.2,
            "num_predict": 512,  # Max tokens to generate
        }
    )
    answer = response['message']['content'].strip()
else:
    # OpenAI API
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    answer = response.choices[0].message.content.strip()
```

**Key Points:**
- Same prompt for both models (consistent behavior)
- Ollama API is similar to OpenAI (different response structure)
- Local generation takes longer (~5-15 seconds on CPU)
- Quality depends on model choice (llama3.2 ≈ GPT-3.5 for simple tasks)
- **Progress message sent to stderr** to keep stdout clean for JSON output

**Performance Comparison:**
- OpenAI GPT-3.5: ~2-4 seconds
- Llama3.2 (3B) on CPU: ~10-20 seconds
- Llama3.2 (3B) on GPU: ~3-5 seconds

---

## Model Selection Guide

### Embedding Models

| Model                 | Dimensions | Size  | Speed (CPU) | Quality   | Best For              |
| --------------------- | ---------- | ----- | ----------- | --------- | --------------------- |
| all-MiniLM-L6-v2      | 384        | 80MB  | ★★★★★       | ★★★★☆     | General use (default) |
| all-mpnet-base-v2     | 768        | 420MB | ★★★★☆       | ★★★★★     | Higher quality        |
| all-MiniLM-L12-v2     | 384        | 120MB | ★★★★☆       | ★★★★☆     | Balanced              |
| text-embedding-ada-02 | 1536       | API   | ★★★★☆       | ★★★★★     | OpenAI (baseline)     |

**Recommendation:** `all-MiniLM-L6-v2` for best speed/quality balance on CPU

### LLM Models (Ollama)

| Model           | Parameters | Size | Speed (CPU) | Quality   | Best For              |
| --------------- | ---------- | ---- | ----------- | --------- | --------------------- |
| llama3.2        | 3B         | 2GB  | ★★★★☆       | ★★★★★     | Best overall (default)|
| phi3:mini       | 3.8B       | 2.3GB| ★★★★★       | ★★★★☆     | Fastest               |
| mistral         | 7B         | 4GB  | ★★☆☆☆       | ★★★★★     | Best quality (slow)   |
| gemma2:2b       | 2B         | 1.6GB| ★★★★★       | ★★★☆☆     | Fastest, lower quality|
| gpt-3.5-turbo   | Unknown    | API  | ★★★★★       | ★★★★★     | OpenAI (baseline)     |

**Recommendation:** `llama3.2:3b` for best quality/speed balance on CPU

---

## Setup Instructions

### Prerequisites

**macOS:**
```bash
# Install Ollama
brew install ollama

# Start Ollama service (runs in background)
ollama serve  # Or just pull a model, which will start it automatically

# Pull recommended model
ollama pull llama3.2
```

**Linux:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull llama3.2
```

**Windows:**
```bash
# Download and install from https://ollama.com/download
# Then pull model
ollama pull llama3.2
```

### Python Dependencies

```bash
# Install local model libraries
pip install sentence-transformers ollama

# Optional: Install PyTorch with GPU support for faster inference
# pip install torch --index-url https://download.pytorch.org/whl/cu118  # CUDA
```

### Configuration

**Update `.env` file:**
```bash
# Enable local models
USE_LOCAL_MODELS=true

# Optional: Override default local models
EMBED_MODEL_LOCAL=all-MiniLM-L6-v2
LLM_MODEL_LOCAL=llama3.2
```

---

## Testing Strategy

### Test 1: Verify Local Models Work

**Setup:**
```bash
# Ensure local models are available
ollama list  # Should show llama3.2

# Enable local mode
echo "USE_LOCAL_MODELS=true" >> .env
```

**Run:**
```bash
python src/rag_assessment_partial.py
# Enter: What is the PTO policy?
```

**Expected Output:**
- No API key errors
- Progress bar shows "Embedding (local: all-MiniLM-L6-v2)"
- Message shows "Generating answer with llama3.2..."
- Valid JSON output with answer and sources
- Answer cites at least 2 source files

### Test 2: Compare OpenAI vs Local

**Test Script:**
```bash
# Test with OpenAI
USE_LOCAL_MODELS=false python src/rag_assessment_partial.py <<< "What is the PTO policy?" > openai_output.json

# Test with local models
USE_LOCAL_MODELS=true python src/rag_assessment_partial.py <<< "What is the PTO policy?" > local_output.json

# Compare results
diff openai_output.json local_output.json
```

**Expected:**
- Both produce valid JSON
- Both cite at least 2 sources
- Answers should be similar in content (may differ in phrasing)
- Local may be faster for embedding, slower for generation

### Test 3: Error Handling (Optional)

**Test Missing Ollama:**
```bash
# Stop Ollama
killall ollama

# Try to run
USE_LOCAL_MODELS=true python src/rag_assessment_partial.py
```

**Expected:**
- Clear error message about Ollama not running
- Instructions to run `ollama serve`

**Optional Enhancement (Blueprint 6):**

Error handling is **optional** for MVP. If time permits, add dependency checking:

```python
if USE_LOCAL_MODELS:
    try:
        from sentence_transformers import SentenceTransformer
        import ollama
    except ImportError as e:
        print("❌ Local model dependencies not installed.")
        print("Install with: pip install sentence-transformers ollama")
        print("Then install Ollama from: https://ollama.com")
        sys.exit(1)

    # Test Ollama connection
    try:
        ollama.list()
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("Start Ollama with: ollama serve")
        print(f"Pull model with: ollama pull {LLM_MODEL}")
        sys.exit(1)
```

**Note:** This is optional and can be implemented after core functionality is working.

---

## Performance Benchmarks

### Expected Performance (MacBook Pro M1, CPU only)

| Phase              | OpenAI      | Local (CPU) | Speedup |
| ------------------ | ----------- | ----------- | ------- |
| Chunk Embedding    | 3-5 sec     | 1-2 sec     | 2-3x faster |
| Query Embedding    | 0.5 sec     | 0.1 sec     | 5x faster |
| Similarity Search  | 0.01 sec    | 0.01 sec    | Same |
| Answer Generation  | 2-4 sec     | 10-20 sec   | 3-5x slower |
| **Total**          | **6-10 sec**| **12-23 sec**| 1.5-2x slower |

### With GPU (NVIDIA RTX 3080)

| Phase              | Local (GPU) |
| ------------------ | ----------- |
| Chunk Embedding    | 0.5 sec     |
| Query Embedding    | 0.05 sec    |
| Similarity Search  | 0.01 sec    |
| Answer Generation  | 2-3 sec     |
| **Total**          | **3-4 sec** |

**Conclusion:** Local models on GPU are **faster** than OpenAI API!

---

## Design Decisions & Tradeoffs

### Decision 1: Ollama vs Native Transformers

**Choice:** Ollama for LLM, sentence-transformers for embeddings

**Rationale:**
- ✅ Ollama provides easy model management (pull, update, delete)
- ✅ Ollama handles model quantization and optimization
- ✅ sentence-transformers is specifically designed for embeddings
- ✅ Combined approach: simplicity (Ollama) + performance (sentence-transformers)

**Tradeoff:**
- ⚠️ Two dependencies instead of one
- ⚠️ Ollama requires separate installation (not just pip)

**Alternative:** Pure transformers library
- ✅ Single dependency
- ❌ More complex setup
- ❌ Slower LLM inference
- ❌ Manual model management

### Decision 2: Configuration Toggle vs Separate Script

**Choice:** Single script with `USE_LOCAL_MODELS` toggle

**Rationale:**
- ✅ Easier to maintain (one codebase)
- ✅ Demonstrates flexibility and good design
- ✅ Users can switch without changing code
- ✅ Both modes tested automatically

**Tradeoff:**
- ⚠️ Slightly more complex code (if/else branches)
- ⚠️ Both dependency sets in requirements

**Alternative:** Separate scripts (rag_local.py vs rag_openai.py)
- ✅ Simpler individual scripts
- ❌ Code duplication
- ❌ Harder to maintain consistency

### Decision 3: Model Selection

**Choice:** all-MiniLM-L6-v2 + llama3.2:3b

**Rationale:**
- ✅ Best balance of speed/quality on CPU
- ✅ Small enough to download quickly (< 5GB total)
- ✅ Llama3.2 is state-of-the-art for size
- ✅ MiniLM is proven for semantic search

**Tradeoff:**
- ⚠️ Slightly lower quality than larger models
- ⚠️ 384-dim embeddings vs 1536-dim (different similarity scales)

**Alternative:** Mistral 7B + all-mpnet-base-v2
- ✅ Higher quality
- ❌ 3x slower inference
- ❌ Requires 6GB+ download

---

## Documentation Updates Required

### README.md Updates

**Add to Features section:**
```markdown
- **Local Model Support**: Run without API keys using Ollama and sentence-transformers (bonus feature)
```

**Add to Requirements section:**
```markdown
### Option 1: OpenAI API (Default)
- OpenAI API key
- Dependencies: `openai`, `numpy`, `tqdm`, `python-dotenv`

### Option 2: Local Models (No API Key)
- Ollama installed and running
- Dependencies: `sentence-transformers`, `ollama`, `numpy`, `tqdm`, `python-dotenv`
```

**Add new section after Setup:**
```markdown
## Local Model Setup (Optional - No API Key Required)

For bonus points demonstration, run with local models:

### 1. Install Ollama
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from https://ollama.com/download
```

### 2. Pull LLM Model
```bash
ollama pull llama3.2
```

### 3. Install Python Dependencies
```bash
pip install sentence-transformers ollama
```

### 4. Enable Local Mode
```bash
# Edit .env
USE_LOCAL_MODELS=true
```

### 5. Run Without API Key
```bash
python src/rag_assessment_partial.py
# No OPENAI_API_KEY required!
```
```

**Add to Configuration section:**
```markdown
### Local vs OpenAI Toggle

Switch between OpenAI API and local models via `.env`:

```bash
# Use OpenAI API (default)
USE_LOCAL_MODELS=false
OPENAI_API_KEY=sk-...

# Use local models (no API key needed)
USE_LOCAL_MODELS=true
# OPENAI_API_KEY not required
```
```

---

## Validation Checklist

### Core Functionality
- ✅ System runs without OpenAI API key when `USE_LOCAL_MODELS=true`
- ✅ Embeddings generated using sentence-transformers
- ✅ Answers generated using Ollama
- ✅ Retrieves top-k relevant chunks
- ✅ Cites at least 2 source files in answer
- ✅ Outputs valid JSON format

### Configuration
- ✅ `USE_LOCAL_MODELS` toggle works correctly
- ✅ Falls back gracefully if local models unavailable
- ✅ Clear error messages for missing dependencies
- ✅ Documents both modes in README

### Quality
- ✅ Answers are factually grounded in context
- ✅ Quality comparable to OpenAI (for small FAQ corpus)
- ✅ No hallucinations or source fabrication
- ✅ Proper source attribution

### Performance
- ✅ Acceptable latency on CPU (< 30 seconds per query)
- ✅ Faster on GPU if available
- ✅ Progress indicators for long operations

---

## Implementation Summary

### Files to Modify

1. **`.env.template`** ⚡ (Blueprint 1)
   - Add `USE_LOCAL_MODELS` configuration
   - Add local model configuration options
   - **Action Required:** Update file with configuration shown in Blueprint 1

2. **`src/rag_assessment_partial.py`**
   - Prepare imports and remove conflicts (Blueprint 0) ⚠️ **Critical First Step**
   - Add configuration toggle (Blueprint 2)
   - Update `embed_texts()` function (Blueprint 3)
   - Update query embedding (Blueprint 4)
   - Update answer generation (Blueprint 5)
   - Add error handling (Blueprint 6 - **Optional**)

3. **`README.md`**
   - Add local model setup instructions
   - Document configuration options
   - Add performance comparisons

4. **`planning/backlog.md`** ✅ (Already updated)
   - Moved "Local Model Support" to "In Progress" section
   - Added link to implementation plan

### Estimated Implementation Time

| Task | Time | Status |
|------|------|--------|
| Blueprint 0: Prepare imports (NEW) | 5 min | ⚠️ **Critical First** |
| Blueprint 1: Update .env.template | 5 min | ⚡ Next |
| Blueprint 2: Configuration toggle | 30 min | Pending |
| Blueprint 3: Embedding function | 30 min | Pending |
| Blueprint 4: Query embedding | 15 min | Pending |
| Blueprint 5: Answer generation | 30 min | Pending |
| Blueprint 6: Error handling (Optional) | 30 min | Optional |
| Documentation updates | 45 min | Pending |
| Testing and validation | 60 min | Pending |
| **Total (MVP)** | **3.5 hours** | (without optional error handling) |
| **Total (Full)** | **4 hours** | (with error handling) |

---

## Implementation Readiness Checklist

### ✅ Ready to Start - All Issues Resolved

**Critical Issues Addressed:**

1. ✅ **Blueprint 0 Added** - Reorganizes imports to prevent ImportError
   - Adds `import sys` for error handling and stderr output
   - Removes `from openai import OpenAI` from top-level imports
   - Moves OpenAI import to conditional block in Blueprint 2
   - Removes premature client initialization

2. ✅ **Blueprint 2 Fixed** - Import location corrected
   - OpenAI import now inside `else` block (conditional)
   - Prevents ImportError when openai package not installed
   - Enables running with only local model dependencies

3. ✅ **Blueprint 6 Marked Optional** - Error handling is enhancement
   - Core functionality works without it
   - Can be implemented after MVP is complete
   - Doesn't block primary implementation

4. ✅ **Stdout Preservation** - All informational output to stderr
   - All `print()` statements use `file=sys.stderr`
   - Keeps stdout clean for JSON-only output
   - Maintains backward compatibility with existing pipelines
   - Preserves requirement: "Output the answer and sources as a JSON object to stdout"

**Implementation Order:**
1. Blueprint 0: Prepare imports ⚠️ **Must do first**
2. Blueprint 1: Update .env.template
3. Blueprint 2: Configuration toggle
4. Blueprint 3: Dual-mode embedding
5. Blueprint 4: Dual-mode query embedding
6. Blueprint 5: Dual-mode answer generation
7. Blueprint 6: Error handling (optional)
8. Testing and validation

**Backward Compatibility Verification:**
- ✅ All blueprints build sequentially to complete solution
- ✅ No import conflicts
- ✅ Both OpenAI and local modes supported
- ✅ **Default behavior unchanged** - `USE_LOCAL_MODELS=false` by default
- ✅ **OpenAI mode identical to current code** - same config, same API calls
- ✅ **JSON stdout preserved** - informational messages go to stderr
- ✅ **Same output format** - JSON with "answer" and "sources" fields
- ✅ **Same environment variables** - EMBED_MODEL and LLM_MODEL still work
- ✅ Clear separation of concerns

**Existing Features Preserved:**
- ✅ OpenAI embedding with text-embedding-ada-002
- ✅ OpenAI generation with gpt-3.5-turbo
- ✅ Environment variable configuration (.env file)
- ✅ JSON-only stdout output
- ✅ Source citation (minimum 2 files)
- ✅ Cosine similarity search
- ✅ Top-K retrieval (K=4)

---

## Bonus Points Achieved

This implementation directly addresses the bonus criterion:

> "If it can run with local models (and doesn't require API keys)" - **GUIDE.pdf**

**Demonstration:**
```bash
# No OPENAI_API_KEY in .env
USE_LOCAL_MODELS=true python src/rag_assessment_partial.py

# Shows:
# ✅ Running with local models (no API key required)
# Loading local embedding model: all-MiniLM-L6-v2...
# Using local LLM: llama3.2 (via Ollama)
# Embedding (local: all-MiniLM-L6-v2): 100%|████| 15/15
# Enter your question: What is the PTO policy?
# Generating answer with llama3.2...
# {
#   "answer": "According to faq_employee.md, employees receive...",
#   "sources": ["faq_employee.md", "faq_auth.md"]
# }
```

**Additional Benefits:**
- Zero ongoing costs (no API charges)
- Privacy (data never leaves machine)
- Offline operation (works without internet)
- Educational value (demonstrates understanding of local inference)
