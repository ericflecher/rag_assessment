# RAG FAQ Question Answering System

A minimal Retrieval-Augmented Generation (RAG) prototype that answers questions using a small corpus of FAQ documents. Built as a technical demonstration for semantic search and LLM-powered question answering.

## Features

- **Semantic Search**: Finds relevant information using OpenAI's text-embedding-ada-002
- **LLM-Powered Answers**: Generates natural language answers with GPT-3.5-turbo
- **Source Citations**: Automatically cites at least 2 source files in responses
- **Simple CLI**: Interactive command-line interface for queries
- **Vector Similarity**: Uses cosine similarity for relevance scoring
- **Configurable**: Easy configuration via environment variables

## Architecture

1. **Load & Chunk**: Reads markdown FAQ files and splits into ~200 character chunks
2. **Embed**: Converts text chunks to vector embeddings via OpenAI API
3. **Query**: Accepts user questions and embeds them using same model
4. **Retrieve**: Finds top-4 most relevant chunks using cosine similarity
5. **Generate**: Uses GPT-3.5-turbo to create answers based on retrieved context
6. **Output**: Returns JSON with answer and source citations

## Requirements

- Python 3.10+
- OpenAI API key
- Dependencies: `openai`, `numpy`, `tqdm`, `python-dotenv`

## Setup

### 1. Clone and Navigate
```bash
cd rag_assessment
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install openai numpy tqdm python-dotenv
```

### 4. Configure Environment
```bash
# Copy template
cp .env.template .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-...

# Optional: Customize model configuration
# EMBED_MODEL=text-embedding-ada-002
# LLM_MODEL=gpt-3.5-turbo
```

### 5. Verify FAQ Documents
Ensure `faqs/` directory contains your FAQ markdown files:
```
faqs/
├── faq_auth.md
├── faq_employee.md
└── faq_sso.md
```

## Usage

### Basic Usage

Run the application:
```bash
python src/rag_assessment_partial.py
```

You'll be prompted to enter a question:
```
Embedding: 100%|████████████████████| 15/15 [00:03<00:00,  4.2it/s]
Enter your question: How do I reset my password?
```

### Example Output

```json
{
  "answer": "To reset your password, use the reset link on the login page (faq_auth.md).",
  "sources": [
    "faq_auth.md",
    "faq_sso.md"
  ]
}
```

### Sample Queries

**Authentication:**
```bash
python src/rag_assessment_partial.py
# Enter: How do I reset my password?
```

**Employee Policies:**
```bash
python src/rag_assessment_partial.py
# Enter: What is the PTO policy?
```

**Complex Query (Multiple Sources):**
```bash
python src/rag_assessment_partial.py
# Enter: Tell me about SSO and authentication
```

## Configuration

### Environment Variables (.env)

Model configuration is managed via `.env` file:

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional - Override default models
EMBED_MODEL=text-embedding-ada-002  # Embedding model
LLM_MODEL=gpt-3.5-turbo            # LLM for answer generation
```

### Code Configuration

Edit `src/rag_assessment_partial.py` for additional settings:

```python
FAQ_DIR = "faqs"     # FAQ directory path
CHUNK_SIZE = 200     # Characters per chunk
TOP_K = 4            # Number of chunks to retrieve
```

## Project Structure

```
rag_assessment/
├── .env                      # API keys (not in git)
├── .env.template            # Template for .env
├── .gitignore               # Git ignore rules
├── README.md                # This file
├── CLAUDE.md                # Claude Code guidance
├── faqs/                    # FAQ documents
│   ├── faq_auth.md
│   ├── faq_employee.md
│   └── faq_sso.md
├── src/
│   └── rag_assessment_partial.py  # Main implementation
├── docs/                    # Technical exercise docs
└── planning/                # Implementation plans
    ├── mvp_implementation_plan.md
    └── backlog.md
```

## Documentation

- **[Technical Exercise Requirements](docs/)** - Original assessment guidelines and acceptance criteria
- **[Implementation Plan](planning/mvp_implementation_plan.md)** - Detailed MVP implementation blueprints
- **[Feature Backlog](planning/backlog.md)** - Future enhancements and improvements

## How It Works

### 1. Text Chunking
FAQ documents are split into 200-character chunks to create focused, searchable pieces while maintaining context.

### 2. Embedding Generation
Each chunk is converted to a 1536-dimensional vector using OpenAI's `text-embedding-ada-002` model, capturing semantic meaning.

### 3. Similarity Search
User queries are embedded using the same model, then compared to chunk embeddings using cosine similarity:

```python
similarity = np.dot(query_vec, chunk_vec) / (||query_vec|| * ||chunk_vec||)
```

### 4. Context Retrieval
The top 4 most similar chunks are retrieved and formatted with source attribution.

### 5. Answer Generation
Retrieved context is provided to GPT-3.5-turbo with instructions to:
- Answer based only on provided context
- Cite at least 2 source files
- Provide accurate, factual responses

## Design Decisions

### Character-Based Chunking
- **Pro**: Simple, predictable, fast implementation
- **Con**: May split sentences awkwardly
- **Alternative**: Semantic chunking (sentence/paragraph boundaries)

### In-Memory Storage
- **Pro**: Simple for small FAQ sets
- **Con**: Not scalable to large document sets
- **Production**: Use vector database (Pinecone, Weaviate, ChromaDB)

### No Caching
- **Pro**: Keeps code simple for demo
- **Con**: Re-embeds chunks on every run
- **Production**: Cache embeddings to file/database

### Individual API Calls
- **Pro**: Progress tracking with tqdm
- **Con**: Slower than batch calls
- **Production**: Batch API calls (up to 2048 inputs per request)

## Troubleshooting

### API Key Error
```
Error: OPENAI_API_KEY not set
```
**Solution**: Ensure `.env` file exists with valid API key

### No FAQ Files Found
```
Error: No .md files found in 'faqs/'
```
**Solution**: Verify FAQ directory and file extensions

### Rate Limit Error
```
OpenAI API error: Rate limit exceeded
```
**Solution**: Wait and retry, or upgrade API plan

## Performance

Typical performance metrics:
- **Embedding**: ~1-2 seconds per chunk (depends on API latency)
- **Query Embedding**: ~0.5 seconds
- **Retrieval**: <0.1 seconds (in-memory)
- **Answer Generation**: ~2-4 seconds
- **Total**: ~3-5 seconds per query

## Testing

Test with diverse queries to verify:
1. Single-source questions retrieve correct file
2. Multi-source questions cite at least 2 files
3. Answers are factual and grounded in context
4. JSON output is well-formed

### Test Queries
```bash
# Test 1: Single source (faq_auth.md)
python src/rag_assessment_partial.py
# Enter: How do I reset my password?

# Test 2: Single source (faq_employee.md)
python src/rag_assessment_partial.py
# Enter: What is the PTO policy?

# Test 3: Single source (faq_employee.md)
python src/rag_assessment_partial.py
# Enter: How does equity vesting work?

# Test 4: Multiple sources
python src/rag_assessment_partial.py
# Enter: Tell me about SSO and authentication
```

## Production Considerations

This is a minimal prototype. For production:

**Scalability:**
- Use vector database (Pinecone, Weaviate, Qdrant)
- Implement embedding caching
- Batch API calls

**Quality:**
- Add re-ranking after retrieval
- Use semantic chunking with overlap
- Implement query expansion

**Reliability:**
- Add retry logic for API failures
- Implement fallback responses
- Add input validation and sanitization

**Observability:**
- Log query-answer pairs
- Track latency metrics
- Monitor API costs

See `planning/backlog.md` for detailed feature roadmap.

## License

This is a technical demonstration project.

## Acknowledgments

- Built with OpenAI API
- Uses numpy for vector operations
- tqdm for progress visualization
