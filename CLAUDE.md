# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) assessment project for demonstrating a simple semantic search and question-answering system using OpenAI's APIs.

**Core Architecture:**
- `src/rag_assessment_partial.py`: Main implementation file containing the RAG pipeline
- `faqs/`: Directory containing markdown FAQ documents that serve as the knowledge base
  - `faq_sso.md`: SSO-related questions
  - `faq_auth.md`: Authentication questions
  - `faq_employee.md`: Employee handbook content (PTO, equity vesting)

## RAG Pipeline Flow

The system follows a standard RAG pattern:

1. **Load & Chunk**: Read markdown files from `faqs/` and split into ~200 character chunks
2. **Embed**: Convert text chunks to vectors using OpenAI's `text-embedding-ada-002` model
3. **Query**: Accept user question and embed it
4. **Retrieve**: Find top-k (4) most relevant chunks using cosine similarity
5. **Generate**: Use `gpt-3.5-turbo` to answer the question based on retrieved context
6. **Output**: Return JSON with answer and at least 2 cited source files

## Running the Code

```bash
# Set OpenAI API key in .env file
# .env should contain:
# OPENAI_API_KEY=your-key-here

# Run the main script (loads key from environment)
python src/rag_assessment_partial.py
```

The script will prompt for a question and output a JSON response to stdout.

## Key Configuration

Constants defined in `src/rag_assessment_partial.py`:
- `FAQ_DIR = "faqs"`: Source directory for FAQ markdown files
- `EMBED_MODEL = "text-embedding-ada-002"`: OpenAI embedding model
- `LLM_MODEL = "gpt-3.5-turbo"`: OpenAI chat model for answer generation
- `CHUNK_SIZE = 200`: Characters per text chunk
- `TOP_K = 4`: Number of chunks to retrieve for context

## Implementation Notes

The codebase uses:
- OpenAI Python client for embeddings and chat completions
- NumPy for vector operations (cosine similarity)
- tqdm for progress bars during embedding
- Simple character-based chunking (no overlap)

When working on this code, note that the partial implementation has TODOs for:
- Embedding API calls in `embed_texts()`
- Query embedding
- Similarity computation and retrieval
- Answer generation using chat completions
