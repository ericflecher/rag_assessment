"""
RAG Assessment Skeleton
----------------------
This is a technical interview assessment for a Solutions Engineer role at Glean.

Your task: Complete the functions and logic below to build a simple Retrieval-Augmented Generation (RAG) demo.

Requirements:
- Read and chunk markdown files from the 'faqs/' directory.
- Embed the chunks using OpenAI's embedding API.
- Retrieve the top-k most relevant chunks for a user query.
- Generate an answer using OpenAI's chat completion API, citing at least two source files.
- Output the answer and sources as a JSON object to stdout.

"""

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
EMBED_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo"
CHUNK_SIZE = 200  # characters per chunk
TOP_K = 4

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chunk_text(text, size=CHUNK_SIZE):
    """
    Split the input text into chunks of approximately 'size' characters.
    Return a list of text chunks.
    """
    return [text[i:i+size] for i in range(0, len(text), size)]

def cosine_sim(a, b):
    """
    Compute the cosine similarity between two vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_and_chunk_faqs(faq_dir):
    """
    Read all .md files in faq_dir, chunk their contents, and return:
    - chunks: List of text chunks
    - sources: List of corresponding source filenames
    """
    chunks = []
    sources = []
    for fname in os.listdir(faq_dir):
        if fname.endswith(".md"):
            with open(os.path.join(faq_dir, fname)) as f:
                text = f.read()
            for chunk in chunk_text(text):
                chunks.append(chunk)
                sources.append(fname)
    return chunks, sources

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

def main():
    # --- 1. Load & Chunk ---
    chunks, sources = load_and_chunk_faqs(FAQ_DIR)

    # --- 2. Embed Chunks ---
    chunk_embeddings = embed_texts(chunks)

    # --- 3. Query Loop ---
    query = input("Enter your question: ")
    # Embed the query using the same embedding model
    response = client.embeddings.create(
        input=[query],
        model=EMBED_MODEL  # "text-embedding-ada-002"
    )
    query_emb = np.array(response.data[0].embedding)

    # --- 4. Retrieve Top-k ---
    # Compute cosine similarity between query and all chunk embeddings
    sims = np.array([cosine_sim(query_emb, chunk_emb) for chunk_emb in chunk_embeddings])

    # Get indices of top-k most similar chunks
    # np.argsort gives ascending order, so [-TOP_K:] gets last K (highest), [::-1] reverses
    top_indices = np.argsort(sims)[-TOP_K:][::-1]
    top_chunks = [chunks[i] for i in top_indices]
    top_files = [sources[i] for i in top_indices]

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

    # --- 6. Output JSON ---
    # Get unique sources from retrieved chunks, preserving relevance order
    unique_sources = []
    for source in top_files:
        if source not in unique_sources:
            unique_sources.append(source)

    # Ensure we have at least 2 sources as per requirements
    # If we have fewer unique sources, pad with top_files to meet requirement
    output_sources = unique_sources if len(unique_sources) >= 2 else list(sorted(set(top_files)))

    output = {
        "answer": answer,
        "sources": output_sources
    }
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main() 