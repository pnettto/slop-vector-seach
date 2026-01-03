import numpy as np
from openai import OpenAI
from config import OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, EMBEDDING_BATCH_SIZE
import time

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def generate_embedding(text: str) -> np.ndarray:
    """Generate embedding for a single text."""
    if not client:
        raise ValueError("OpenAI API key not configured")
    
    # Truncate if too long (model limit ~8k tokens)
    text = text[:32000]
    
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        dimensions=EMBEDDING_DIMENSIONS
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def batch_generate_embeddings(texts: list[str], progress_callback=None) -> list[np.ndarray]:
    """Generate embeddings for multiple texts with batching and rate limiting."""
    if not client:
        raise ValueError("OpenAI API key not configured")
    
    embeddings = []
    total = len(texts)
    
    for i in range(0, total, EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        # Truncate each text
        batch = [t[:32000] for t in batch]
        
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
                dimensions=EMBEDDING_DIMENSIONS
            )
            for item in response.data:
                embeddings.append(np.array(item.embedding, dtype=np.float32))
            
            if progress_callback:
                progress_callback(min(i + EMBEDDING_BATCH_SIZE, total), total)
                
        except Exception as e:
            if "rate_limit" in str(e).lower():
                time.sleep(60)  # Wait and retry
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                    dimensions=EMBEDDING_DIMENSIONS
                )
                for item in response.data:
                    embeddings.append(np.array(item.embedding, dtype=np.float32))
            else:
                raise
    
    return embeddings

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(normalize(a), normalize(b)))

def mix_vectors(vectors: dict[str, tuple[np.ndarray, float]]) -> np.ndarray:
    """
    Create weighted combination of vectors.
    vectors: dict of name -> (vector, weight)
    """
    result = np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)
    for name, (vec, weight) in vectors.items():
        result += weight * vec
    return normalize(result)

def debias_vector(main: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Remove projection of bias from main vector."""
    bias_normalized = normalize(bias)
    projection = np.dot(main, bias_normalized) * bias_normalized
    return normalize(main - projection)

def estimate_tokens(text: str) -> int:
    """Rough token count estimate (4 chars per token)."""
    return len(text) // 4

def estimate_cost(texts: list[str]) -> float:
    """Estimate API cost for embedding texts (text-embedding-3-small: $0.02/1M tokens)."""
    total_tokens = sum(estimate_tokens(t) for t in texts)
    return (total_tokens / 1_000_000) * 0.02
