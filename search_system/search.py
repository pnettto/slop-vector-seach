import time
import database as db
from embeddings import (
    generate_embedding,
    cosine_similarity,
    normalize,
    mix_vectors,
    debias_vector
)
import numpy as np

def keyword_search(query: str, limit: int = 20) -> dict:
    """Full-text search using SQLite FTS5 with BM25 ranking."""
    start = time.time()
    
    # FTS5 search
    results = db.fts_search(query, limit)
    
    # Fetch document details
    documents = []
    for doc_id, score in results:
        doc = db.get_document(doc_id)
        if doc:
            documents.append({
                "id": doc["id"],
                "title": doc["title"],
                "filepath": doc["filepath"],
                "word_count": doc["word_count"],
                "snippet": _extract_snippet(doc["content"], query),
                "score": round(score, 4),
                "modified_at": doc["modified_at"]
            })
    
    elapsed = (time.time() - start) * 1000
    
    return {
        "results": documents,
        "execution_details": {
            "mode": "keyword",
            "processing_time_ms": round(elapsed, 2),
            "operations": ["fts5_bm25_search"],
            "total_candidates": len(results),
            "returned": len(documents)
        }
    }

def semantic_search(query: str, limit: int = 20) -> dict:
    """Vector similarity search using cosine similarity."""
    start = time.time()
    
    # Embed query
    query_embedding = generate_embedding(query)
    embed_time = time.time()
    
    # Get all embeddings
    all_embeddings = db.get_all_embeddings()
    
    # Compute similarities
    scores = []
    for doc_id, embedding in all_embeddings.items():
        sim = cosine_similarity(query_embedding, embedding)
        scores.append((doc_id, sim))
    
    # Sort by similarity
    scores.sort(key=lambda x: x[1], reverse=True)
    top_results = scores[:limit]
    
    # Fetch document details
    documents = []
    for doc_id, score in top_results:
        doc = db.get_document(doc_id)
        if doc:
            documents.append({
                "id": doc["id"],
                "title": doc["title"],
                "filepath": doc["filepath"],
                "word_count": doc["word_count"],
                "snippet": doc["content"][:300] + "..." if len(doc["content"]) > 300 else doc["content"],
                "score": round(score, 4),
                "modified_at": doc["modified_at"]
            })
    
    elapsed = (time.time() - start) * 1000
    
    return {
        "results": documents,
        "execution_details": {
            "mode": "semantic",
            "processing_time_ms": round(elapsed, 2),
            "embedding_time_ms": round((embed_time - start) * 1000, 2),
            "operations": ["embed_query", f"cosine_similarity({len(all_embeddings)})"],
            "total_candidates": len(all_embeddings),
            "returned": len(documents)
        }
    }

def hybrid_search(query: str, limit: int = 20) -> dict:
    """Keyword search followed by semantic reranking."""
    start = time.time()
    
    # Get top 100 keyword results
    keyword_results = db.fts_search(query, 100)
    keyword_time = time.time()
    
    if not keyword_results:
        return {
            "results": [],
            "execution_details": {
                "mode": "hybrid",
                "processing_time_ms": round((time.time() - start) * 1000, 2),
                "operations": ["fts5_bm25_search(0)"],
                "total_candidates": 0,
                "returned": 0
            }
        }
    
    # Embed query
    query_embedding = generate_embedding(query)
    embed_time = time.time()
    
    # Rerank by semantic similarity
    reranked = []
    for doc_id, bm25_score in keyword_results:
        embedding = db.get_embedding(doc_id)
        if embedding is not None:
            sem_score = cosine_similarity(query_embedding, embedding)
            # Combine scores (weighted average)
            combined = 0.3 * (bm25_score / 10) + 0.7 * sem_score
            reranked.append((doc_id, combined, bm25_score, sem_score))
    
    reranked.sort(key=lambda x: x[1], reverse=True)
    top_results = reranked[:limit]
    
    # Fetch document details
    documents = []
    for doc_id, combined, bm25, sem in top_results:
        doc = db.get_document(doc_id)
        if doc:
            documents.append({
                "id": doc["id"],
                "title": doc["title"],
                "filepath": doc["filepath"],
                "word_count": doc["word_count"],
                "snippet": _extract_snippet(doc["content"], query),
                "score": round(combined, 4),
                "bm25_score": round(bm25, 4),
                "semantic_score": round(sem, 4),
                "modified_at": doc["modified_at"]
            })
    
    elapsed = (time.time() - start) * 1000
    
    return {
        "results": documents,
        "execution_details": {
            "mode": "hybrid",
            "processing_time_ms": round(elapsed, 2),
            "keyword_time_ms": round((keyword_time - start) * 1000, 2),
            "embedding_time_ms": round((embed_time - keyword_time) * 1000, 2),
            "operations": [
                f"fts5_bm25_search({len(keyword_results)})",
                "embed_query",
                f"semantic_rerank({len(reranked)})"
            ],
            "total_candidates": len(keyword_results),
            "returned": len(documents)
        }
    }

def store_concept(name: str, text: str) -> dict:
    """Store a reusable concept embedding."""
    start = time.time()
    
    embedding = generate_embedding(text)
    db.insert_concept(name, embedding, text)
    
    elapsed = (time.time() - start) * 1000
    
    return {
        "name": name,
        "source_text": text[:200] + "..." if len(text) > 200 else text,
        "processing_time_ms": round(elapsed, 2)
    }

def vector_search(concept_mix: dict[str, float], limit: int = 20) -> dict:
    """
    Search using weighted concept combinations.
    concept_mix: dict of concept_name -> weight (e.g., {"ai": 0.6, "safety": 0.4, "hype": -0.2})
    """
    start = time.time()
    
    # Load concept vectors
    vectors = {}
    for name, weight in concept_mix.items():
        vec, _ = db.get_concept(name)
        if vec is not None:
            vectors[name] = (vec, weight)
    
    if not vectors:
        return {
            "results": [],
            "execution_details": {
                "mode": "concept",
                "processing_time_ms": round((time.time() - start) * 1000, 2),
                "operations": ["no_valid_concepts"],
                "error": "No valid concepts found"
            }
        }
    
    # Create mixed vector
    mixed = mix_vectors(vectors)
    
    # Search with mixed vector
    all_embeddings = db.get_all_embeddings()
    
    scores = []
    for doc_id, embedding in all_embeddings.items():
        sim = cosine_similarity(mixed, embedding)
        scores.append((doc_id, sim))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    top_results = scores[:limit]
    
    # Fetch document details
    documents = []
    for doc_id, score in top_results:
        doc = db.get_document(doc_id)
        if doc:
            documents.append({
                "id": doc["id"],
                "title": doc["title"],
                "filepath": doc["filepath"],
                "word_count": doc["word_count"],
                "snippet": doc["content"][:300] + "..." if len(doc["content"]) > 300 else doc["content"],
                "score": round(score, 4),
                "modified_at": doc["modified_at"]
            })
    
    elapsed = (time.time() - start) * 1000
    
    concept_str = ", ".join(f"{k}:{v}" for k, v in concept_mix.items())
    
    return {
        "results": documents,
        "execution_details": {
            "mode": "concept",
            "processing_time_ms": round(elapsed, 2),
            "operations": [
                f"load_concepts({len(vectors)})",
                f"mix_vectors({concept_str})",
                f"cosine_similarity({len(all_embeddings)})"
            ],
            "total_candidates": len(all_embeddings),
            "returned": len(documents)
        }
    }

def debias_search(main_concept: str, remove_concept: str, limit: int = 20) -> dict:
    """Search for main_concept with remove_concept factored out."""
    start = time.time()
    
    main_vec, _ = db.get_concept(main_concept)
    remove_vec, _ = db.get_concept(remove_concept)
    
    if main_vec is None or remove_vec is None:
        return {
            "results": [],
            "execution_details": {
                "mode": "debias",
                "error": "One or both concepts not found"
            }
        }
    
    # Create debiased vector
    debiased = debias_vector(main_vec, remove_vec)
    
    # Search with debiased vector
    all_embeddings = db.get_all_embeddings()
    
    scores = []
    for doc_id, embedding in all_embeddings.items():
        sim = cosine_similarity(debiased, embedding)
        scores.append((doc_id, sim))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    top_results = scores[:limit]
    
    # Fetch document details
    documents = []
    for doc_id, score in top_results:
        doc = db.get_document(doc_id)
        if doc:
            documents.append({
                "id": doc["id"],
                "title": doc["title"],
                "filepath": doc["filepath"],
                "word_count": doc["word_count"],
                "snippet": doc["content"][:300] + "..." if len(doc["content"]) > 300 else doc["content"],
                "score": round(score, 4),
                "modified_at": doc["modified_at"]
            })
    
    elapsed = (time.time() - start) * 1000
    
    return {
        "results": documents,
        "execution_details": {
            "mode": "debias",
            "processing_time_ms": round(elapsed, 2),
            "operations": [
                f"load_concept({main_concept})",
                f"load_concept({remove_concept})",
                f"debias({main_concept} - {remove_concept})",
                f"cosine_similarity({len(all_embeddings)})"
            ],
            "total_candidates": len(all_embeddings),
            "returned": len(documents)
        }
    }

def _extract_snippet(content: str, query: str, context_chars: int = 150) -> str:
    """Extract snippet around query terms."""
    content_lower = content.lower()
    query_terms = query.lower().split()
    
    # Find first occurrence of any query term
    best_pos = len(content)
    for term in query_terms:
        pos = content_lower.find(term)
        if 0 <= pos < best_pos:
            best_pos = pos
    
    if best_pos == len(content):
        # No match found, return beginning
        return content[:300] + "..." if len(content) > 300 else content
    
    # Extract context around match
    start = max(0, best_pos - context_chars)
    end = min(len(content), best_pos + context_chars + len(query))
    
    snippet = content[start:end]
    
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."
    
    return snippet
