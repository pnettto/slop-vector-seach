import sqlite3
import os
import numpy as np
from contextlib import contextmanager
from config import DATABASE_PATH

def init_db():
    """Initialize database with schema."""
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filepath TEXT UNIQUE,
                title TEXT,
                content TEXT,
                word_count INTEGER,
                file_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT UNIQUE,
                embedding BLOB,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS concept_vectors (
                name TEXT PRIMARY KEY,
                embedding BLOB,
                source_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS sync_status (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                last_sync TIMESTAMP,
                files_processed INTEGER DEFAULT 0
            );
        """)
        
        # Create FTS5 table if not exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents_fts'"
        )
        if not cursor.fetchone():
            conn.execute("""
                CREATE VIRTUAL TABLE documents_fts USING fts5(
                    document_id UNINDEXED,
                    title,
                    content,
                    tokenize='porter unicode61'
                )
            """)
        
        conn.commit()

@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()

# Document operations
def insert_document(doc_id, filepath, title, content, word_count, file_hash):
    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO documents (id, filepath, title, content, word_count, file_hash, modified_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (doc_id, filepath, title, content, word_count, file_hash))
        
        # Update FTS
        conn.execute("DELETE FROM documents_fts WHERE document_id = ?", (doc_id,))
        conn.execute(
            "INSERT INTO documents_fts (document_id, title, content) VALUES (?, ?, ?)",
            (doc_id, title, content)
        )
        conn.commit()

def get_document(doc_id):
    with get_connection() as conn:
        return conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()

def get_document_by_path(filepath):
    with get_connection() as conn:
        return conn.execute("SELECT * FROM documents WHERE filepath = ?", (filepath,)).fetchone()

def get_all_documents(limit=100, offset=0, search=None):
    with get_connection() as conn:
        if search:
            return conn.execute("""
                SELECT d.* FROM documents d
                JOIN documents_fts fts ON d.id = fts.document_id
                WHERE documents_fts MATCH ?
                ORDER BY d.modified_at DESC
                LIMIT ? OFFSET ?
            """, (search, limit, offset)).fetchall()
        return conn.execute(
            "SELECT * FROM documents ORDER BY modified_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()

def delete_document(doc_id):
    with get_connection() as conn:
        conn.execute("DELETE FROM documents_fts WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM embeddings WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()

def get_document_count():
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

# Embedding operations
def insert_embedding(doc_id, embedding):
    blob = embedding.astype(np.float32).tobytes()
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (document_id, embedding) VALUES (?, ?)",
            (doc_id, blob)
        )
        conn.commit()

def get_embedding(doc_id):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT embedding FROM embeddings WHERE document_id = ?", (doc_id,)
        ).fetchone()
        if row:
            return np.frombuffer(row[0], dtype=np.float32)
        return None

def get_all_embeddings():
    """Return dict of doc_id -> embedding."""
    with get_connection() as conn:
        rows = conn.execute("SELECT document_id, embedding FROM embeddings").fetchall()
        return {
            row[0]: np.frombuffer(row[1], dtype=np.float32)
            for row in rows
        }

def get_embedding_count():
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

# Concept vector operations
def insert_concept(name, embedding, source_text):
    blob = embedding.astype(np.float32).tobytes()
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO concept_vectors (name, embedding, source_text, created_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            (name, blob, source_text)
        )
        conn.commit()

def get_concept(name):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT embedding, source_text FROM concept_vectors WHERE name = ?", (name,)
        ).fetchone()
        if row:
            return np.frombuffer(row[0], dtype=np.float32), row[1]
        return None, None

def get_all_concepts():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT name, source_text, created_at FROM concept_vectors ORDER BY name"
        ).fetchall()
        return [dict(row) for row in rows]

def delete_concept(name):
    with get_connection() as conn:
        conn.execute("DELETE FROM concept_vectors WHERE name = ?", (name,))
        conn.commit()

# FTS search
def fts_search(query, limit=100):
    """Full-text search returning document IDs and BM25 scores."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT document_id, bm25(documents_fts) as score
            FROM documents_fts
            WHERE documents_fts MATCH ?
            ORDER BY score
            LIMIT ?
        """, (query, limit)).fetchall()
        return [(row[0], -row[1]) for row in rows]  # BM25 returns negative scores

# Stats
def get_stats():
    with get_connection() as conn:
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        emb_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        concept_count = conn.execute("SELECT COUNT(*) FROM concept_vectors").fetchone()[0]
        
        last_sync = conn.execute(
            "SELECT last_sync FROM sync_status WHERE id = 1"
        ).fetchone()
        
        db_size = os.path.getsize(DATABASE_PATH) if os.path.exists(DATABASE_PATH) else 0
        
        return {
            "document_count": doc_count,
            "embedding_count": emb_count,
            "concept_count": concept_count,
            "last_sync": last_sync[0] if last_sync else None,
            "database_size_mb": round(db_size / (1024 * 1024), 2)
        }

def update_sync_status(files_processed):
    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO sync_status (id, last_sync, files_processed)
            VALUES (1, CURRENT_TIMESTAMP, ?)
        """, (files_processed,))
        conn.commit()
