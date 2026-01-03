# Document Search System with Semantic + Lexical Search

Build a Python-based document search system with vector embeddings and full-text search capabilities.

## Core Requirements

### 1. Document Processing Pipeline
- Accept a folder path containing documents (.txt, .md, .pdf, .docx)
- Extract text from each document
- Generate embeddings using OpenAI's text-embedding-3-small model
- Store in SQLite with FTS5 (full-text search) and vector capabilities
- Support incremental updates (don't re-process unchanged files)

### 2. Database Schema
```sql
-- documents table
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    filepath TEXT UNIQUE,
    title TEXT,
    content TEXT,
    word_count INTEGER,
    created_at TIMESTAMP,
    modified_at TIMESTAMP
);

-- embeddings table
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY,
    document_id TEXT,
    embedding BLOB,  -- store as numpy array bytes
    FOREIGN KEY(document_id) REFERENCES documents(id)
);

-- full-text search
CREATE VIRTUAL TABLE documents_fts USING fts5(
    document_id UNINDEXED,
    content,
    tokenize='porter unicode61'
);

-- stored concept vectors (for reuse)
CREATE TABLE concept_vectors (
    name TEXT PRIMARY KEY,
    embedding BLOB,
    source_text TEXT,
    created_at TIMESTAMP
);
```

### 3. Search Capabilities

Implement these search modes:

**A. Keyword Search (BM25-style)**
```python
def keyword_search(query: str, limit: int = 20) -> List[Dict]:
    """Full-text search using SQLite FTS5"""
    pass
```

**B. Semantic Search**
```python
def semantic_search(query: str, limit: int = 20) -> List[Dict]:
    """
    1. Embed the query
    2. Compute cosine similarity with all document embeddings
    3. Return top-k results
    """
    pass
```

**C. Hybrid Search**
```python
def hybrid_search(query: str, limit: int = 20) -> List[Dict]:
    """
    1. Get top 100 keyword results
    2. Re-rank by semantic similarity
    3. Return top limit results
    """
    pass
```

**D. Concept Vector Operations**
```python
def store_concept(name: str, text: str):
    """Store a reusable concept embedding"""
    pass

def vector_search(concept_mix: Dict[str, float], limit: int = 20):
    """
    Search using weighted concept combinations.
    Example: {"interpretability": 0.6, "governance": 0.4, "hype": -0.2}
    """
    pass

def debias_search(main_concept: str, remove_concept: str, limit: int = 20):
    """
    Search for main_concept but remove overlap with remove_concept.
    Example: "humble tone" debiased by "humility topic"
    """
    pass
```

### 4. Admin Interface (Flask)

Create `/admin` routes:

- `GET /admin` - Dashboard showing:
  - Total documents indexed
  - Total embeddings
  - Storage size
  - Last sync time
  
- `POST /admin/sync` - Process documents folder
  - Show progress (X/Y files processed)
  - Return summary of added/updated/unchanged files

- `GET /admin/concepts` - List stored concept vectors

- `POST /admin/concepts` - Create new concept vector
  - Input: name, description text
  
- `DELETE /admin/concepts/<name>` - Delete concept vector

- `GET /admin/documents` - Paginated document list with search

- `DELETE /admin/documents/<id>` - Remove document and its embedding

### 5. User Interface (Flask)

Create `/` routes:

**Main Search Page** (`GET /`)
- Single search input
- Radio buttons: Keyword / Semantic / Hybrid / Concept Mix
- Search button
- Results area showing:
  - Query execution details (which mode, processing time)
  - Generated SQL or operation (in collapsible section)
  - Results list with title, snippet, relevance score

**Concept Search Page** (`GET /concepts`)
- Interface to build concept mixes:
  - Add concept with weight slider (-1.0 to 1.0)
  - List of stored concepts to choose from
  - Live preview of the vector operation
- Search button
- Same results display as main page

**API Endpoints:**
```python
POST /api/search
{
    "query": "mesa-optimization",
    "mode": "hybrid",  # keyword | semantic | hybrid | concept
    "limit": 20,
    "concept_mix": {"mesa_opt": 1.0, "hype": -0.3}  # optional
}

Response:
{
    "results": [...],
    "execution_details": {
        "mode": "hybrid",
        "processing_time_ms": 145,
        "operations": ["keyword_search(100)", "semantic_rerank"],
        "total_candidates": 100,
        "returned": 20
    }
}
```

### 6. Technical Stack

- **Framework:** Flask
- **Database:** SQLite with FTS5
- **Embeddings:** OpenAI text-embedding-3-small (1536 dims)
- **Vector ops:** numpy + scipy (cosine_similarity)
- **Document parsing:** 
  - pypdf2 for PDFs
  - python-docx for DOCX
  - Plain text for .txt/.md
- **Frontend:** Minimal HTML + Tailwind CSS + Alpine.js (no build step)

### 7. Key Implementation Details

**Cosine Similarity (fast):**
```python
from scipy.spatial.distance import cosine

def cosine_similarity(a, b):
    return 1 - cosine(a, b)

def search_similar(query_embedding, limit=20):
    # Load all embeddings
    # Compute similarities
    # Sort and return top-k
    pass
```

**Vector Operations:**
```python
def normalize(v):
    return v / np.linalg.norm(v)

def mix_vectors(concepts: Dict[str, float]) -> np.ndarray:
    """Weighted sum of concept vectors"""
    result = np.zeros(1536)
    for name, weight in concepts.items():
        vec = load_concept_vector(name)
        result += weight * vec
    return normalize(result)

def debias_vector(main: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Remove projection of bias from main"""
    projection = np.dot(main, bias) * bias
    return normalize(main - projection)
```

**Progress Display:**
For document sync, use SSE (Server-Sent Events) or simple polling:
```python
@app.route('/admin/sync', methods=['POST'])
def sync_documents():
    # Start background task
    # Return task_id
    
@app.route('/admin/sync/status/<task_id>')
def sync_status(task_id):
    # Return current progress
    return {"processed": 45, "total": 120, "current_file": "doc.pdf"}
```

### 8. UI Design (Minimal)

**Color Scheme:**
- Background: #fafafa
- Cards: white with subtle shadow
- Primary action: #2563eb (blue)
- Text: #1f2937 (dark gray)
- Borders: #e5e7eb (light gray)

**Layout:**
- Max width 1200px, centered
- Single column for search
- Results in cards with hover effect
- Monospace font for execution details
- Collapsible sections for advanced info

**Search Results Card:**
```
┌─────────────────────────────────────┐
│ Document Title                       │
│ ─────────────────────────────────── │
│ Snippet of content showing context   │
│ with search terms highlighted...     │
│                                      │
│ 📄 document.pdf  📅 2024-01-15       │
│ ⭐ Relevance: 0.87                   │
└─────────────────────────────────────┘
```

### 9. Configuration

Store in `config.py`:
```python
DOCUMENTS_FOLDER = "./documents"
DATABASE_PATH = "./search.db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
MAX_CHUNK_SIZE = 8000  # characters per document chunk
```

### 10. Project Structure

```
search_system/
├── app.py              # Flask app
├── config.py           # Configuration
├── database.py         # DB initialization and queries
├── embeddings.py       # Embedding generation and vector ops
├── indexer.py          # Document processing pipeline
├── search.py           # Search implementations
├── templates/
│   ├── index.html      # Main search UI
│   ├── admin.html      # Admin dashboard
│   └── concepts.html   # Concept mixer UI
├── static/
│   └── style.css       # Minimal CSS (or use CDN Tailwind)
└── documents/          # User's document folder
```

### 11. Key Differences from ExoPriors

**Simplified:**
- No PostgreSQL, just SQLite
- No distributed architecture
- Smaller scale (thousands of docs, not millions)
- No complex SQL exposure (pre-built queries only)
- Single-user by default

**Kept:**
- Core vector operations (mix, debias, similarity)
- Hybrid search capability
- Concept storage and reuse
- Transparent execution details
- Clean minimal UI

### 12. Getting Started

The system should work like this:

1. User runs: `python app.py`
2. Opens `http://localhost:5000/admin`
3. Points to documents folder and clicks "Sync"
4. Watches progress bar as documents are processed
5. Goes to `http://localhost:5000` and searches
6. Can create concept vectors in `/concepts` for advanced searches

### Success Criteria

- Process 1000 documents in under 5 minutes (excluding API rate limits)
- Keyword search returns results in < 100ms
- Semantic search returns results in < 500ms
- UI clearly shows what's happening under the hood
- No JavaScript frameworks required (Alpine.js for interactivity only)
- Single command to start: `python app.py`

## Implementation Notes

- Use connection pooling for SQLite
- Batch embedding API calls (max 100 per request)
- Cache embeddings in memory for frequently used concepts
- Show API costs in admin panel (estimate based on token count)
- Handle rate limits gracefully with exponential backoff
- Validate file types before processing
- Skip hidden files and common non-document files (.DS_Store, etc.)

Build this as a single working application, not a framework. Optimize for understanding and ease of use over flexibility.

Last thing: the app should run in a Docker container.