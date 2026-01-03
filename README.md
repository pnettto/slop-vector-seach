Disclaimer: absolutely everything in this project was created by an coding agent, including the prompt below, which it outputted after I asked it for a prompt to re-create something like this: https://exopriors.com/scry. The result is not super useful but still really impressive and good studying material.

# Document Search System

A document search system with semantic + lexical search using OpenAI embeddings and SQLite FTS5.

## Quick Start

```bash
# Set API key
export OPENAI_API_KEY=your_key

# Run with Docker
docker compose up --build

# Or run directly
pip install -r requirements.txt
python app.py
```

Open http://localhost:5005

## Features

- **Keyword Search**: BM25-ranked full-text search
- **Semantic Search**: Vector similarity search
- **Hybrid Search**: Keyword + semantic reranking  
- **Concept Vectors**: Store and combine concepts for advanced search
- **Admin Dashboard**: Document sync, stats, concept management
