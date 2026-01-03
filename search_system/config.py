import os

# Paths
DOCUMENTS_FOLDER = os.getenv("DOCUMENTS_FOLDER", "./documents")
DATABASE_PATH = os.getenv("DATABASE_PATH", "./data/search.db")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
EMBEDDING_BATCH_SIZE = 100

# Processing
MAX_CHUNK_SIZE = 8000  # characters per document
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}

# Flask
DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
HOST = os.getenv("FLASK_HOST", "0.0.0.0")
PORT = int(os.getenv("FLASK_PORT", "5000"))
