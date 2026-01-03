import os
import hashlib
import uuid
from pathlib import Path
from config import DOCUMENTS_FOLDER, SUPPORTED_EXTENSIONS, MAX_CHUNK_SIZE
import database as db
from embeddings import generate_embedding

# Document extractors
def extract_text_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_text_pdf(filepath: str) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except ImportError:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except ImportError:
            return "[PDF extraction requires pypdf or PyPDF2]"

def extract_text_docx(filepath: str) -> str:
    try:
        from docx import Document
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)
    except ImportError:
        return "[DOCX extraction requires python-docx]"

def extract_text(filepath: str) -> str:
    """Extract text from document based on extension."""
    ext = Path(filepath).suffix.lower()
    
    if ext in {".txt", ".md"}:
        return extract_text_txt(filepath)
    elif ext == ".pdf":
        return extract_text_pdf(filepath)
    elif ext == ".docx":
        return extract_text_docx(filepath)
    else:
        return ""

def compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of file content."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_title(filepath: str, content: str) -> str:
    """Extract title from filename or first line."""
    filename = Path(filepath).stem
    # Try first non-empty line as title
    for line in content.split("\n"):
        line = line.strip()
        if line:
            # Remove markdown headers
            if line.startswith("#"):
                line = line.lstrip("#").strip()
            return line[:200] if len(line) > 200 else line
    return filename

def process_document(filepath: str, force: bool = False) -> dict:
    """
    Process a single document.
    Returns dict with status: 'added', 'updated', 'unchanged', or 'error'.
    """
    try:
        abs_path = os.path.abspath(filepath)
        file_hash = compute_file_hash(abs_path)
        
        # Check if already processed
        existing = db.get_document_by_path(abs_path)
        if existing and existing["file_hash"] == file_hash and not force:
            return {"status": "unchanged", "filepath": abs_path}
        
        # Extract content
        content = extract_text(abs_path)
        if not content.strip():
            return {"status": "error", "filepath": abs_path, "error": "No content extracted"}
        
        # Truncate for embedding
        content_for_embedding = content[:MAX_CHUNK_SIZE]
        
        # Generate embedding
        embedding = generate_embedding(content_for_embedding)
        
        # Generate or reuse ID
        doc_id = existing["id"] if existing else str(uuid.uuid4())
        
        # Get title and word count
        title = get_title(abs_path, content)
        word_count = len(content.split())
        
        # Store in database
        db.insert_document(doc_id, abs_path, title, content, word_count, file_hash)
        db.insert_embedding(doc_id, embedding)
        
        status = "updated" if existing else "added"
        return {"status": status, "filepath": abs_path, "doc_id": doc_id}
        
    except Exception as e:
        return {"status": "error", "filepath": filepath, "error": str(e)}

def find_documents(folder: str) -> list[str]:
    """Find all supported documents in folder."""
    documents = []
    folder_path = Path(folder)
    
    if not folder_path.exists():
        return documents
    
    for ext in SUPPORTED_EXTENSIONS:
        documents.extend(str(p) for p in folder_path.rglob(f"*{ext}"))
    
    # Filter out hidden files and common junk
    documents = [
        d for d in documents
        if not any(part.startswith(".") for part in Path(d).parts)
        and "__pycache__" not in d
        and "node_modules" not in d
    ]
    
    return sorted(documents)

def sync_folder(folder: str = None, progress_callback=None) -> dict:
    """
    Sync all documents in folder with database.
    Returns summary of operations.
    """
    folder = folder or DOCUMENTS_FOLDER
    documents = find_documents(folder)
    
    summary = {
        "total": len(documents),
        "added": 0,
        "updated": 0,
        "unchanged": 0,
        "errors": [],
        "processed": 0
    }
    
    for i, filepath in enumerate(documents):
        result = process_document(filepath)
        summary["processed"] = i + 1
        
        if result["status"] == "added":
            summary["added"] += 1
        elif result["status"] == "updated":
            summary["updated"] += 1
        elif result["status"] == "unchanged":
            summary["unchanged"] += 1
        elif result["status"] == "error":
            summary["errors"].append({"file": filepath, "error": result.get("error")})
        
        if progress_callback:
            progress_callback(i + 1, len(documents), filepath)
    
    # Update sync status
    db.update_sync_status(summary["processed"])
    
    return summary

def remove_deleted_documents(folder: str = None) -> int:
    """Remove documents from DB that no longer exist on disk."""
    folder = folder or DOCUMENTS_FOLDER
    removed = 0
    
    with db.get_connection() as conn:
        docs = conn.execute("SELECT id, filepath FROM documents").fetchall()
        
    for doc in docs:
        if not os.path.exists(doc["filepath"]):
            db.delete_document(doc["id"])
            removed += 1
    
    return removed
