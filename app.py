import os
import uuid
import threading
from flask import Flask, render_template, request, jsonify
import config
import database as db
import search
import indexer
from embeddings import estimate_cost

app = Flask(__name__)

# Sync task state
sync_tasks = {}

# Initialize database on startup
db.init_db()

# ============== User Routes ==============

@app.route("/")
def index():
    """Main search page."""
    stats = db.get_stats()
    concepts = db.get_all_concepts()
    return render_template("index.html", stats=stats, concepts=concepts)

@app.route("/concepts")
def concepts_page():
    """Concept mixer page."""
    concepts = db.get_all_concepts()
    return render_template("concepts.html", concepts=concepts)

@app.route("/api/search", methods=["POST"])
def api_search():
    """Unified search API endpoint."""
    data = request.get_json()
    
    query = data.get("query", "").strip()
    mode = data.get("mode", "hybrid")
    limit = min(data.get("limit", 20), 100)
    concept_mix = data.get("concept_mix")
    
    if mode == "concept" and concept_mix:
        return jsonify(search.vector_search(concept_mix, limit))
    elif mode == "debias":
        main = data.get("main_concept")
        remove = data.get("remove_concept")
        if main and remove:
            return jsonify(search.debias_search(main, remove, limit))
        return jsonify({"error": "main_concept and remove_concept required"}), 400
    elif not query:
        return jsonify({"error": "Query required"}), 400
    elif mode == "keyword":
        return jsonify(search.keyword_search(query, limit))
    elif mode == "semantic":
        return jsonify(search.semantic_search(query, limit))
    else:  # hybrid
        return jsonify(search.hybrid_search(query, limit))

# ============== Admin Routes ==============

@app.route("/admin")
def admin():
    """Admin dashboard."""
    stats = db.get_stats()
    concepts = db.get_all_concepts()
    return render_template("admin.html", stats=stats, concepts=concepts)

@app.route("/admin/sync", methods=["POST"])
def admin_sync():
    """Start document sync."""
    data = request.get_json() or {}
    folder = data.get("folder", config.DOCUMENTS_FOLDER)
    
    # Check folder exists
    if not os.path.exists(folder):
        return jsonify({"error": f"Folder not found: {folder}"}), 400
    
    # Find documents
    documents = indexer.find_documents(folder)
    
    # Estimate cost
    cost_estimate = 0
    for doc in documents[:10]:  # Sample first 10
        try:
            content = indexer.extract_text(doc)
            cost_estimate += estimate_cost([content])
        except:
            pass
    cost_estimate = cost_estimate * len(documents) / min(10, len(documents)) if documents else 0
    
    # Create task
    task_id = str(uuid.uuid4())
    sync_tasks[task_id] = {
        "id": task_id,
        "status": "running",
        "folder": folder,
        "total": len(documents),
        "processed": 0,
        "current_file": "",
        "added": 0,
        "updated": 0,
        "unchanged": 0,
        "errors": [],
        "cost_estimate_usd": round(cost_estimate, 4)
    }
    
    # Start background sync
    def run_sync():
        def progress(current, total, filepath):
            sync_tasks[task_id]["processed"] = current
            sync_tasks[task_id]["current_file"] = os.path.basename(filepath)
        
        result = indexer.sync_folder(folder, progress)
        
        sync_tasks[task_id]["status"] = "completed"
        sync_tasks[task_id]["added"] = result["added"]
        sync_tasks[task_id]["updated"] = result["updated"]
        sync_tasks[task_id]["unchanged"] = result["unchanged"]
        sync_tasks[task_id]["errors"] = result["errors"]
    
    thread = threading.Thread(target=run_sync)
    thread.start()
    
    return jsonify({"task_id": task_id, "total": len(documents)})

@app.route("/admin/sync/status/<task_id>")
def sync_status(task_id):
    """Get sync task status."""
    task = sync_tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(task)

@app.route("/admin/concepts", methods=["GET"])
def get_concepts():
    """List stored concepts."""
    concepts = db.get_all_concepts()
    return jsonify(concepts)

@app.route("/admin/concepts", methods=["POST"])
def create_concept():
    """Create new concept vector."""
    data = request.get_json()
    name = data.get("name", "").strip()
    text = data.get("text", "").strip()
    
    if not name or not text:
        return jsonify({"error": "Name and text required"}), 400
    
    result = search.store_concept(name, text)
    return jsonify(result)

@app.route("/admin/concepts/<name>", methods=["DELETE"])
def delete_concept(name):
    """Delete concept vector."""
    db.delete_concept(name)
    return jsonify({"deleted": name})

@app.route("/admin/documents")
def list_documents():
    """List documents with pagination."""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    search_query = request.args.get("search", "")
    
    offset = (page - 1) * per_page
    docs = db.get_all_documents(per_page, offset, search_query or None)
    total = db.get_document_count()
    
    return jsonify({
        "documents": [dict(d) for d in docs],
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page
    })

@app.route("/admin/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    """Delete document and its embedding."""
    db.delete_document(doc_id)
    return jsonify({"deleted": doc_id})

@app.route("/admin/cleanup", methods=["POST"])
def cleanup_deleted():
    """Remove documents that no longer exist on disk."""
    removed = indexer.remove_deleted_documents()
    return jsonify({"removed": removed})

if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
