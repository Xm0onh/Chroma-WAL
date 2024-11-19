import os
import json
import time
import glob
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from config import CHROMA_PATH, SNAPSHOT_PATH
import gzip 

def create_snapshot():
    """
    Create a delta snapshot of the current ChromaDB state compared to the previous snapshot.
    """
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )
    current_items = db.get(include=["metadatas", "embeddings"])

    if "embeddings" in current_items:
        current_items["embeddings"] = [
            emb.tolist() if hasattr(emb, 'tolist') else emb
            for emb in current_items["embeddings"]
        ]

    previous_snapshot = get_latest_snapshot()
    if previous_snapshot:
        with gzip.open(previous_snapshot, "rt", encoding="utf-8") as f:
            previous_items = json.load(f)
    else:
        previous_items = {"ids": [], "embeddings": [], "metadatas": []}

    delta = compute_delta(previous_items, current_items)

    snapshot_file = os.path.join(SNAPSHOT_PATH, f"snapshot_{int(time.time())}.json.gz")
    with gzip.open(snapshot_file, "wt", encoding="utf-8") as f:
        json.dump(delta, f, separators=(',', ':'))
    print(f"✅ Delta snapshot saved: {snapshot_file}")

def get_latest_snapshot():
    """
    Retrieve the most recent snapshot file.
    """
    snapshot_files = glob.glob(os.path.join(SNAPSHOT_PATH, "snapshot_*.json.gz"))
    if not snapshot_files:
        return None
    latest_snapshot = max(snapshot_files, key=os.path.getctime)
    return latest_snapshot

def compute_delta(prev, curr):
    """
    Compute the delta between previous and current items.
    """
    prev_ids_set = set(prev.get("ids", []))
    curr_ids_set = set(curr.get("ids", []))

    new_ids = list(curr_ids_set - prev_ids_set)
    new_indices = [curr["ids"].index(id_) for id_ in new_ids]

    delta = {
        "ids": [curr["ids"][i] for i in new_indices],
        "embeddings": [curr["embeddings"][i] for i in new_indices],
        "metadatas": [curr["metadatas"][i] for i in new_indices],
    }

    return delta

class HuggingFaceEmbeddings:
    """
    Wrapper for Hugging Face SentenceTransformer to integrate with Chroma.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts/documents.
        """
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Generate an embedding for a single query.
        """
        return self.model.encode(text, convert_to_tensor=False).tolist()

def get_embedding_function():
    """
    Initialize Hugging Face embeddings.
    """
    return HuggingFaceEmbeddings()
