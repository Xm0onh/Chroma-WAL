from transformers import AutoTokenizer, AutoModel
import torch
import chromadb
import os
import shutil
import json
from datetime import datetime
import sqlite3
from typing import List, Dict, Any
import glob

class ChromaDBReconstructor:
    def __init__(self):
        self.BACKUP_DIR = "./enhanced_backups"
        self.RECONSTRUCTION_DIR = "./reconstructed_chroma"
        
    def get_original_collection_id(self):
        """Get the collection ID from the original database."""
        original_db = "./chroma_db"
        if not os.path.exists(original_db):
            return None
            
        # Find the UUID directory
        for item in os.listdir(original_db):
            if len(item) == 36 and '-' in item:  # UUID format check
                return item
        return None

    def reconstruct_chroma(self, target_timestamp: str = None) -> str:
        """
        Reconstruct ChromaDB from backups up to a specific timestamp.
        """
        backups = sorted(os.listdir(self.BACKUP_DIR)) if os.path.exists(self.BACKUP_DIR) else []
        if not backups:
            raise ValueError("No backups found to reconstruct from")
            
        # Get original collection ID
        original_id = self.get_original_collection_id()
        if not original_id:
            raise ValueError("Could not find original collection ID")
            
        print(f"Using original collection ID: {original_id}")
        
        # Create reconstruction directory
        reconstruction_path = os.path.join(
            self.RECONSTRUCTION_DIR,
            f"reconstruction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(reconstruction_path, exist_ok=True)
        
        # First, copy the entire original database structure
        print("Copying original database structure...")
        shutil.copy2(
            os.path.join("./chroma_db", "chroma.sqlite3"),
            os.path.join(reconstruction_path, "chroma.sqlite3")
        )
        
        # Copy the index directory
        original_index_path = os.path.join("./chroma_db", original_id)
        reconstructed_index_path = os.path.join(reconstruction_path, original_id)
        if os.path.exists(original_index_path):
            shutil.copytree(original_index_path, reconstructed_index_path)
        
        client = None
        try:
            # Initialize client with the copied structure
            client = chromadb.PersistentClient(path=reconstruction_path)
            
            # Get the existing collection
            collection = client.get_collection("document_embeddings")
            
            # Process backups
            total_documents = 0
            for backup_timestamp in backups:
                backup_dir = os.path.join(self.BACKUP_DIR, backup_timestamp)
                if not os.path.isdir(backup_dir):
                    continue
                    
                embedding_data = self.load_embedding_data(backup_dir)
                for data in embedding_data:
                    metadatas = [{
                        "original_timestamp": data['timestamp'],
                        "model": data.get('model', 'unknown'),
                        "reconstruction_time": datetime.now().strftime('%Y%m%d_%H%M%S')
                    }]
                    
                    collection.add(
                        embeddings=[data['embedding']],
                        documents=[data['text']],
                        ids=[f"doc_{backup_timestamp}_{total_documents}"],
                        metadatas=metadatas
                    )
                    total_documents += 1
                    print(f"Reconstructed document {total_documents} from backup {backup_timestamp}")
            
            return reconstruction_path
            
        finally:
            if client:
                del client

    def load_embedding_data(self, backup_dir: str) -> List[Dict[str, Any]]:
        """Load embedding data from a backup directory."""
        embedding_files = glob.glob(os.path.join(backup_dir, '**/embedding_data.json'), recursive=True)
        embedding_data = []
        
        for file_path in embedding_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                embedding_data.append(data)
        
        return embedding_data

    def compare_directories(self, dir1, dir2):
        """Compare two ChromaDB directories."""
        print("\nComparing directories:")
        print(f"Original: {dir1}")
        print(f"Reconstructed: {dir2}")
        
        def get_dir_info(path):
            total_size = 0
            files = []
            for root, _, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(root, f)
                    total_size += os.path.getsize(fp)
                    files.append((f, os.path.getsize(fp)))
            return total_size, files
        
        size1, files1 = get_dir_info(dir1)
        size2, files2 = get_dir_info(dir2)
        
        print(f"\nOriginal size: {size1} bytes")
        print(f"Reconstructed size: {size2} bytes")
        
        print("\nDirectory contents:")
        print("\nOriginal directory:")
        os.system(f"ls -la {dir1}")
        print("\nReconstructed directory:")
        os.system(f"ls -la {dir2}")

def main():
    reconstructor = ChromaDBReconstructor()
    
    try:
        # Perform reconstruction
        reconstructed_path = reconstructor.reconstruct_chroma()
        print(f"\nReconstruction completed!")
        print(f"Reconstructed database location: {reconstructed_path}")
        
        # Compare original and reconstructed directories
        reconstructor.compare_directories("./chroma_db", reconstructed_path)
        
    except Exception as e:
        print(f"Error during reconstruction: {e}")

if __name__ == "__main__":
    main()