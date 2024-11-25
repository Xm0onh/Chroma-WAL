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
import zlib
import base64
import numpy as np

class ChromaDBReconstructor:
    def __init__(self):
        self.BACKUP_DIR = "./enhanced_backups"
        self.RECONSTRUCTION_DIR = "./reconstructed_chroma"
    
    def decompress_numpy_array(self, compressed_str: str, shape: tuple, dtype: str) -> np.ndarray:
        """Decompress numpy array from string."""
        compressed_bytes = base64.b64decode(compressed_str)
        decompressed_bytes = zlib.decompress(compressed_bytes)
        arr = np.frombuffer(decompressed_bytes, dtype=dtype)
        return arr.reshape(shape)

    def get_original_collection_id(self):
        """Get the collection ID from the original database."""
        original_db = "./chroma_db"
        if not os.path.exists(original_db):
            return None
            
        for item in os.listdir(original_db):
            if len(item) == 36 and '-' in item:
                return item
        return None

    def reconstruct_chroma(self, target_timestamp: str = None) -> str:
        """Reconstruct ChromaDB from optimized backups."""
        backups = sorted(os.listdir(self.BACKUP_DIR)) if os.path.exists(self.BACKUP_DIR) else []
        if not backups:
            raise ValueError("No backups found to reconstruct from")
        
        original_id = self.get_original_collection_id()
        if not original_id:
            raise ValueError("Could not find original collection ID")
        
        print(f"Using original collection ID: {original_id}")
        
        reconstruction_path = os.path.join(
            self.RECONSTRUCTION_DIR,
            f"reconstruction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(reconstruction_path, exist_ok=True)
        
        # Copy database structure
        print("Copying original database structure...")
        shutil.copy2(
            os.path.join("./chroma_db", "chroma.sqlite3"),
            os.path.join(reconstruction_path, "chroma.sqlite3")
        )
        
        # Copy index directory
        original_index_path = os.path.join("./chroma_db", original_id)
        reconstructed_index_path = os.path.join(reconstruction_path, original_id)
        if os.path.exists(original_index_path):
            shutil.copytree(original_index_path, reconstructed_index_path)
        
        client = None
        try:
            client = chromadb.PersistentClient(path=reconstruction_path)
            collection = client.get_collection("document_embeddings")
            
            total_documents = 0
            for backup_timestamp in backups:
                backup_dir = os.path.join(self.BACKUP_DIR, backup_timestamp)
                if not os.path.isdir(backup_dir):
                    continue
                
                # Load backup data
                backup_data = self.load_backup_data(backup_dir)
                
                for data in backup_data:
                    # Handle both old and new backup formats
                    if 'version' in data and data['version'] == '2.0':
                        # New format with compression
                        embedding = self.decompress_numpy_array(
                            data['embedding_compressed'],
                            tuple(data['embedding_shape']),
                            data['embedding_dtype']
                        )
                    else:
                        # Old format
                        embedding = np.array(data['embedding'])
                    
                    metadatas = [{
                        "original_timestamp": data['timestamp'],
                        "model": data.get('model', 'unknown'),
                        "reconstruction_time": datetime.now().strftime('%Y%m%d_%H%M%S')
                    }]
                    
                    collection.add(
                        embeddings=[embedding.tolist()],
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

    def load_backup_data(self, backup_dir: str) -> List[Dict[str, Any]]:
        """Load backup data handling both old and new formats."""
        backup_data = []
        
        # Try new format first
        backup_file = os.path.join(backup_dir, 'backup_data.json')
        if os.path.exists(backup_file):
            with open(backup_file, 'r') as f:
                data = json.load(f)
                backup_data.append(data)
        else:
            # Fall back to old format
            embedding_files = glob.glob(os.path.join(backup_dir, '**/embedding_data.json'), recursive=True)
            for file_path in embedding_files:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    backup_data.append(data)
        
        return backup_data

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
        
        print(f"\nOriginal size: {size1:,} bytes")
        print(f"Reconstructed size: {size2:,} bytes")
        
        print("\nDirectory contents:")
        print("\nOriginal directory:")
        os.system(f"ls -la {dir1}")
        print("\nReconstructed directory:")
        os.system(f"ls -la {dir2}")

def main():
    reconstructor = ChromaDBReconstructor()
    
    try:
        reconstructed_path = reconstructor.reconstruct_chroma()
        print(f"\nReconstruction completed!")
        print(f"Reconstructed database location: {reconstructed_path}")
        
        reconstructor.compare_directories("./chroma_db", reconstructed_path)
        
    except Exception as e:
        print(f"Error during reconstruction: {e}")

if __name__ == "__main__":
    main()