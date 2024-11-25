from transformers import AutoTokenizer, AutoModel
import torch
import chromadb
import os
import shutil
import sqlite3
import json
from datetime import datetime
import numpy as np
import zlib
import base64

class EnhancedBackupSystem:
    def __init__(self):
        self.DB_DIR = "./chroma_db"
        self.BACKUP_DIR = "./enhanced_backups"
        self.model_name = "sentence-transformers/all-mpnet-base-v2"
        
        # Initialize directories
        os.makedirs(self.DB_DIR, exist_ok=True)
        os.makedirs(self.BACKUP_DIR, exist_ok=True)
    
    def get_embedding(self, text):
        """Generate embeddings using Hugging Face model."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        
        return embeddings[0]

    def configure_sqlite_wal(self, db_path):
        """Configure SQLite to use WAL mode."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('PRAGMA journal_mode=WAL;')
        cursor.execute('PRAGMA wal_autocheckpoint=10000;')
        conn.commit()
        conn.close()

    def setup_chroma(self):
        """Setup ChromaDB with persistent storage."""
        settings = chromadb.Settings(
            persist_directory=self.DB_DIR,
            allow_reset=True,
            anonymized_telemetry=False
        )
        
        client = chromadb.PersistentClient(
            path=self.DB_DIR,
            settings=settings
        )
        
        sqlite_path = os.path.join(self.DB_DIR, 'chroma.sqlite3')
        while not os.path.exists(sqlite_path):
            pass
        
        self.configure_sqlite_wal(sqlite_path)
        
        return client.get_or_create_collection(
            name="document_embeddings",
            metadata={"hnsw:space": "cosine"}
        )

    def compress_numpy_array(self, arr):
        """Compress numpy array to bytes."""
        arr_bytes = arr.tobytes()
        compressed = zlib.compress(arr_bytes)
        return base64.b64encode(compressed).decode('utf-8')

    def create_enhanced_backup(self, text, embedding, timestamp):
        """Create an optimized backup with compressed embeddings."""
        backup_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(self.BACKUP_DIR, backup_timestamp)
        os.makedirs(backup_dir, exist_ok=True)
        
        # 1. Save compressed embedding
        compressed_embedding = self.compress_numpy_array(embedding)
        
        # 2. Save WAL files with compression
        wal_files = []
        for root, _, files in os.walk(self.DB_DIR):
            for file in files:
                if file.endswith(('-wal', '.wal', '.shm')):
                    source_path = os.path.join(root, file)
                    dest_path = os.path.join(backup_dir, file)
                    with open(source_path, 'rb') as src, open(dest_path + '.gz', 'wb') as dst:
                        dst.write(zlib.compress(src.read()))
                    wal_files.append(file + '.gz')

        # 3. Save metadata and compressed embedding
        backup_data = {
            'text': text,
            'embedding_compressed': compressed_embedding,
            'embedding_shape': embedding.shape,
            'embedding_dtype': str(embedding.dtype),
            'timestamp': backup_timestamp,
            'model': self.model_name,
            'wal_files': wal_files,
            'version': '2.0'  # Version tracking for compatibility
        }
        
        metadata_path = os.path.join(backup_dir, 'backup_data.json')
        with open(metadata_path, 'w') as f:
            json.dump(backup_data, f)
        
        return backup_dir, wal_files

    def load_backup(self, timestamp):
        """Load a specific backup's data."""
        backup_dir = os.path.join(self.BACKUP_DIR, timestamp)
        if not os.path.exists(backup_dir):
            raise ValueError(f"No backup found for timestamp: {timestamp}")
        
        metadata_path = os.path.join(backup_dir, 'backup_data.json')
        with open(metadata_path, 'r') as f:
            backup_data = json.load(f)
        
        return backup_data

def main():
    backup_system = EnhancedBackupSystem()
    collection = backup_system.setup_chroma()
    
    print("Enter your text:")
    text = input().strip()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    embedding = backup_system.get_embedding(text)
    
    collection.add(
        embeddings=[embedding.tolist()],
        documents=[text],
        ids=[f"doc_{timestamp}"]
    )
    
    backup_dir, wal_files = backup_system.create_enhanced_backup(text, embedding, timestamp)
    
    print("\nBackup created successfully!")
    print(f"Backup location: {backup_dir}")
    print("\nContents:")
    print("1. Compressed WAL files:")
    for wal in wal_files:
        print(f"  - {wal}")
    print("2. Backup data: backup_data.json")
    
    loaded_data = backup_system.load_backup(timestamp)
    print(f"\nVerification:")
    print(f"Original text: {loaded_data['text']}")
    print(f"Embedding shape: {loaded_data['embedding_shape']}")
    
    # Print size comparison
    original_size = len(str(embedding.tolist()))
    compressed_size = len(loaded_data['embedding_compressed'])
    print(f"\nStorage efficiency:")
    print(f"Original size: {original_size:,} bytes")
    print(f"Compressed size: {compressed_size:,} bytes")
    print(f"Compression ratio: {original_size/compressed_size:.2f}x")

if __name__ == "__main__":
    main()