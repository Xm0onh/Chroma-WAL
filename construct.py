from transformers import AutoTokenizer, AutoModel
import torch
import chromadb
import os
import shutil
import sqlite3
import json
from datetime import datetime
import numpy as np

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
        
        # Wait for SQLite database creation
        sqlite_path = os.path.join(self.DB_DIR, 'chroma.sqlite3')
        while not os.path.exists(sqlite_path):
            pass
        
        self.configure_sqlite_wal(sqlite_path)
        
        return client.get_or_create_collection(
            name="document_embeddings",
            metadata={"hnsw:space": "cosine"}
        )

    def create_enhanced_backup(self, text, embedding, timestamp):
        """Create a backup including both WAL and embedding data."""
        backup_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create timestamp-specific backup directory
        backup_dir = os.path.join(self.BACKUP_DIR, backup_timestamp)
        os.makedirs(backup_dir, exist_ok=True)
        
        # 1. Backup WAL files
        wal_files = []
        for root, _, files in os.walk(self.DB_DIR):
            for file in files:
                if file.endswith(('-wal', '.wal', '.shm')):
                    source_path = os.path.join(root, file)
                    dest_path = os.path.join(backup_dir, file)
                    shutil.copy2(source_path, dest_path)
                    wal_files.append(file)

        # 2. Save embedding and text data
        embedding_data = {
            'text': text,
            'embedding': embedding.tolist(),
            'timestamp': backup_timestamp,
            'model': self.model_name
        }
        
        embedding_file = os.path.join(backup_dir, 'embedding_data.json')
        with open(embedding_file, 'w') as f:
            json.dump(embedding_data, f, indent=2)
        
        return backup_dir, wal_files

    def load_backup(self, timestamp):
        """Load a specific backup's data."""
        backup_dir = os.path.join(self.BACKUP_DIR, timestamp)
        if not os.path.exists(backup_dir):
            raise ValueError(f"No backup found for timestamp: {timestamp}")
            
        # Load embedding data
        embedding_file = os.path.join(backup_dir, 'embedding_data.json')
        with open(embedding_file, 'r') as f:
            embedding_data = json.load(f)
            
        return embedding_data

def main():
    # Initialize the backup system
    backup_system = EnhancedBackupSystem()
    
    # Setup ChromaDB
    collection = backup_system.setup_chroma()
    
    # Get input and process
    # Todo -> do it for files
    print("Enter your text:")
    text = input().strip()
    
    # Generate embedding
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    embedding = backup_system.get_embedding(text)
    
    # Store in ChromaDB
    collection.add(
        embeddings=[embedding.tolist()],
        documents=[text],
        ids=[f"doc_{timestamp}"]
    )
    
    # Create enhanced backup
    backup_dir, wal_files = backup_system.create_enhanced_backup(text, embedding, timestamp)
    
    print("\nBackup created successfully!")
    print(f"Backup location: {backup_dir}")
    print("\nContents:")
    print("1. WAL files:")
    for wal in wal_files:
        print(f"  - {wal}")
    print("2. Embedding data: embedding_data.json")
    
    # Demonstrate loading backup
    print("\nLoading backup to verify...")
    loaded_data = backup_system.load_backup(timestamp)
    print(f"Original text: {loaded_data['text']}")
    print(f"Embedding shape: {len(loaded_data['embedding'])} dimensions")

if __name__ == "__main__":
    main()