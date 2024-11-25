import os
import shutil
import json
from datetime import datetime
import sqlite3
from typing import List, Dict, Any, Optional
import glob
import zlib
import numpy as np
import chromadb
import io
from langchain.schema.document import Document

class ChromaDBReconstructor:
    def __init__(self):
        self.BACKUP_DIR = "./enhanced_backups"
        self.RECONSTRUCTION_DIR = "./reconstructed_chroma"
        self.TEMP_DIR = "./temp_reconstruction"
        
        # Create necessary directories
        for dir_path in [self.RECONSTRUCTION_DIR, self.TEMP_DIR]:
            os.makedirs(dir_path, exist_ok=True)

    def load_compressed_embeddings(self, file_path: str) -> np.ndarray:
        """Load and decompress embeddings from .npy.gz file."""
        try:
            with open(file_path, 'rb') as f:
                compressed_data = f.read()
            
            # Decompress data
            decompressed_data = zlib.decompress(compressed_data)
            
            # Load numpy array from decompressed data
            return np.load(io.BytesIO(decompressed_data))
        except Exception as e:
            print(f"Error loading embeddings from {file_path}: {e}")
            raise

    def get_original_collection_id(self) -> Optional[str]:
        """Get the collection ID from the original database."""
        original_db = "./chroma_db"
        if not os.path.exists(original_db):
            return None
            
        for item in os.listdir(original_db):
            if len(item) == 36 and '-' in item:  # UUID format
                return item
        return None

    def decompress_wal_file(self, compressed_path: str, target_path: str):
        """Decompress a WAL file to target location."""
        try:
            with open(compressed_path, 'rb') as src:
                compressed_data = src.read()
                decompressed_data = zlib.decompress(compressed_data)
                with open(target_path, 'wb') as dst:
                    dst.write(decompressed_data)
        except Exception as e:
            print(f"Error decompressing WAL file {compressed_path}: {e}")
            raise

    def reconstruct_chroma(self, target_timestamp: Optional[str] = None) -> str:
        """Reconstruct ChromaDB from optimized backups."""
        # Get available backups
        backups = sorted(os.listdir(self.BACKUP_DIR)) if os.path.exists(self.BACKUP_DIR) else []
        if not backups:
            raise ValueError("No backups found to reconstruct from")
        
        # Filter backups by target timestamp
        if target_timestamp:
            backups = [b for b in backups if b <= target_timestamp]
            if not backups:
                raise ValueError(f"No backups found before timestamp {target_timestamp}")
        
        # Get original collection ID
        original_id = self.get_original_collection_id()
        if not original_id:
            raise ValueError("Could not find original collection ID")
        
        print(f"Using original collection ID: {original_id}")
        
        # Create reconstruction directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        reconstruction_path = os.path.join(
            self.RECONSTRUCTION_DIR,
            f"reconstruction_{timestamp}"
        )
        os.makedirs(reconstruction_path, exist_ok=True)
        
        # Initialize ChromaDB client
        client = None
        total_documents = 0
        processing_stats = {
            'processed_backups': 0,
            'total_chunks': 0,
            'total_embeddings': 0
        }
        
        try:
            # Setup ChromaDB
            client = chromadb.PersistentClient(path=reconstruction_path)
            collection = client.create_collection(
                name="document_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Process each backup chronologically
            for backup_timestamp in backups:
                backup_dir = os.path.join(self.BACKUP_DIR, backup_timestamp)
                if not os.path.isdir(backup_dir):
                    continue
                
                print(f"\nProcessing backup: {backup_timestamp}")
                
                try:
                    # Load metadata
                    with open(os.path.join(backup_dir, 'metadata.json'), 'r') as f:
                        metadata = json.load(f)
                    
                    # Load embeddings
                    embeddings = self.load_compressed_embeddings(
                        os.path.join(backup_dir, 'embeddings.npy.gz')
                    )
                    
                    # Reconstruct documents
                    chunks = metadata['chunks']
                    
                    # Add to collection
                    collection.add(
                        embeddings=embeddings.tolist(),
                        documents=[chunk['page_content'] for chunk in chunks],
                        metadatas=[{
                            **chunk['metadata'],
                            'reconstruction_time': timestamp,
                            'original_backup': backup_timestamp
                        } for chunk in chunks],
                        ids=[f"doc_{backup_timestamp}_{i}" 
                             for i in range(len(chunks))]
                    )
                    
                    # Update statistics
                    total_documents += len(chunks)
                    processing_stats['processed_backups'] += 1
                    processing_stats['total_chunks'] += len(chunks)
                    processing_stats['total_embeddings'] += len(embeddings)
                    
                    print(f"Added {len(chunks)} documents from backup {backup_timestamp}")
                    
                except Exception as e:
                    print(f"Error processing backup {backup_timestamp}: {e}")
                    continue
            
            # Print final statistics
            print("\nReconstruction completed!")
            print(f"Processed {processing_stats['processed_backups']} backups")
            print(f"Total documents: {processing_stats['total_chunks']}")
            print(f"Total embeddings: {processing_stats['total_embeddings']}")
            
            return reconstruction_path
            
        except Exception as e:
            print(f"Error during reconstruction: {e}")
            raise
            
        finally:
            if client:
                del client
            # Clean up temporary directory
            if os.path.exists(self.TEMP_DIR):
                shutil.rmtree(self.TEMP_DIR)

    def verify_reconstruction(self, original_dir: str, reconstructed_dir: str) -> Dict:
        """Verify the reconstruction by comparing databases."""
        try:
            original_client = chromadb.PersistentClient(path=original_dir)
            reconstructed_client = chromadb.PersistentClient(path=reconstructed_dir)
            
            original_collection = original_client.get_collection("document_embeddings")
            reconstructed_collection = reconstructed_client.get_collection("document_embeddings")
            
            # Get all documents from both collections
            original_docs = original_collection.get()
            reconstructed_docs = reconstructed_collection.get()
            
            verification = {
                'original_count': len(original_docs['ids']),
                'reconstructed_count': len(reconstructed_docs['ids']),
                'matches': True,
                'differences': []
            }
            
            # Compare counts
            if verification['original_count'] != verification['reconstructed_count']:
                verification['matches'] = False
                verification['differences'].append(
                    f"Document count mismatch: Original={verification['original_count']}, "
                    f"Reconstructed={verification['reconstructed_count']}"
                )
            
            # Compare embeddings
            for i, (orig_emb, rec_emb) in enumerate(
                zip(original_docs['embeddings'], reconstructed_docs['embeddings'])
            ):
                if not np.allclose(np.array(orig_emb), np.array(rec_emb), atol=1e-5):
                    verification['matches'] = False
                    verification['differences'].append(f"Embedding mismatch at index {i}")
            
            return verification
            
        finally:
            if 'original_client' in locals():
                del original_client
            if 'reconstructed_client' in locals():
                del reconstructed_client

def main():
    parser = argparse.ArgumentParser(description='Reconstruct ChromaDB from backups')
    parser.add_argument(
        '--timestamp', 
        help='Target timestamp for reconstruction (format: YYYYMMDD_HHMMSS)',
        required=False
    )
    parser.add_argument(
        '--verify', 
        action='store_true',
        help='Verify reconstruction against original database'
    )
    
    args = parser.parse_args()
    
    reconstructor = ChromaDBReconstructor()
    
    try:
        print("Starting reconstruction process...")
        reconstructed_path = reconstructor.reconstruct_chroma(args.timestamp)
        print(f"\nReconstruction completed!")
        print(f"Reconstructed database location: {reconstructed_path}")
        
        if args.verify and os.path.exists("./chroma_db"):
            print("\nVerifying reconstruction...")
            verification = reconstructor.verify_reconstruction("./chroma_db", reconstructed_path)
            
            print("\nVerification Results:")
            print(f"Original documents: {verification['original_count']}")
            print(f"Reconstructed documents: {verification['reconstructed_count']}")
            print(f"Exact matches: {verification['matches']}")
            
            if not verification['matches']:
                print("\nDifferences found:")
                for diff in verification['differences']:
                    print(f"- {diff}")
        
    except Exception as e:
        print(f"Error during reconstruction: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    import argparse
    main()