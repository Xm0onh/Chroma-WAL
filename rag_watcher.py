import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import shutil
from datetime import datetime
import zlib
import json
import numpy as np
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import chromadb
from transformers import AutoTokenizer, AutoModel
import torch
import io
from typing import List, Tuple, Dict, Any

class DocumentProcessor:
    def __init__(self):
        self.DATA_DIR = "./data_folder"
        self.DB_DIR = "./chroma_db"
        self.BACKUP_DIR = "./enhanced_backups"
        self.PROCESSED_DIR = "./processed_files"
        self.model_name = "sentence-transformers/all-mpnet-base-v2"
        
        # Initialize directories
        for dir_path in [self.DATA_DIR, self.DB_DIR, self.BACKUP_DIR, self.PROCESSED_DIR]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize model and tokenizer
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Initialize ChromaDB
        self.setup_chroma()

    def setup_chroma(self):
        """Initialize ChromaDB with WAL mode."""
        try:
            settings = chromadb.Settings(
                persist_directory=self.DB_DIR,
                allow_reset=True,
                anonymized_telemetry=False
            )
            
            self.client = chromadb.PersistentClient(
                path=self.DB_DIR,
                settings=settings
            )
            
            # Configure SQLite WAL
            sqlite_path = os.path.join(self.DB_DIR, 'chroma.sqlite3')
            if os.path.exists(sqlite_path):
                import sqlite3
                conn = sqlite3.connect(sqlite_path)
                cursor = conn.cursor()
                cursor.execute('PRAGMA journal_mode=WAL;')
                cursor.execute('PRAGMA wal_autocheckpoint=10000;')
                cursor.execute('PRAGMA synchronous=NORMAL;')
                conn.commit()
                conn.close()
            
            self.collection = self.client.get_or_create_collection(
                name="document_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"Error setting up ChromaDB: {e}")
            raise

    def process_documents(self) -> List[Document]:
        """Process all PDFs in the data directory."""
        try:
            loader = PyPDFDirectoryLoader(self.DATA_DIR)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=80,
                length_function=len,
                is_separator_regex=False,
            )
            
            chunks = text_splitter.split_documents(documents)
            
            # Add additional metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'processing_time': datetime.now().isoformat()
                })
            
            return chunks
        except Exception as e:
            print(f"Error processing documents: {e}")
            return []

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        try:
            inputs = self.tokenizer(
                text, 
                padding=True, 
                truncation=True,
                max_length=512, 
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            
            return embeddings[0]
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise

    def create_backup(self, chunks: List[Document], embeddings: List[np.ndarray], 
                     timestamp: str) -> Tuple[str, List[str]]:
        """Create backup of embeddings and WAL files."""
        backup_dir = os.path.join(self.BACKUP_DIR, timestamp)
        os.makedirs(backup_dir, exist_ok=True)
        
        try:
            # 1. Save compressed embeddings
            embeddings_array = np.array(embeddings)
            embeddings_path = os.path.join(backup_dir, 'embeddings.npy.gz')
            
            # Use BytesIO to handle the compression properly
            buffer = io.BytesIO()
            np.save(buffer, embeddings_array)
            buffer.seek(0)
            compressed = zlib.compress(buffer.read(), level=9)
            
            with open(embeddings_path, 'wb') as f:
                f.write(compressed)

            # 2. Save metadata
            metadata = {
                'chunks': [
                    {
                        'page_content': chunk.page_content,
                        'metadata': chunk.metadata
                    } for chunk in chunks
                ],
                'embedding_shape': embeddings_array.shape,
                'embedding_dtype': str(embeddings_array.dtype),
                'timestamp': timestamp,
                'model': self.model_name,
                'backup_version': '3.0'
            }
            
            metadata_path = os.path.join(backup_dir, 'metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            # 3. Backup WAL files
            wal_files = []
            for root, _, files in os.walk(self.DB_DIR):
                for file in files:
                    if file.endswith(('-wal', '.wal', '.shm')):
                        source_path = os.path.join(root, file)
                        try:
                            with open(source_path, 'rb') as src:
                                content = src.read()
                                if content:
                                    compressed_path = os.path.join(backup_dir, f"{file}.gz")
                                    with open(compressed_path, 'wb') as dst:
                                        dst.write(zlib.compress(content, level=9))
                                    wal_files.append(file)
                        except IOError as e:
                            print(f"Warning: Could not backup WAL file {file}: {e}")

            return backup_dir, wal_files
            
        except Exception as e:
            print(f"Error creating backup: {e}")
            raise

    def move_to_processed(self, file_path: str) -> str:
        """Move processed file to processed directory."""
        filename = os.path.basename(file_path)
        dest_path = os.path.join(self.PROCESSED_DIR, filename)
        shutil.move(file_path, dest_path)
        return dest_path

class PDFHandler(FileSystemEventHandler):
    def __init__(self, processor: DocumentProcessor):
        self.processor = processor
        self.processed_files = set()

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.pdf'):
            self.process_pdf(event.src_path)

    def process_pdf(self, pdf_path: str):
        """Process a new PDF file."""
        if pdf_path in self.processed_files:
            print(f"File already processed: {os.path.basename(pdf_path)}")
            return
            
        print(f"\nProcessing new PDF: {os.path.basename(pdf_path)}")
        
        try:
            # 1. Process and chunk documents
            chunks = self.processor.process_documents()
            if not chunks:
                print("No chunks created from document")
                return
                
            print(f"Created {len(chunks)} chunks")
            
            # 2. Generate embeddings
            embeddings = []
            for chunk in chunks:
                embedding = self.processor.get_embedding(chunk.page_content)
                embeddings.append(embedding)
            
            print(f"Generated embeddings for all chunks")
            
            # 3. Create timestamp for this update
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 4. Update ChromaDB
            self.processor.collection.add(
                embeddings=[emb.tolist() for emb in embeddings],
                documents=[chunk.page_content for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
                ids=[f"doc_{timestamp}_{i}" for i in range(len(chunks))]
            )
            
            print(f"Updated ChromaDB with new chunks")
            
            # 5. Create backup
            backup_dir, wal_files = self.processor.create_backup(
                chunks=chunks,
                embeddings=embeddings,
                timestamp=timestamp
            )
            
            print(f"Created backup at: {backup_dir}")
            if wal_files:
                print(f"Backed up WAL files: {', '.join(wal_files)}")
            else:
                print("No WAL files to backup")
            
            # 6. Move processed file
            processed_path = self.processor.move_to_processed(pdf_path)
            print(f"Moved file to: {processed_path}")
            
            self.processed_files.add(pdf_path)
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            import traceback
            print(traceback.format_exc())

def main():
    try:
        processor = DocumentProcessor()
        event_handler = PDFHandler(processor)
        observer = Observer()
        observer.schedule(event_handler, processor.DATA_DIR, recursive=False)
        observer.start()
        
        print(f"Watching directory: {processor.DATA_DIR}")
        print("Waiting for PDF files...")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            print("\nStopped watching directory")
        
        observer.join()
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()