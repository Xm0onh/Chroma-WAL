import os
import logging
from typing import List
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from config import CHROMA_PATH, DATA_PATH
from snapshot import get_embedding_function

logging.basicConfig(level=logging.INFO)

def process_files():
    """
    Process all files in the `data` directory and add their embeddings to ChromaDB.
    """
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents() -> List[Document]:
    """
    Load PDF documents from the `data` directory.
    """
    documents = []
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        try:
            logging.info(f"📄 Loading: {pdf_file}")
            loader = PyPDFDirectoryLoader(DATA_PATH, glob=pdf_file)
            documents.extend(loader.load())
        except Exception as e:
            logging.error(f"❌ Error loading {pdf_file}: {str(e)}")
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=2)
    return splitter.split_documents(documents)

def add_to_chroma(chunks: List[Document]):
    """
    Add document chunks to ChromaDB using Hugging Face embeddings.
    """
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()  
    )

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    logging.info(f"🗂️ Existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks if chunk.metadata.get("source") not in existing_ids]

    if not new_chunks:
        logging.info("✅ No new documents to add.")
        return

    embedding_function = get_embedding_function()
    texts = [chunk.page_content for chunk in new_chunks]
    embeddings = embedding_function.embed_documents(texts)

    logging.info(f"➕ Adding {len(new_chunks)} new documents to ChromaDB.")
    db.add_texts(texts=texts, metadatas=[chunk.metadata for chunk in new_chunks], embeddings=embeddings)
