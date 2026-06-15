"""
Milestone 4: Embedding and Retrieval

This script implements the embedding and retrieval pipeline for the
Unofficial Guide project. It takes chunks from Milestone 3, embeds them
using sentence-transformers (all-MiniLM-L6-v2), stores them in ChromaDB,
and provides a retrieval interface for semantic search.

Pipeline:
1. Load and chunk documents (using Milestone 3 functions)
2. Create embeddings using all-MiniLM-L6-v2
3. Store embeddings and metadata in ChromaDB
4. Retrieve top-5 most relevant chunks per query
5. Display results with distance scores and source attribution
"""

import sys
from typing import List, Dict, Tuple
import chromadb
from sentence_transformers import SentenceTransformer

# Import Milestone 3 functions
from milestone3 import load_documents, clean_text, chunk_text, create_chunks_with_metadata


def initialize_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Initialize the sentence-transformers embedding model.
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        Initialized SentenceTransformer model
    """
    print(f"[*] Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"    ✓ Model loaded successfully")
    return model


def initialize_chromadb_collection(
    collection_name: str = "professor_reviews"
) -> Tuple[chromadb.Client, chromadb.Collection]:
    """
    Initialize ChromaDB client and create/get a collection.
    
    Args:
        collection_name: Name of the ChromaDB collection
        
    Returns:
        Tuple of (ChromaDB client, ChromaDB collection)
    """
    print(f"[*] Initializing ChromaDB collection: '{collection_name}'...")
    
    # Create an ephemeral (in-memory) ChromaDB client
    client = chromadb.Client()
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"    ✓ ChromaDB collection initialized")
    return client, collection


def store_chunks_in_chromadb(
    collection: chromadb.Collection,
    chunks_with_metadata: List[Dict[str, str]],
    embedding_model: SentenceTransformer
) -> int:
    """
    Embed chunks and store them in ChromaDB with metadata.
    
    Args:
        collection: ChromaDB collection to store in
        chunks_with_metadata: List of chunks with source metadata
        embedding_model: SentenceTransformer model for creating embeddings
        
    Returns:
        Number of chunks stored
    """
    print(f"[*] Creating embeddings and storing {len(chunks_with_metadata)} chunks...")
    
    # Extract chunk texts
    texts = [chunk["chunk"] for chunk in chunks_with_metadata]
    
    # Create embeddings
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    
    # Prepare metadata for ChromaDB
    # ChromaDB requires: ids, documents, embeddings, metadatas
    ids = [f"chunk_{i}" for i in range(len(chunks_with_metadata))]
    metadatas = [{"source": chunk["source"]} for chunk in chunks_with_metadata]
    
    # Add to collection
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )
    
    print(f"    ✓ Stored {len(chunks_with_metadata)} chunks in ChromaDB")
    return len(chunks_with_metadata)


def retrieve_chunks(
    collection: chromadb.Collection,
    query: str,
    embedding_model: SentenceTransformer,
    top_k: int = 5
) -> List[Dict]:
    """
    Retrieve top-k most relevant chunks for a query.
    
    Args:
        collection: ChromaDB collection to query
        query: User query text
        embedding_model: SentenceTransformer model for embedding the query
        top_k: Number of results to retrieve
        
    Returns:
        List of retrieved chunks with metadata and distance scores
    """
    # Embed the query
    query_embedding = embedding_model.encode(query)
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    # Parse and structure results
    retrieved = []
    if results and results["documents"] and len(results["documents"]) > 0:
        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            retrieved.append({
                "rank": i + 1,
                "chunk": doc,
                "source": metadata["source"],
                "distance": distance,  # Lower is better (cosine distance)
            })
    
    return retrieved


def print_retrieval_results(query: str, results: List[Dict]) -> None:
    """
    Pretty-print retrieval results with formatting.
    
    Args:
        query: The query that was executed
        results: List of retrieved chunks with metadata
    """
    print()
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)
    print()
    
    if not results:
        print("No results found.")
        return
    
    for result in results:
        print(f"[Result {result['rank']}] Source: {result['source']}")
        print(f"Distance Score: {result['distance']:.4f}")
        print("-" * 80)
        print(result['chunk'][:500])  # Print first 500 chars
        if len(result['chunk']) > 500:
            print("...")
        print()


def main():
    """Main pipeline: load, embed, store, and retrieve."""
    
    print("=" * 80)
    print("MILESTONE 4: Embedding and Retrieval")
    print("=" * 80)
    print()
    
    # Configuration
    documents_dir = "documents"
    chunk_size = 400
    overlap = 50
    embedding_model_name = "all-MiniLM-L6-v2"
    collection_name = "professor_reviews"
    top_k = 5
    
    # Step 1: Load and chunk documents (Milestone 3)
    print("[1] Loading and chunking documents...")
    documents = load_documents(documents_dir)
    chunks_with_metadata = create_chunks_with_metadata(
        documents,
        chunk_size=chunk_size,
        overlap=overlap
    )
    print(f"    ✓ Generated {len(chunks_with_metadata)} chunks")
    print()
    
    # Step 2: Initialize embedding model
    print("[2] Initializing embedding model...")
    embedding_model = initialize_embedding_model(embedding_model_name)
    print()
    
    # Step 3: Initialize ChromaDB
    print("[3] Setting up ChromaDB vector store...")
    client, collection = initialize_chromadb_collection(collection_name)
    print()
    
    # Step 4: Store chunks in ChromaDB
    print("[4] Creating embeddings and storing in ChromaDB...")
    num_stored = store_chunks_in_chromadb(collection, chunks_with_metadata, embedding_model)
    print()
    
    # Step 5: Run example retrieval tests
    print("[5] Running example retrieval tests...")
    print()
    
    test_queries = [
        "Which professor gives detailed feedback?",
        "What is the workload like for CSCI courses?",
        "Which professors are good at explaining concepts?",
        "What do students say about exam difficulty?",
        "Which professor is known for being accessible during office hours?",
    ]
    
    for query in test_queries:
        results = retrieve_chunks(collection, query, embedding_model, top_k=top_k)
        print_retrieval_results(query, results)
    
    print("=" * 80)
    print("MILESTONE 4 COMPLETE")
    print("=" * 80)
    print(f"Total chunks stored: {num_stored}")
    print(f"Retrieval model: {embedding_model_name}")
    print(f"Vector store: ChromaDB ({collection_name})")
    print(f"Top-k results: {top_k}")


if __name__ == "__main__":
    main()
