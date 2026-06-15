"""
Milestone 3: Document Ingestion and Chunking

This script implements the document loading and chunking pipeline for the
Unofficial Guide project (CSI Computer Science professor reviews).

Pipeline:
1. Load all .txt files from the documents/ folder
2. Clean whitespace and remove empty lines
3. Create chunks of 400 characters with 50-character overlap
4. Preserve source filename metadata with each chunk
5. Report total chunk count and display 5 representative chunks
"""

import os
from pathlib import Path
from typing import List, Dict


def load_documents(documents_dir: str) -> Dict[str, str]:
    """
    Load all .txt files from the documents directory.
    
    Args:
        documents_dir: Path to the documents folder
        
    Returns:
        Dictionary mapping filename to file contents
    """
    documents = {}
    docs_path = Path(documents_dir)
    
    if not docs_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")
    
    txt_files = sorted(docs_path.glob("*.txt"))
    
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {documents_dir}")
    
    for file_path in txt_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                documents[file_path.name] = f.read()
        except Exception as e:
            print(f"Warning: Could not read {file_path.name}: {e}")
    
    return documents


def clean_text(text: str) -> str:
    """
    Clean whitespace and remove empty lines from text.
    
    - Normalize newlines
    - Strip leading/trailing whitespace
    - Remove multiple consecutive spaces
    - Remove multiple consecutive newlines
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    # Normalize newlines to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Collapse multiple spaces into single space
    lines = []
    for line in text.split("\n"):
        # Strip each line and collapse internal spaces
        cleaned_line = " ".join(line.split())
        if cleaned_line:  # Only keep non-empty lines
            lines.append(cleaned_line)
    
    # Join with single newlines, ensuring no trailing newline
    text = "\n".join(lines)
    
    return text


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: The text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Number of characters to overlap between consecutive chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    
    if len(text) == 0:
        return chunks
    
    # Create chunks with overlap
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start position forward by (chunk_size - overlap)
        # This ensures overlap between consecutive chunks
        start += chunk_size - overlap
    
    return chunks


def create_chunks_with_metadata(
    documents: Dict[str, str],
    chunk_size: int = 400,
    overlap: int = 50
) -> List[Dict[str, str]]:
    """
    Create chunks from all documents with source metadata.
    
    Args:
        documents: Dictionary mapping filename to file contents
        chunk_size: Size of each chunk in characters
        overlap: Number of characters to overlap between consecutive chunks
        
    Returns:
        List of dictionaries, each containing 'chunk' and 'source' keys
    """
    chunks_with_metadata = []
    
    for filename, content in documents.items():
        # Clean the content
        cleaned_content = clean_text(content)
        
        # Create chunks
        chunks = chunk_text(cleaned_content, chunk_size, overlap)
        
        # Attach metadata
        for chunk in chunks:
            chunks_with_metadata.append({
                "chunk": chunk,
                "source": filename
            })
    
    return chunks_with_metadata


def main():
    """Main pipeline: load, clean, chunk, and report."""
    
    # Configuration
    documents_dir = "documents"
    chunk_size = 400
    overlap = 50
    
    print("=" * 70)
    print("MILESTONE 3: Document Ingestion and Chunking")
    print("=" * 70)
    print()
    
    # Step 1: Load documents
    print(f"[1] Loading documents from '{documents_dir}/'...")
    documents = load_documents(documents_dir)
    print(f"    ✓ Loaded {len(documents)} document(s)")
    for filename in sorted(documents.keys()):
        print(f"      - {filename}")
    print()
    
    # Step 2: Create chunks with metadata
    print(f"[2] Creating chunks (size={chunk_size}, overlap={overlap})...")
    chunks_with_metadata = create_chunks_with_metadata(
        documents,
        chunk_size=chunk_size,
        overlap=overlap
    )
    print(f"    ✓ Generated {len(chunks_with_metadata)} total chunks")
    print()
    
    # Step 3: Report total chunks
    print("=" * 70)
    print(f"TOTAL CHUNKS GENERATED: {len(chunks_with_metadata)}")
    print("=" * 70)
    print()
    
    # Step 4: Display 5 representative chunks
    print("=" * 70)
    print("5 REPRESENTATIVE CHUNKS FOR INSPECTION")
    print("=" * 70)
    print()
    
    # Select evenly-spaced chunks for inspection
    if len(chunks_with_metadata) == 0:
        print("No chunks available for inspection.")
    else:
        # Calculate indices for 5 representative chunks
        indices = []
        if len(chunks_with_metadata) <= 5:
            indices = list(range(len(chunks_with_metadata)))
        else:
            step = len(chunks_with_metadata) // 5
            indices = [i * step for i in range(5)]
        
        for i, idx in enumerate(indices, 1):
            chunk_data = chunks_with_metadata[idx]
            print(f"[Chunk {i}] Source: {chunk_data['source']}")
            print(f"Length: {len(chunk_data['chunk'])} characters")
            print("-" * 70)
            print(chunk_data['chunk'])
            print()


if __name__ == "__main__":
    main()
