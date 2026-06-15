"""
Milestone 5: Grounded Generation and Gradio Interface

This script implements grounded answer generation using Groq's llama-3.3-70b
model. It ensures all answers are based only on retrieved context from student
reviews about CSI Computer Science professors.

Pipeline:
1. Accept user question
2. Retrieve top-5 relevant chunks from ChromaDB (via Milestone 4)
3. Pass retrieved chunks to Groq with grounding prompt
4. Generate answer constrained to retrieved context
5. Include source attribution in response
6. Provide Gradio interface for user interaction
"""

import os
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import gradio as gr

# Import Milestone 3 and 4 functions
from milestone3 import load_documents, clean_text, chunk_text, create_chunks_with_metadata
from milestone4 import retrieve_chunks, initialize_embedding_model, initialize_chromadb_collection


# Global state: embedding model and ChromaDB collection
# These are initialized once when the module loads
_embedding_model = None
_chromadb_collection = None
_chunks_with_metadata = None


def initialize_pipeline():
    """Initialize the embedding, retrieval, and Groq pipeline."""
    global _embedding_model, _chromadb_collection, _chunks_with_metadata
    
    print("[*] Initializing Milestone 5 pipeline...")
    
    # Load and chunk documents (Milestone 3)
    print("[1] Loading and chunking documents...")
    documents = load_documents("documents")
    _chunks_with_metadata = create_chunks_with_metadata(
        documents,
        chunk_size=400,
        overlap=50
    )
    print(f"    ✓ Generated {len(_chunks_with_metadata)} chunks")
    
    # Initialize embedding model (Milestone 4)
    print("[2] Initializing embedding model...")
    _embedding_model = initialize_embedding_model("all-MiniLM-L6-v2")
    
    # Initialize ChromaDB (Milestone 4)
    print("[3] Setting up ChromaDB...")
    _, _chromadb_collection = initialize_chromadb_collection("professor_reviews")
    
    # Store chunks in ChromaDB (Milestone 4)
    print("[4] Storing chunks in ChromaDB...")
    texts = [chunk["chunk"] for chunk in _chunks_with_metadata]
    ids = [f"chunk_{i}" for i in range(len(_chunks_with_metadata))]
    embeddings = _embedding_model.encode(texts, show_progress_bar=False)
    metadatas = [{"source": chunk["source"]} for chunk in _chunks_with_metadata]
    
    _chromadb_collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )
    print(f"    ✓ Stored {len(_chunks_with_metadata)} chunks in ChromaDB")
    print("[*] Pipeline initialized successfully\n")


def initialize_groq_client() -> Groq:
    """
    Initialize Groq client with API key from environment.
    
    Returns:
        Initialized Groq client
        
    Raises:
        ValueError: If GROQ_API_KEY is not set
    """
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in environment. "
            "Please set it in .env file or as an environment variable."
        )
    
    return Groq(api_key=api_key)


def extract_sources(retrieved: List[Dict]) -> str:
    """
    Extract unique source filenames from retrieved chunks.
    
    Args:
        retrieved: List of retrieved chunks with metadata
        
    Returns:
        Comma-separated source filenames
    """
    if not retrieved:
        return "No sources"
    
    sources = list(set(chunk["source"] for chunk in retrieved))
    return ", ".join(sorted(sources))


def ask(question: str) -> str:
    """
    Answer a question using retrieved chunks and Groq generation.
    
    The answer is grounded in the retrieved context. If sufficient information
    is not available in the retrieved chunks, returns a "not enough information"
    message.
    
    Args:
        question: User's question about CSI Computer Science professors
        
    Returns:
        Grounded answer with source attribution
    """
    global _embedding_model, _chromadb_collection
    
    if not _embedding_model or not _chromadb_collection:
        return "Error: Pipeline not initialized. Please restart the application."
    
    try:
        # Step 1: Retrieve top-5 chunks
        retrieved = retrieve_chunks(
            _chromadb_collection,
            question,
            _embedding_model,
            top_k=5
        )
        
        # Step 2: Check if we have relevant chunks
        if not retrieved:
            return "I don't have enough information on that."
        
        # Step 3: Format context from retrieved chunks
        context_text = "\n\n".join([
            f"[{chunk['source']}] {chunk['chunk']}"
            for chunk in retrieved
        ])
        
        # Step 4: Create grounding system prompt
        system_prompt = """You are a helpful assistant answering questions about CSI Computer Science professors and courses based on student reviews.

IMPORTANT RULES:
1. Answer ONLY based on the provided context from student reviews.
2. Do NOT add information from outside knowledge or general experience.
3. If the context does not contain enough information to answer the question, respond EXACTLY with: "I don't have enough information on that."
4. Be honest about what the reviews say, including both positive and negative feedback.
5. If opinions are mixed, acknowledge the variation in student experiences.

You will be provided with retrieved context below. Use only this context to answer the user's question."""
        
        # Step 5: Call Groq API
        groq_client = initialize_groq_client()
        
        message = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Context from student reviews:\n\n{context_text}\n\nQuestion: {question}"
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2,  # Low temperature for factual consistency
            max_tokens=500
        )
        
        answer = message.choices[0].message.content.strip()
        
        # Step 6: Add source attribution
        sources = extract_sources(retrieved)
        full_response = f"{answer}\n\n---\n**Sources:** {sources}"
        
        return full_response
    
    except ValueError as e:
        return f"Configuration error: {str(e)}"
    except Exception as e:
        return f"Error generating answer: {str(e)}"


def create_gradio_interface() -> gr.Interface:
    """
    Create a Gradio interface for the question-answering system.
    
    Returns:
        Configured Gradio Interface
    """
    interface = gr.Interface(
        fn=ask,
        inputs=gr.Textbox(
            lines=3,
            placeholder="Ask a question about CSI Computer Science professors...",
            label="Question"
        ),
        outputs=gr.Textbox(
            lines=8,
            label="Answer",
            interactive=False
        ),
        title="CSI Computer Science Professor Reviews - Question Answering",
        description="""Ask questions about CSI Computer Science professors based on student reviews.
The system retrieves relevant student feedback and generates grounded answers.
All responses are sourced from actual student reviews.""",
        examples=[
            ["Which professor gives detailed feedback?"],
            ["What is the workload like for CSCI courses?"],
            ["Which professors are good at explaining concepts?"],
            ["What do students say about exam difficulty?"],
            ["Which professor is known for being accessible during office hours?"],
            ["How hard is CSC335?"],
            ["Do professors curve their grades?"],
        ]
    )
    
    return interface


def main():
    """Main entry point: initialize pipeline and launch Gradio."""
    print("=" * 80)
    print("MILESTONE 5: Grounded Generation and Gradio Interface")
    print("=" * 80)
    print()
    
    # Initialize the pipeline
    initialize_pipeline()
    
    # Create and launch Gradio interface
    print("=" * 80)
    print("Launching Gradio interface...")
    print("=" * 80)
    print()
    
    interface = create_gradio_interface()
    interface.launch(
        share=False,
        show_error=True,
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
