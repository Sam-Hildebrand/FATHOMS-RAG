import os
import time
import argparse
import glob
import requests
import json
from pathlib import Path
from typing import Any, List, Dict, Tuple

# LlamaIndex core imports
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import SimpleVectorStore

# PDF processing
import fitz

# Ollama imports
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from evaluation.evaluator import load_questions, evaluate_model


def get_ollama_models(base_url: str = "http://localhost:11435") -> List[str]:
    """Get list of all available models from Ollama API."""
    try:
        response = requests.get(f"{base_url}/api/tags")
        response.raise_for_status()
        
        models_data = response.json()
        model_names = []
        
        print("Available Ollama models:")
        for model in models_data.get("models", []):
            model_name = model.get("name", "")
            if model_name:
                model_names.append(model_name)
                size = model.get("size", 0)
                size_gb = size / (1024**3) if size > 0 else 0
                print(f"  - {model_name} (Size: {size_gb:.1f}GB)")
        
        return model_names
        
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama API: {e}")
        print("Make sure Ollama is running with: ollama serve")
        return []


def load_and_process_documents(data_path: str = "data/papers") -> Dict[str, Any]:
    """Load and process all documents into nodes."""
    # Check if data folder exists
    if not os.path.isdir(data_path):
        raise SystemExit(f"Error: Data folder not found at {data_path}")
    
    # Check for PDF files
    pdf_files = glob.glob(os.path.join(data_path, "*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDF files found in {data_path}")
    
    print(f"Loading documents from {data_path}...")
    documents = {}
    
    # Load documents
    for pdf_file in pdf_files:
        try:
            doc = fitz.open(pdf_file)
            file_name = os.path.basename(pdf_file)
            documents[file_name] = doc
            print(f"Loaded: {file_name} ({len(doc)} pages)")
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
    
    # Create text chunks
    text_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    text_chunks = []
    doc_metadata = []
    
    for file_name, doc in documents.items():
        for page_idx, page in enumerate(doc):
            page_text = page.get_text("text")
            if not page_text.strip():
                continue
                
            cur_text_chunks = text_parser.split_text(page_text)
            text_chunks.extend(cur_text_chunks)
            
            for chunk_idx, chunk in enumerate(cur_text_chunks):
                metadata = {
                    "file_name": file_name,
                    "page_number": page_idx + 1,
                    "chunk_index": chunk_idx,
                    "total_pages": len(doc)
                }
                doc_metadata.append(metadata)
    
    # Create nodes
    nodes = []
    for idx, (text_chunk, metadata) in enumerate(zip(text_chunks, doc_metadata)):
        node = TextNode(text=text_chunk, metadata=metadata, id_=f"node_{idx}")
        nodes.append(node)
    
    print(f"Created {len(nodes)} nodes from {len(documents)} documents")
    
    return {
        'documents': documents,
        'nodes': nodes,
        'text_chunks': text_chunks,
        'doc_metadata': doc_metadata
    }


def create_rag_system_with_port(model_name: str, document_data: Dict[str, Any], port: str):
    """Create RAG system for a specific model with custom port."""
    print(f"Creating RAG system for model: {model_name}")
    
    base_url = f"http://localhost:{port}"
    
    # Setup models
    llm = Ollama(
        model=model_name, 
        request_timeout=300.0, 
        temperature=0.1,
        base_url=base_url
    )
    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url=base_url
    )
    
    # Set global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Create vector store and index
    vector_store = SimpleVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print("Building vector index...")
    index = VectorStoreIndex(
        document_data['nodes'],
        storage_context=storage_context,
        show_progress=True,
    )
    
    # Create query engine
    query_engine = index.as_query_engine(similarity_top_k=3, response_mode="compact")
    
    class SimpleRAG:
        def __init__(self, query_engine, model_name):
            self.query_engine = query_engine
            self.model_name = model_name
        
        def query(self, question: str, required_file_names: List[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
            try:
                if required_file_names:
                    file_context = f"Based on information from {', '.join(required_file_names)}, "
                    enhanced_question = file_context + question
                else:
                    enhanced_question = question
                
                response = self.query_engine.query(enhanced_question)
                
                # Extract context from source nodes
                context_info = []
                if hasattr(response, 'source_nodes'):
                    for i, source_node in enumerate(response.source_nodes):
                        context_info.append({
                            'chunk_id': i + 1,
                            'file_name': source_node.node.metadata.get('file_name', 'Unknown'),
                            'page_number': source_node.node.metadata.get('page_number', 'Unknown'),
                            'relevance_score': float(source_node.score) if hasattr(source_node, 'score') else 0.0,
                            'content': source_node.node.text[:500] + "..." if len(source_node.node.text) > 500 else source_node.node.text
                        })
                
                return str(response).strip().lower(), context_info
                
            except Exception as e:
                print(f"Error querying RAG system: {e}")
                return "error occurred during query", []
    
    rag_system = SimpleRAG(query_engine, model_name)
    
    # Test the system
    try:
        test_response, test_context = rag_system.query("Hello, this is a test query.")
        print(f"✓ RAG system for {model_name} is working")
        if test_context:
            print(f"  Test context: {len(test_context)} chunks retrieved")
    except Exception as e:
        print(f"✗ RAG system for {model_name} test failed: {e}")
        raise
    
    return rag_system


def create_rag_system(model_name: str, document_data: Dict[str, Any]):
    """Create RAG system for a specific model."""
    print(f"Creating RAG system for model: {model_name}")
    
    # Setup models
    llm = Ollama(
        model=model_name, 
        request_timeout=300.0, 
        temperature=0.1,
        base_url="http://localhost:11435"
    )
    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11435"
    )
    
    # Set global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Create vector store and index
    vector_store = SimpleVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print("Building vector index...")
    index = VectorStoreIndex(
        document_data['nodes'],
        storage_context=storage_context,
        show_progress=True,
    )
    
    # Create query engine
    query_engine = index.as_query_engine(similarity_top_k=3, response_mode="compact")
    
    class SimpleRAG:
        def __init__(self, query_engine, model_name):
            self.query_engine = query_engine
            self.model_name = model_name
        
        def query(self, question: str, required_file_names: List[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
            try:
                if required_file_names:
                    file_context = f"Based on information from {', '.join(required_file_names)}, "
                    enhanced_question = file_context + question
                else:
                    enhanced_question = question
                
                response = self.query_engine.query(enhanced_question)
                
                # Extract context from source nodes
                context_info = []
                if hasattr(response, 'source_nodes'):
                    for i, source_node in enumerate(response.source_nodes):
                        context_info.append({
                            'chunk_id': i + 1,
                            'file_name': source_node.node.metadata.get('file_name', 'Unknown'),
                            'page_number': source_node.node.metadata.get('page_number', 'Unknown'),
                            'score': float(source_node.score) if hasattr(source_node, 'score') else 0.0,
                            'content': source_node.node.text[:500] + "..." if len(source_node.node.text) > 500 else source_node.node.text
                        })
                
                return str(response).strip().lower(), context_info
                
            except Exception as e:
                print(f"Error querying RAG system: {e}")
                return "error occurred during query", []
    
    rag_system = SimpleRAG(query_engine, model_name)
    
    # Test the system
    try:
        test_response, test_context = rag_system.query("Hello, this is a test query.")
        print(f"✓ RAG system for {model_name} is working")
        if test_context:
            print(f"  Test context: {len(test_context)} chunks retrieved")
    except Exception as e:
        print(f"✗ RAG system for {model_name} test failed: {e}")
        raise
    
    return rag_system


def ask_model(prompt: str, model_name: str, required_file_names: List[str],
              max_retries: int = 2, backoff: float = 30.0, rag_system=None) -> Tuple[str, List[Dict[str, Any]]]:
    """Query the RAG system with a prompt and return both answer and context."""
    if not rag_system:
        return "rag system not initialized", []
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [Attempt {attempt}] Querying RAG system...")
            response, context = rag_system.query(prompt, required_file_names)
            return response, context
            
        except Exception as exc:
            print(f"  [Attempt {attempt}] Query error: {exc}")
            if attempt < max_retries:
                time.sleep(backoff * attempt)
            else:
                print("  All retries exhausted.")
                return "timed out", []


def main():
    parser = argparse.ArgumentParser(description="Evaluate LlamaIndex RAG system")
    parser.add_argument("--suffix", required=True, help="Output name suffix")
    parser.add_argument("--model", help="Specific Ollama model to use")
    parser.add_argument("--full", action="store_true", help="Evaluate all available models")
    parser.add_argument("--data-path", default="data/papers", help="Path to PDF documents")
    parser.add_argument("--port", default="11435", help="Ollama port (default: 11435)")
    args = parser.parse_args()

    if not args.model and not args.full:
        parser.error("Must specify either --model or --full")

    print("Initializing LlamaIndex RAG Evaluation")
    print(f"Connecting to Ollama on port {args.port}")
    print("=" * 60)
    
    # Load and process documents once
    try:
        document_data = load_and_process_documents(args.data_path)
    except Exception as e:
        print(f"Failed to load documents: {e}")
        return
    
    # Get available models
    ollama_base_url = f"http://localhost:{args.port}"
    if args.full:
        models_to_evaluate = get_ollama_models(ollama_base_url)
        if not models_to_evaluate:
            print("No Ollama models found.")
            return
        print(f"\nWill evaluate {len(models_to_evaluate)} models")
    else:
        models_to_evaluate = [args.model]
        print(f"\nWill evaluate model: {args.model}")
    
    # Load questions
    qa_data = load_questions("data/questions_answers.json")
    print(f"Loaded {len(qa_data)} question categories")
    
    # Track results
    results_summary = {}
    successful = 0
    failed = 0
    
    # Evaluate each model
    for i, model_name in enumerate(models_to_evaluate, 1):
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL {i}/{len(models_to_evaluate)}: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Create RAG system for this model
            rag_system = create_rag_system_with_port(model_name, document_data, args.port)
            
            # Create wrapper function
            def rag_ask_model(prompt: str, model_name_inner: str, required_file_names: List[str],
                              max_retries: int = 2, backoff: float = 30.0) -> Tuple[str, List[Dict[str, Any]]]:
                return ask_model(prompt, model_name_inner, required_file_names, 
                                max_retries, backoff, rag_system)
            
            # Run evaluation
            evaluate_model(model_name, rag_ask_model, args.suffix, qa_data)
            
            results_summary[model_name] = "SUCCESS"
            successful += 1
            print(f"✓ Successfully evaluated {model_name}")
            
        except Exception as e:
            print(f"✗ Failed to evaluate {model_name}: {e}")
            results_summary[model_name] = f"FAILED: {str(e)}"
            failed += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total models: {len(models_to_evaluate)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    print(f"\nResults:")
    for model, result in results_summary.items():
        status = "✓" if result == "SUCCESS" else "✗"
        print(f"  {status} {model}: {result}")
    
    # Cleanup
    print(f"\nCleaning up documents...")
    for file_name, doc in document_data['documents'].items():
        try:
            doc.close()
        except Exception as e:
            print(f"Error closing {file_name}: {e}")
    
    print(f"\nEvaluation completed! Results saved as <model_name>_{args.suffix}.json")


if __name__ == "__main__":
    main()