import os
import time
import argparse
import requests
import json
from typing import List, Dict, Any, Tuple
from evaluation.evaluator import load_questions, evaluate_model
from dotenv import load_dotenv

# Use docling directly
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TesseractCliOcrOptions,
    )
    from docling.document_converter import PdfFormatOption
    DOCLING_AVAILABLE = True
except ImportError as e:
    print(f"ImportError loading docling modules: {e}")
    DOCLING_AVAILABLE = False

# Store extracted content
filename_to_content: Dict[str, str] = {}

def extract_documents_with_ocr():
    """Extract document content using docling with EasyOCR enabled"""
    if not DOCLING_AVAILABLE:
        print("ERROR: docling not available. Install with: pip install docling")
        return {}
        
    folder = os.path.join(os.path.dirname(__file__), 'data', 'papers')
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    
    # Switch to EasyOCR
    from docling.datamodel.pipeline_options import EasyOcrOptions
    ocr_options = EasyOcrOptions(force_full_page_ocr=True)
    pipeline_options.ocr_options = ocr_options
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )
    
    for fname in os.listdir(folder):
        if not fname.lower().endswith('.pdf'):
            continue
        path = os.path.join(folder, fname)
        
        print(f"Extracting {fname} with OCR (EasyOCR)...")
        try:
            result = converter.convert(path)
            content = result.document.export_to_markdown()
            filename_to_content[fname] = content
            print(f"✓ Extracted {len(content)} characters from {fname} (with EasyOCR)")
            
            preview = content[:500] + "..." if len(content) > 500 else content
            print(f"  Preview: {preview}\n")
            
        except Exception as e:
            print(f"✗ Failed to extract {fname} with EasyOCR: {e}")
    
    return filename_to_content


def get_ollama_models() -> List[str]:
    """Get available models from Ollama"""
    try:
        resp = requests.get("http://localhost:11435/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = [model['name'] for model in data.get('models', [])]
        return models
    except Exception as e:
        print(f"Failed to get Ollama models: {e}")
        return []


def ask_ollama(prompt: str, model_name: str, required_files: List[str]) -> Tuple[str, List[str]]:
    """Ask Ollama model with manual RAG context - returns (answer, context_chunks)"""
    
    # Build context from required files
    context_parts = []
    context_chunks = []
    
    for fname in required_files:
        if fname in filename_to_content:
            content = filename_to_content[fname]
            if len(content) > 15000:
                content = content[:15000] + "\n[... content truncated ...]"
            
            context_parts.append(f"=== Document: {fname} ===\n{content}\n")
            context_chunks.append(f"File: {fname}\nContent: {content}")
    
    # Build enhanced prompt with context
    if context_parts:
        context_text = "\n".join(context_parts)
        enhanced_prompt = f"""Based on the following document content, answer the question precisely:

{context_text}

Question: {prompt}

Answer based on the document content above."""
    else:
        enhanced_prompt = prompt
    
    # Ollama API payload
    payload = {
        "model": model_name,
        "prompt": enhanced_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 2048
        }
    }
    
    try:
        resp = requests.post(
            "http://localhost:11435/api/generate", 
            json=payload, 
            timeout=300
        )
        
        if resp.status_code == 200:
            data = resp.json()
            answer = data.get('response', 'No response received')
            return answer, context_chunks
        else:
            error_msg = f"Ollama error {resp.status_code}: {resp.text}"
            return error_msg, context_chunks
            
    except Exception as e:
        error_msg = f"Request failed: {e}"
        return error_msg, context_chunks


def test_single_question():
    """Test with a single question"""
    print("=== Testing Single Question ===")
    
    extract_documents_with_ocr()
    if not filename_to_content:
        print("No documents extracted!")
        return
    
    models = get_ollama_models()
    if not models:
        print("No Ollama models found!")
        return
    
    print(f"Available models: {models}")
    model = models[0]
    
    prompt = "There is a typo in the version of The Path To Autonomous Cyber Defense that you have. What is 'cyber' misspelled as?"
    required_files = ['2404.10788v1.pdf']
    
    print(f"Testing with model: {model}")
    print(f"Required files: {required_files}")
    print(f"Available files: {list(filename_to_content.keys())}")
    
    result, context = ask_ollama(prompt, model, required_files)
    print(f"\nResult: {result}")
    print(f"\nContext chunks: {len(context)}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate with Ollama direct + docling")
    parser.add_argument("--suffix", required=True, help="Output name suffix")
    parser.add_argument("--test", action="store_true", help="Test single question")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--models", help="Comma-separated model names")
    group.add_argument("--full", action="store_true", help="Use all available models")
    args = parser.parse_args()

    if args.test:
        test_single_question()
        return
    
    if not (args.models or args.full):
        print("ERROR: Must specify --models, --full, or --test")
        return

    extract_documents_with_ocr()
    
    if not filename_to_content:
        print("ERROR: No documents extracted.")
        return
    
    available_models = get_ollama_models()
    if not available_models:
        print("ERROR: No Ollama models available")
        return
    
    if args.full:
        selected = available_models
    else:
        selected = [m.strip() for m in args.models.split(',')]
        selected = [m for m in selected if m in available_models]
        if not selected:
            print(f"ERROR: None of the specified models found. Available: {available_models}")
            return

    print(f"Documents loaded: {list(filename_to_content.keys())}")
    print(f"Evaluating models: {selected}")
    
    qa_data = load_questions(os.path.join('data', 'questions_answers.json'))
    for model in selected:
        print(f"--- Evaluating {model} ---")
        evaluate_model(model, ask_ollama, args.suffix, qa_data)
        print(f"--- Done {model} ---")


if __name__ == '__main__':
    main()