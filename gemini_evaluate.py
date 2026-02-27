import os
import time
import argparse
import google.generativeai as genai
from typing import Any, List, Dict, Tuple

from evaluator import load_questions, evaluate_model

filename_to_gemini_file_obj: Dict[str, Any] = {}

def initialize_gemini() -> Tuple[List[str], Dict[str, Any]]:
    """
    Initializes Gemini API, uploads/registers files, and returns
    available models and a mapping of display names to uploaded file objects.
    """
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.isfile(dotenv_path):
        load_dotenv(dotenv_path)

    key = os.getenv("GOOGLE_API_KEY") or input("Google Gemini API key: ")
    genai.configure(api_key=key.strip())

    # fetch and upload files
    print("Checking for existing files on Gemini File API...")
    existing_files_on_gemini = list(genai.list_files())
    
    # Populate the global mapping with existing files
    for fobj in existing_files_on_gemini:
        filename_to_gemini_file_obj[fobj.display_name] = fobj
    
    print(f"Found {len(filename_to_gemini_file_obj)} files already uploaded on Gemini File API.")

    folder = "data/papers"
    if not os.path.isdir(folder):
        raise SystemExit(f"Error: Data folder not found at {folder}")

    for fname in os.listdir(folder):
        # Only process PDF, TXT, MD files
        if not fname.lower().endswith((".pdf", ".txt", ".md")):
            continue

        # If file is not already in our global map (meaning it wasn't listed by genai.list_files())
        if fname not in filename_to_gemini_file_obj:
            path = os.path.join(folder, fname)
            try:
                uploaded_file = genai.upload_file(path=path, display_name=fname)
                filename_to_gemini_file_obj[fname] = uploaded_file
                print(f"Uploaded new file: {fname} (URI: {uploaded_file.uri})")
            except Exception as exc:
                print(f"ERROR: Failed to upload {fname}: {exc}")
        else:
            print(f"Re-using existing file: {fname} (URI: {filename_to_gemini_file_obj[fname].uri})")

    if not filename_to_gemini_file_obj:
        raise SystemExit("No files available for RAG. Please check 'data/papers' and API key.")

    # list usable models
    print("\nFetching available Gemini models...")
    models = [
        m.name
        for m in genai.list_models()
        if "generateContent" in m.supported_generation_methods
    ]
    if not models:
        raise SystemExit("No models found that support generateContent. Check API key and region.")
    
    print("Available models:")
    for i, m in enumerate(models):
        print(f" [{i}] {m}")

    # We return the filename_to_gemini_file_obj for potential use, though ask_model now uses the global
    return models, filename_to_gemini_file_obj

def ask_model(prompt: str, model_name: str, required_file_names: List[str],
              max_retries: int = 2, backoff: float = 30.0) -> str:
    """
    Sends a user prompt with specific attached files to the Gemini model.
    """
    model = genai.GenerativeModel(model_name=model_name)
    
    time.sleep(120)
    # Construct parts based on required_file_names
    parts = [prompt]
    if required_file_names:
        for file_name in required_file_names:
            file_obj = filename_to_gemini_file_obj.get(file_name)
            if file_obj:
                parts.append(file_obj)
            else:
                print(f"WARNING: Required file '{file_name}' not found among uploaded files.")
                # You might want to handle this more robustly, e.g., raise an error
                # or log a more severe warning if a required file is truly missing.

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [Attempt {attempt}] Calling model with {len(parts)-1} documents...")
            res = model.generate_content(
                contents=parts,
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ],
                request_options={"timeout": 600}
            )
            return res.text.strip().lower()
        except Exception as exc:
            print(f"  [Attempt {attempt}] Generation error: {exc}")
            if attempt < max_retries:
                time.sleep(backoff * attempt)
            else:
                print("  All retries exhausted.")
                return "timed out" # Consistent with evaluator's scoring logic

def main():
    parser = argparse.ArgumentParser(description="Evaluate via Gemini")
    parser.add_argument("--suffix", required=True, help="Output name suffix")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--models", help="Comma-separated model indices")
    group.add_argument("--full", action="store_true", help="All except last")
    args = parser.parse_args()

    # initialize_gemini now populates the global filename_to_gemini_file_obj
    models, _ = initialize_gemini() # We don't need the second return value explicitly here

    if args.full:
        # Evaluate all models except potentially the last one (often an experimental/arena model)
        selected_models = models[:-1] if models else []
    else:
        try:
            idx = [int(i.strip()) for i in args.models.split(",")]
            selected_models = [models[i] for i in idx]
        except (ValueError, IndexError):
            raise SystemExit("Invalid --models format or index out of range. Use comma-separated indices.")

    print(f"\nSelected models for evaluation: {selected_models}")
    qa_data = load_questions("data/questions_answers.json")

    for m in selected_models:
        # Pass the ask_model function with its new signature to evaluate_model
        evaluate_model(m, ask_model, args.suffix, filename_to_gemini_file_obj, qa_data) # files_param is now the map

    # Clean up: Delete uploaded files from Gemini File API after evaluation
    print("\nDeleting uploaded files from Gemini File API...")
    for file_name, file_obj in filename_to_gemini_file_obj.items():
        try:
            # Check if the file was actually uploaded by this run or was reused.
            # Only delete files explicitly uploaded or whose lifecycle you manage.
            # For simplicity, this example will attempt to delete all files in the map,
            # but in a real app, you might only delete those uploaded *in this session*.
            genai.delete_file(file_obj.name)
            print(f"Deleted {file_obj.display_name} (URI: {file_obj.uri})")
        except Exception as exc:
            print(f"Failed to delete {file_obj.display_name}: {exc}")

if __name__ == "__main__":
    main()