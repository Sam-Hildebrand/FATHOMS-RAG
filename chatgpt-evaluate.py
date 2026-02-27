import os
import time
import argparse
from openai import OpenAI
from typing import List, Dict, Any
from evaluator import load_questions, evaluate_model
from dotenv import load_dotenv

# Default maximum tokens for model output
DEFAULT_MAX_TOKENS = 2048
# Maps for local paths and uploaded file IDs
auth_client: OpenAI
filename_to_path: Dict[str, str] = {}
filename_to_file_id: Dict[str, str] = {}


def initialize_openai() -> None:
    """
    Initialize OpenAI client, scan data/papers for PDFs,
    and upload each under purpose='user_data'.
    """
    global auth_client, filename_to_path, filename_to_file_id

    # Load .env if present
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.isfile(dotenv_path):
        load_dotenv(dotenv_path)

    api_key = os.getenv("OPENAI_API_KEY") or input("OpenAI API key: ")
    auth_client = OpenAI(api_key=api_key.strip())

    papers_dir = os.path.join(os.path.dirname(__file__), "data", "papers")
    if not os.path.isdir(papers_dir):
        raise SystemExit(f"Error: Papers folder not found at {papers_dir}")

    # Scan and upload each PDF
    for fname in os.listdir(papers_dir):
        if not fname.lower().endswith('.pdf'):
            continue
        path = os.path.join(papers_dir, fname)
        filename_to_path[fname] = path
        try:
            resp = auth_client.files.create(file=open(path, 'rb'), purpose="user_data")
            filename_to_file_id[fname] = resp.id
            print(f"Uploaded {fname} - {resp.id}")
        except Exception as e:
            print(f"ERROR uploading {fname}: {e}")

    if not filename_to_file_id:
        raise SystemExit("No PDFs uploaded. Check data/papers folder.")


def ask_model(prompt: str,
              model_name: str,
              required_file_names: List[str],
              max_retries: int = 3,
              backoff: float = 30.0,
              max_output_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """
    Send prompt + ONLY the PDFs listed in required_file_names to the model.
    """
    time.sleep(60) 

    # Build content: only include requested files
    content_items: List[Dict[str, Any]] = []
    for fname in required_file_names:
        fid = filename_to_file_id.get(fname)
        if not fid:
            print(f"WARNING: no file_id for {fname}, skipping")
            continue
        content_items.append({
            "type": "input_file",
            "file_id": fid
        })

    # Finally add the user text
    content_items.append({
        "type": "input_text",
        "text": prompt
    })

    payload_input = [{
        "role": "user",
        "content": content_items
    }]

    for attempt in range(1, max_retries + 1):
        try:
            resp = auth_client.responses.create(
                model=model_name,
                instructions="Please read the attached PDFs before answering.",
                input=payload_input,
                temperature=0.0,
                max_output_tokens=max_output_tokens
            )
            return resp.output_text.strip()
        except Exception as exc:
            print(f"Attempt {attempt} failed: {exc}")
            if attempt < max_retries:
                time.sleep(backoff * attempt)
    return "timed out"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate via Responses API with PDF attachments"
    )
    parser.add_argument("--suffix", required=True, help="Output name suffix")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--models", help="Comma-separated model names (e.g. gpt-4o, gpt-4.1)"
    )
    group.add_argument(
        "--full", action="store_true", help="Use default GPT-4 models"
    )
    args = parser.parse_args()

    initialize_openai()

    if args.full:
        models = ["gpt-4o", "gpt-4.1"]
    else:
        models = [m.strip() for m in args.models.split(",")]

    print(f"Models: {models}, max_output_tokens={DEFAULT_MAX_TOKENS}")
    qa_data = load_questions("data/questions_answers.json")

    # Evaluate each question: uses ask_model() which attaches only required PDFs
    for model in models:
        evaluate_model(
            model,
            ask_model,
            args.suffix,
            qa_data
        )


if __name__ == "__main__":
    main()
