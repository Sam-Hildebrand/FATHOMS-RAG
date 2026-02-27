import os
import time
import argparse
import base64
import anthropic
from typing import Any, List, Tuple, Dict
from evaluation.evaluator import load_questions, evaluate_model

claude_client = None
available_files: Dict[str, Dict[str, Any]] = {}  # Cache of loaded files

def initialize_claude() -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Initialize Claude API client and load all available files"""
    global claude_client, available_files
    
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.isfile(dotenv_path):
        load_dotenv(dotenv_path)
    
    key = os.getenv("ANTHROPIC_API_KEY") or input("Anthropic API key: ")
    
    # Debug: Check if key was loaded
    if key:
        print(f"API key loaded: {key[:20]}..." if len(key) > 20 else f"API key: {key}")
    else:
        print("No API key found in environment!")
    
    claude_client = anthropic.Anthropic(api_key=key.strip())
    
    # Load all available files into cache
    folder = "data/papers"
    max_text_chars = 200000  # Limit text file size
    
    if os.path.exists(folder):
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".pdf", ".txt", ".md")): 
                continue
            path = os.path.join(folder, fname)
            
            if fname.lower().endswith(".pdf"):
                # Handle PDF files
                try:
                    file_size = os.path.getsize(path)
                    if file_size > 30 * 1024 * 1024:  # 30MB limit per PDF
                        print(f"Error: {fname} exceeds 30MB limit, skipping")
                        quit()
                    
                    with open(path, 'rb') as f:
                        pdf_data = base64.b64encode(f.read()).decode('utf-8')
                        available_files[fname] = {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_data
                            }
                        }
                        print(f"Cached PDF: {fname} ({file_size / 1024 / 1024:.1f}MB)")
                except Exception as e:
                    print(f"Error loading PDF {fname}: {e}")
            else:
                # Handle text files with size limits
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Truncate if too long
                        if len(content) > max_text_chars:
                            content = content[:max_text_chars] + "\n\n[Content truncated due to size limits]"
                            print(f"Truncated {fname} to {max_text_chars} characters")
                        
                        available_files[fname] = {
                            "type": "text",
                            "text": f"File: {fname}\n\n{content}\n\n---\n"
                        }
                        print(f"Cached text file: {fname} ({len(content)} chars)")
                except UnicodeDecodeError:
                    print(f"Warning: Could not read {fname} (encoding issue)")
                except Exception as e:
                    print(f"Error reading {fname}: {e}")
    
    print(f"Total files cached: {len(available_files)}")
    
    # Available Claude models
    models = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022", 
        "claude-3-opus-20240229",
        "claude-sonnet-4-20250514"
    ]
    
    return models, available_files

def ask_model_with_files(prompt: str, model_name: str, required_files: List[str],
                        max_retries: int = 2, backoff: float = 120.0) -> str:
    """Ask Claude model with specific files"""
    global claude_client, available_files
    
    if claude_client is None:
        return "error: client not initialized"
    
    # 1 minute between requests
    time.sleep(180)

    # Build message content with only the required files
    content = []
    
    # Add only the required files
    files_added = 0
    for filename in required_files:
        if filename in available_files:
            content.append(available_files[filename])
            files_added += 1
            print(f"  Using file: {filename}")
        else:
            print(f"  Warning: Required file {filename} not found in cache")
    
    if files_added == 0 and required_files:
        print(f"  Warning: None of the required files {required_files} were available")
    
    # Add the prompt
    content.append({
        "type": "text",
        "text": prompt
    })
    
    for attempt in range(1, max_retries + 1):
        try:
            message = claude_client.messages.create(
                model=model_name,
                max_tokens=500,
                messages=[
                    {"role": "user", "content": content}
                ]
            )
            return message.content[0].text.strip().lower()
        except anthropic.RateLimitError:
            if attempt < max_retries:
                print(f"Rate limit hit, waiting {backoff * attempt} seconds...")
                time.sleep(backoff * attempt)
            else:
                return "rate limit exceeded"
        except anthropic.APITimeoutError:
            if attempt < max_retries:
                print(f"Timeout, retrying in {backoff * attempt} seconds...")
                time.sleep(backoff * attempt)
            else:
                return "timed out"
        except Exception as e:
            if attempt < max_retries:
                print(f"Error: {e}, retrying in {backoff * attempt} seconds...")
                time.sleep(backoff * attempt)
            else:
                return f"error: {str(e)}"
    
    return "timed out"

def main():
    parser = argparse.ArgumentParser(description="Evaluate via Claude API")
    parser.add_argument("--suffix", required=True, help="Output name suffix")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--models", help="Comma-separated model indices")
    group.add_argument("--full", action="store_true", help="All except last")
    args = parser.parse_args()
    
    models, files = initialize_claude()
    
    if args.full:
        selected = models[:-1]
    else:
        idx = [int(i) for i in args.models.split(",")]
        selected = [models[i] for i in idx]
    
    print("Available models:")
    for i, model in enumerate(models):
        print(f"  {i}: {model}")
    print("Evaluating: ", selected)
    
    qa = load_questions("data/questions_answers.json")
    
    for m in selected:
        evaluate_model(m, ask_model_with_files, args.suffix, files, qa)

if __name__ == "__main__":
    main()