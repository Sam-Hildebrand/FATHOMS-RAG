import os
import json
import numpy as np
from typing import Any, Callable, Dict, List, Tuple

from evaluation.hallucination import HallucinationDetector

# Initialize once (dim=128 as placeholder)
detector = HallucinationDetector(data_file="evaluation/statement_abstention_training.json")

# Uncertainty cues for hallucination detection
UNCERTAINTY_CUES = [
    "not sure", "unfortunately", "unable", "unsure",
    "not provide", "no information", "don't have the information",
    "can't assist", "not assist", "not able to", "don't have access",
    "not have access", "not explicitly mention", "not specifically mention", "not mention",
    "not include", "no mention", "no specific information",
    "not contain", "not specify", "not explicitly detailed",
    "not detailed", "no detail", "not specifically detailed", "not explicitly stated",
    "not stated", "not specifically stated"
]

def load_questions(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load a JSON file containing categories of questions and their expected answers.
    """
    if not os.path.exists(path):
        raise SystemExit(f"Questions file not found: {path}")
    with open(path, 'r') as f:
        return json.load(f)

def evaluate_model(
    model_name: str,
    ask_fn: Callable[[str, str, List[str]], Tuple[str, List[Dict[str, Any]]]], # Updated to return context
    suffix: str,
    qa_data: Dict[str, List[Dict[str, Any]]],
) -> None:
    """
    Evaluate a single model over loaded Q&A data, printing debug info and
    writing out a summary JSON file with RAG context information and hallucination detection.
    
    Args:
        model_name: Name of the API model to call.
        ask_fn: Function(prompt, model_name, required_files) -> (answer_text, context_info).
        suffix: Suffix to append to output filename.
        qa_data: Dict mapping categories to lists of {question, answers, files} items.
    """
    print(f"\n=== Evaluating model: {model_name} ===")
    results: List[Dict[str, Any]] = []
    category_scores: Dict[str, float] = {}
    category_hallucinations: Dict[str, float] = {}
    
    # Loop categories
    for cat, qlist in qa_data.items():
        print(f"\n-- Category: {cat} --")
        total_score = 0.0
        total_hallucinations = 0
        
        for item in qlist:
            q = item['question']
            required_files = item.get('files', []) # Get required files for this question
            print(f"Question: {q}")
            print(f"Required files: {required_files}")
            
            # Call the API wrapper with specific files - now returns answer and context
            try:
                response = ask_fn(q, model_name, required_files)
                if isinstance(response, tuple) and len(response) == 2:
                    ans, context_info = response
                else:
                    # Fallback for functions that only return answer
                    ans = response if isinstance(response, str) else str(response)
                    context_info = []
            except Exception as e:
                print(f"Error calling ask_fn: {e}")
                ans = "error occurred"
                context_info = []
            
            print(f"-> Answer: {ans}")
            
            # Print context information
            if context_info:
                print(f"-> RAG Context ({len(context_info)} chunks):")
                for ctx in context_info:
                    print("\n", ctx)
            else:
                print("-> No RAG context available")
            
            # Scoring
            best = 0.0
            if ans not in ['TIMED OUT', 'timed out', 'rate limit exceeded'] and not ans.startswith('error'):
                for pattern in item.get('answers', []):
                    toks = [t.lower() for t in pattern]
                    score = sum(t.lower().replace("-", " ") in ans.lower().replace("-", " ") for t in toks) / len(toks) if toks else 0.0
                    best = max(best, score)
            
            # Hallucination detection
            hallucinating = detector.is_hallucinating(ans, best)
            if hallucinating:
                total_hallucinations += 1
            
            print(f"Score: {best}")
            print(f"Hallucinating: {hallucinating}\n")
            
            total_score += best
            
            results.append({
                'category': cat,
                'question': q,
                'required_files': required_files,
                'rag_context': context_info,
                'pred_answer': ans,
                'ground_truth': item.get('answers'),
                'correctness_score': best,
                'suspected_hallucination': hallucinating,
            })
        
        # Category averages
        avg_score = total_score / len(qlist) if qlist else 0.0
        hallucination_rate = total_hallucinations / len(qlist) if qlist else 0.0
        
        category_scores[cat] = avg_score
        category_hallucinations[f"{cat} Hallucination Rate"] = hallucination_rate
        
        print(f"Average score for {cat}: {avg_score}")
        print(f"Hallucination rate for {cat}: {hallucination_rate} ({total_hallucinations}/{len(qlist)})\n")
    
    # Overall averages
    overall_score = float(np.mean(list(category_scores.values()))) if category_scores else 0.0
    overall_hallucinations = float(np.mean(list(category_hallucinations.values()))) if category_hallucinations else 0.0
    
    print(f"Overall score: {overall_score}")
    print(f"Overall hallucination rate: {overall_hallucinations}\n")
    
    # Combine scores and hallucination rates
    all_scores = {**category_scores, **category_hallucinations}
    
    out = {
        'model': f"{model_name.lower().replace('/', '_')}_{suffix}",
        'api_model': model_name,
        'scores': all_scores,
        'overall_score': overall_score,
        'total_hallucination_rate': overall_hallucinations,
        'responses': results,
    }
    
    fname = out['model'].replace(" ", "_").replace("-", "_").replace(".", "_") + '.json'
    
    # Write to file
    try:
        with open(fname, 'w') as of:
            json.dump(out, of, indent=2)
        print(f"Results saved to {fname}")
    except Exception as exc:
        print(f"Error writing results file {fname}: {exc}")