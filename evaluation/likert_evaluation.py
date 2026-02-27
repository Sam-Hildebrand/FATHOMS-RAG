import os
import sys
import json
import random
from pathlib import Path

def collect_json_files(paths):
    """Collect all JSON files from the provided directories."""
    json_files = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            json_files.extend(path.glob("*.json"))
        elif path.is_file() and path.suffix == ".json":
            json_files.append(path)
    return json_files

def format_ground_truth(ground_truth):
    """
    Convert the nested ground_truth list of lists into a readable string.
    Each inner list is a set of required phrases.
    Each outer list is an alternative acceptable set.
    """
    if not ground_truth:
        return "N/A"
    formatted = []
    for option in ground_truth:
        formatted.append(" AND ".join(option))
    return " OR ".join(formatted)

def review_file(file_path):
    """Load JSON file, pick 3 random responses, and ask user for feedback."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    responses = data.get("responses", [])
    if not responses:
        return None

    sampled = random.sample(responses, min(3, len(responses)))
    results = []

    print(f"\nReviewing file: {file_path}")
    print("=" * 60)

    for i, resp in enumerate(sampled, 1):
        question = resp.get("question")
        pred_answer = resp.get("pred_answer")
        ground_truth = format_ground_truth(resp.get("ground_truth", []))
        correctness_score = resp.get("correctness_score")
        suspected_hallucination = resp.get("suspected_hallucination")

        print(f"\nQuestion {i}: {question}\n")
        print(f"\nPredicted Answer:\n{pred_answer}\n")
        print(f"\nActual Answer (from ground_truth):\n{ground_truth}\n")
        print(f"\nSystem Correctness Score: {correctness_score}")
        
        # User rates correctness agreement
        while True:
            try:
                correctness_rating = int(input("Do you agree? (1=Strongly Disagree, 3=Neutral, 5=Strongly Agree): "))
                if 1 <= correctness_rating <= 5:
                    break
            except ValueError:
                pass
            print("Please enter a number between 1 and 5.")

        print(f"\nSystem Suspected Hallucination: {suspected_hallucination}")
        
        # User rates hallucination agreement
        while True:
            try:
                hallucination_rating = int(input("Do you agree with hallucination detection? (1=Strongly Disagree, 3=Neutral, 5=Strongly Agree): "))
                if 1 <= hallucination_rating <= 5:
                    break
            except ValueError:
                pass
            print("Please enter a number between 1 and 5.")

        results.append({
            "question": question,
            "pred_answer": pred_answer,
            "ground_truth": ground_truth,
            "correctness_score": correctness_score,
            "suspected_hallucination": suspected_hallucination,
            "correctness_rating": correctness_rating,
            "hallucination_rating": hallucination_rating
        })

    return {
        "file": str(file_path),
        "model": data.get("model"),
        "api_model": data.get("api_model"),
        "reviews": results
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python likert_evaluation.py <folder1> [<folder2> ...]")
        sys.exit(1)

    paths = sys.argv[1:]
    json_files = collect_json_files(paths)

    if not json_files:
        print("No JSON files found in the given paths.")
        sys.exit(1)

    all_results = []
    for jf in json_files:
        review = review_file(jf)
        if review:
            all_results.append(review)

    output_file = "review_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nReviews saved to {output_file}")

if __name__ == "__main__":
    main()
