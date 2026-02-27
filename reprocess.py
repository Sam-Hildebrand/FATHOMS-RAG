#!/usr/bin/env python3
"""
Reprocess old evaluation JSON files to add hallucination detection data and rescore questions.
Takes one or more JSON files as command line arguments and updates them with:
- hallucinating field for each response
- hallucination rates for each category
- total_hallucination_rate overall metric
- rescored correctness_score using current ground truth from data/questions_answers.json
- updated ground_truth from current questions file
- pred_answer normalized to lowercase

Usage: python reprocess.py file1.json file2.json ...
"""

import sys
import json
import numpy as np
from typing import Dict, List, Optional, Any
from evaluation.evaluator import load_questions

from evaluation.hallucination import HallucinationDetector
detector = HallucinationDetector(data_file="evaluation/statement_abstention_training.json")


def score_answer(pred_answer: str, ground_truth_patterns: List[List[str]]) -> float:
    """Score a predicted answer against ground truth patterns using simple keyword overlap."""
    if pred_answer in ['TIMED OUT', 'timed out', 'rate limit exceeded'] or pred_answer.startswith('error'):
        return 0.0

    best_score = 0.0
    for pattern in ground_truth_patterns:
        if not pattern:
            continue
        toks = [t.lower() for t in pattern]
        score = sum(t in pred_answer.lower().replace("-", " ") for t in toks) / len(toks)
        best_score = max(best_score, score)

    return best_score


def find_question_data(question: str, required_files: List[str], qa_data: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Find the question data in the loaded QA dataset by matching question text and required files."""
    for category, questions in qa_data.items():
        for q_data in questions:
            if (q_data.get('question', '').strip() == question.strip() and
                q_data.get('files', []) == required_files):
                return q_data
    return None


def reprocess_file(filename: str, qa_data: Dict[str, List[Dict[str, Any]]]) -> None:
    print(f"Processing {filename}...")

    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {filename}: {e}")
        return

    responses = data.get('responses', [])
    if not responses:
        print(f"  No responses found in {filename}")
        return

    # Process each response
    category_hallucinations: Dict[str, List[bool]] = {}
    category_scores: Dict[str, List[float]] = {}

    for response in responses:
        pred_answer = response.get('pred_answer', '')
        # Normalize pred_answer to lowercase
        response['pred_answer'] = pred_answer.lower()
        pred_answer = response['pred_answer']

        question = response.get('question', '')
        required_files = response.get('required_files', [])
        category = response.get('category', 'Unknown')

        # Ensure rag_context is always present
        if 'rag_context' not in response:
            response['rag_context'] = []

        # Find current ground truth
        current_q_data = find_question_data(question, required_files, qa_data)
        if current_q_data:
            response['ground_truth'] = current_q_data.get('answers', response.get('ground_truth', []))
            print(f"  Updated ground truth for: {question[:50]}...")
        else:
            print(f"  Warning: Could not find current data for question: {question[:50]}...")

        ground_truth = response.get('ground_truth', [])

        # Rescore
        new_score = score_answer(pred_answer, ground_truth)
        old_score = response.get('correctness_score', 0.0)
        response['correctness_score'] = new_score

        if abs(new_score - old_score) > 0.001:
            print(f"  Rescored: {question[:50]}... | Old: {old_score:.3f} -> New: {new_score:.3f}")

        # Detect hallucination
        hallucinating = detector.is_hallucinating(pred_answer, new_score)
        response['hallucinating'] = hallucinating
        if 'suspected_hallucination' in response:
            del response['suspected_hallucination']

        # Track by category
        category_hallucinations.setdefault(category, []).append(hallucinating)
        category_scores.setdefault(category, []).append(new_score)

    # Aggregate per-category
    updated_scores = {}
    hallucination_rates = []

    for category in category_hallucinations:
        hallucinations = category_hallucinations[category]
        scores = category_scores[category]

        if hallucinations and scores:
            avg_score = sum(scores) / len(scores)
            rate = sum(hallucinations) / len(hallucinations)

            updated_scores[category] = avg_score
            updated_scores[f"{category} Hallucination Rate"] = rate
            hallucination_rates.append(rate)

            print(f"  {category}: {avg_score:.3f} avg score, {rate:.3f} hallucination rate")

    # Overall metrics
    category_avg_scores = [v for k, v in updated_scores.items() if not k.endswith(" Hallucination Rate")]
    overall_score = float(np.mean(category_avg_scores)) if category_avg_scores else 0.0
    overall_hallucination_rate = float(np.mean(hallucination_rates)) if hallucination_rates else 0.0

    data['scores'] = updated_scores
    data['overall_score'] = overall_score
    data['total_hallucination_rate'] = overall_hallucination_rate

    if 'overall' in data:
        del data['overall']

    print(f"  Overall score: {overall_score:.3f}")
    print(f"  Overall hallucination rate: {overall_hallucination_rate:.3f}")

    # Save
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Updated {filename} successfully")
    except Exception as e:
        print(f"  Error writing {filename}: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python reprocess.py file1.json file2.json ...")
        sys.exit(1)

    json_files = sys.argv[1:]
    questions_file = "data/questions_answers.json"

    try:
        qa_data = load_questions(questions_file)
        print(f"Loaded questions from {questions_file}")
        print(f"Found {sum(len(qlist) for qlist in qa_data.values())} total questions across {len(qa_data)} categories\n")
    except Exception as e:
        print(f"Error: Could not load questions file {questions_file}: {e}")
        sys.exit(1)

    print(f"Reprocessing {len(json_files)} file(s)...\n")
    for filename in json_files:
        reprocess_file(filename, qa_data)
        print()

    print("Reprocessing complete!")


if __name__ == "__main__":
    main()
