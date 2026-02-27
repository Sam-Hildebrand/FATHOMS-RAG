import json
import os
import numpy as np
from typing import List
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

class HallucinationDetector:
    def __init__(self, data_file: str):
        # Load sentence transformer
        self.model = SentenceTransformer("tomaarsen/static-retrieval-mrl-en-v1")
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Prepare storage
        self.texts: List[str] = []
        self.labels: List[str] = []  # parallel to texts
        self.embeddings: np.ndarray = None
        self.nn: NearestNeighbors = None

        if data_file and os.path.exists(data_file):
            self._load_training_data(data_file)
        else:
            raise ValueError(f"Training JSON file not found: {data_file}")

    def _embed(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], convert_to_numpy=True)[0]
        norm = np.linalg.norm(vec)
        if norm == 0.0 or np.isnan(norm):
            # fallback: return zero vector or small epsilon-normalized
            return np.zeros_like(vec)
        return vec / norm


    def _load_training_data(self, data_file: str):
        with open(data_file, 'r') as f:
            data = json.load(f)

        for label, samples in data.items():
            for s in samples:
                self.texts.append(s)
                self.labels.append(label)

        # Compute embeddings and normalize
        self.embeddings = np.array([self._embed(t) for t in self.texts])

        # Fit nearest neighbor index
        self.nn = NearestNeighbors(n_neighbors=1, metric='cosine')
        self.nn.fit(self.embeddings)

    def classify(self, text: str) -> str:
        vec = self._embed(text).reshape(1, -1)
        distance, idx = self.nn.kneighbors(vec, n_neighbors=1)
        label = self.labels[idx[0][0]]
        return label  # always 'statement' or 'abstention'

    def is_hallucinating(self, pred_answer: str, score: float) -> bool:
        """
        Hallucination = predicted answer classified as 'statement' but score != 1.0
        """
        if pred_answer == "":
            return False
        classification = self.classify(pred_answer)
        confident = classification == "statement"
        return confident and (score != 1.0)
