import json
from pathlib import Path

import numpy as np


class ExportedEmbeddings:
    """Utility class for loading and using exported word embeddings."""

    def __init__(self, vectors_path, word_to_id_path, id_to_word_path):
        self.vectors_path = Path(vectors_path)
        self.word_to_id_path = Path(word_to_id_path)
        self.id_to_word_path = Path(id_to_word_path)

        self.vectors = np.load(self.vectors_path)

        with open(self.word_to_id_path, "r", encoding="utf-8") as f:
            self.word_to_id = json.load(f)

        with open(self.id_to_word_path, "r", encoding="utf-8") as f:
            raw_id_to_word = json.load(f)

        self.id_to_word = {int(k): v for k, v in raw_id_to_word.items()}

        if self.vectors.shape[0] != len(self.word_to_id):
            raise ValueError("vectors rows must match word_to_id size")

    def __contains__(self, word):
        return word in self.word_to_id

    def vector(self, word):
        idx = self.word_to_id[word]
        return self.vectors[idx]

    def cosine_similarity(self, word_a, word_b):
        va = self.vector(word_a)
        vb = self.vector(word_b)
        denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-12
        return float(np.dot(va, vb) / denom)

    def most_similar(self, query_word, top_k=10):
        if query_word not in self.word_to_id:
            return []

        qv = self.vector(query_word)
        norms = np.linalg.norm(self.vectors, axis=1) + 1e-12
        sims = (self.vectors @ qv) / (norms * (np.linalg.norm(qv) + 1e-12))

        best = np.argsort(-sims)[: top_k + 1]
        out = []
        for i in best:
            w = self.id_to_word[int(i)]
            if w == query_word:
                continue
            out.append((w, float(sims[i])))
            if len(out) == top_k:
                break
        return out

    def analogy(self, a, b, c, top_k=10):
        missing = [w for w in [a, b, c] if w not in self.word_to_id]
        if missing:
            return []

        target = self.vector(a) - self.vector(b) + self.vector(c)
        norms = np.linalg.norm(self.vectors, axis=1) + 1e-12
        sims = (self.vectors @ target) / (norms * (np.linalg.norm(target) + 1e-12))

        for w in [a, b, c]:
            sims[self.word_to_id[w]] = -1e9

        best = np.argsort(-sims)[:top_k]
        return [(self.id_to_word[int(i)], float(sims[i])) for i in best]
