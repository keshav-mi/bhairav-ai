# ============================================================
# BHAIRAV AI — RERANKER v4
# ============================================================
# KEY CHANGES vs v3:
#   - Reranks on COMBINED text: hindi_summary + text
#     (v3 only used hindi_summary, missing Sanskrit text matches)
#   - RERANK_SCORE_FLOOR drops obvious noise chunks before LLM
#   - Logs score distribution for debugging
# ============================================================

from sentence_transformers import CrossEncoder
from typing import List, Dict

from config import RERANKER_MODEL, RERANK_TOP_N, RERANK_SCORE_FLOOR


class Reranker:
    def __init__(self):
        print(f" Loading reranker: {RERANKER_MODEL}")
        self.model = CrossEncoder(RERANKER_MODEL)
        print(" Reranker ready")

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_n: int = RERANK_TOP_N,
    ) -> List[Dict]:
        """
        Cross-encoder reranking.

        Scores (query, passage) pairs jointly — far more accurate
        than the bi-encoder FAISS scores.

        Passage = hindi_summary + Sanskrit text (if present).
        This gives the cross-encoder both the conceptual summary
        and the original Sanskrit tokens for matching.

        SCORE FLOOR: Chunks below RERANK_SCORE_FLOOR are dropped
        as clear noise. This prevents the LLM from being confused
        by irrelevant context even when top_n is large.
        """
        if not chunks:
            return []

        # Build rich passage for reranking
        pairs = []
        for chunk in chunks:
            summary = chunk.get("hindi_summary", "") or ""
            text    = chunk.get("text", "") or ""
            passage = f"{summary}\n{text}".strip()
            pairs.append((query, passage))

        scores = self.model.predict(pairs, show_progress_bar=False)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        # Sort descending
        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

        # Debug: log top-5 scores
        top5_scores = [round(c["rerank_score"], 3) for c in reranked[:5]]
        print(f"   Reranker top-5 scores: {top5_scores}")

        # Apply score floor — drop clear noise
        filtered = [c for c in reranked if c["rerank_score"] >= RERANK_SCORE_FLOOR]

        if not filtered:
            # If everything is below floor, return top chunk anyway
            print(f"   All chunks below floor ({RERANK_SCORE_FLOOR}), returning top-1")
            return reranked[:1]

        return filtered[:top_n]
