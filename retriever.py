# ============================================================
# BHAIRAV AI — RETRIEVER v4 (FIXED)
# ============================================================
import json
import pickle
import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple

from config import (
    FAISS_PATH, BM25_PATH, METADATA_PATH, ID_MAP_PATH,
    EMBEDDING_MODEL, FAISS_TOP_K, BM25_TOP_K, RRF_K,
    DOMAIN_BOOST, ENTITY_BOOST, GROQ_API_KEY,
)
from query_expander import expand_query
from query_normalizer import QueryNormalizer

class Retriever:
    def __init__(self):
        print(" Loading retriever components...")
        
        # 1. Initialize the new Normalizer (CRITICAL FIX)
        self.normalizer = QueryNormalizer() 

        # 2. Initialize Embedding Model
        print(f"   Embedding model : {EMBEDDING_MODEL}")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        self.embed_model.max_seq_length = 512

        # 3. Load FAISS index
        print("   FAISS index...")
        self.faiss_index = faiss.read_index(FAISS_PATH)
        print(f"   FAISS vectors   : {self.faiss_index.ntotal:,}")

        # 4. Load BM25 index
        print("   BM25 index...")
        with open(BM25_PATH, "rb") as f:
            self.bm25 = pickle.load(f)

        # 5. Load Metadata store
        print("   Metadata store...")
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # 6. Load ID map
        print("   ID map...")
        with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)

        # FIX: Safe integer casting for the reverse map
        self.chunk_ids = [self.id_map[str(i)] for i in range(len(self.id_map))]
        self.rev_id_map = {}
        for k, v in self.id_map.items():
            try:
                self.rev_id_map[v] = int(k)
            except ValueError:
                pass # Skips non-sequential/corrupt keys silently
        
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        print(f" Retriever ready — {len(self.metadata):,} chunks")
        
        
    def get_neighbor_chunks(self, ranked_ids: List[Tuple[str, float]], window: int = 1) -> List[Dict]:
        """Bridges information gaps by pulling surrounding verses."""
        expanded_chunks = []
        seen_ids = set()

        for cid, score in ranked_ids:
            try:
                # Use pre-computed reverse map for speed
                current_idx = self.rev_id_map.get(cid)
                if current_idx is None: continue
            
                for i in range(current_idx - window, current_idx + window + 1):
                    neighbor_cid = self.id_map.get(str(i))
                    if neighbor_cid and neighbor_cid not in seen_ids:
                        if neighbor_cid in self.metadata:
                            chunk = dict(self.metadata[neighbor_cid])
                            chunk["score"] = score 
                            expanded_chunks.append(chunk)
                            seen_ids.add(neighbor_cid)
            except Exception:
                continue

        return expanded_chunks

    def embed_query(self, query: str) -> np.ndarray:
        vector = self.embed_model.encode(
            [query.strip().lower()],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector.astype(np.float32)

    def faiss_search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        scores, positions = self.faiss_index.search(query_vector, top_k)
        results = []
        for score, pos in zip(scores[0], positions[0]):
            if pos != -1:
                cid = self.id_map[str(pos)]
                results.append((cid, float(score)))
        return results

    def bm25_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.chunk_ids[idx], float(scores[idx])))
        return results

    def reciprocal_rank_fusion(
        self,
        faiss_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        boosted_sources: List[str],
        k: int = RRF_K,
    ) -> List[Tuple[str, float]]:
        rrf_scores: Dict[str, float] = {}
        for rank, (cid, _) in enumerate(faiss_results):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        for rank, (cid, _) in enumerate(bm25_results):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

        if boosted_sources:
            for cid in rrf_scores:
                source = self.metadata.get(cid, {}).get("source", "")
                if source in boosted_sources:
                    rrf_scores[cid] *= DOMAIN_BOOST
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(
        self, query: str, top_k: int = FAISS_TOP_K
    ) -> Tuple[List[Dict], str, str, List[str]]:
        
        # 1. Normalize the query (Gemini + Wikidata)
        norm_result = self.normalizer.normalize(query)
        
        # 2. Expand the query (IndicXlit + Groq + Domain extraction)
        exp_faiss, exp_bm25, domains = expand_query(query, self.groq_client) 

        # 3. FIX: Merge and deduplicate words from both pipelines
        combined_faiss = f"{norm_result['augmented']} {exp_faiss}"
        faiss_query = " ".join(dict.fromkeys(combined_faiss.split()))
        
        combined_bm25 = f"{norm_result['augmented']} {exp_bm25}"
        bm25_query = " ".join(dict.fromkeys(combined_bm25.split()))

        print(f"   Original    : {query}")
        print(f"   Augmented   : {faiss_query[:120]}...")

        # 4. Search
        query_vector = self.embed_query(faiss_query)
        faiss_results = self.faiss_search(query_vector, top_k)
        bm25_results  = self.bm25_search(bm25_query, top_k)

        # 5. Merge and Expand context
        ranked_ids = self.reciprocal_rank_fusion(faiss_results, bm25_results, domains)
        chunks = self.get_neighbor_chunks(ranked_ids, window=0)

        return chunks, faiss_query, bm25_query, domains