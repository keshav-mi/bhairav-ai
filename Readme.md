# Bhairav AI — Multilingual Retrieval-Augmented Generation System

Bhairav AI is a **production-oriented Retrieval-Augmented Generation (RAG) system** designed to answer queries grounded in **Dharmic primary texts** such as the Vedas, Mahabharata, Ramayana, and Bhagavad Gita.

The system combines **hybrid search, multilingual query understanding, and LLM-based reasoning** to generate **accurate, citation-backed responses**.

---

## Overview

This project focuses on building a **real-world AI system**, not just isolated models.
It integrates retrieval, ranking, and generation into a **modular backend architecture**.

Key capabilities include:

* Hybrid retrieval (BM25 + FAISS)
* Multilingual query handling (Hindi, English, Hinglish)
* Entity-aware query normalization and expansion
* Cross-encoder reranking for precision
* Structured, citation-grounded answer generation

---

## System Architecture

```
User Query
   ↓
Query Normalization (Entity extraction + Wikidata + Transliteration)
   ↓
Query Expansion (Synonyms + IndicXlit + LLM-assisted expansion)
   ↓
Hybrid Retrieval
   ├── BM25 (lexical search)
   └── FAISS (semantic search)
   ↓
Reciprocal Rank Fusion (RRF)
   ↓
Cross-Encoder Reranker
   ↓
LLM Generation (Groq)
   ↓
Answer + Citations
```

---

## Tech Stack

**Backend**

* FastAPI
* Python

**Machine Learning / Retrieval**

* FAISS (vector search)
* BM25 (rank-bm25)
* SentenceTransformers
* CrossEncoder (BGE reranker)

**LLM & External Systems**

* Groq (LLaMA-based inference)
* Wikidata API
* Indic Transliteration
* Monier-Williams Sanskrit Lexicon

---

## Project Structure

```
bhairav-ai/
│
├── main.py                # FastAPI application
├── config.py              # Configuration and model setup
├── retriever.py           # Hybrid retrieval pipeline
├── reranker.py            # Cross-encoder reranking
├── generator.py           # LLM-based answer generation
├── query_expander.py      # Query expansion logic
├── query_normalizer.py    # Entity normalization pipeline
├── canonical_maps.py      # Canonical naming system
│
├── requirements.txt
├── .gitignore
```

---

## Setup

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file:

```
GROQ_API_KEY=your_key
GEMINI_API_KEY=your_key
APP_API_KEY=your_key
ALLOWED_ORIGINS=http://localhost:3000
```

### 3. Run the server

```
uvicorn main:app --reload
```

---

## API

### POST /query

Headers:

```
x-api-key: <your_api_key>
```

Body:

```json
{
  "query": "arjun aur krishna ka sambandh kya tha",
  "top_k": 10,
  "include_citations": true
}
```

---

## Security Considerations

* Environment-based secret management
* API key authentication
* Rate limiting (10 requests/minute)
* Input validation and prompt filtering

---

## Project Status

**This project is actively under development.**

The current repository represents a **public showcase version** of the system architecture and core pipeline.

The following components are intentionally excluded:

* Dataset / text corpus
* FAISS and BM25 indexes
* Production optimizations and internal tooling

This repository is designed to demonstrate:

* System design
* Retrieval pipeline
* Backend architecture

---

## Notes

To run the system end-to-end, additional components (data and indexes) are required, which are not part of this repository.

---

## Author

Keshav Mishra
Computer Science Student
