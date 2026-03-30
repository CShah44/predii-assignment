# RAG Pipeline (PyMuPDF + Chroma)

This project provides a local RAG ingestion and retrieval pipeline for instruction manuals.

## 1) Clone & Install

```bash
git clone <your-repo-url>
cd predii-assignment
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
pip install -r requirements.txt
```

## 2) Prepare Data & Ingest

First, create the required data directories and place your target PDF manual inside:

```bash
mkdir -p data/raw
# Copy or move your PDF manual into data/raw/manual.pdf
```

Then, run the ingestion script to process the PDF and populate the vector database:

```bash
python src/rag.py ingest --pdf data/raw/manual.pdf --db data/processed/chroma --collection manual --chunks-out data/processed/chunks.jsonl
```

## 3) Query

```bash
python src/rag.py query --db data/processed/chroma --collection manual --q "torque for tie-rod end nut" --k 5
```

## 4) Query With Structured Ollama Output (JSON)

```bash
python src/rag.py query --db data/processed/chroma --collection manual --q "torque for tie-rod end nut" --k 5 --llm-structured --ollama-model "gemma3:4b"
```

Expected JSON shape:

```json
[
  {
    "action_or_component": "Brake Caliper Bolt",
    "description": "Torque specification",
    "value": "35",
    "unit": "Nm"
  }
]
```

## Retrieval Flow Architecture

1. **Semantic Embedding:** The incoming text query is converted into a dense vector representation using the same embedding model utilized during the document ingestion phase.
2. **Approximate Nearest Neighbors (ANN) Search:** `ChromaDB` leverages an ANN index to rapidly retrieve an initial pool of text chunks that are mathematically most similar in vector space, efficiently filtering out irrelevant context.
3. **Lexical Scoring (BM25):** The retrieved chunks are then evaluated using a custom BM25 algorithm to measure exact keyword overlaps. This ensures queries for highly specific terminology (e.g., part names or numbers) accurately surface appropriately matched text.
4. **Intent-Based Reranking:** The pipeline normalizes and merges the dense semantic similarity score with the sparse lexical (BM25) score. Additional domain-specific heuristics (e.g., detecting "torque" queries and preferring chunks containing numerical data or tables) further refine the final chunk weights (their score is boosted).
5. **LLM Synthesis:** The finalized, re-sorted chunks are truncated to the top `k` results (configurable via `--k`) and injected as context into the LLM's prompt for final structured JSON extraction.

## System Design & Tools Used

- **PDF Extraction**: `pymupdf4llm` to extract precise visual layout into markdown while keeping semantic flow and stripping redundant headers.
- **Chunking Strategy**: A semantic logic engine strips apart headers, paragraphs, lists, and tables without brute-force overlapping character splits, guaranteeing higher context boundaries.
- **Embeddings**: Local, offline execution utilizing `sentence-transformers/all-MiniLM-L6-v2`. Chosen for an excellent trade-off between semantic density and real-time compute speeds.
- **Vector Database**: `ChromaDB` was implemented as the persistent memory store handling raw text distance lookups.
- **Hybrid Retrieval System (BM25 + Semantic)**: Built from scratch, we use both vector searches directly with Chroma, joined immediately by a custom algorithm calculating BM25 frequencies across text strings.
- **Intent Analysis Mapping**: Regular expression metadata assigns weights to words based purely on expected output format ("procedure", "torque", "lubricant").
- **LLM Agent**: Leverages a local `Ollama` endpoint with `gemma3` or `qwen2.5`. Rather than unstructured blob text, we strictly bind the return output using the `<model>.api/generate(format: schema)` enforcement.

## Architectural Decisions

#### Why not Langchain?

Langchain is not used because this assignment was meant to demonstrate fundamental proficiency with embeddings, similarity metrics, layout chunking, and JSON enforcement. Langchain could be useful later strictly as an orchestration layer, saving duplicate function calls for things like map-reduce querying or prompt templates.

#### Limitation of Testing Data

Having only one example query (e.g., about torque specs) made it hard to fine-tune the system for all possible questions. Automotive manuals cover many different things, like step-by-step procedures, troubleshooting, and general part information. Because of this, the JSON schema was kept flexible. Instead of just returning rigid numerical specs, it can also return general steps and descriptions. This ensures the code doesn't fail when someone asks a "how-to" question instead of a direct "what is the value" question.

## Ideas for Improvement

- **Framework Integration**: Using a framework like LlamaIndex could be beneficial for building the pipeline, as it provides many out-of-the-box features for data ingestion, chunking, and retrieval indexing.
- **OCR Support**: Adding Optical Character Recognition (OCR) support would allow the pipeline to extract text from scanned PDFs or images within the manuals that are not natively text-selectable.
- **RAG Evaluation Pipeline**: Setting up a testing framework (eg. RAGAS) to evaluate the RAG pipeline with a variety of test queries would be a great next step. By scoring how accurate the search results are, we could scientifically find the sweet spot for the `--target-tokens` and `--overlap-tokens` variables. This would make sure chunks are perfectly sized to isolate correct data blocks without pulling in useless noise.
- **Dynamic Chunk Sizing**: Experimenting with dynamic chunk sizes based on the type of content (e.g., smaller chunks for dense tables, larger chunks for long procedures) could further improve search accuracy.
- **Better LLM Prompting**: Testing and providing more examples (few-shot prompting) directly in the LLM's system prompt could further guide the model to output even more consistent JSON data for edge-case queries.
