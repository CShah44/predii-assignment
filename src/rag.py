import argparse
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import chromadb
import pymupdf4llm
from sentence_transformers import SentenceTransformer

from llm_extract import extract_structured_specs


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
NUMBERED_STEP_RE = re.compile(r"^\s*\d+[\.)]\s+")
BULLET_RE = re.compile(r"^\s*[-*]\s+")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_DIVIDER_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WORKSHOP_PAGE_RE = re.compile(r"workshop manual\s+page\s+\d+\s+sur\s+\d+", re.IGNORECASE)
BOLD_LINE_RE = re.compile(r"^\*\*([^*]{3,120})\*\*$")


@dataclass
class ChunkConfig:
    target_tokens: int = 800
    max_tokens: int = 1000
    hard_max_tokens: int = 1200
    overlap_tokens: int = 120
    min_tokens: int = 200


class LocalEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> List[List[float]]:
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return vectors.tolist()


def approx_token_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def bm25_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def compute_bm25_scores(query: str, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    tokenized_docs = [bm25_tokenize(doc) for doc in documents]
    doc_count = len(tokenized_docs)
    if doc_count == 0:
        return []

    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    avgdl = sum(doc_lengths) / doc_count if doc_count else 0.0
    query_terms = bm25_tokenize(query)
    if not query_terms:
        return [0.0] * doc_count

    doc_freq: Dict[str, int] = {}
    for term in set(query_terms):
        doc_freq[term] = sum(1 for tokens in tokenized_docs if term in set(tokens))

    scores: List[float] = []
    for tokens, dl in zip(tokenized_docs, doc_lengths):
        tf_counter = Counter(tokens)
        score = 0.0
        for term in query_terms:
            n_qi = doc_freq.get(term, 0)
            if n_qi == 0:
                continue
            idf = max(0.0, (doc_count - n_qi + 0.5) / (n_qi + 0.5))
            idf = math.log(1.0 + idf)
            tf = tf_counter.get(term, 0)
            if tf == 0:
                continue
            denom = tf + k1 * (1.0 - b + b * (dl / avgdl if avgdl > 0 else 0.0))
            score += idf * ((tf * (k1 + 1.0)) / denom)
        scores.append(score)

    return scores


def tail_words(text: str, token_budget: int) -> str:
    words = text.split()
    if not words:
        return ""
    return " ".join(words[-token_budget:])


def stable_chunk_id(source: str, index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{Path(source).stem}-{index}-{digest}"


def sanitize_markdown(text: str) -> str:
    out_lines: List[str] = []

    for raw in text.splitlines():
        line = raw.strip().replace("\ufffd", "-")
        if not line:
            out_lines.append("")
            continue
        if "file:///" in line.lower() or "repair4less" in line.lower():
            continue
        if WORKSHOP_PAGE_RE.search(line):
            continue

        de_emph = line.replace("*", "").strip()
        if de_emph.upper().startswith("SECTION "):
            out_lines.append(f"# {de_emph}")
            continue

        bold_match = BOLD_LINE_RE.match(line)
        if bold_match:
            title = bold_match.group(1).strip()
            title_tokens = title.split()
            if 1 <= len(title_tokens) <= 8 and not title.lower().startswith("note"):
                out_lines.append(f"## {title}")
                continue

        out_lines.append(line)

    compressed: List[str] = []
    previous_blank = False
    for line in out_lines:
        is_blank = line.strip() == ""
        if is_blank and previous_blank:
            continue
        compressed.append(line)
        previous_blank = is_blank
    return "\n".join(compressed).strip()


def strip_repeated_page_boilerplate(pages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not pages:
        return []

    head_counter: Counter[str] = Counter()
    tail_counter: Counter[str] = Counter()
    total = len(pages)

    for page in pages:
        lines = [ln.strip() for ln in (page.get("text") or "").splitlines() if ln.strip()]
        for ln in lines[:3]:
            head_counter[ln] += 1
        for ln in lines[-3:]:
            tail_counter[ln] += 1

    min_hits = max(3, int(total * 0.6))
    repeated_heads = {k for k, v in head_counter.items() if v >= min_hits}
    repeated_tails = {k for k, v in tail_counter.items() if v >= min_hits}

    cleaned: List[Dict[str, Any]] = []
    for page in pages:
        lines = (page.get("text") or "").splitlines()
        if not lines:
            cleaned.append(page)
            continue

        start = 0
        end = len(lines)
        while start < end and lines[start].strip() in repeated_heads:
            start += 1
        while end > start and lines[end - 1].strip() in repeated_tails:
            end -= 1

        new_page = dict(page)
        new_page["text"] = sanitize_markdown("\n".join(lines[start:end]))
        cleaned.append(new_page)

    return cleaned


def extract_pages(pdf_path: Path) -> List[Dict[str, Any]]:
    page_chunks = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
        ignore_images=True,
        table_strategy="lines_strict",
        show_progress=False,
    )
    if isinstance(page_chunks, str):
        return [{"metadata": {"page": 1}, "text": sanitize_markdown(page_chunks)}]
    return strip_repeated_page_boilerplate(page_chunks)


def blockify(markdown_text: str) -> List[Dict[str, str]]:
    lines = [line.rstrip() for line in markdown_text.splitlines()]
    blocks: List[Dict[str, str]] = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        heading_match = HEADING_RE.match(line)
        if heading_match:
            blocks.append({"type": "heading", "text": line})
            i += 1
            continue

        if TABLE_ROW_RE.match(line):
            table_lines = [lines[i]]
            i += 1
            while i < len(lines) and TABLE_ROW_RE.match(lines[i].strip()):
                table_lines.append(lines[i])
                i += 1
            if len(table_lines) >= 2 and any(TABLE_DIVIDER_RE.match(t.strip()) for t in table_lines):
                blocks.append({"type": "table", "text": "\n".join(table_lines).strip()})
            else:
                blocks.append({"type": "paragraph", "text": "\n".join(table_lines).strip()})
            continue

        if NUMBERED_STEP_RE.match(line) or BULLET_RE.match(line):
            list_lines = [lines[i]]
            i += 1
            while i < len(lines):
                nxt = lines[i].rstrip()
                stripped = nxt.strip()
                if not stripped:
                    if i + 1 < len(lines) and (NUMBERED_STEP_RE.match(lines[i + 1].strip()) or BULLET_RE.match(lines[i + 1].strip())):
                        list_lines.append("")
                        i += 1
                        continue
                    break
                if NUMBERED_STEP_RE.match(stripped) or BULLET_RE.match(stripped) or nxt.startswith(" "):
                    list_lines.append(nxt)
                    i += 1
                    continue
                break
            blocks.append({"type": "procedure", "text": "\n".join(list_lines).strip()})
            continue

        para_lines = [lines[i]]
        i += 1
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped:
                break
            if HEADING_RE.match(stripped) or TABLE_ROW_RE.match(stripped) or NUMBERED_STEP_RE.match(stripped) or BULLET_RE.match(stripped):
                break
            para_lines.append(lines[i])
            i += 1
        blocks.append({"type": "paragraph", "text": "\n".join(para_lines).strip()})

    return blocks


def split_large_text(text: str, max_tokens: int) -> List[str]:
    words = text.split()
    if len(words) <= max_tokens:
        return [text]
    parts: List[str] = []
    start = 0
    step = max(200, int(max_tokens * 0.8))
    while start < len(words):
        end = min(len(words), start + max_tokens)
        parts.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return parts


def get_default_flags() -> Dict[str, bool]:
    return {
        "contains_procedure": False,
        "contains_table": False,
        "contains_warning": False,
        "contains_torque_specs": False,
        "contains_materials": False,
    }


class ChunkBuilder:
    def __init__(self, source: str, config: ChunkConfig):
        self.source = source
        self.config = config
        self.chunks: List[Dict[str, Any]] = []
        self.heading_path: List[str] = []
        
        self.current_parts: List[str] = []
        self.current_tokens = 0
        self.current_start_page = None
        self.current_end_page = None
        self.current_flags = get_default_flags()
        self.last_chunk_body = ""
        self.last_heading_path = ""

    def flush(self) -> None:
        if not self.current_parts:
            return

        body = "\n\n".join(self.current_parts).strip()
        if approx_token_count(body) < self.config.min_tokens and self.chunks:
            self.chunks[-1]["body"] = self.chunks[-1]["body"] + "\n\n" + body
            self.chunks[-1]["text"] = self.chunks[-1]["text"] + "\n\n" + body
            self.chunks[-1]["page_end"] = self.current_end_page
            for key, val in self.current_flags.items():
                self.chunks[-1][key] = self.chunks[-1].get(key, False) or val
        else:
            section_path = " > ".join(self.heading_path).strip()
            prefix = f"Section: {section_path}\n\n" if section_path else ""
            text_for_embedding = prefix + body
            chunk_index = len(self.chunks)
            chunk = {
                "id": stable_chunk_id(self.source, chunk_index, text_for_embedding),
                "source": self.source,
                "chunk_index": chunk_index,
                "section_path": section_path,
                "page_start": self.current_start_page,
                "page_end": self.current_end_page,
                "body": body,
                "text": text_for_embedding,
                **self.current_flags,
            }
            self.chunks.append(chunk)
            self.last_chunk_body = body
            self.last_heading_path = section_path

        self.current_parts = []
        self.current_tokens = 0
        self.current_start_page = None
        self.current_end_page = None
        self.current_flags = get_default_flags()


def build_chunks(pages: Sequence[Dict[str, Any]], source: str, config: ChunkConfig) -> List[Dict[str, Any]]:
    builder = ChunkBuilder(source, config)

    for page in pages:
        page_number = page.get("metadata", {}).get("page") or page.get("metadata", {}).get("page_number")
        blocks = blockify(page.get("text", ""))

        for block in blocks:
            btype = block["type"]
            btext = block["text"].strip()
            if not btext:
                continue

            if btype == "heading":
                heading_match = HEADING_RE.match(btext)
                if heading_match:
                    builder.flush()
                    level = len(heading_match.group(1))
                    title = heading_match.group(2).strip()
                    while len(builder.heading_path) >= level:
                        builder.heading_path.pop()
                    builder.heading_path.append(title)
                continue

            btoken = approx_token_count(btext)
            if btoken > config.hard_max_tokens:
                split_parts = split_large_text(btext, config.max_tokens)
            else:
                split_parts = [btext]

            for part in split_parts:
                part_tokens = approx_token_count(part)
                if builder.current_tokens and builder.current_tokens + part_tokens > config.target_tokens:
                    builder.flush()
                    if builder.last_chunk_body and " > ".join(builder.heading_path) == builder.last_heading_path and config.overlap_tokens > 0:
                        overlap = tail_words(builder.last_chunk_body, config.overlap_tokens)
                        if overlap:
                            builder.current_parts.append(f"Context overlap: {overlap}")
                            builder.current_tokens += approx_token_count(overlap)
                if builder.current_start_page is None:
                    builder.current_start_page = page_number
                builder.current_end_page = page_number
                builder.current_parts.append(part)
                builder.current_tokens += part_tokens
                
                lowered = part.lower()
                builder.current_flags["contains_procedure"] |= (btype == "procedure")
                builder.current_flags["contains_table"] |= (btype == "table")
                builder.current_flags["contains_warning"] |= any(w in lowered for w in ("warning", "caution"))
                builder.current_flags["contains_torque_specs"] |= any(w in lowered for w in ("torque", "lb-ft", "nm"))
                builder.current_flags["contains_materials"] |= any(w in lowered for w in ("material", "lubricant", "sealant"))

    builder.flush()
    return builder.chunks


def write_chunks_jsonl(chunks: Iterable[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=True) + "\n")


def ingest(
    pdf_path: Path,
    db_path: Path,
    collection_name: str,
    model_name: str,
    chunks_out: Path,
) -> None:
    pages = extract_pages(pdf_path)
    config = ChunkConfig()
    chunks = build_chunks(pages, source=str(pdf_path), config=config)

    if not chunks:
        raise RuntimeError("No chunks were generated from the PDF.")

    write_chunks_jsonl(chunks, chunks_out)

    embedder = LocalEmbedder(model_name)
    vectors = embedder.encode([c["text"] for c in chunks], batch_size=64)

    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    collection.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "section_path": c["section_path"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "contains_procedure": c["contains_procedure"],
                "contains_table": c["contains_table"],
                "contains_warning": c["contains_warning"],
                "contains_torque_specs": c["contains_torque_specs"],
                "contains_materials": c["contains_materials"],
            }
            for c in chunks
        ],
    )

    print(f"Indexed {len(chunks)} chunks into collection '{collection_name}'.")
    print(f"Chunks file: {chunks_out}")
    print(f"Chroma DB path: {db_path}")


def detect_intent(query: str) -> Dict[str, bool]:
    q = query.lower()
    return {
        "procedure": any(k in q for k in ["how", "step", "procedure", "remove", "install", "adjust"]),
        "torque": any(k in q for k in ["torque", "nm", "lb-ft", "lb in", "tighten"]),
        "materials": any(k in q for k in ["material", "fluid", "lubricant", "sealant", "cleaner"]),
    }


def rerank(query: str, rows: List[Dict[str, Any]], bm25_weight: float = 0.35) -> List[Dict[str, Any]]:
    intent = detect_intent(query)
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    query_normalized = " ".join(re.findall(r"[a-z0-9]+", query.lower()))

    for row in rows:
        dense_score = 1.0 - float(row["distance"])
        bm25_score = float(row.get("bm25_score", 0.0))
        base = (1.0 - bm25_weight) * dense_score + bm25_weight * bm25_score
        bonus = 0.0
        meta = row.get("metadata") or {}
        section_terms = set(re.findall(r"[a-z0-9]+", (meta.get("section_path") or "").lower()))
        overlap = len(query_terms & section_terms)
        bonus += min(0.08, overlap * 0.02)

        doc_normalized = " ".join(re.findall(r"[a-z0-9]+", (row.get("document") or "").lower()))
        doc_terms = set(doc_normalized.split())
        lexical_overlap = len(query_terms & doc_terms)
        bonus += min(0.2, lexical_overlap * 0.01)
        if query_normalized and len(query_normalized.split()) >= 2 and query_normalized in doc_normalized:
            bonus += 0.2

        if intent["procedure"] and meta.get("contains_procedure"):
            bonus += 0.08
        if intent["torque"] and meta.get("contains_torque_specs"):
            bonus += 0.1
        if intent["torque"] and meta.get("contains_table"):
            bonus += 0.15
        if intent["torque"] and "torque specifications" in (meta.get("section_path") or "").lower():
            bonus += 0.35
        if intent["materials"] and meta.get("contains_materials"):
            bonus += 0.1

        row["score"] = base + bonus

    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows


def query_collection(
    db_path: Path,
    collection_name: str,
    model_name: str,
    query_text: str,
    top_k: int,
    bm25_weight: float = 0.35,
    llm_structured: bool = False,
    ollama_model: str = "qwen2.5:7b",
    ollama_url: str = "http://localhost:11434",
    ollama_timeout: int = 180,
) -> None:
    embedder = LocalEmbedder(model_name)
    qvec = embedder.encode([query_text], batch_size=1)[0]

    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(name=collection_name)

    raw = collection.query(
        query_embeddings=[qvec],
        n_results=max(top_k * 10, 30),
        include=["documents", "metadatas", "distances"],
    )

    rows: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0]):
        rows.append({"document": doc, "metadata": meta, "distance": dist})

    bm25_raw_scores = compute_bm25_scores(query_text, [row["document"] for row in rows])
    max_bm25 = max(bm25_raw_scores) if bm25_raw_scores else 0.0
    for row, raw_score in zip(rows, bm25_raw_scores):
        row["bm25_score"] = (raw_score / max_bm25) if max_bm25 > 0 else 0.0

    intent = detect_intent(query_text)
    if intent["torque"]:
        torque_rows = [
            row
            for row in rows
            if (row.get("metadata") or {}).get("contains_table")
            or "torque" in ((row.get("metadata") or {}).get("section_path") or "").lower()
            or "|description|nm|" in (row.get("document") or "").lower()
        ]
        if torque_rows:
            rows = torque_rows
    if intent["materials"]:
        material_rows = [
            row for row in rows if (row.get("metadata") or {}).get("contains_materials")
        ]
        if material_rows:
            rows = material_rows

    ranked = rerank(query_text, rows, bm25_weight=bm25_weight)[:top_k]

    if not ranked:
        print("No results.")
        return

    if llm_structured:
        contexts = [row["document"] for row in ranked]
        try:
            structured = extract_structured_specs(
                query=query_text,
                contexts=contexts,
                model=ollama_model,
                base_url=ollama_url,
                timeout_seconds=ollama_timeout
            )
            print(structured)
        except Exception as exc:
            print(f"Structured extraction failed: {exc}")
        return

    for idx, row in enumerate(ranked, start=1):
        meta = row.get("metadata") or {}
        print("=" * 80)
        print(f"Result {idx} | score={row['score']:.4f} | distance={float(row['distance']):.4f}")
        print(f"Section: {meta.get('section_path') or 'N/A'}")
        print(f"Pages: {meta.get('page_start')} - {meta.get('page_end')}")
        print(row["document"][:1200])
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal PyMuPDF + Chroma RAG pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Extract, chunk, embed, and index a PDF")
    ingest_parser.add_argument("--pdf", required=True, type=Path, help="Path to source PDF")
    ingest_parser.add_argument("--db", default=Path("data/processed/chroma"), type=Path, help="Chroma persistence directory")
    ingest_parser.add_argument("--collection", default="manual", help="Chroma collection name")
    ingest_parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    ingest_parser.add_argument(
        "--chunks-out",
        default=Path("data/processed/chunks.jsonl"),
        type=Path,
        help="Path to write generated chunks as JSONL",
    )

    query_parser = subparsers.add_parser("query", help="Query existing Chroma index")
    query_parser.add_argument("--db", default=Path("data/processed/chroma"), type=Path, help="Chroma persistence directory")
    query_parser.add_argument("--collection", default="manual", help="Chroma collection name")
    query_parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    query_parser.add_argument("--q", required=True, help="Query string")
    query_parser.add_argument("--k", default=5, type=int, help="Number of final results")
    query_parser.add_argument(
        "--bm25-weight",
        default=0.35,
        type=float,
        help="Hybrid blend weight for BM25 lexical scoring (0.0 to 1.0)",
    )
    query_parser.add_argument(
        "--llm-structured",
        action="store_true",
        help="Use direct Ollama call to return structured extraction from retrieved context",
    )
    query_parser.add_argument(
        "--ollama-model",
        default="qwen2.5:7b",
        help="Ollama model name for structured extraction",
    )
    query_parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama base URL",
    )
    query_parser.add_argument(
        "--ollama-timeout",
        default=600,
        type=int,
        help="Timeout in seconds for Ollama response",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "ingest":
        ingest(
            pdf_path=args.pdf,
            db_path=args.db,
            collection_name=args.collection,
            model_name=args.model,
            chunks_out=args.chunks_out,
        )
        return

    if args.command == "query":
        query_collection(
            db_path=args.db,
            collection_name=args.collection,
            model_name=args.model,
            query_text=args.q,
            top_k=args.k,
            bm25_weight=max(0.0, min(1.0, args.bm25_weight)),
            llm_structured=args.llm_structured,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
            ollama_timeout=args.ollama_timeout,
        )


if __name__ == "__main__":
    main()
