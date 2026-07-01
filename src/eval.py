"""
Evaluation Harness
──────────────────
Quantifies retrieval + answer quality against a labeled QA set.
Run: python eval.py --pdf path/to/doc.pdf --qa path/to/qa.json

qa.json format:
[
  {"question": "...", "expected_answer_contains": "...", "expected_page": 2}
]

Outputs: Recall@k, MRR, and a simple keyword-match answer accuracy,
printed as a table and saved to eval_results.json.
"""

from __future__ import annotations
import argparse
import json
import os
import time

from src.document_processor import process_document
from src.embedder import Embedder
from src.vector_store import FAISSVectorStore
from src.rag_chain import answer_with_history


class _FakeFile:
    def __init__(self, path):
        self.name = os.path.basename(path)
        self._path = path

    def read(self):
        with open(self._path, "rb") as f:
            return f.read()


def build_index(pdf_path: str):
    chunks, doc_info = process_document(_FakeFile(pdf_path))
    embedder = Embedder()
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_texts(texts)
    store = FAISSVectorStore(dimension=embedder.dimension)
    store.add(embeddings, texts, sources=[c.source for c in chunks], pages=[c.page for c in chunks])
    return store, embedder, doc_info


def evaluate(pdf_path: str, qa_path: str, groq_api_key: str, top_k: int = 5):
    store, embedder, doc_info = build_index(pdf_path)
    with open(qa_path) as f:
        qa_set = json.load(f)

    results_log = []
    hits, mrr_sum, answer_correct, total_latency = 0, 0.0, 0, 0.0

    for item in qa_set:
        q = item["question"]
        expected_page = item.get("expected_page")
        expected_str = item.get("expected_answer_contains", "").lower()

        t0 = time.time()
        answer, sources = answer_with_history(
            query=q, history=[], vector_store=store, embedder=embedder,
            groq_api_key=groq_api_key, top_k=top_k, stream=False,
        )
        latency = time.time() - t0
        total_latency += latency

        pages_retrieved = [s.page for s in sources]
        hit = expected_page in pages_retrieved if expected_page is not None else None
        if hit:
            hits += 1
            rank = pages_retrieved.index(expected_page) + 1
            mrr_sum += 1.0 / rank

        correct = expected_str in answer.lower() if expected_str else None
        if correct:
            answer_correct += 1

        results_log.append({
            "question": q, "answer": answer, "retrieval_hit": hit,
            "answer_correct": correct, "latency_sec": round(latency, 2),
        })

    n = len(qa_set)
    summary = {
        "n_questions": n,
        "recall_at_k": round(hits / n, 3) if n else 0,
        "mrr": round(mrr_sum / n, 3) if n else 0,
        "answer_accuracy": round(answer_correct / n, 3) if n else 0,
        "avg_latency_sec": round(total_latency / n, 2) if n else 0,
        "doc_chunks": doc_info.total_chunks,
    }

    print(json.dumps(summary, indent=2))
    with open("eval_results.json", "w") as f:
        json.dump({"summary": summary, "details": results_log}, f, indent=2)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--qa", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    api_key = os.getenv("GROQ_API_KEY", "")
    evaluate(args.pdf, args.qa, api_key, args.top_k)