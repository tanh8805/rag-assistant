"""Đánh giá retrieval bằng Recall@k và MRR trên eval/golden.jsonl.

Cách chấm: một văn bản tìm được là "đúng" nếu chứa nguyên văn một đoạn trong evidence_spans.
Cần ingest tài liệu trước. Chạy từ thư mục gốc:  python -m eval.run_retrieval
"""

import json

from sqlalchemy import text

from eval.metrics import recall_at_k, reciprocal_rank
from src.config import ROOT, ChunkingStrategy
from src.db import create_session
from src.retrieval import RetrieverName, get_parents, get_retriever

STRATEGIES: list[ChunkingStrategy] = ["parent_child", "flat"]  # các database cần đánh giá
RETRIEVERS: list[RetrieverName] = ["dense", "bm25", "rrf"]
KS = [1, 3, 5, 10, 20]
LEVEL = "chunk"  # "chunk": chunk tìm được | "context": parent đưa cho LLM


def load_questions():
    questions = []
    with open(ROOT / "eval" / "golden.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["answerable"]:  # câu không có đáp án thì không có gì để tìm
                questions.append(q)
    return questions


def search(db, retriever, strategy, question):
    """Trả về nội dung các văn bản tìm được, xếp từ cao xuống thấp."""
    chunks = retriever.retrieve(question, top_k=max(KS))
    if LEVEL == "context":
        return [p.content for p in get_parents(db, chunks, strategy)]
    return [c.content for c in chunks]


def main():
    questions = load_questions()
    print(f"mức: {LEVEL}, số câu hỏi: {len(questions)}\n")
    print(" | ".join(["strategy", "retriever"] + [f"Recall@{k}" for k in KS] + ["MRR"]))

    for strategy in STRATEGIES:
        db = create_session(strategy)
        if db.execute(text("SELECT count(*) FROM chunks")).scalar_one() == 0:
            print(f"{strategy} | (database trống, hãy ingest trước)")
            continue

        for name in RETRIEVERS:
            retriever = get_retriever(name, db)
            recall_sum = {k: 0.0 for k in KS}
            mrr_sum = 0.0
            for q in questions:
                results = search(db, retriever, strategy, q["question"])
                for k in KS:
                    recall_sum[k] += recall_at_k(results, q["evidence_spans"], k)
                mrr_sum += reciprocal_rank(results, q["evidence_spans"])

            n = len(questions)
            row = [strategy, name] + [f"{recall_sum[k] / n:.3f}" for k in KS] + [f"{mrr_sum / n:.3f}"]
            print(" | ".join(row))


if __name__ == "__main__":
    main()
