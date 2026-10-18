"""Đánh giá retrieval bằng Recall@k, MRR và số token đưa cho LLM, trên eval/golden.jsonl.

Danh sách được chấm là "context" (thứ thực sự đưa cho LLM):
- parent_child: retriever tìm trên chunk con, mỗi chunk con đổi thành chunk cha của nó,
  bỏ cha trùng (giữ thứ hạng lần xuất hiện đầu tiên). k tính trên số cha.
- flat: chính chunk tìm được.
Một văn bản đúng nếu chứa nguyên văn một đoạn trong evidence_spans.

Cần ingest tài liệu trước. Chạy từ thư mục gốc:  python -m eval.run_retrieval
"""

import json

from sqlalchemy import text

from eval.metrics import recall_at_k, reciprocal_rank
from src.config import ROOT, ChunkingStrategy, settings
from src.db import create_session
from src.embeddings import get_model
from src.retrieval import RetrieverName, get_parents, get_retriever

STRATEGIES: list[ChunkingStrategy] = ["parent_child", "flat"]  # các database cần đánh giá
RETRIEVERS: list[RetrieverName] = ["dense", "bm25", "rrf"]
KS = [1, 3, 5, 10, 20]
TOP_N = settings.rerank.top_n  # số văn bản đầu danh sách đưa cho LLM (giống API /query)


def load_questions():
    questions = []
    with open(ROOT / "eval" / "golden.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["answerable"]:  # câu không có đáp án thì không có gì để tìm
                questions.append(q)
    return questions


def search(db, retriever, strategy, question):
    """Trả về nội dung các văn bản đưa cho LLM, xếp từ cao xuống thấp."""
    chunks = retriever.retrieve(question, top_k=max(KS))
    return [p.content for p in get_parents(db, chunks, strategy)]


def count_tokens(tokenizer, texts):
    return sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in texts)


def main():
    questions = load_questions()
    tokenizer = get_model().tokenizer  # đếm token bằng tokenizer của model embedding (chỉ để so sánh)
    print(f"số câu hỏi: {len(questions)}, token tính trên top-{TOP_N} văn bản\n")
    header = ["strategy", "retriever"] + [f"Recall@{k}" for k in KS] + ["MRR", f"Token TB (top-{TOP_N})"]
    print(" | ".join(header))

    for strategy in STRATEGIES:
        db = create_session(strategy)
        if db.execute(text("SELECT count(*) FROM chunks")).scalar_one() == 0:
            print(f"{strategy} | (database trống, hãy ingest trước)")
            continue

        for name in RETRIEVERS:
            retriever = get_retriever(name, db)
            recall_sum = {k: 0.0 for k in KS}
            mrr_sum = 0.0
            token_sum = 0
            for q in questions:
                results = search(db, retriever, strategy, q["question"])
                for k in KS:
                    recall_sum[k] += recall_at_k(results, q["evidence_spans"], k)
                mrr_sum += reciprocal_rank(results, q["evidence_spans"])
                token_sum += count_tokens(tokenizer, results[:TOP_N])

            n = len(questions)
            row = [strategy, name]
            row += [f"{recall_sum[k] / n:.3f}" for k in KS]
            row += [f"{mrr_sum / n:.3f}", f"{token_sum / n:.0f}"]
            print(" | ".join(row))


if __name__ == "__main__":
    main()
