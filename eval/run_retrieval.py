import json

from sqlalchemy import text

from eval.metrics import recall_at_k, reciprocal_rank
from src.config import ROOT, ChunkingStrategy, settings
from src.db import create_session
from src.embeddings import get_model
from src.rerank import rerank
from src.retrieval import RetrieverName, get_parents, get_retriever

STRATEGIES: list[ChunkingStrategy] = ["parent_child", "flat"]  # các database cần đánh giá
RETRIEVERS: list[RetrieverName] = ["dense", "bm25", "fts", "rrf"]
KS = [1, 3, 5, 10, 20]
VARIANTS = ["không", "có"]  
TOP_N = settings.rerank.top_n  


def load_questions():
    questions = []
    with open(ROOT / "eval" / "golden.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["answerable"]:  # câu không có đáp án thì không có gì để tìm
                questions.append(q)
    return questions


def search(db, retriever, strategy, question):
    """Trả về các văn bản (chunk cha hoặc chunk) đưa cho LLM, xếp từ cao xuống thấp."""
    chunks = retriever.retrieve(question, top_k=max(KS))
    return get_parents(db, chunks, strategy)


def count_tokens(tokenizer, texts):
    return sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in texts)


def main():
    questions = load_questions()
    tokenizer = get_model().tokenizer  
    print(f"số câu hỏi: {len(questions)}, token tính trên top-{TOP_N} văn bản\n")
    header = ["strategy", "retriever", "rerank"] + [f"Recall@{k}" for k in KS] + ["MRR", f"Token TB (top-{TOP_N})"]
    print(" | ".join(header))

    for strategy in STRATEGIES:
        db = create_session(strategy)
        if db.execute(text("SELECT count(*) FROM chunks")).scalar_one() == 0:
            print(f"{strategy} | (database trống, hãy ingest trước)")
            continue

        for name in RETRIEVERS:
            retriever = get_retriever(name, db)
            recall_sum = {v: {k: 0.0 for k in KS} for v in VARIANTS}
            mrr_sum = {v: 0.0 for v in VARIANTS}
            token_sum = {v: 0 for v in VARIANTS}
            for q in questions:
                parents = search(db, retriever, strategy, q["question"])
                reranked = rerank(q["question"], parents, top_n=len(parents))
                for variant, ranked in [("không", parents), ("có", reranked)]:
                    results = [p.content for p in ranked]
                    for k in KS:
                        recall_sum[variant][k] += recall_at_k(results, q["evidence_spans"], k)
                    mrr_sum[variant] += reciprocal_rank(results, q["evidence_spans"])
                    token_sum[variant] += count_tokens(tokenizer, results[:TOP_N])

            n = len(questions)
            for variant in VARIANTS:
                row = [strategy, name, variant]
                row += [f"{recall_sum[variant][k] / n:.3f}" for k in KS]
                row += [f"{mrr_sum[variant] / n:.3f}", f"{token_sum[variant] / n:.0f}"]
                print(" | ".join(row))


if __name__ == "__main__":
    main()
