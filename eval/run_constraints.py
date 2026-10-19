"""Kiểm tra 3 ràng buộc của hệ thống trên toàn bộ 40 câu hỏi trong eval/golden.jsonl:
- Độ trễ P95 < 2 giây (truy xuất + rerank + LLM, đúng các bước của /query)
- Chi phí < 0,03 USD mỗi câu hỏi (tính từ số token Gemini báo về)
- Trích dẫn chính xác: nguồn trả về có chứa đoạn bằng chứng không

Gọi Gemini thật nên cần GEMINI_API_KEY (tốn vài cent). Thêm --skip-llm để chỉ đo truy xuất + rerank.
Chạy từ thư mục gốc:  python -m eval.run_constraints
"""

import json
import math
import re
import sys
import time

from eval.metrics import norm
from src.config import ROOT, settings
from src.db import create_session
from src.generation import SYSTEM_PROMPT, build_prompt
from src.generation.gemini_llm import GeminiLLM
from src.rerank import get_reranker, rerank
from src.retrieval import get_parents, get_retriever

SKIP_LLM = "--skip-llm" in sys.argv
RETRIEVER = "rrf"  # giống mặc định của /query
PRICE_PER_M_TOKENS = {"gemini-2.5-flash": (0.30, 2.50)}  # (đầu vào, đầu ra) USD / 1 triệu token
LATENCY_LIMIT_S, COST_LIMIT_USD = 2.0, 0.03


def percentile(values, p):
    """Percentile theo nearest-rank."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def summarize(name, values, unit="s"):
    fmt = "{:.3f}" if unit == "s" else "{:.5f}"
    print(f"{name:<22} P50 {fmt.format(percentile(values, 50))}  P95 {fmt.format(percentile(values, 95))}  max {fmt.format(max(values))} {unit}")


def generate(llm, prompt):
    """Giống GeminiLLM.generate nhưng trả thêm số token để tính tiền (token 'suy nghĩ' tính như token ra)
    và thời gian của lần gọi thành công. Gặp 429 (hết hạn mức theo phút) thì chờ rồi gọi lại, thời gian chờ không tính."""
    from google.genai import types

    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
    for attempt in range(5):
        start = time.perf_counter()
        try:
            response = llm.client.models.generate_content(model=llm.model, contents=prompt, config=config)
        except Exception as e:
            if "429" not in str(e) or attempt == 4:
                raise
            time.sleep(20)
            continue
        seconds = time.perf_counter() - start
        usage = response.usage_metadata
        tokens_in = usage.prompt_token_count or 0
        tokens_out = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        return response.text or "", tokens_in, tokens_out, seconds


def main():
    questions = [json.loads(line) for line in open(ROOT / "eval" / "golden.jsonl", encoding="utf-8")]
    strategy = settings.chunking.strategy
    db = create_session(strategy)
    llm = None if SKIP_LLM else GeminiLLM()
    price_in, price_out = PRICE_PER_M_TOKENS[settings.llm.gemini_model]

    # chạy thử một lần để nạp model, tách riêng khỏi số đo
    t = time.perf_counter()
    retriever = get_retriever(RETRIEVER, db)
    rerank("khởi động", get_parents(db, retriever.retrieve("khởi động"), strategy))
    device = get_reranker().device
    print(f"retriever: {RETRIEVER}, strategy: {strategy}, rerank trên: {device}, nạp model lần đầu: {time.perf_counter() - t:.1f}s")
    print(f"số câu hỏi: {len(questions)}, LLM: {'bỏ qua' if SKIP_LLM else settings.llm.gemini_model}\n")

    t_retrieve, t_rerank, t_llm, t_total, costs = [], [], [], [], []
    answerable = hit = shown = relevant = with_marker = unanswerable = unanswerable_with_sources = 0
    errors = 0
    for q in questions:
        retriever = get_retriever(RETRIEVER, db)  # như /query: tạo retriever mỗi request
        start = time.perf_counter()
        parents = get_parents(db, retriever.retrieve(q["question"], settings.retrieval.top_k), strategy)
        mid = time.perf_counter()
        sources = rerank(q["question"], parents)
        end = time.perf_counter()
        t_retrieve.append(mid - start)
        t_rerank.append(end - mid)

        total = end - start
        answer = ""
        if llm is not None:
            try:
                answer, tokens_in, tokens_out, seconds = generate(llm, build_prompt(q["question"], sources))
            except Exception as e:  # lỗi mạng hoặc vượt hạn mức: bỏ qua câu này
                errors += 1
                print(f"{q['id']}: lỗi LLM: {str(e)[:100]}")
                continue
            t_llm.append(seconds)
            total += seconds
            costs.append((tokens_in * price_in + tokens_out * price_out) / 1e6)
        t_total.append(total)

        # trích dẫn: nguồn có chứa nguyên văn đoạn bằng chứng không
        contents = [norm(s.content) for s in sources]
        if q["answerable"]:
            answerable += 1
            flags = [any(norm(span) in c for span in q["evidence_spans"]) for c in contents]
            hit += any(flags)
            shown += len(flags)
            relevant += sum(flags)
        else:
            unanswerable += 1
            unanswerable_with_sources += bool(sources)
        with_marker += bool(re.search(r"\[\d+\]", answer))

    print("=== Độ trễ (giây) ===")
    summarize("truy xuất (RRF)", t_retrieve)
    summarize("rerank", t_rerank)
    if t_llm:
        summarize("LLM", t_llm)
    summarize("tổng", t_total)
    print(f"P95 tổng {'ĐẠT' if percentile(t_total, 95) < LATENCY_LIMIT_S else 'KHÔNG ĐẠT'} (< {LATENCY_LIMIT_S}s)")

    if costs:
        print("\n=== Chi phí (USD / câu hỏi, chỉ tính LLM) ===")
        summarize("chi phí", costs, unit="USD")
        print(f"max {'ĐẠT' if max(costs) < COST_LIMIT_USD else 'KHÔNG ĐẠT'} (< {COST_LIMIT_USD} USD)")

    print("\n=== Trích dẫn ===")
    print(f"câu có đáp án: nguồn trả về chứa bằng chứng ở {hit}/{answerable} câu")
    print(f"độ chính xác của danh sách nguồn: {relevant}/{shown} nguồn có chứa bằng chứng ({relevant / shown:.0%})")
    print(f"câu không có đáp án vẫn trả về nguồn: {unanswerable_with_sources}/{unanswerable}")
    print(f"câu trả lời có đánh dấu trích dẫn dạng [n]: {with_marker}/{len(t_total)}")
    if errors:
        print(f"\ncó {errors} câu lỗi LLM, không tính vào số đo")


if __name__ == "__main__":
    main()
