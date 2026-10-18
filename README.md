# RAG Assistant

Hệ thống hỏi đáp dựa trên tài liệu (Retrieval-Augmented Generation): ingest → chunk → embed → retrieve → rerank → generate.

## Chạy thử

```bash
cp .env.example .env        # rồi chỉnh giá trị trong .env
docker compose up -d db     # khởi tạo Postgres + pgvector, tự tạo 2 database
```

Mỗi chunking strategy có database riêng (`rag_parent_child`, `rag_flat`) để so sánh mà không ghi đè nhau;
mỗi database có schema riêng (`src/db/init.sql`, `src/db/init_1.sql`, mỗi file tự `CREATE DATABASE`). Hai file này chỉ chạy khi volume `pgdata` còn trống. Để chạy lại từ đầu (xóa hết dữ liệu):

```bash
docker compose down -v && docker compose up -d db
```

## API

```bash
uvicorn src.main:app --reload    # hoặc docker compose up
```

- `POST /ingest`: upload file (`.pdf .md .txt .html .htm`), lưu vào `data/uploads/`. Trả về `document_id`, hoặc `duplicate` nếu nội dung đã có.
- `POST /query`: `{"question": "...", "retriever": "rrf", "top_k": 20, "document_ids": [1, 2]}` (`retriever`: `dense | bm25 | rrf`; bỏ `document_ids` để tìm trong tất cả tài liệu). Trả về `answer` và `sources`.

```bash
curl -F "file=@tai_lieu.pdf" localhost:8000/ingest
curl -X POST localhost:8000/query -H "Content-Type: application/json" -d '{"question": "..."}'
```

Swagger UI: http://localhost:8000/docs

## Đánh giá retrieval

Đo Recall@k và MRR của dense, BM25, RRF trên `eval/golden.jsonl` (một văn bản tìm được là đúng nếu chứa nguyên văn một đoạn trong `evidence_spans`). Cần ingest tài liệu trước:

```bash
python -m eval.run_retrieval
```

Muốn đổi strategy, retriever, các mức k hoặc chấm trên parent đưa cho LLM (`LEVEL = "context"`), sửa các hằng số ở đầu `eval/run_retrieval.py`.

### Kết quả

Tài liệu: Sổ tay Sinh viên D26 (PDF, 266 trang có text). 28 câu hỏi có đáp án trong `eval/golden.jsonl`. Model embedding `AITeamVN/Vietnamese_Embedding`. Chunking: parent_child (parent 1500, child 600, overlap 80, tổng 541 parent / 1.254 chunk con) và flat (1000, overlap 100, 740 chunk).

Mức `context` là nội dung đưa cho LLM (parent với parent_child, chính chunk với flat), nên là mức so sánh công bằng giữa hai strategy:

| Strategy | Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@20 | MRR |
|---|---|---|---|---|---|---|---|
| parent_child | dense | 0.562 | 0.798 | 0.857 | 0.946 | 0.967 | 0.811 |
| parent_child | bm25 | 0.628 | 0.711 | 0.771 | 0.866 | 0.911 | 0.796 |
| parent_child | rrf | 0.640 | 0.896 | 0.938 | 0.955 | 0.955 | **0.887** |
| flat | dense | 0.473 | 0.789 | 0.869 | 0.914 | 0.967 | 0.787 |
| flat | bm25 | 0.533 | 0.735 | 0.771 | 0.839 | 0.857 | 0.762 |
| flat | rrf | 0.574 | 0.807 | 0.833 | 0.893 | 0.964 | 0.824 |

Mức `chunk` (chấm trên chunk tìm được, chunk con của parent_child ngắn hơn chunk flat nên không so sánh trực tiếp được):

| Strategy | Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@20 | MRR |
|---|---|---|---|---|---|---|---|
| parent_child | dense | 0.473 | 0.679 | 0.714 | 0.804 | 0.917 | 0.762 |
| parent_child | bm25 | 0.539 | 0.658 | 0.682 | 0.777 | 0.857 | 0.789 |
| parent_child | rrf | 0.542 | 0.771 | 0.807 | 0.860 | 0.938 | 0.843 |
| flat | dense | 0.473 | 0.789 | 0.869 | 0.914 | 0.967 | 0.787 |
| flat | bm25 | 0.533 | 0.735 | 0.771 | 0.839 | 0.857 | 0.762 |
| flat | rrf | 0.574 | 0.807 | 0.833 | 0.893 | 0.964 | 0.824 |

Nhận xét:
- RRF tốt nhất ở hầu hết các mức; với parent_child, MRR 0.887 và Recall@5 0.938 (flat: 0.824 và 0.833).
- Ở Recall@20 hai strategy gần như bằng nhau; parent_child chủ yếu xếp đúng lên đầu tốt hơn.
- Chỉ có 28 câu hỏi nên chênh lệch vài điểm phần trăm chưa đủ để kết luận chắc chắn.

## Schema

Mỗi chunking strategy một database, schema riêng:

**`rag_parent_child`** (`src/db/init.sql`)
- `documents`: mỗi file đã ingest, có `content_hash` (SHA-256 nội dung file) để bỏ qua file đã có
- `parents`: chunk cha (~1500 ký tự), phần được trả về cho LLM
- `chunks`: chunk con (~600 ký tự) + `embedding vector(1024)` (HNSW cosine), có `parent_id` trỏ về chunk cha

**`rag_flat`** (`src/db/init_1.sql`)
- `documents`: mỗi file đã ingest, có `content_hash` (SHA-256 nội dung file) để bỏ qua file đã có
- `chunks`: chunk thường (1000 ký tự, overlap 100) + `embedding vector(1024)`, không có parent

> Nếu đổi embedding model, sửa `vector(1024)` trong `init.sql`, `init_1.sql` cho khớp.
