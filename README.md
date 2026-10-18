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

Đo Recall@k, MRR và số token đưa cho LLM của dense, BM25, RRF trên `eval/golden.jsonl`. Cần ingest tài liệu trước:

```bash
python -m eval.run_retrieval
```

Muốn đổi strategy, retriever hoặc các mức k, sửa các hằng số ở đầu `eval/run_retrieval.py`.

### Cách tính

Chỉ đánh giá **context**, tức thứ thực sự đưa cho LLM:
- **parent_child:** retriever vẫn tìm trên chunk con (top 20). Mỗi chunk con được đổi thành chunk cha của nó, cha trùng bị bỏ và giữ thứ hạng của lần xuất hiện đầu tiên. Danh sách cha sau khi bỏ trùng là danh sách xếp hạng, nên **k tính trên số cha, không phải số chunk con**. Vì nhiều chunk con thuộc cùng một cha, 20 chunk con chỉ cho khoảng 16 cha (nhỏ nhất 12), nên Recall@20 thực chất là recall của toàn bộ danh sách cha đó.
- **flat:** danh sách xếp hạng là chính các chunk tìm được.
- **Đúng/sai:** một văn bản (cha hoặc chunk) được coi là đúng nếu chứa nguyên văn một đoạn trong `evidence_spans` của câu hỏi. Recall@k là tỉ lệ đoạn bằng chứng nằm trong top-k, MRR là 1 chia thứ hạng của văn bản đúng đầu tiên.
- **Token TB (top-5):** số token trung bình của 5 văn bản đầu danh sách (số văn bản API `/query` đưa cho LLM). Đếm bằng tokenizer của model embedding, chỉ có ý nghĩa để so sánh giữa các strategy, không phải số token thật của LLM.

### Kết quả

Tài liệu: Sổ tay Sinh viên D26 (PDF, 266 trang có text). 28 câu hỏi có đáp án trong `eval/golden.jsonl`. Model embedding `AITeamVN/Vietnamese_Embedding`. Chunking: parent_child (parent 1500, child 600, overlap 80; 541 parent, 1.254 chunk con) và flat (1000, overlap 100; 740 chunk).

| Strategy | Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@20 | MRR | Token TB (top-5) |
|---|---|---|---|---|---|---|---|---|
| parent_child | dense | 0.562 | 0.798 | 0.857 | 0.946 | 0.967 | 0.811 | 1627 |
| parent_child | bm25 | 0.628 | 0.711 | 0.771 | 0.866 | 0.911 | 0.796 | 1699 |
| parent_child | rrf | 0.640 | 0.896 | 0.938 | 0.955 | 0.955 | **0.887** | 1638 |
| flat | dense | 0.473 | 0.789 | 0.869 | 0.914 | 0.967 | 0.787 | 1176 |
| flat | bm25 | 0.533 | 0.735 | 0.771 | 0.839 | 0.857 | 0.762 | 1200 |
| flat | rrf | 0.574 | 0.807 | 0.833 | 0.893 | 0.964 | 0.824 | 1181 |

Nhận xét:
- RRF tốt nhất ở hầu hết các mức; với parent_child, MRR 0.887 và Recall@5 0.938 (flat: 0.824 và 0.833).
- parent_child tốn nhiều token hơn flat khoảng 40% (khoảng 1.640 so với 1.180 token cho top-5), nên phần chênh lệch về chất lượng cần cân nhắc cùng chi phí này.
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
