# RAG Assistant

Hệ thống hỏi đáp dựa trên tài liệu (Retrieval-Augmented Generation): ingest → chunk → embed → retrieve → rerank → generate.

## Chạy thử

```bash
cp .env.example .env        # rồi chỉnh giá trị trong .env
docker compose up -d db     # khởi tạo Postgres + pgvector, tự chạy src/db/init.sh
```

Mỗi chunking strategy có database riêng (`rag_parent_child`, `rag_flat`) để so sánh mà không ghi đè nhau;
mỗi database dùng schema trong `src/db/init.sql`. `init.sh` chỉ chạy khi volume `pgdata` còn trống. Để chạy lại từ đầu:

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

## Schema

Mỗi chunking strategy một database, schema riêng:

**`rag_parent_child`** (`src/db/init.sql`)
- `documents`: mỗi file đã ingest, có `content_hash` (SHA-256 nội dung file) để bỏ qua file đã có
- `parents`: chunk cha (~3000 ký tự), phần được trả về cho LLM
- `chunks`: chunk con (~600 ký tự) + `embedding vector(1024)` (HNSW cosine), có `parent_id` trỏ về chunk cha

**`rag_flat`** (`src/db/init_1.sql`)
- `documents`: mỗi file đã ingest, có `content_hash` (SHA-256 nội dung file) để bỏ qua file đã có
- `chunks`: chunk thường (1000 ký tự, overlap 100) + `embedding vector(1024)`, không có parent

> Nếu đổi embedding model, sửa `vector(1024)` trong `init.sql`, `init_1.sql` cho khớp.
