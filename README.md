# RAG Assistant

Hệ thống hỏi đáp dựa trên tài liệu (Retrieval-Augmented Generation): ingest → chunk → embed → retrieve → rerank → generate.

## Chạy thử

```bash
cp .env.example .env        # rồi chỉnh giá trị trong .env
docker compose up -d db     # khởi tạo Postgres + pgvector, tự chạy src/db/init.sql
```

`init.sql` chỉ chạy khi volume `pgdata` còn trống. Để chạy lại từ đầu:

```bash
docker compose down -v && docker compose up -d db
```

## Schema

- `documents`: mỗi file đã ingest
- `chunks`: đoạn văn bản + `embedding vector(1024)` (HNSW cosine)

> Nếu đổi embedding model, sửa `vector(1024)` trong `init.sql` và `EMBEDDING_DIM` trong `.env` cho khớp.
