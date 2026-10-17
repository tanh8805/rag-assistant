# RAG Assistant

Hệ thống hỏi đáp dựa trên tài liệu (Retrieval-Augmented Generation): ingest → chunk → embed → retrieve → rerank → generate.

## Cấu trúc

```
rag-assistant/
├── data/              # tài liệu nguồn để ingest
├── eval/              # bộ câu hỏi và script đánh giá
└── src/
    ├── api/           # FastAPI endpoints
    ├── ingestion/     # đọc PDF/DOCX/HTML/...
    ├── chunking/      # chia nhỏ văn bản
    ├── embeddings/    # tạo vector embedding
    ├── retrieval/     # vector + keyword (hybrid) search
    ├── rerank/        # sắp xếp lại kết quả
    ├── generation/    # gọi LLM sinh câu trả lời
    └── db/            # kết nối DB, init.sql (schema)
```

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

- `documents`: mỗi nguồn đã ingest (có `content_hash` để bỏ qua file không đổi)
- `chunks`: đoạn văn bản + `embedding vector(1024)` (HNSW cosine) + `content_tsv` (full-text)
- `query_logs`: câu hỏi, chunk đã dùng, câu trả lời, độ trễ — phục vụ eval

> Nếu đổi embedding model, sửa `vector(1024)` trong `init.sql` và `EMBEDDING_DIM` trong `.env` cho khớp.
