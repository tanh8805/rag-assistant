-- Database riêng cho chunking strategy flat. Postgres tự chạy file này khi khởi tạo volume lần đầu.
CREATE DATABASE rag_flat;
\c rag_flat

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id         SERIAL PRIMARY KEY,
    source     TEXT NOT NULL UNIQUE,
    title      TEXT,
    content_hash TEXT,
    metadata   JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    document_id INT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    content     TEXT NOT NULL,
    embedding   vector(1024)
);

CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents (content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
