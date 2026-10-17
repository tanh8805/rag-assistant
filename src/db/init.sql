-- Khởi tạo schema cho RAG assistant (PostgreSQL + pgvector)
-- Tự chạy lần đầu khi container postgres khởi động (mount vào /docker-entrypoint-initdb.d/).

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- gen_random_uuid()

-- ---------------------------------------------------------------
-- documents: một bản ghi cho mỗi file/nguồn đã ingest
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source        TEXT        NOT NULL,              -- đường dẫn file hoặc URL
    title         TEXT,
    mime_type     TEXT,
    content_hash  TEXT        NOT NULL,              -- sha256 nội dung, dùng để bỏ qua file không đổi
    metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source)
);

CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents (content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING gin (metadata);

-- ---------------------------------------------------------------
-- chunks: các đoạn văn bản + vector embedding
-- Số chiều vector (1024) phải khớp với EMBEDDING_DIM trong .env
-- (vd: bge-m3, multilingual-e5-large). Đổi model khác => sửa số này.
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID        NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index  INTEGER     NOT NULL,
    content      TEXT        NOT NULL,
    token_count  INTEGER,
    embedding    vector(1024),
    metadata     JSONB       NOT NULL DEFAULT '{}'::jsonb,   -- page, heading, ...
    -- full-text search ('simple' để không phụ thuộc ngôn ngữ, phù hợp tiếng Việt)
    content_tsv  TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks (document_id);

-- Vector search (cosine)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Keyword search cho hybrid retrieval
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING gin (content_tsv);
CREATE INDEX IF NOT EXISTS idx_chunks_trgm ON chunks USING gin (content gin_trgm_ops);

-- ---------------------------------------------------------------
-- query_logs: lưu câu hỏi, context đã truy xuất và câu trả lời (phục vụ eval/debug)
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS query_logs (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question      TEXT        NOT NULL,
    answer        TEXT,
    chunk_ids     UUID[]      NOT NULL DEFAULT '{}',
    latency_ms    INTEGER,
    metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_query_logs_created ON query_logs (created_at DESC);

-- ---------------------------------------------------------------
-- Tự cập nhật documents.updated_at
-- ---------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_documents_updated_at ON documents;
CREATE TRIGGER trg_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
