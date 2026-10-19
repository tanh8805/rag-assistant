# RAG Assistant

Hệ thống hỏi đáp tiếng Việt dựa trên tài liệu của riêng bạn, xây theo mô hình **Retrieval-Augmented Generation**. Tài liệu (PDF, HTML, Markdown, văn bản thuần) được nạp, chia nhỏ, vector hóa và lưu vào PostgreSQL + pgvector. Khi có câu hỏi, hệ thống tìm các đoạn liên quan rồi nhờ LLM trả lời kèm nguồn trích dẫn.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-009688)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791)

## Tính năng

- **Nạp tài liệu** qua API hoặc code: PDF (theo từng trang), HTML, Markdown, TXT.
- **Chống nạp trùng** bằng mã băm SHA-256 của nội dung file: file đã có (dù đổi tên) sẽ bị bỏ qua, file sửa nội dung sẽ thay bản cũ.
- **Hai chiến lược chia chunk** để so sánh: `parent_child` (tìm trên chunk nhỏ, trả về chunk cha giàu ngữ cảnh) và `flat` (chunk cố định). Mỗi chiến lược dùng một database riêng.
- **Ba cách truy xuất**: dense (vector, cosine), BM25 (từ khóa) và RRF (kết hợp hai cách trên), có thể lọc theo tài liệu.
- **Hai LLM** chọn bằng cấu hình: Google Gemini hoặc OpenAI.
- **Bộ đánh giá** kèm theo: 40 câu hỏi có đáp án chuẩn và đoạn bằng chứng, đo Recall@k, MRR và số token đưa cho LLM.

## Kiến trúc

```mermaid
flowchart LR
    D["Tài liệu<br/>PDF · HTML · MD · TXT"] --> L1[Loader]
    L1 --> C["Chunker<br/>parent_child | flat"]
    C --> E["Embedding<br/>Vietnamese_Embedding"]
    E --> DB[("PostgreSQL<br/>+ pgvector")]
    Q[Câu hỏi] --> R["Retriever<br/>dense · BM25 · RRF"]
    DB --> R
    R --> P["Chunk cha / chunk"]
    P --> G["LLM<br/>Gemini · OpenAI"]
    G --> A["Câu trả lời + nguồn"]
```

| Thành phần | Công nghệ |
|---|---|
| API | FastAPI, Uvicorn |
| Cơ sở dữ liệu | PostgreSQL 16 + pgvector (HNSW, cosine), SQLAlchemy 2 + psycopg 3 |
| Embedding | `AITeamVN/Vietnamese_Embedding` (1024 chiều) qua sentence-transformers |
| Chia chunk | LangChain `RecursiveCharacterTextSplitter` |
| BM25 | `rank-bm25` (BM25Plus) |
| LLM | Gemini (`gemini-2.5-flash`) hoặc OpenAI (`gpt-4o-mini`) |

## Bắt đầu nhanh

**Yêu cầu:** Python 3.11+, Docker, và một API key của Gemini hoặc OpenAI. Có GPU NVIDIA thì embedding nhanh hơn nhiều nhưng không bắt buộc.

```bash
# 1. Cài thư viện
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Cấu hình: điền POSTGRES_PASSWORD và GEMINI_API_KEY (hoặc OPENAI_API_KEY)
cp .env.example .env

# 3. Khởi động database (cổng 5432 phải đang trống); tự tạo rag_parent_child và rag_flat
docker compose up -d db

# 4. Chạy API
uvicorn src.main:app --reload
```

Nạp tài liệu và đặt câu hỏi:

```bash
curl -F "file=@tai_lieu.pdf" localhost:8000/ingest
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
     -d '{"question": "Điều kiện để được xét học bổng là gì?"}'
```

Giao diện Swagger: http://localhost:8000/docs

> Lần nạp đầu tiên chậm vì phải tải và khởi tạo model embedding (khoảng 2 GB). Muốn chạy trên CPU, đặt `CUDA_VISIBLE_DEVICES=""` trước lệnh `uvicorn`.
> Service `api` trong `docker-compose.yml` cũng chạy được (`docker compose up -d`) nhưng build lâu, image lớn và chỉ dùng CPU; khi phát triển nên chạy `uvicorn` trực tiếp như trên.

## API

### `POST /ingest`

Upload một file (`.pdf`, `.md`, `.txt`, `.html`, `.htm`). File được lưu trong `data/uploads/`.

```json
{ "filename": "tai_lieu.pdf", "status": "ingested", "document_id": 1 }
```

`status` là `"duplicate"` (và `document_id` là `null`) nếu nội dung file đã có trong hệ thống. Định dạng không hỗ trợ trả về `400`.

### `POST /query`

| Trường | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `question` | string | bắt buộc | Câu hỏi |
| `retriever` | `"dense"` \| `"bm25"` \| `"rrf"` | `"rrf"` | Cách truy xuất |
| `top_k` | int (1–100) | `20` | Số chunk lấy ra ban đầu |
| `document_ids` | int[] \| null | `null` | Chỉ tìm trong các tài liệu này; `null` là tìm tất cả |

```json
{
  "answer": "…",
  "sources": [{ "source": "data/uploads/tai_lieu.pdf", "content": "…" }]
}
```

Các `sources` là những chunk cha xếp hạng cao nhất (tối đa `rerank.top_n`) được đưa cho LLM. Nếu không tìm thấy gì, API trả lời "không tìm thấy" mà không gọi LLM. Thiếu API key trả về `503`.

## Cấu hình

Cài đặt chung nằm trong [`config.yaml`](config.yaml); khóa bí mật nằm trong `.env`.

| Mục | Khóa | Ý nghĩa |
|---|---|---|
| `chunking` | `strategy` | `parent_child` hoặc `flat`; quyết định API ghi vào database nào |
| | `parent_size`, `child_size`, `child_overlap` | Kích thước chunk cha, chunk con và phần chồng lấn (parent_child) |
| | `chunk_size`, `chunk_overlap` | Kích thước chunk và phần chồng lấn (flat) |
| `retrieval` | `top_k` | Số chunk lấy ra mặc định |
| `rerank` | `top_n` | Số chunk cha đưa cho LLM |
| `llm` | `provider` | `gemini` hoặc `openai` |
| | `gemini_model`, `openai_model` | Tên model |
| `embedding` | `model`, `dim` | Model embedding và số chiều vector |

Biến môi trường (`.env`): `POSTGRES_PASSWORD`, `GEMINI_API_KEY`, `OPENAI_API_KEY`.

## Dữ liệu và chiến lược chunking

Mỗi chiến lược có database riêng (`rag_<strategy>`) để so sánh công bằng mà không ghi đè lẫn nhau. Hai file SQL khởi tạo chỉ chạy khi volume `pgdata` còn trống; để làm lại từ đầu (xóa hết dữ liệu): `docker compose down -v && docker compose up -d db`.

| | `rag_parent_child` ([`init.sql`](src/db/init.sql)) | `rag_flat` ([`init_1.sql`](src/db/init_1.sql)) |
|---|---|---|
| Bảng | `documents`, `parents`, `chunks` | `documents`, `chunks` |
| Embedding | chỉ chunk con (~600 ký tự) | mỗi chunk (~1000 ký tự) |
| Trả cho LLM | chunk cha (~1500 ký tự) chứa chunk con tìm được | chính chunk tìm được |

Cả hai đều có cột `embedding vector(1024)` với chỉ mục HNSW cosine. Nếu đổi model embedding, sửa `vector(1024)` trong hai file SQL và `embedding.dim` trong `config.yaml` cho khớp, rồi nạp lại dữ liệu.

**Đổi cấu hình chunking:** vì nạp trùng bị bỏ qua theo mã băm, sau khi đổi kích thước chunk cần xóa tài liệu cũ (`DELETE FROM documents`) rồi nạp lại.

## Đánh giá retrieval

Bộ dữ liệu thử nghiệm là *Sổ tay Sinh viên khóa D26* của Học viện Công nghệ Bưu chính Viễn thông ([nguồn](https://tsv.ptit.edu.vn/S%E1%BB%95%20tay%20sinh%20vi%C3%AAn%20D26.pdf), 268 trang). File PDF không nằm trong git; hãy tải về và đặt vào `data/`.

[`eval/golden.jsonl`](eval/golden.jsonl) có 40 câu hỏi: 28 câu có đáp án (loại `fact`, `procedure`, `multi_section`) và 12 câu không có đáp án trong tài liệu. Mỗi câu có đáp án kèm trang nguồn và `evidence_spans`, các đoạn trích nguyên văn từ tài liệu.

```bash
python -m eval.run_retrieval
```

Muốn đổi strategy, retriever hoặc các mức k, sửa các hằng số ở đầu [`eval/run_retrieval.py`](eval/run_retrieval.py). Cần nạp tài liệu vào database của strategy cần đánh giá trước.

### Cách tính

Chỉ đánh giá **context**, tức thứ thực sự đưa cho LLM:

- **parent_child:** retriever tìm trên chunk con (top 20). Mỗi chunk con được đổi thành chunk cha của nó, cha trùng bị bỏ và giữ thứ hạng của lần xuất hiện đầu tiên. Danh sách cha sau khi bỏ trùng là danh sách xếp hạng, nên **k tính trên số cha, không phải số chunk con**. Vì nhiều chunk con thuộc cùng một cha, 20 chunk con chỉ cho khoảng 16 cha (nhỏ nhất 12), nên Recall@20 thực chất là recall của toàn bộ danh sách cha đó.
- **flat:** danh sách xếp hạng là chính các chunk tìm được.
- **Đúng/sai:** một văn bản được coi là đúng nếu chứa nguyên văn một đoạn trong `evidence_spans`. Recall@k là tỉ lệ đoạn bằng chứng nằm trong top-k; MRR là 1 chia thứ hạng của văn bản đúng đầu tiên.
- **Token TB (top-5):** số token trung bình của 5 văn bản đầu danh sách (số văn bản mà `/query` đưa cho LLM), đếm bằng tokenizer của model embedding. Chỉ dùng để so sánh tương đối giữa các strategy, không phải số token thật của LLM.

### Kết quả

28 câu có đáp án; parent_child dùng parent 1500 / child 600 (541 cha, 1.254 chunk con), flat dùng chunk 1000 / overlap 100 (740 chunk).

| Strategy | Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@20 | MRR | Token TB (top-5) |
|---|---|---|---|---|---|---|---|---|
| parent_child | dense | 0.562 | 0.798 | 0.857 | 0.946 | 0.967 | 0.811 | 1627 |
| parent_child | bm25 | 0.628 | 0.711 | 0.771 | 0.866 | 0.911 | 0.796 | 1699 |
| parent_child | rrf | 0.640 | 0.896 | 0.938 | 0.955 | 0.955 | **0.887** | 1638 |
| flat | dense | 0.473 | 0.789 | 0.869 | 0.914 | 0.967 | 0.787 | 1176 |
| flat | bm25 | 0.533 | 0.735 | 0.771 | 0.839 | 0.857 | 0.762 | 1200 |
| flat | rrf | 0.574 | 0.807 | 0.833 | 0.893 | 0.964 | 0.824 | 1181 |

- RRF tốt nhất ở hầu hết các mức. Với parent_child, MRR đạt 0.887 và Recall@5 đạt 0.938 (flat: 0.824 và 0.833).
- parent_child đưa cho LLM nhiều token hơn flat khoảng 40% (~1.640 so với ~1.180 cho top-5), nên cần cân nhắc phần chất lượng tăng thêm cùng chi phí này.
- Ở Recall@20 hai strategy gần như bằng nhau; ưu thế của parent_child nằm ở việc xếp đúng lên đầu.
- Chỉ có 28 câu hỏi, nên chênh lệch vài điểm phần trăm chưa đủ để kết luận chắc chắn.

## Cấu trúc thư mục

```
.
├── config.yaml              # cấu hình chung
├── docker-compose.yml       # Postgres + pgvector (và API)
├── eval/
│   ├── golden.jsonl         # bộ câu hỏi đánh giá
│   ├── metrics.py           # Recall@k, MRR
│   └── run_retrieval.py     # chạy đánh giá
├── src/
│   ├── main.py              # ứng dụng FastAPI, gom các router
│   ├── api/                 # ingest.py, query.py
│   ├── ingestion/           # loader PDF/HTML/MD/TXT, băm file, pipeline nạp
│   ├── chunking/            # parent_child, flat, factory
│   ├── embeddings/          # sentence-transformers
│   ├── db/                  # schema SQL, lưu chunk, kết nối
│   ├── retrieval/           # dense, BM25, RRF, factory
│   └── generation/          # LLM Gemini/OpenAI, prompt, factory
└── tests/
```

## Kiểm thử

```bash
pytest
```

Các test cần database (retrieval, nạp trùng, API) tự bỏ qua nếu không kết nối được. Có thể trỏ sang database khác bằng biến `TEST_DATABASE_URL_PARENT_CHILD` và `TEST_DATABASE_URL_FLAT`. Các test dùng embedding và LLM giả nên không cần tải model hay API key.

## Hạn chế và hướng phát triển

- Chưa có bước **rerank** (`src/rerank/` còn trống, `rerank.model` trong cấu hình chưa được dùng); hiện chỉ lấy `rerank.top_n` chunk cha xếp hạng cao nhất.
- Chỉ số BM25 được dựng lại từ database ở mỗi lần truy vấn, sẽ chậm khi dữ liệu lớn.
- Chưa có đánh giá chất lượng câu trả lời của LLM, mới đo ở bước truy xuất.
- Bộ đánh giá nhỏ (28 câu có đáp án, một tài liệu).
