"""
Module hiện thực các bộ truy xuất (Retrievers)
Gồm DenseRetriever kết nối với PostgreSQL pgvector từ Day 6 và OpenAI Embeddings.
"""
import json
import os
from typing import Any
import psycopg
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class DenseRetriever:
    """Truy vấn vector store pgvector của Day 6 bằng OpenAI text-embedding-3-small."""

    def __init__(self, database_url: str | None = None, dimensions: int = 768) -> None:
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab"
        )
        self.dimensions = dimensions
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self._ensure_schema_and_data()

    def _ensure_schema_and_data(self) -> None:
        """Đảm bảo bảng document_chunks tồn tại và có dữ liệu 325 chunks từ Day 6."""
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        chunk_id VARCHAR(255) PRIMARY KEY,
                        strategy VARCHAR(50) NOT NULL,
                        document_id VARCHAR(100) NOT NULL,
                        source VARCHAR(255) NOT NULL,
                        page INT NOT NULL,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL,
                        embedding vector({self.dimensions}) NOT NULL
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_strategy ON document_chunks (strategy);")
                conn.commit()

                # Kiểm tra số lượng chunk hiện tại của chiến lược B (by-paragraph)
                cur.execute("SELECT COUNT(*) FROM document_chunks WHERE strategy = 'B';")
                count = cur.fetchone()[0]

        # Tính năng tự phục hồi: Nếu database chưa có vector, tự động nạp 325 chunks từ data/chunks.json
        if count == 0:
            print("-> Phát hiện database chưa có vector; đang tự động nạp 325 chunks từ data/chunks.json...")
            chunks_path = os.path.join(os.path.dirname(__file__), "..", "data", "chunks.json")
            if os.path.exists(chunks_path):
                with open(chunks_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                texts = [c["text"] for c in chunks]
                resp = self.client.embeddings.create(input=texts, model=self.model_name, dimensions=self.dimensions)
                vectors = [d.embedding for d in resp.data]

                with psycopg.connect(self.database_url) as conn:
                    with conn.cursor() as cur:
                        records = [
                            (
                                c["id"],
                                "B",
                                c["source_id"],
                                c["source"],
                                c["page"],
                                c["text"],
                                json.dumps(c["metadata"]),
                                v,
                            )
                            for c, v in zip(chunks, vectors)
                        ]
                        cur.executemany(
                            """
                            INSERT INTO document_chunks 
                            (chunk_id, strategy, document_id, source, page, text, metadata, embedding)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (chunk_id) DO NOTHING;
                        """,
                            records,
                        )
                        conn.commit()
                print(f"-> Đã nạp thành công {len(records)} vectors vào PostgreSQL pgvector.")

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Nhúng query và tìm kiếm top-k chunks gần nhất theo khoảng cách cosine."""
        # 1. Tạo query embedding tương thích index Day 6
        resp = self.client.embeddings.create(
            input=[query], model=self.model_name, dimensions=self.dimensions
        )
        query_vector = resp.data[0].embedding

        # 2. Xây dựng câu truy vấn SQL có hỗ trợ metadata filter (nếu có)
        conditions = ["strategy = %(strategy)s"]
        params: dict[str, Any] = {
            "strategy": "B",
            "query_vec": str(query_vector),
            "top_k": top_k,
        }

        if filters:
            for key, val in filters.items():
                param_key = f"filter_{key}"
                conditions.append(f"metadata->>'{key}' = %({param_key})s")
                params[param_key] = str(val)

        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT 
                chunk_id,
                document_id,
                source,
                page,
                text,
                metadata,
                1 - (embedding <=> %(query_vec)s::vector) AS score
            FROM document_chunks
            WHERE {where_clause}
            ORDER BY embedding <=> %(query_vec)s::vector
            LIMIT %(top_k)s;
        """

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        # Chuẩn hóa đầu ra khớp với Protocol Retriever trong starter/agentic.py
        results = []
        for r in rows:
            results.append({
                "id": r[0],
                "source_id": r[1],
                "source": r[2],
                "page": r[3],
                "text": r[4],
                "metadata": r[5] if isinstance(r[5], dict) else json.loads(r[5]),
                "score": float(r[6]),
                "strategy": "dense_pgvector",
            })
        return results
