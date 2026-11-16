"""
Module hiện thực các bộ truy xuất (Retrievers) cho Day 7.
Gồm DenseRetriever (pgvector), TfidfRetriever (Lexical), và HybridRetriever (RRF).
"""
import json
import os
import sys
from pathlib import Path
from typing import Any
import psycopg
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "starter") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "starter"))
from baseline import TfidfIndex


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
                "source_id": r[2],
                "source": r[2],
                "page": r[3],
                "text": r[4],
                "metadata": r[5] if isinstance(r[5], dict) else json.loads(r[5]),
                "score": float(r[6]),
                "strategy": "dense_pgvector",
            })
        return results

class TfidfRetriever:
    """Bộ truy xuất từ khóa Lexical TF-IDF (Phương pháp 1)."""

    def __init__(self, chunks_path: str | None = None) -> None:
        if chunks_path is None:
            chunks_path = os.path.join(os.path.dirname(__file__), "..", "data", "chunks.json")
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.index = TfidfIndex(self.chunks)

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Tìm kiếm từ khóa qua TF-IDF và áp dụng bộ lọc metadata (nếu có)."""
        results = self.index.search(query, top_k=top_k)
        if filters:
            filtered = []
            for r in results:
                meta = r.get("metadata", {})
                if all(str(meta.get(k)) == str(v) for k, v in filters.items()):
                    filtered.append(r)
            return filtered[:top_k]
        return results


class HybridRetriever:
    """Bộ truy xuất lai Hybrid RRF kết hợp Dense Retrieval và Lexical TF-IDF (Phương pháp 3)."""

    def __init__(self, dense_retriever: DenseRetriever, tfidf_retriever: TfidfRetriever) -> None:
        self.dense = dense_retriever
        self.tfidf = tfidf_retriever

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """
        Lấy candidate dư từ cả 2 nguồn, sau đó tính điểm Reciprocal Rank Fusion (RRF)
        với hằng số chuẩn k = 60 để xếp hạng lại.
        """
        # Lấy dư ứng viên (gấp đôi top_k) để đảm bảo không bỏ sót đoạn văn tiềm năng
        candidate_k = max(top_k * 2, 6)
        dense_hits = self.dense.search(query, filters=filters, top_k=candidate_k)
        tfidf_hits = self.tfidf.search(query, filters=filters, top_k=candidate_k)

        # Tính điểm RRF cho từng chunk
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            cid = hit["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60.0 + rank))
            chunk_map[cid] = hit

        for rank, hit in enumerate(tfidf_hits, start=1):
            cid = hit["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60.0 + rank))
            if cid not in chunk_map:
                chunk_map[cid] = hit

        # Sắp xếp lại theo điểm RRF giảm dần và lấy top_k cao nhất
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: -rrf_scores[cid])[:top_k]

        results = []
        for cid in sorted_cids:
            item = dict(chunk_map[cid])
            item["score"] = rrf_scores[cid]
            item["strategy"] = "hybrid_rrf"
            results.append(item)
        return results

