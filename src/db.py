
import psycopg
from typing import List, Optional
from .config import PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASS

def get_conn():
    return psycopg.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASS,
    )

def insert_chunks(doc_id: str, chunks: List[str], embeddings: List[List[float]]):
    assert len(chunks) == len(embeddings)
    with get_conn() as conn, conn.cursor() as cur:
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                """
                INSERT INTO document_chunks (doc_id, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (doc_id, i, chunk, emb),
            )


def search_similar(
    query_emb: List[float],
    top_k: int = 5,
    doc_ids: Optional[List[str]] = None,
):
    
    vec_str = "[" + ",".join(f"{x:.6f}" for x in query_emb) + "]"

    with get_conn() as conn, conn.cursor() as cur:
        if doc_ids:
            
            cur.execute(
                """
                SELECT content
                FROM document_chunks
                WHERE doc_id = ANY(%s::uuid[])
                ORDER BY embedding <-> %s::vector
                LIMIT %s
                """,
                (doc_ids, vec_str, top_k),
            )
        else:
            
            cur.execute(
                """
                SELECT content
                FROM document_chunks
                ORDER BY embedding <-> %s::vector
                LIMIT %s
                """,
                (vec_str, top_k),
            )

        return [row[0] for row in cur.fetchall()]
