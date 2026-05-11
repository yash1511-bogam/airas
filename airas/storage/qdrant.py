"""Qdrant vector storage for antigen matching.

Why Qdrant: HNSW gives <5ms search at 100K vectors. Payload filtering
enables two-phase matching (embedding similarity + condition predicates)
in a single query. Embedded mode available for dev, gRPC for production.
"""
import os
import logging
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

COLLECTION = "antigens"
VECTOR_DIM = 384  # all-MiniLM-L6-v2

_client: QdrantClient | None = None
_embedder: SentenceTransformer | None = None


def get_qdrant() -> QdrantClient:
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        _client = QdrantClient(url=url, timeout=5)
        _ensure_collection(_client)
    return _client


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _ensure_collection(client: QdrantClient):
    """Create collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION not in collections:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=VECTOR_DIM, distance=models.Distance.COSINE),
        )
        # Create payload index for filtered search
        client.create_payload_index(COLLECTION, "failure_class", models.PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION, "active", models.PayloadSchemaType.BOOL)
        logger.info("Created Qdrant collection: %s", COLLECTION)


def upsert_antigen(antigen_id: str, signature: str, failure_class: str, conditions: dict, danger_score: float):
    """Insert or update an antigen in Qdrant."""
    client = get_qdrant()
    embedding = get_embedder().encode([signature], normalize_embeddings=True)[0].tolist()
    client.upsert(
        collection_name=COLLECTION,
        points=[models.PointStruct(
            id=antigen_id,
            vector=embedding,
            payload={
                "failure_class": failure_class,
                "signature": signature,
                "conditions": conditions,
                "danger_score": danger_score,
                "active": True,
            },
        )],
    )


def search_antigens(
    signature: str,
    failure_class: str | None = None,
    limit: int = 5,
    score_threshold: float = 0.80,
) -> list[dict]:
    """Search for matching antigens. Returns list of {id, score, payload}."""
    client = get_qdrant()
    embedding = get_embedder().encode([signature], normalize_embeddings=True)[0].tolist()

    query_filter = models.Filter(must=[
        models.FieldCondition(key="active", match=models.MatchValue(value=True)),
    ])
    if failure_class:
        query_filter.must.append(
            models.FieldCondition(key="failure_class", match=models.MatchValue(value=failure_class))
        )

    results = client.query_points(
        collection_name=COLLECTION,
        query=embedding,
        query_filter=query_filter,
        limit=limit,
        score_threshold=score_threshold,
    )

    return [
        {"id": str(r.id), "score": r.score, "payload": r.payload}
        for r in results.points
    ]


def delete_antigen(antigen_id: str):
    """Remove an antigen from the index."""
    get_qdrant().delete(collection_name=COLLECTION, points_selector=models.PointIdsList(points=[antigen_id]))


def count_antigens() -> int:
    """Get total number of antigens in the collection."""
    return get_qdrant().count(collection_name=COLLECTION).count
