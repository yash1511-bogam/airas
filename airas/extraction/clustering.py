"""Cluster extracted antigens into failure pattern classes.

Uses sentence-transformers for embedding + HDBSCAN for density-based clustering.
No need to specify K — HDBSCAN finds natural clusters.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from airas.models import FailureAntigen

# Lazy-loaded singleton
_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def embed_antigens(antigens: list[FailureAntigen]) -> np.ndarray:
    """Embed antigen signatures into vector space."""
    model = get_embedder()
    texts = [a.signature for a in antigens]
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


def cluster_antigens(
    antigens: list[FailureAntigen],
    min_cluster_size: int = 3,
    min_samples: int = 2,
) -> list[FailureAntigen]:
    """Cluster antigens and assign cluster_id to each.

    Returns the same antigens with cluster_id populated.
    Noise points (cluster_id=-1) are antigens that don't fit any cluster.
    """
    if len(antigens) < min_cluster_size:
        for a in antigens:
            a.cluster_id = 0
        return antigens

    embeddings = embed_antigens(antigens)
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embeddings)

    for antigen, label in zip(antigens, labels):
        antigen.cluster_id = int(label)

    return antigens


def get_cluster_summary(antigens: list[FailureAntigen]) -> dict[int, dict]:
    """Summarize each cluster: count, dominant failure class, representative signature."""
    clusters: dict[int, list[FailureAntigen]] = {}
    for a in antigens:
        clusters.setdefault(a.cluster_id, []).append(a)

    summary = {}
    for cid, members in clusters.items():
        if cid == -1:
            continue  # skip noise
        # Dominant failure class
        class_counts: dict[str, int] = {}
        for m in members:
            class_counts[m.failure_class.value] = class_counts.get(m.failure_class.value, 0) + 1
        dominant = max(class_counts, key=class_counts.get)

        summary[cid] = {
            "count": len(members),
            "dominant_class": dominant,
            "representative": members[0].signature,
            "all_classes": class_counts,
        }
    return summary


# Example:
# antigens = [extract_antigen(...) for each failure]
# antigens = cluster_antigens(antigens)
# summary = get_cluster_summary(antigens)
# → {0: {"count": 12, "dominant_class": "wrong_tool", ...}, 1: {...}}
