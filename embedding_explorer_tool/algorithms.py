"""Embedding-space algorithms for the explorer tool.

Self-contained port of the pipeline's ``TopKMeansXNN`` selector
(``selection/kmeans_xnn.py``): quality and farthest-point seeds, k-means
clustering with ``k = n``, and the constrained ``{centroid} ∪ xNN`` candidate
sets with medoid fallback.  Also provides the 3D MDS projection of the pool
and the image/mask overlay used by both frontends.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

DEFAULT_XNN_K = 3


def load_snapshot(output_dir: str | Path):
    """Load a pipeline snapshot from an output directory.

    Reads ``embeddings.npy``, ``selection_pool_ids.npy``, ``quality.csv`` and
    ``report.json`` (when present).  ``pool_ids[i]`` is the frame id of
    ``embeddings[i]``; ``quality`` is aligned with the same rows.
    """
    out = Path(output_dir)
    if not (out / "embeddings.npy").exists():
        raise FileNotFoundError(f"No embeddings.npy in {out}")

    embeddings = np.load(out / "embeddings.npy")
    pool_ids = np.load(out / "selection_pool_ids.npy")
    if len(pool_ids) != len(embeddings):
        raise ValueError(
            f"selection_pool_ids.npy ({len(pool_ids)}) does not match "
            f"embeddings.npy ({len(embeddings)})"
        )

    quality = _pool_quality(out, pool_ids)

    selected_ids = None
    sel_path = out / "selected_indices.npy"
    if sel_path.exists():
        idx = np.load(sel_path).astype(int)
        selected_ids = [int(pool_ids[i]) for i in idx if i < len(pool_ids)]

    report = None
    rep_path = out / "report.json"
    if rep_path.exists():
        report = json.loads(rep_path.read_text())

    data_root = (report or {}).get("data_root", "")
    return {
        "embeddings": embeddings,
        "pool_ids": pool_ids.astype(int),
        "quality": quality,
        "selected_ids": selected_ids,
        "report": report,
        "data_root": data_root,
    }


def _pool_quality(out: Path, pool_ids):
    qcsv = out / "quality.csv"
    if not qcsv.exists():
        return np.ones(len(pool_ids), dtype=float)
    df = pd.read_csv(qcsv)
    if "id" not in df.columns or "quality" not in df.columns:
        return np.ones(len(pool_ids), dtype=float)
    mapping = dict(zip(df["id"].astype(int), df["quality"].astype(float)))
    return np.array([mapping.get(int(i), 1.0) for i in pool_ids], dtype=float)


def quality_seeds(quality, k: int):
    """Indices of the ``k`` highest-quality samples (stable, ties by index)."""
    q = np.asarray(quality, dtype=float)
    order = np.argsort(-q, kind="stable")
    return [int(i) for i in order[:k]]


def fps_seeds(embeddings, quality, k: int):
    """Deterministic farthest-point seeds over the embedding space.

    Starts at the highest-quality sample, then repeatedly adds the point with
    the largest minimum cosine distance to the already-chosen seeds.
    """
    n = len(embeddings)
    dist = pairwise_distances(embeddings, metric="cosine")
    q = np.asarray(quality, dtype=float)
    seeds = [int(q.argmax())]
    while len(seeds) < k:
        min_dists = dist[:, seeds].min(axis=1)
        min_dists[seeds] = -1.0
        seeds.append(int(min_dists.argmax()))
    return seeds


def constrained_candidates(dist_center, labels, cluster: int, x: int):
    """Constrained ``{centroid} ∪ xNN`` candidate set for one cluster centroid.

    ``dist_center`` is the (N,) cosine distance of every pool sample to the
    centroid; ``labels`` the k-means assignment.  The raw neighbourhood is the
    ``x + 1`` pool samples nearest to the centroid (the medoid plus its ``x``
    nearest neighbours).  A candidate may not be closer to another centroid
    than to this one, so neighbours whose k-means label differs from
    ``cluster`` are dropped.  Falls back to the cluster medoid when nothing
    survives.
    """
    order = np.argsort(dist_center)
    raw = [int(p) for p in order[: x + 1]]
    candidates = [p for p in raw if labels[p] == cluster]
    if not candidates:
        members = np.where(labels == cluster)[0]
        medoid = int(members[dist_center[members].argmin()])
        candidates = [medoid]
    return candidates


def run_kmeans_xnn(embeddings, quality, k: int, init: str = "farthest", x: int = DEFAULT_XNN_K):
    """Run k-means with ``k = n`` and resolve each cluster's xNN pick.

    Returns a dict with seeds, labels, cluster centers and, for every cluster,
    its medoid (nearest pool sample to the center), its constrained candidate
    set and the best-quality pick.
    """
    embeddings = np.asarray(embeddings, dtype=float)
    quality = np.asarray(quality, dtype=float)
    n = len(embeddings)
    k = max(1, min(int(k), n))
    x = max(1, int(x))

    if init == "best_quality":
        seeds = quality_seeds(quality, k)
    else:
        init = "farthest"
        seeds = fps_seeds(embeddings, quality, k)

    km = KMeans(n_clusters=k, init=embeddings[seeds], n_init=1, random_state=0)
    labels = km.fit_predict(embeddings)
    centers = km.cluster_centers_
    dist_centers = pairwise_distances(embeddings, centers, metric="cosine")

    clusters = []
    picks = []
    for c in range(k):
        members = np.where(labels == c)[0]
        medoid = int(members[dist_centers[members, c].argmin()])
        candidates = constrained_candidates(dist_centers[:, c], labels, c, x)
        pick = int(max(candidates, key=lambda p: quality[p]))
        clusters.append(
            {
                "cluster": int(c),
                "medoid": medoid,
                "candidates": candidates,
                "pick": pick,
            }
        )
        picks.append(pick)

    return {
        "k": k,
        "init": init,
        "x": x,
        "seeds": seeds,
        "labels": labels.tolist(),
        "centers": centers,
        "clusters": clusters,
        "picks": picks,
    }


def project_mds(embeddings, random_state: int = 0):
    """Project the pool into 3D with metric MDS over cosine distance."""
    dist = pairwise_distances(embeddings, metric="cosine")
    mds = MDS(
        n_components=3,
        metric=True,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=4,
        max_iter=300,
        normalized_stress="auto",
    )
    return mds.fit_transform(dist)


def compose_mask_overlay(image, mask, mask_alpha: float = 0.75, content_alpha: float = 0.25, bg_alpha: float = 0.66):
    """Compose the frame + mask overlay for the explorer viewers.

    The mask (object) region keeps only ``content_alpha`` of the original
    pixels and gets a ``mask_alpha``-strength green tint; the background is
    dimmed to ``bg_alpha``.
    """
    img = np.asarray(image, dtype=float)
    m = np.asarray(mask) > 0
    out = img.copy()
    out[~m] = out[~m] * bg_alpha
    green = np.zeros_like(img)
    green[:, :, 1] = 255.0
    out[m] = content_alpha * img[m] + mask_alpha * green[m]
    return out.astype(np.uint8)


def build_text(result, pool_ids, quality):
    """Render the text output: centroid frame IDs then the xNN dictionary."""
    ids = np.asarray(pool_ids, dtype=int)
    quality = np.asarray(quality, dtype=float)
    clusters = result["clusters"]

    centroid_ids = [int(ids[c["medoid"]]) for c in clusters]
    picks = [int(ids[c["pick"]]) for c in clusters]
    xnn = {
        str(int(ids[c["medoid"]])): [int(ids[p]) for p in c["candidates"]]
        for c in clusters
    }

    lines = [
        f"k = {result['k']}   init = {result['init']}   xNN = {result['x']}",
        "",
        f"Centroid frame IDs ({len(centroid_ids)}):",
        repr(centroid_ids),
        "",
        "Constrained xNN per centroid (keys = centroid frame IDs):",
        json.dumps(xnn, indent=2),
        "",
        f"Final picks (best quality in xNN, {len(picks)}):",
        repr(picks),
        "",
        "Final pick qualities:",
        repr([round(float(quality[c["pick"]]), 3) for c in clusters]),
    ]
    return "\n".join(lines)
