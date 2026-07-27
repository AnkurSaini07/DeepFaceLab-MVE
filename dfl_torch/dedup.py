"""
Deduplication / pose-balancing — requirements.md Section 5.3, applied independently to src and
dst (not paired — DFL/SAEHD training isn't paired-frame dependent).

Implements two of Section 5.3's three near-duplicate signals:
- Perceptual hashing (dHash) for near-exact duplicate frames — fast, CPU-only, self-contained
  (no `imagehash` package dependency for what's a simple algorithm).
- Landmark-based pose-bucket clustering (yaw/pitch/roll via dfl_torch.alignment's pose
  estimation) — shared with Section 5.2's pose-balancing per Section 5.3.

Sharpness scoring for representative selection reuses DFL's existing
`core.imagelib.estimate_sharpness` (Section 5.1's "reuse/extend DFL's existing blur-sort mode").

Not implemented: ArcFace embedding similarity (Section 5.3's third signal, for
same-pose-different-pixel duplicates that dHash won't catch) — needs an ArcFace model, a heavier
dependency (InsightFace model zoo or a standalone ONNX checkpoint) not pulled in yet, same
category of deferral as Phase 5's mic detector / SAM fallback.
"""
import cv2
import numpy as np

from core.imagelib import estimate_sharpness


def compute_dhash(image, hash_size=8):
    """
    Difference hash: resize to (hash_size+1, hash_size) grayscale, compare adjacent columns.
    image: HWC BGR array, either uint8 [0, 255] or float [0, 1] (auto-detected by dtype).
    Returns a (hash_size * hash_size,) boolean array.
    """
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def hamming_distance(hash_a, hash_b):
    return int(np.count_nonzero(hash_a != hash_b))


def sharpness_score(image):
    """Wraps core.imagelib.estimate_sharpness, which silently returns 0.0 (not an error) on
    float [0, 1] input — it expects uint8 [0, 255], so that conversion happens here."""
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return estimate_sharpness(image)


def cluster_by_hash(hashes, max_distance=5):
    """
    Union-find clustering: two frames land in the same cluster if their dHash Hamming distance is
    <= max_distance (and transitively, via any chain of such pairs). Returns a list of clusters,
    each a list of frame indices, in first-seen order.
    """
    n = len(hashes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if hamming_distance(hashes[i], hashes[j]) <= max_distance:
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def select_cluster_representatives(cluster_indices, sharpness_scores, max_representatives=4):
    """Keeps the max_representatives sharpest frames in a cluster — Section 5.3's cap of ~3-5
    representative frames per cluster, not 1, to retain some redundancy for lighting/micro-angle
    variation."""
    ranked = sorted(cluster_indices, key=lambda i: sharpness_scores[i], reverse=True)
    return ranked[:max_representatives]


def deduplicate(images, sharpness_scores=None, hash_max_distance=5, max_representatives=4, hash_size=8):
    """
    images: list of HWC BGR arrays (uint8 or float [0, 1]).
    Returns the sorted list of frame indices to keep.
    """
    hashes = [compute_dhash(img, hash_size=hash_size) for img in images]
    if sharpness_scores is None:
        sharpness_scores = [sharpness_score(img) for img in images]

    clusters = cluster_by_hash(hashes, max_distance=hash_max_distance)
    keep = []
    for cluster in clusters:
        keep.extend(select_cluster_representatives(cluster, sharpness_scores, max_representatives))
    return sorted(keep)


def bucket_by_pose(poses, yaw_bin_size=15.0, pitch_bin_size=15.0):
    """
    poses: (N, 3) yaw/pitch/roll in degrees (roll isn't used for bucketing — it's an in-plane
    framing detail, not a distinct facial pose for this purpose). Returns
    {(yaw_bucket, pitch_bucket): [frame_indices]}, shared between dedup (Section 5.3) and
    pose-balancing (Section 5.2) per Section 5.3's "combine this with pose-balancing as one
    shared pipeline stage."
    """
    poses = np.asarray(poses, dtype=np.float64)
    buckets = {}
    for i, (yaw, pitch, _roll) in enumerate(poses):
        key = (int(np.floor(yaw / yaw_bin_size)), int(np.floor(pitch / pitch_bin_size)))
        buckets.setdefault(key, []).append(i)
    return buckets
