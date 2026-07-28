"""
Deduplication / pose-balancing — requirements.md Section 5.3, applied independently to src and
dst (not paired — DFL/SAEHD training isn't paired-frame dependent).

Implements all three of Section 5.3's near-duplicate signals:
- Perceptual hashing (dHash) for near-exact duplicate frames — fast, CPU-only, self-contained
  (no `imagehash` package dependency for what's a simple algorithm).
- Embedding-based similarity for same-pose-different-pixel duplicates dHash won't catch — see
  `FaceEmbedder`'s docstring for why this uses `facenet-pytorch` rather than the literal
  ArcFace/InsightFace name requirements.md uses.
- Landmark-based pose-bucket clustering (yaw/pitch/roll via dfl_torch.alignment's pose
  estimation) — shared with Section 5.2's pose-balancing per Section 5.3.

Sharpness scoring for representative selection reuses DFL's existing
`core.imagelib.estimate_sharpness` (Section 5.1's "reuse/extend DFL's existing blur-sort mode").
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from core.imagelib import estimate_sharpness


def _union_find_cluster(n, is_similar_fn):
    """Shared union-find clustering: groups indices [0, n) where `is_similar_fn(i, j)` is True,
    transitively. Returns a list of clusters (each a list of indices), first-seen order. Used by
    both cluster_by_hash (Hamming distance) and cluster_by_embedding_similarity (cosine
    similarity) — same algorithm, different pairwise-similarity criterion."""
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
            if is_similar_fn(i, j):
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


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
    Two frames land in the same cluster if their dHash Hamming distance is <= max_distance (and
    transitively, via any chain of such pairs). Returns a list of clusters, each a list of frame
    indices, in first-seen order.
    """
    return _union_find_cluster(len(hashes), lambda i, j: hamming_distance(hashes[i], hashes[j]) <= max_distance)


class FaceEmbedder:
    """
    Wraps `facenet-pytorch`'s `InceptionResnetV1` (VGGFace2-pretrained) for embedding-similarity
    dedup — Section 5.3's third near-duplicate signal, for same-pose-different-pixel duplicates
    dHash won't catch. requirements.md names ArcFace specifically; this uses `facenet-pytorch`
    instead for consistency with `dfl_torch.losses.IdentityLoss` (one face-embedding dependency,
    not two) — see that class's docstring for the full reasoning (there, the substitution is a
    hard requirement since ONNX-based ArcFace isn't autograd-differentiable; here, no gradient is
    needed at all since this is pure inference, so it's purely about not maintaining two
    face-embedding stacks for two similar purposes).
    """

    def __init__(self, device="cpu"):
        from facenet_pytorch import InceptionResnetV1

        self.device = device
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def embed(self, image):
        """image: HWC BGR array (uint8 [0,255] or float [0,1]). Returns a (512,) numpy embedding."""
        if image.dtype != np.float32 and image.dtype != np.float64:
            image = image.astype(np.float32) / 255.0
        img_t = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().unsqueeze(0).to(self.device)
        img_t = F.interpolate(img_t, size=(160, 160), mode="bilinear", align_corners=False)
        img_t = img_t.flip(1)  # BGR -> RGB
        img_t = (img_t * 255.0 - 127.5) / 128.0
        return self.model(img_t)[0].cpu().numpy()


def embedding_cosine_similarity(embedding_a, embedding_b):
    a = embedding_a / (np.linalg.norm(embedding_a) + 1e-8)
    b = embedding_b / (np.linalg.norm(embedding_b) + 1e-8)
    return float(np.dot(a, b))


def cluster_by_embedding_similarity(embeddings, similarity_threshold=0.6):
    """Two frames land in the same cluster if their face-embedding cosine similarity is
    >= similarity_threshold (transitively). Same clustering algorithm as cluster_by_hash, a
    different pairwise-similarity criterion suited to same-pose-different-pixel duplicates."""
    return _union_find_cluster(
        len(embeddings),
        lambda i, j: embedding_cosine_similarity(embeddings[i], embeddings[j]) >= similarity_threshold,
    )


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
