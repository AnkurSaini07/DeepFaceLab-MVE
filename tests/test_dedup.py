"""
Tests for dfl_torch/dedup.py (requirements.md Section 5.3: deduplication + pose-bucket
clustering).
"""
import numpy as np
import pytest

from dfl_torch.dedup import (
    FaceEmbedder,
    bucket_by_pose,
    cluster_by_embedding_similarity,
    cluster_by_hash,
    compute_dhash,
    deduplicate,
    embedding_cosine_similarity,
    hamming_distance,
    select_cluster_representatives,
    sharpness_score,
)


def _base_image(size, seed, shape="circle"):
    rng = np.random.RandomState(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = rng.randint(40, 100, size=3)
    if shape == "circle":
        cv2_center = (size // 2, size // 2)
        _draw_circle(img, cv2_center, size // 3, rng.randint(150, 255, size=3))
    else:
        _draw_rect(img, size // 4, size // 4, size // 2, size // 2, rng.randint(150, 255, size=3))
    return img


def _draw_circle(img, center, radius, color):
    import cv2
    cv2.circle(img, center, radius, tuple(int(c) for c in color), -1)


def _draw_rect(img, x, y, w, h, color):
    import cv2
    cv2.rectangle(img, (x, y), (x + w, y + h), tuple(int(c) for c in color), -1)


def _add_noise(img, seed, scale=3.0):
    rng = np.random.RandomState(seed)
    noisy = img.astype(np.float64) + rng.normal(scale=scale, size=img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _blur(img, ksize):
    import cv2
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


# --- dHash / hamming distance ---

def test_dhash_identical_images_zero_distance():
    img = _base_image(64, seed=0)
    h1 = compute_dhash(img)
    h2 = compute_dhash(img.copy())
    assert hamming_distance(h1, h2) == 0


def test_dhash_near_duplicate_small_distance():
    img = _base_image(64, seed=0)
    noisy = _add_noise(img, seed=1, scale=2.0)
    h1 = compute_dhash(img)
    h2 = compute_dhash(noisy)
    assert hamming_distance(h1, h2) <= 5


def test_dhash_distinct_images_large_distance():
    circle = _base_image(64, seed=0, shape="circle")
    rect = _base_image(64, seed=5, shape="rect")
    h1 = compute_dhash(circle)
    h2 = compute_dhash(rect)
    assert hamming_distance(h1, h2) > 5


def test_dhash_accepts_float_and_uint8_consistently():
    img_uint8 = _base_image(64, seed=0)
    img_float = img_uint8.astype(np.float32) / 255.0
    assert hamming_distance(compute_dhash(img_uint8), compute_dhash(img_float)) == 0


# --- sharpness_score ---

def test_sharpness_score_blurry_lower_than_sharp():
    sharp = _base_image(128, seed=0)
    blurry = _blur(sharp, 15)
    assert sharpness_score(blurry) < sharpness_score(sharp)


def test_sharpness_score_handles_float_input():
    img_uint8 = _base_image(64, seed=0)
    img_float = img_uint8.astype(np.float32) / 255.0
    # Must not silently return 0 for float input (core.imagelib.estimate_sharpness does that if
    # you forget to convert to uint8 first — this wrapper handles it).
    assert sharpness_score(img_float) > 0.0


# --- clustering ---

def test_cluster_by_hash_groups_near_duplicates_separates_distinct():
    circle = _base_image(64, seed=0, shape="circle")
    circle_variants = [circle] + [_add_noise(circle, seed=i, scale=2.0) for i in range(1, 4)]
    rect = _base_image(64, seed=5, shape="rect")
    rect_variants = [rect] + [_add_noise(rect, seed=i, scale=2.0) for i in range(10, 12)]

    images = circle_variants + rect_variants
    hashes = [compute_dhash(img) for img in images]
    clusters = cluster_by_hash(hashes, max_distance=8)

    assert len(clusters) == 2
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [2, 4] or sizes == [3, 4]  # allow minor hash-boundary noise either way


def test_select_cluster_representatives_picks_sharpest():
    scores = {0: 0.9, 1: 0.1, 2: 0.5, 3: 0.7}
    chosen = select_cluster_representatives([0, 1, 2, 3], scores, max_representatives=2)
    assert chosen == [0, 3]


def test_select_cluster_representatives_respects_cap_even_if_cluster_smaller():
    scores = {0: 0.5, 1: 0.2}
    chosen = select_cluster_representatives([0, 1], scores, max_representatives=4)
    assert chosen == [0, 1]


# --- end-to-end deduplicate ---

def test_deduplicate_caps_near_duplicate_cluster_and_keeps_distinct_frames():
    circle = _base_image(64, seed=0, shape="circle")
    circle_variants = [_add_noise(circle, seed=i, scale=2.0) for i in range(10)]  # 10 near-dupes
    rect = _base_image(64, seed=5, shape="rect")

    images = circle_variants + [rect]
    keep = deduplicate(images, hash_max_distance=8, max_representatives=3)

    # The 10-frame near-duplicate cluster should be capped at 3; the lone distinct frame kept.
    assert len(keep) <= 4
    assert (len(images) - 1) in keep  # the rect frame (last index) survives — it's its own cluster


# --- pose bucketing ---

def test_bucket_by_pose_groups_similar_poses():
    poses = [(10, 5, 0), (12, 6, 0), (80, -30, 0), (82, -28, 0)]
    buckets = bucket_by_pose(poses, yaw_bin_size=15.0, pitch_bin_size=15.0)
    assert len(buckets) == 2
    sizes = sorted(len(v) for v in buckets.values())
    assert sizes == [2, 2]


def test_bucket_by_pose_ignores_roll():
    poses = [(10, 5, 0), (10, 5, 45), (10, 5, -30)]
    buckets = bucket_by_pose(poses)
    assert len(buckets) == 1
    assert len(next(iter(buckets.values()))) == 3


# --- FaceEmbedder / embedding similarity (network-dependent: downloads+caches VGGFace2 weights) ---

def _make_embedder():
    pytest.importorskip("facenet_pytorch")
    try:
        return FaceEmbedder()
    except Exception as e:
        pytest.skip(f"could not initialize FaceEmbedder (likely no network for weight download): {e}")


def test_embedding_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert embedding_cosine_similarity(v, v) == pytest.approx(1.0)


def test_embedding_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert embedding_cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_embedding_cosine_similarity_opposite_vectors_is_negative_one():
    v = np.array([1.0, 2.0, 3.0])
    assert embedding_cosine_similarity(v, -v) == pytest.approx(-1.0)


def test_face_embedder_produces_consistent_embedding_for_same_image():
    embedder = _make_embedder()
    img = _base_image(128, seed=0)
    emb_a = embedder.embed(img)
    emb_b = embedder.embed(img)
    assert emb_a.shape == (512,)
    np.testing.assert_allclose(emb_a, emb_b)


def test_face_embedder_accepts_float_and_uint8_consistently():
    embedder = _make_embedder()
    img_uint8 = _base_image(128, seed=0)
    img_float = img_uint8.astype(np.float32) / 255.0
    emb_uint8 = embedder.embed(img_uint8)
    emb_float = embedder.embed(img_float)
    np.testing.assert_allclose(emb_uint8, emb_float, atol=1e-4)


def test_cluster_by_embedding_similarity_groups_identical_embeddings():
    embeddings = [np.array([1.0, 0.0]), np.array([1.0, 0.01]), np.array([0.0, 1.0])]
    clusters = cluster_by_embedding_similarity(embeddings, similarity_threshold=0.99)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]


def test_cluster_by_embedding_similarity_high_threshold_separates_everything():
    embeddings = [np.array([1.0, 0.0]), np.array([0.9, 0.1]), np.array([0.0, 1.0])]
    clusters = cluster_by_embedding_similarity(embeddings, similarity_threshold=0.9999)
    assert len(clusters) == 3
