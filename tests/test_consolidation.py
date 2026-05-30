"""Tests for pure functions in consolidation.py.

No Qdrant connection needed — tests _cosine_similarity, _batch_cosine_similarity,
_avg_group_similarity, and helper functions.
"""
import math
import pytest

from plugin.consolidation import (
    _cosine_similarity,
    _batch_cosine_similarity,
    _avg_group_similarity,
    DUPLICATE_THRESHOLD,
    STALE_DAYS,
)


class TestCosineSimilarity:
    """Tests for _cosine_similarity."""

    def test_identical_vectors(self):
        vec = [1.0, 0.0, 0.0, 0.0]
        assert _cosine_similarity(vec, vec) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = [1.0, 0.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_45_degree_vectors(self):
        a = [1.0, 0.0]
        b = [1.0, 1.0]
        expected = 1.0 / math.sqrt(2)
        assert _cosine_similarity(a, b) == pytest.approx(expected, abs=1e-6)

    def test_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0

    def test_none_vectors(self):
        assert _cosine_similarity(None, [1.0]) == 0.0
        assert _cosine_similarity([1.0], None) == 0.0

    def test_different_length_vectors(self):
        assert _cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_high_dimensional_vectors(self):
        """Test with realistic embedding dimensions (2048)."""
        import random
        random.seed(42)
        a = [random.gauss(0, 1) for _ in range(2048)]
        b = [random.gauss(0, 1) for _ in range(2048)]
        sim = _cosine_similarity(a, b)
        # Two random vectors should have near-zero cosine similarity
        assert -0.1 < sim < 0.1

    def test_scaled_vectors_same_direction(self):
        """Scaling a vector shouldn't change cosine similarity."""
        a = [1.0, 2.0, 3.0]
        b = [2.0, 4.0, 6.0]  # 2x scaling
        assert _cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-6)


class TestBatchCosineSimilarity:
    """Tests for _batch_cosine_similarity."""

    def test_empty_input(self):
        assert _batch_cosine_similarity([]) == []

    def test_single_vector(self):
        result = _batch_cosine_similarity([[1.0, 0.0]])
        assert len(result) == 1
        assert result[0][0] == pytest.approx(1.0, abs=1e-6)

    def test_identity_matrix(self):
        """Three orthogonal vectors should produce identity-like matrix."""
        vecs = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        result = _batch_cosine_similarity(vecs)
        assert len(result) == 3
        for i in range(3):
            assert result[i][i] == pytest.approx(1.0, abs=1e-6)
            for j in range(3):
                if i != j:
                    assert result[i][j] == pytest.approx(0.0, abs=1e-6)

    def test_symmetry(self):
        """Similarity matrix should be symmetric."""
        vecs = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        result = _batch_cosine_similarity(vecs)
        for i in range(len(vecs)):
            for j in range(len(vecs)):
                assert result[i][j] == pytest.approx(result[j][i], abs=1e-6)

    def test_consistency_with_pairwise(self):
        """Batch result should match individual pairwise calls."""
        vecs = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.1, 0.2, 0.3],
        ]
        result = _batch_cosine_similarity(vecs)
        for i in range(len(vecs)):
            for j in range(len(vecs)):
                expected = _cosine_similarity(vecs[i], vecs[j])
                assert result[i][j] == pytest.approx(expected, abs=1e-4)


class TestAvgGroupSimilarity:
    """Tests for _avg_group_similarity."""

    def test_empty_group(self):
        assert _avg_group_similarity([]) == 0.0

    def test_single_point(self):
        assert _avg_group_similarity([{"vector": [1.0, 0.0]}]) == 0.0

    def test_two_identical_points(self):
        group = [
            {"vector": [1.0, 0.0, 0.0]},
            {"vector": [1.0, 0.0, 0.0]},
        ]
        assert _avg_group_similarity(group) == pytest.approx(1.0, abs=1e-6)

    def test_points_without_vectors(self):
        group = [
            {"content": "hello"},
            {"content": "world"},
        ]
        assert _avg_group_similarity(group) == 0.0

    def test_orthogonal_points(self):
        group = [
            {"vector": [1.0, 0.0]},
            {"vector": [0.0, 1.0]},
        ]
        assert _avg_group_similarity(group) == pytest.approx(0.0, abs=1e-6)


class TestConsolidationConstants:
    """Sanity checks on constants."""

    def test_duplicate_threshold_range(self):
        assert 0.0 <= DUPLICATE_THRESHOLD <= 1.0

    def test_stale_days_positive(self):
        assert STALE_DAYS > 0
