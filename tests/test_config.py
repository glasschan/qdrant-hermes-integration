"""Tests for config.py — config loading and validation.

Tests _validate_config with various edge cases.
"""
import logging
import pytest

from plugin.config import (
    _validate_config,
    DEDUP_THRESHOLD,
    SEARCH_RECENCY_WEIGHT,
    PREFETCH_SCORE_THRESHOLD,
    PREFETCH_TOP_K,
    PREFETCH_MIN_TURNS,
    SEARCH_MIN_PRIORITY,
)


class TestConfigValidation:
    """Tests for _validate_config."""

    def _make_valid_config(self) -> dict:
        """Return a valid default config dict."""
        return {
            "dedup_threshold": 0.85,
            "search_recency_weight": 0.0,
            "prefetch_score_threshold": 0.4,
            "prefetch_top_k": 8,
            "prefetch_min_turns": 3,
            "orphan_age_days": 60,
            "correction_topic_threshold": 3,
            "auto_stale_days": 90,
            "auto_prune_days": 180,
            "search_min_priority": 1,
            "collection_name": "test_collection",
        }

    def test_valid_config_passes(self):
        """A perfectly valid config should not be modified."""
        config = self._make_valid_config()
        original = dict(config)
        _validate_config(config)
        for key in ("dedup_threshold", "search_recency_weight", "prefetch_score_threshold"):
            assert config[key] == original[key]

    def test_dedup_threshold_out_of_range_high(self):
        config = self._make_valid_config()
        config["dedup_threshold"] = 1.5
        _validate_config(config)
        assert config["dedup_threshold"] == DEDUP_THRESHOLD

    def test_dedup_threshold_out_of_range_low(self):
        config = self._make_valid_config()
        config["dedup_threshold"] = -0.1
        _validate_config(config)
        assert config["dedup_threshold"] == DEDUP_THRESHOLD

    def test_dedup_threshold_not_a_number(self):
        config = self._make_valid_config()
        config["dedup_threshold"] = "not_a_float"
        _validate_config(config)
        assert config["dedup_threshold"] == DEDUP_THRESHOLD

    def test_recency_weight_boundary_values(self):
        """0.0 and 1.0 should be accepted."""
        config = self._make_valid_config()
        config["search_recency_weight"] = 0.0
        _validate_config(config)
        assert config["search_recency_weight"] == 0.0

        config["search_recency_weight"] = 1.0
        _validate_config(config)
        assert config["search_recency_weight"] == 1.0

    def test_prefetch_top_k_negative(self):
        config = self._make_valid_config()
        config["prefetch_top_k"] = -1
        _validate_config(config)
        assert config["prefetch_top_k"] == PREFETCH_TOP_K

    def test_prefetch_top_k_not_a_number(self):
        config = self._make_valid_config()
        config["prefetch_top_k"] = "abc"
        _validate_config(config)
        # Should not crash

    def test_search_min_priority_out_of_range(self):
        config = self._make_valid_config()
        config["search_min_priority"] = 0
        _validate_config(config)
        assert config["search_min_priority"] == SEARCH_MIN_PRIORITY

        config["search_min_priority"] = 6
        _validate_config(config)
        assert config["search_min_priority"] == SEARCH_MIN_PRIORITY

    def test_search_min_priority_valid_range(self):
        config = self._make_valid_config()
        for val in (1, 2, 3, 4, 5):
            config["search_min_priority"] = val
            _validate_config(config)
            assert config["search_min_priority"] == val

    def test_empty_collection_name_is_ok(self):
        """Empty collection name should be handled (auto-generated)."""
        config = self._make_valid_config()
        config["collection_name"] = ""
        _validate_config(config)
        # Should not crash

    def test_none_values_handled(self):
        """None values in optional fields should not crash."""
        config = self._make_valid_config()
        config["dedup_threshold"] = None
        _validate_config(config)
        # Should not crash

    def test_orphan_age_days_negative(self):
        config = self._make_valid_config()
        config["orphan_age_days"] = -10
        _validate_config(config)
        assert config["orphan_age_days"] == 60


class TestEmbeddingCircuitBreaker:
    """Tests for EmbeddingCircuitBreaker from embeddings.py."""

    def test_initial_state_is_closed(self):
        from plugin.embeddings import EmbeddingCircuitBreaker
        cb = EmbeddingCircuitBreaker(threshold=3, cooldown_secs=1.0)
        assert not cb.is_open

    def test_opens_after_threshold_failures(self):
        from plugin.embeddings import EmbeddingCircuitBreaker
        cb = EmbeddingCircuitBreaker(threshold=3, cooldown_secs=10.0)
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open  # Not yet
        cb.record_failure()
        assert cb.is_open  # Now open

    def test_success_resets_count(self):
        from plugin.embeddings import EmbeddingCircuitBreaker
        cb = EmbeddingCircuitBreaker(threshold=3, cooldown_secs=10.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert not cb.is_open  # Reset after success, only 1 failure
