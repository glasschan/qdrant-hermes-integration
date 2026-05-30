"""Tests for store.py pure functions.

Tests _completeness_score helper function.
"""
import pytest

from plugin.store import _completeness_score


class TestCompletenessScore:
    """Tests for _completeness_score (dedup quality scoring)."""

    def test_empty_point(self):
        assert _completeness_score({}) == 0

    def test_all_fields_filled(self):
        point = {
            "tags": ["career", "salary"],
            "category": "preference",
            "priority": 2,
            "origin": "explicit",
            "evolved_from": "abc-123",
        }
        assert _completeness_score(point) == 5

    def test_some_fields_filled(self):
        point = {
            "tags": ["career"],
            "category": "fact",
        }
        assert _completeness_score(point) == 2

    def test_empty_tags_list(self):
        """Empty list should NOT count as filled."""
        point = {"tags": []}
        assert _completeness_score(point) == 0

    def test_empty_string_fields(self):
        """Empty strings should NOT count as filled."""
        point = {
            "category": "",
            "origin": "  ",
        }
        assert _completeness_score(point) == 0

    def test_zero_priority(self):
        """Priority 0 is falsy — _completeness_score checks `if val` which skips 0."""
        point = {"priority": 0}
        # 0 is int but truthy check (`if val`) excludes it — this is by design
        assert _completeness_score(point) == 0

    def test_evolved_from_present(self):
        point = {"evolved_from": "some-uuid"}
        assert _completeness_score(point) == 1

    def test_partial_overlap(self):
        """Real-world scenario: existing point has more fields than new."""
        existing = {
            "tags": ["career", "remote"],
            "category": "preference",
            "priority": 1,
            "origin": "user_correction",
            "evolved_from": "old-id",
        }
        new = {
            "tags": [],
            "category": "preference",
            "priority": 3,
        }
        assert _completeness_score(existing) > _completeness_score(new)
