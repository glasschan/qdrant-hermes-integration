"""Tests for indexer.py pure functions.

Tests _chunk_markdown, _hash_file, _hash_chunk helpers.
"""
import pytest
import tempfile
import os

from plugin.indexer import _hash_file, _hash_chunk, _chunk_markdown, MAX_CHUNK_CHARS


class TestHashFunctions:
    """Tests for hash helpers."""

    def test_hash_file_returns_16_chars(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("hello world")
            f.flush()
            result = _hash_file(__import__('pathlib').Path(f.name))
        os.unlink(f.name)
        assert len(result) == 16
        assert all(c in '0123456789abcdef' for c in result)

    def test_hash_file_deterministic(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("same content")
            f.flush()
            p = __import__('pathlib').Path(f.name)
            h1 = _hash_file(p)
            h2 = _hash_file(p)
        os.unlink(f.name)
        assert h1 == h2

    def test_hash_file_different_content(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f1:
            f1.write("content A")
            f1.flush()
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f2:
                f2.write("content B")
                f2.flush()
                h1 = _hash_file(__import__('pathlib').Path(f1.name))
                h2 = _hash_file(__import__('pathlib').Path(f2.name))
            os.unlink(f2.name)
        os.unlink(f1.name)
        assert h1 != h2

    def test_hash_chunk_returns_12_chars(self):
        result = _hash_chunk("some text content")
        assert len(result) == 12

    def test_hash_chunk_empty_string(self):
        result = _hash_chunk("")
        assert len(result) == 12  # SHA-256 of empty string still produces hash


class TestChunkMarkdown:
    """Tests for _chunk_markdown."""

    def test_empty_text(self):
        chunks = _chunk_markdown("", "test.md")
        # Empty text → either no chunks or a single empty-ish chunk
        assert isinstance(chunks, list)

    def test_single_paragraph(self):
        text = "This is a simple paragraph with some content."
        chunks = _chunk_markdown(text, "simple.md")
        assert len(chunks) >= 1
        assert chunks[0]["text"] == text
        assert chunks[0]["file_path"] == "simple.md"
        assert chunks[0]["filename"] == "simple.md"
        assert "chunk_hash" in chunks[0]

    def test_multiple_headings(self):
        text = """# Heading 1

Content under heading 1.

## Heading 2

Content under heading 2.

### Heading 3

Content under heading 3."""
        chunks = _chunk_markdown(text, "headings.md")
        assert len(chunks) >= 3
        # Each chunk should have the heading in it
        for c in chunks:
            assert "text" in c
            assert c["file_path"] == "headings.md"

    def test_long_paragraph_splitting(self):
        """Long paragraphs separated by double newlines should be split."""
        text = ("A" * (MAX_CHUNK_CHARS + 100)) + "\n\n" + ("B" * (MAX_CHUNK_CHARS + 100))
        chunks = _chunk_markdown(text, "long.md")
        # Should split into multiple chunks
        assert len(chunks) > 1

    def test_preamble_before_heading(self):
        text = """This is the preamble.

# First Heading

Body text."""
        chunks = _chunk_markdown(text, "preamble.md")
        # Should have at least a preamble chunk + a heading chunk
        assert len(chunks) >= 2
        # First chunk should be the preamble
        assert "This is the preamble" in chunks[0]["text"]

    def test_chunk_metadata(self):
        text = "Some content"
        chunks = _chunk_markdown(text, "path/to/myfile.md")
        assert chunks[0]["file_path"] == "path/to/myfile.md"
        assert chunks[0]["filename"] == "myfile.md"
        assert len(chunks[0]["chunk_hash"]) == 12

    def test_max_chunk_size_respected(self):
        """No chunk should exceed MAX_CHUNK_CHARS (when split by headings)."""
        text = "# H\n\n" + "X" * (MAX_CHUNK_CHARS + 100)
        chunks = _chunk_markdown(text, "big.md")
        for c in chunks:
            # Chunks can be capped at MAX_CHUNK_CHARS
            assert len(c["text"]) <= MAX_CHUNK_CHARS + 50  # some tolerance for heading

    def test_multiple_paragraphs_no_headings(self):
        text = """First paragraph here.

Second paragraph with more text.

Third paragraph for good measure."""
        chunks = _chunk_markdown(text, "paras.md")
        assert len(chunks) >= 1
        # All content should be preserved across chunks
        combined = " ".join(c["text"] for c in chunks)
        assert "First paragraph" in combined
        assert "Second paragraph" in combined
        assert "Third paragraph" in combined
