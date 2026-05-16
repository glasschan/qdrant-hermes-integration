"""File indexer for Qdrant memory plugin.

Indexes .md and .txt files into the Qdrant memory collection.
Features: manifest sync (hash-based change detection), dry-run first,
directory scanning with exclusions, chunking by headings.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── defaults ─────────────────────────────────────────────────────────────────

DEFAULT_EXTENSIONS = {".md", ".txt"}
DEFAULT_EXCLUDE_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__",
    "dist", "build", "target", ".next", ".cache", ".hermes",
}
MAX_CHUNK_CHARS = 2000  # ~500 tokens
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


# ── helpers ──────────────────────────────────────────────────────────────────

def _hash_file(path: Path) -> str:
    """SHA-256 hash of file contents for manifest sync."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _hash_chunk(text: str) -> str:
    """Hash for a chunk to detect content changes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _chunk_markdown(text: str, file_path: str) -> list[dict]:
    """Split markdown text into chunks by headings, with fallback to paragraphs."""
    chunks = []
    sections = HEADING_PATTERN.split(text)

    # If no headings found, fall back to paragraph splitting
    if len(sections) <= 1:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        current = ""
        for para in paragraphs:
            if len(current) + len(para) > MAX_CHUNK_CHARS and current:
                chunks.append(current.strip())
                current = para
            else:
                current = current + "\n\n" + para if current else para
        if current.strip():
            chunks.append(current.strip())
        if not chunks:
            chunks = [text[:MAX_CHUNK_CHARS]]
    else:
        # sections[0] = text before first heading (preamble)
        preamble = sections[0].strip()
        if preamble:
            chunks.append(preamble[:MAX_CHUNK_CHARS])

        # sections[1:] = pairs of (heading_markers, heading_text, body)
        for i in range(1, len(sections), 3):
            if i + 2 < len(sections):
                level_markers = sections[i]
                heading_text = sections[i + 1]
                body = sections[i + 2].strip()
                heading = f"{level_markers} {heading_text}"
                chunk = f"{heading}\n\n{body}"
                if len(chunk) > MAX_CHUNK_CHARS:
                    # If too long, include heading + first part of body
                    chunks.append(chunk[:MAX_CHUNK_CHARS])
                else:
                    chunks.append(chunk)

    # Add metadata
    filename = Path(file_path).name
    return [
        {
            "text": c,
            "file_path": file_path,
            "filename": filename,
            "chunk_hash": _hash_chunk(c),
        }
        for c in chunks
    ]


# ── FileIndexer ──────────────────────────────────────────────────────────────

class FileIndexer:
    """Indexes markdown/text files into Qdrant, with manifest sync support."""

    def __init__(
        self,
        store,            # _QdrantStore instance
        embed_fn,         # embedding function: (list[str]) -> list[list[float]]
        config: dict,
    ):
        self._store = store
        self._embed = embed_fn
        self._config = config
        self._collection = store._collection

        # Configurable
        self._extensions: set[str] = DEFAULT_EXTENSIONS.copy()
        self._exclude_dirs: set[str] = DEFAULT_EXCLUDE_DIRS.copy()
        self._max_files: int = int(config.get("index_max_files", 500))
        self._max_chunk_tokens: int = int(config.get("max_chunk_tokens", 128))

        # Manifest cache: {file_path: file_hash}
        self._manifest: dict[str, str] = {}

    def set_extensions(self, exts: set[str]) -> None:
        self._extensions = exts

    def set_exclude_dirs(self, dirs: set[str]) -> None:
        self._exclude_dirs = dirs

    # ── public API ────────────────────────────────────────────────────────

    def index(
        self,
        paths: list[str],
        dry_run: bool = True,
        max_files: Optional[int] = None,
        user_id: str = "",
        agent_id: str = "",
    ) -> dict:
        """Index files from paths. Always dry-run first for safety.

        Returns a dict with:
            dry_run: bool
            files_scanned: int
            chunks_prepared: int
            new_files: list[str]
            changed_files: list[str]
            deleted_files: list[str]
            stale_ids: list[str] (point IDs to delete if not dry-run)
            chunks: list[dict] (only when dry_run=True, preview of what would be indexed)
        """
        max_f = max_files or self._max_files

        # 1. Collect files
        files = self._collect_files(paths, max_f)

        if not files:
            return {"dry_run": dry_run, "files_scanned": 0, "chunks_prepared": 0,
                    "new_files": [], "changed_files": [], "deleted_files": []}

        # 2. Load existing manifest
        self._load_manifest()

        # 3. Classify files
        new_files = []
        changed_files = []
        for fp in files:
            fhash = _hash_file(Path(fp))
            if fp not in self._manifest:
                new_files.append(fp)
            elif self._manifest[fp] != fhash:
                changed_files.append(fp)

        # 4. Detect deleted files (files in manifest but not on disk)
        deleted_files = [fp for fp in self._manifest if fp not in set(files)]

        # 5. Find stale point IDs from changed + deleted files
        stale_ids = self._find_stale_ids(changed_files + deleted_files)

        # 6. Prepare chunks for new + changed files
        chunks = []
        for fp in new_files + changed_files:
            try:
                text = Path(fp).read_text(encoding="utf-8")
                chunks.extend(_chunk_markdown(text, fp))
            except Exception as e:
                logger.warning("Failed to read %s: %s", fp, e)

        result = {
            "dry_run": dry_run,
            "files_scanned": len(files),
            "chunks_prepared": len(chunks),
            "new_files": new_files,
            "changed_files": changed_files,
            "deleted_files": deleted_files,
            "stale_ids": stale_ids,
        }

        if dry_run:
            # Show preview of first 20 chunks
            preview = [
                {"file": c["file_path"], "text_preview": c["text"][:80] + "..."}
                for c in chunks[:20]
            ]
            result["chunks_preview"] = preview
            result["total_chunks"] = len(chunks)
            if len(chunks) > 20:
                result["chunks_preview_note"] = f"Showing 20 of {len(chunks)} chunks"
            return result

        # 7. Live indexing: delete stale, upsert new
        if stale_ids:
            for sid in stale_ids:
                self._store.delete(sid)

        if chunks:
            self._embed_and_upsert(chunks, user_id, agent_id)

        # 8. Update manifest
        self._update_manifest(files, deleted_files)

        result["indexed_chunks"] = len(chunks)
        result["deleted_stale"] = len(stale_ids)
        return result

    # ── internals ─────────────────────────────────────────────────────────

    def _collect_files(self, paths: list[str], max_files: int) -> list[str]:
        """Collect .md/.txt files from paths (files or directories)."""
        collected = []
        seen = set()

        for raw in paths:
            p = Path(raw).expanduser().resolve()
            if not p.exists():
                logger.info("Skipping non-existent path: %s", raw)
                continue

            if p.is_file():
                if p.suffix.lower() in self._extensions and str(p) not in seen:
                    collected.append(str(p))
                    seen.add(str(p))
            elif p.is_dir():
                for fpath in p.rglob("*"):
                    if len(collected) >= max_files:
                        break
                    if fpath.is_file() and fpath.suffix.lower() in self._extensions:
                        # Check exclude dirs
                        parts = set(fpath.parts)
                        if parts & self._exclude_dirs:
                            continue
                        if str(fpath) not in seen:
                            collected.append(str(fpath))
                            seen.add(str(fpath))

        return collected[:max_files]

    def _load_manifest(self) -> None:
        """Load manifest from Qdrant — find points with source_type='file'."""
        try:
            from qdrant_client.http import models
            results, _ = self._store._client.scroll(
                collection_name=self._collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_type",
                            match=models.MatchValue(value="file"),
                        )
                    ]
                ),
                limit=10000,
                with_payload=True,
            )
            self._manifest = {}
            for r in results:
                fp = r.payload.get("file_path", "")
                fhash = r.payload.get("file_hash", "")
                if fp:
                    # Keep the most recent hash
                    self._manifest[fp] = fhash
        except Exception:
            self._manifest = {}

    def _find_stale_ids(self, file_paths: list[str]) -> list[str]:
        """Find Qdrant point IDs for given file paths (to be deleted)."""
        if not file_paths:
            return []
        try:
            from qdrant_client.http import models
            ids = []
            for fp in file_paths:
                results, _ = self._store._client.scroll(
                    collection_name=self._collection,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source_type",
                                match=models.MatchValue(value="file"),
                            ),
                            models.FieldCondition(
                                key="file_path",
                                match=models.MatchValue(value=fp),
                            ),
                        ]
                    ),
                    limit=100,
                    with_payload=False,
                )
                for r in results:
                    ids.append(str(r.id))
            return ids
        except Exception:
            return []

    def _embed_and_upsert(
        self, chunks: list[dict], user_id: str, agent_id: str
    ) -> None:
        """Batch embed and upsert chunks to Qdrant."""
        import uuid
        from datetime import datetime, timezone

        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        vectors = self._embed(texts)
        now = datetime.now(timezone.utc).isoformat()

        from qdrant_client.http import models

        points = []
        for i, chunk in enumerate(chunks):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vectors[i],
                    payload={
                        "content": chunk["text"],
                        "source_type": "file",
                        "file_path": chunk["file_path"],
                        "filename": chunk["filename"],
                        "chunk_hash": chunk["chunk_hash"],
                        "file_hash": _hash_file(Path(chunk["file_path"])),
                        "user_id": user_id,
                        "agent_id": agent_id,
                        "category": "file_index",
                        "created_at": now,
                    },
                )
            )

        self._store._client.upsert(
            collection_name=self._collection,
            points=points,
        )

    def _update_manifest(
        self, current_files: list[str], deleted_files: list[str]
    ) -> None:
        """Update in-memory manifest. The Qdrant points carry their own file_hash."""
        for fp in deleted_files:
            self._manifest.pop(fp, None)
        for fp in current_files:
            self._manifest[fp] = _hash_file(Path(fp))
