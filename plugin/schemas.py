"""Tool schemas for Qdrant memory plugin.

All 6 tool definitions in one place. Imported by provider.py.
"""

PROFILE_SCHEMA = {
    "name": "qdrant_profile",
    "description": (
        "Retrieve all stored vector memories about the user — preferences, facts, "
        "project context. Returns everything from Qdrant vector store."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "qdrant_search",
    "description": (
        "Search vector memories by semantic meaning. Uses Qdrant + embeddings "
        "to find the most relevant stored facts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for semantically."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
            "recency_weight": {
                "type": "number",
                "description": "How much to favor recent memories. 0.0=pure relevance, 1.0=50/50 relevance+freshness (default: 0.0).",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags to filter by (AND logic — all must match).",
            },
        },
        "required": ["query"],
    },
}

REMEMBER_SCHEMA = {
    "name": "qdrant_remember",
    "description": (
        "Store a durable fact about the user in Qdrant vector memory. "
        "Use for explicit preferences, corrections, or decisions. "
        "If the same semantic content already exists (dedup), "
        "updates the existing entry instead of creating a duplicate."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to remember."},
            "category": {
                "type": "string",
                "enum": ["preference", "fact", "decision", "goal", "instruction"],
                "description": "Category (default: fact).",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for better filtering, e.g. [\"career\", \"salary\"].",
            },
        },
        "required": ["content"],
    },
}

FORGET_SCHEMA = {
    "name": "qdrant_forget",
    "description": (
        "Delete a vector memory by its point ID. "
        "Dry-run defaults to true — always preview first before live deletion."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "point_id": {"type": "string", "description": "The point ID (UUID) to delete."},
            "dry_run": {
                "type": "boolean",
                "description": "When true (default), only report what would be deleted without deleting.",
            },
        },
        "required": ["point_id"],
    },
}

INDEX_SCHEMA = {
    "name": "qdrant_index",
    "description": (
        "Safely index markdown/text files or directories into Qdrant memory. "
        "Dry-run defaults to true — always preview first. "
        "Supports manifest sync: detects changed and deleted files."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Files or directories to index.",
            },
            "dry_run": {
                "type": "boolean",
                "description": "When true (default), preview chunks without upserting.",
            },
            "max_files": {
                "type": "integer",
                "description": "Max files to scan (default: 500).",
                "minimum": 1,
            },
        },
        "required": ["paths"],
    },
}

CONSOLIDATE_SCHEMA = {
    "name": "qdrant_consolidate",
    "description": (
        "Generate a read-only memory consolidation report. "
        "Finds duplicates, stale memories, and quality warnings. "
        "NEVER mutates data — report-only by design."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "description": "Which collections to scan: memory, learning, or both.",
                "enum": ["memory", "learning", "both"],
            },
            "max_points": {
                "type": "integer",
                "description": "Max points to scan (default: 500).",
                "minimum": 10,
            },
            "max_groups": {
                "type": "integer",
                "description": "Max duplicate groups to return (default: 20).",
                "minimum": 1,
            },
            "include_examples": {
                "type": "boolean",
                "description": "Include redacted content examples in report.",
            },
        },
        "required": [],
    },
}

# Convenience lists
CORE_SCHEMAS = [PROFILE_SCHEMA, SEARCH_SCHEMA, REMEMBER_SCHEMA, FORGET_SCHEMA]
EXTENDED_SCHEMAS = [INDEX_SCHEMA, CONSOLIDATE_SCHEMA]
ALL_SCHEMAS = CORE_SCHEMAS + EXTENDED_SCHEMAS
