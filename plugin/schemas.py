"""Tool schemas for Qdrant memory plugin.

All 10 tool definitions in one place. Imported by provider.py.
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
        },
        "required": ["query"],
    },
}

REMEMBER_SCHEMA = {
    "name": "qdrant_remember",
    "description": (
        "Store a durable fact about the user in Qdrant vector memory. "
        "Use for explicit preferences, corrections, or decisions."
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

LEARNING_STORE_SCHEMA = {
    "name": "qdrant_learning_store",
    "description": (
        "Store an explicit procedural learning in the separate Qdrant learning collection. "
        "Manual/gated only — not automatic."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "lesson": {"type": "string", "description": "The durable lesson/procedure learned."},
            "learning_type": {
                "type": "string",
                "description": "tool_failure_lesson, user_correction, workflow_lesson, or environment_quirk.",
            },
            "trigger": {"type": "string", "description": "Situation that should trigger recall."},
            "mistake": {"type": "string", "description": "What went wrong."},
            "correction": {"type": "string", "description": "The corrected action."},
            "evidence": {"type": "string", "description": "Evidence supporting the lesson."},
            "tool_name": {"type": "string", "description": "Tool involved, if any."},
            "importance": {"type": "integer", "description": "Importance 1-10 (default: 7).", "minimum": 1, "maximum": 10},
            "confidence": {"type": "number", "description": "Confidence 0-1 (default: 0.8).", "minimum": 0, "maximum": 1},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags."},
            "promote_to_skill": {"type": "boolean", "description": "Mark as skill promotion candidate."},
        },
        "required": ["lesson"],
    },
}

LEARNING_SEARCH_SCHEMA = {
    "name": "qdrant_learning_search",
    "description": "Search procedural learnings from the separate Qdrant learning collection.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results (default: 5, max: 20)."},
            "learning_type": {"type": "string", "description": "Optional filter by learning_type."},
        },
        "required": ["query"],
    },
}

LEARNING_PREVIEW_SCHEMA = {
    "name": "qdrant_learning_preview",
    "description": "Preview pending gated learning candidates. Dry-run only — never writes.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

LEARNING_APPROVE_SCHEMA = {
    "name": "qdrant_learning_approve",
    "description": "Approve a pending learning candidate by ID and store it.",
    "parameters": {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string", "description": "Candidate ID to approve."},
            "dry_run": {"type": "boolean", "description": "When true, preview without storing."},
        },
        "required": ["candidate_id"],
    },
}

# Convenience lists
CORE_SCHEMAS = [PROFILE_SCHEMA, SEARCH_SCHEMA, REMEMBER_SCHEMA, FORGET_SCHEMA]
EXTENDED_SCHEMAS = [INDEX_SCHEMA, CONSOLIDATE_SCHEMA]
LEARNING_SCHEMAS = [LEARNING_STORE_SCHEMA, LEARNING_SEARCH_SCHEMA, LEARNING_PREVIEW_SCHEMA, LEARNING_APPROVE_SCHEMA]
ALL_SCHEMAS = CORE_SCHEMAS + EXTENDED_SCHEMAS + LEARNING_SCHEMAS
