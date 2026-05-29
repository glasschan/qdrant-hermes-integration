"""Tool schemas for Qdrant memory plugin.

All tool definitions in one place. Imported by provider.py.
v0.7.0+: min_priority search filter, evolved_from on remember, quick consolidation,
         backfill tool.
v0.8.0+: topics tool.
v0.9.0+: auto_stale/auto_prune consolidation params.
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
            "min_priority": {
                "type": "integer",
                "description": "Minimum priority to include. 1=all (default), 5=only highest quality. Memories with priority > this value are excluded.",
                "minimum": 1,
                "maximum": 5,
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
                "enum": ["preference", "fact", "decision", "goal", "instruction", "correction"],
                "description": "Category (default: fact). Use 'correction' when the user corrects agent behavior.",
            },
            "priority": {
                "type": "integer",
                "description": "Priority 1 (highest) to 5 (lowest). Corrections and critical rules: 1-2. Preferences: 3. Facts: 4-5. Default: 3.",
                "minimum": 1,
                "maximum": 5,
            },
            "origin": {
                "type": "string",
                "enum": ["user_correction", "agent_discovery", "explicit", "auto"],
                "description": "How this memory was created. 'user_correction' = user corrected agent. 'agent_discovery' = agent found this independently. 'explicit' = user said 'remember this'. 'auto' = system-generated. Default: 'explicit'.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for better filtering, e.g. [\"career\", \"salary\"].",
            },
            "evolved_from": {
                "type": "string",
                "description": "ID of memory this evolved from (set automatically during dedup updates or manual evolution).",
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
        "Generate a memory consolidation report. "
        "Finds duplicates, stale memories, quality warnings, and topic clusters. "
        "Supports quick mode (skip expensive dedup), auto-stale, and auto-prune. "
        "Quick mode is safe for frequent runs. Auto-prune DELETES data — use with caution."
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
            "quick": {
                "type": "boolean",
                "description": "Quick mode: skip expensive duplicate detection, only report stats and stale items (default: false).",
            },
            "auto_stale": {
                "type": "boolean",
                "description": "Auto-stale: bump stale low-priority memories to priority 5. Requires QDRANT_AUTO_STALE config enabled (default: false).",
            },
            "auto_prune": {
                "type": "boolean",
                "description": "Auto-prune: DELETE memories with priority 5 older than 180 days. Requires QDRANT_AUTO_PRUNE config enabled (default: false). DANGEROUS — deletes data.",
            },
        },
        "required": [],
    },
}

BACKFILL_SCHEMA = {
    "name": "qdrant_backfill",
    "description": (
        "Backfill missing fields on existing memories. "
        "Scrolls all points and adds default values for fields that are missing. "
        "Dry-run defaults to true — preview changes before applying."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "defaults": {
                "type": "object",
                "description": "Field names and default values to backfill. E.g. {\"priority\": 3, \"origin\": \"auto\"}.",
                "properties": {
                    "priority": {"type": "integer", "minimum": 1, "maximum": 5},
                    "origin": {"type": "string"},
                    "category": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": True,
            },
            "dry_run": {
                "type": "boolean",
                "description": "When true (default), preview changes without applying.",
            },
        },
        "required": ["defaults"],
    },
}

TOPICS_SCHEMA = {
    "name": "qdrant_topics",
    "description": (
        "List topic clusters discovered from memories. "
        "Groups semantically similar memories into clusters with auto-generated topic labels."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "min_cluster_size": {
                "type": "integer",
                "description": "Minimum memories per cluster to include (default: 2).",
                "minimum": 2,
            },
            "similarity_threshold": {
                "type": "number",
                "description": "Cosine similarity threshold for clustering (default: 0.75).",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": [],
    },
}

# Convenience lists
CORE_SCHEMAS = [PROFILE_SCHEMA, SEARCH_SCHEMA, REMEMBER_SCHEMA, FORGET_SCHEMA]
EXTENDED_SCHEMAS = [INDEX_SCHEMA, CONSOLIDATE_SCHEMA, BACKFILL_SCHEMA, TOPICS_SCHEMA]
ALL_SCHEMAS = CORE_SCHEMAS + EXTENDED_SCHEMAS
