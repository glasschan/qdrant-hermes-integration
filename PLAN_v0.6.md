# Self-Healing Evolving Memory System — v0.6.0 Plan

## Goal
Transform Qdrant plugin from passive memory store → active self-healing evolving foundation.
The system should auto-inject relevant context every turn, prioritize corrections, and evolve over time.

## Current State (v0.5.0)
- 6 tools: qdrant_profile, qdrant_search, qdrant_remember, qdrant_forget, qdrant_index, qdrant_consolidate
- prefetch() + queue_prefetch() already implemented but basic (top 5, no category boost)
- Categories: preference, fact, decision, goal, instruction, conversation
- Tags support: AND logic filtering
- Dedup: pre-save semantic dedup (threshold 0.85)
- Circuit breaker: 5 consecutive failures → 120s cooldown
- Embedding: OpenRouter qwen3-embedding-4b, dim=2048

## Architecture

```
User message
  ↓
[Phase 1] queue_prefetch() → background semantic search → prefetch() injects context
  ↓
Agent responds (with memory context already in system prompt)
  ↓
[Phase 2] on_memory_write() → auto-classify, tag corrections, priority metadata
  ↓
[Phase 3] Cron consolidation → dedup, stale cleanup, pattern extraction, skill suggestions
```

## Changes by File

### Phase 1: Enhanced Auto-Context Injection

#### provider.py — Enhanced prefetch
Current: queue_prefetch does basic search(query, top_k=5)
Upgrade:
- Search with TWO queries: user message + extracted intent keywords
- Boost correction/instruction category results (score * 1.3 multiplier)
- Include category + tags in prefetch output for agent context
- Increase top_k to 8 for broader coverage
- Add score threshold (skip results < 0.4 to avoid noise)

#### store.py — Category-boosted search
Current: plain vector search
Upgrade:
- Add `category_boost` parameter to search()
- When enabled, multiply score by boost factor for correction/instruction/preference categories
- This makes "lessons learned" surface above generic facts

#### config.py — New config option
- PREFETCH_TOP_K = 8 (was hardcoded 5)
- PREFETCH_SCORE_THRESHOLD = 0.4
- PREFETCH_CATEGORY_BOOST = {"correction": 1.3, "instruction": 1.2, "preference": 1.15}

### Phase 2: Correction Tagging + Priority Metadata

#### schemas.py — Updated qdrant_remember schema
- Add "correction" to category enum
- Add "priority" field (1-5, default 3)
- Add "origin" field ("user_correction" | "agent_discovery" | "explicit" | "auto")

#### store.py — Priority-aware storage
- Store priority in payload
- Store origin in payload
- Create payload index on "priority" field
- Create payload index on "origin" field

#### provider.py — Enhanced system_prompt_block
- Show correction count and recent corrections
- Show memory health stats (total, by category)

#### provider.py — on_memory_write hook
- Mirror built-in memory writes to Qdrant
- Auto-classify: if content contains correction patterns → tag as correction
- Auto-tag: extract domain tags from content

### Phase 3: Consolidation + Skill Evolution

#### consolidation.py — Enhanced consolidation
- Add `_find_correction_patterns()`: group corrections by topic, suggest skill extraction
- Add `_find_orphans()`: memories with very low search hit rate
- Add `evolution_suggestions` to report output
- Pattern detection: 3+ corrections on same topic → suggest creating a skill

#### provider.py — on_session_end hook
- Extract key facts from full conversation at session end
- Auto-store session insights as facts

#### config.py — New consolidation settings
- CONSOLIDATION_CORRECTION_TOPIC_THRESHOLD = 3 (3 corrections = suggest skill)
- CONSOLIDATION_ORPHAN_AGE_DAYS = 60
- SESSION_END_AUTO_EXTRACT = True

## Implementation Order

1. config.py — Add new constants
2. schemas.py — Add correction category, priority, origin fields
3. store.py — Priority-aware storage + payload indexes
4. provider.py — Enhanced prefetch + category boost + on_memory_write + on_session_end
5. consolidation.py — Correction patterns + evolution suggestions
6. VERSION → v0.6.0, plugin.yaml → 0.6.0
7. Syntax check all files
8. Deploy to ~/.hermes/plugins/hermes-memory-qdrant/
9. Test all features
10. Git commit + push + release v0.6.0

## Safety
- NEVER delete collections
- NEVER auto-mutate data (consolidation is report-only)
- All new features are additive (existing behavior unchanged)
- Backward compatible with v0.5.0 data (priority defaults to 3, origin defaults to "explicit")
