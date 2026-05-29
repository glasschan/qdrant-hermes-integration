# Qdrant Memory Plugin — Roadmap v0.7 → v0.9

Based on: codebase review (2,579 LOC, 10 files) + LCM inspiration analysis.
Current version: v0.6.2 (53 memories, working).
Goal: 從「flat vector store」進化到「smart memory system」。

---

## v0.7.0 — Robustness & Quality Guard

主題：確保記憶系統喺任何情況下都穩定、search 質素可靠。

### 7.1 Circuit Breaker for Embedding API
- **問題：** OpenRouter 掛 → 每個 turn 都 timeout → 成個 agent loop 變慢
- **方案：** 連續 N 次 embedding 失敗後，暫停 embedding calls M 秒
- **改動：** config.py 加入 `BREAKER_THRESHOLD` (已有=5) + `BREAKER_COOLDOWN_SECS` (已有=120)
- **改動：** embeddings.py 加入 `EmbeddingCircuitBreaker` class
- **改動：** store.py / provider.py 嘅 embedding calls 經過 breaker
- **文件：** embeddings.py, config.py
- **估計：** ~30 行新 code

### 7.2 Priority-Based Search Filter
- **問題：** e2e test memories (priority=5)、低質素 memories 污染 search result
- **方案：** search 預設只返 priority >= `search_min_priority`（default=1，即全部）
- **改動：** store.py search() 加入 Qdrant range filter on `priority`
- **改動：** schemas.py SEARCH_SCHEMA 加入 `min_priority` parameter
- **改動：** config.py 加入 `SEARCH_MIN_PRIORITY = 1`
- **文件：** store.py, schemas.py, config.py
- **估計：** ~15 行改動

### 7.3 Memory Evolution Tracking (evolved_from)
- **問題：** 同一條 memory 被更新多次（dedup update），但冇 record 邊條係舊版
- **方案：** qdrant_remember 加入 optional `evolved_from` field → store in payload
- **改動：** schemas.py REMEMBER_SCHEMA 加入 `evolved_from` param
- **改動：** store.py add() 存 `evolved_from` to payload
- **改動：** provider.py handle_tool_call 傳遞 evolved_from
- **改動：** consolidation.py _find_evolution_suggestions() 可以沿 evolved_from 鏈做 analysis
- **文件：** schemas.py, store.py, provider.py, consolidation.py
- **估計：** ~25 行改動

### 7.4 Stale Data Backfill Tool
- **問題：** 現有 53 points 冇 `priority` field → stale detection 唔 work
- **方案：** 新 tool `qdrant_backfill` — batch update 缺少 fields 嘅 points
- **改動：** 新 method in store.py: `backfill_fields(defaults={"priority": 3, "origin": "auto"})`
- **改動：** 新 schema in schemas.py: BACKFILL_SCHEMA
- **改動：** provider.py handle_tool_call 加 case
- **文件：** store.py, schemas.py, provider.py
- **估計：** ~40 行新 code

### 7.5 Enhanced Diagnostics
- **問題：** qdrant_consolidate 只出 report，冇 quick health check
- **方案：** qdrant_consolidate 加入 `quick=True` mode — 只返 stats（唔跑 O(n²) dedup）
- **靈感來自：** LCM 嘅 `lcm_status` + `lcm_doctor` 拆分
- **改動：** consolidation.py consolidate() 加 `quick` param
- **改動：** schemas.py CONSOLIDATE_SCHEMA 加 `quick` param
- **文件：** consolidation.py, schemas.py
- **估計：** ~20 行改動

**v0.7 總改動：** ~130 行新/改 code，5 files
**風險：** 低 — 全部係 additive changes，唔影響現有行為

---

## v0.8.0 — Smart Memory (Topic Clustering + Auto-Extract)

主題：記憶系統開始「自己識諗」— 自動 grouping、自動提取。

### 8.1 Topic Clustering (Mini-DAG)
- **靈感：** LCM 嘅 Summary DAG — 分層 summary 可以 drill-down
- **問題：** 53 條 memories 係 flat 嘅，agent 要逐條睇先知大方向
- **方案：** Consolidation 時自動 cluster 相似 memories，generate topic summary
- **實現：**
  - 新 file: `clustering.py` (~200 行)
  - 使用現有嘅 cosine similarity + connected components 做 clustering
  - 每個 cluster 生成一個 `topic_summary` payload field
  - Store topic summary as special point (category="topic_summary")
  - Topic points 有 `member_ids` field → 可以 drill-down 到成員
- **新 tool：** `qdrant_topics` — list topic clusters + member count + summary
- **改動：** consolidation.py, store.py, schemas.py, provider.py
- **估計：** ~250 行新 code

### 8.2 Auto-Extract Key Facts (Session-End)
- **靈感：** LCM 自動 capture 所有 messages；我哋唔需要咁 aggressive
- **問題：** 依賴 agent 手動 call qdrant_remember → 好多重要 facts 冇被記住
- **方案：** on_session_end hook 自動 extract key facts from conversation
- **實現：**
  - provider.py on_session_end_hook(): 抽取最近 N 條 user messages
  - 用 embedding 搵出「有冇已經記住咗」（dedup check）
  - 新 facts 用 qdrant_remember 存（category="auto", origin="auto_extract"）
  - 加 config: `session_end_auto_extract=True`（已有）
- **改動：** provider.py on_session_end_hook(), config.py
- **估計：** ~80 行新 code

### 8.3 Smart Prefetch (Adaptive Auto-Context)
- **靈感：** LCM 嘅 Zero-Cost Continuity — 短對話零 overhead
- **問題：** 每個 turn 都做 embedding + search，即使對話好短（浪費 API call）
- **方案：** 如果 conversation < N turns，skip prefetch
- **改動：** provider.py on_pre_llm_call() 加 early return
- **改動：** config.py 加入 `PREFETCH_MIN_TURNS = 3`
- **估計：** ~15 行改動

### 8.4 Dedup Quality Score
- **問題：** dedup 只睇 cosine similarity，唔考慮內容質素
- **方案：** dedup 時比較兩條 memory 嘅 completeness（邊條有更多 fields/tags）
- **改動：** store.py _find_duplicate() 加 completeness score
- **估計：** ~20 行改動

**v0.8 總改動：** ~365 行新/改 code，新 file clustering.py
**風險：** 中 — Topic clustering 係新 feature，需要測試
**Dependency：** 需要 v0.7 嘅 circuit breaker + priority filter

---

## v0.9.0 — Production Hardening

主題：Scale 到 500+ memories 時仍然快速穩定。

### 9.1 Numpy Cosine (Replace O(n²) Pure Python)
- **問題：** consolidation 嘅 _cosine_sim() 係 O(n²) pure Python
- **現狀：** 53 points → OK。500+ points → 會慢
- **方案：** 用 numpy batch cosine similarity
- **改動：** consolidation.py 加 `try: import numpy` fallback
- **估計：** ~30 行改動

### 9.2 Incremental Consolidation
- **問題：** 而家每次 consolidate 都要 scan 全部 points
- **方案：** 記住上次 consolidate 嘅 timestamp，只 scan 新增/改動嘅 points
- **改動：** consolidation.py 加 `_last_consolidation_time` tracking
- **改動：** store.py 加 `get_points_since(timestamp)` method
- **估計：** ~40 行改動

### 9.3 Memory Lifecycle (Auto-Stale + Auto-Delete)
- **靈感：** LCM 嘅 transcript GC — 自動清理過時內容
- **方案：** 
  - qdrant_consolidate 加入 `auto_stale=True` option
  - Stale memories (priority < 4, > 90 days, 0 search hits) 被降級 priority=5
  - 加入 `auto_prune=False` option — 刪除 priority=5 + > 180 days 嘅 memories
  - 兩個都要 explicit config enable（唔會自動刪）
- **改動：** consolidation.py, config.py, schemas.py
- **估計：** ~50 行改動

### 9.4 Cross-Session Context Bridge
- **問題：** 新 session 開始時，agent 冇任何 context from previous session
- **方案：** on_pre_llm_call (is_first_turn=True) 時 inject "session starter context"
  - Top 5 highest-priority memories (corrections > instructions > preferences)
  - Recent memories from last 24 hours
  - Active topic clusters
- **改動：** provider.py on_pre_llm_call() 加 first-turn logic
- **估計：** ~40 行改動

### 9.5 Config Validation + Migration
- **問題：** 新 config options 不斷加入，冇 validation
- **方案：** config.py load_config() 加 schema validation
- **改動：** config.py
- **估計：** ~30 行改動

**v0.9 總改動：** ~190 行改 code
**風險：** 低-中 — 9.3 auto-prune 要非常小心（有 safety rules）

---

## Version Timeline (建議)

| Version | Scope | LOC | 風險 | 建議時間 |
|---------|-------|-----|------|----------|
| v0.7.0 | Robustness & Quality | ~130 | 低 | 1-2 days |
| v0.8.0 | Smart Memory | ~365 | 中 | 3-5 days |
| v0.9.0 | Production Hardening | ~190 | 低-中 | 2-3 days |

每個 version 都有：
1. PLAN.md（你 approve 先開工）
2. Code + tests
3. Deploy + e2e verify
4. Git tag + release

---

## 唔會做（明確排除）

以下係 LCM 有但我哋唔需要嘅：

1. **Full message capture** — LCM 自動存所有 messages。我哋係 semantic memory，唔係 conversation archive。Hermes 已經有 session_search 做呢樣嘢。
2. **Summary DAG for conversations** — 我哋嘅 memories 係 user facts/preferences，唔係 conversation turns。DAG 嘅分層概念可以簡化為 topic clusters。
3. **LLM-Map / Agentic-Map** — 呢個係 LCM 嘅 parallel processing primitive，同 memory system 無關。
4. **Sensitive pattern redaction** — 我哋嘅 memories 係 agent-generated，唔係 raw user input。API keys 唔會入到 Qdrant。
5. **Large output externalization** — 我哋嘅 memories 好短（一條 < 500 chars），唔需要 externalize。

---

## Architecture Evolution

```
v0.6.2 (Current)              v0.7.0                    v0.8.0
                              ┌──────────────┐          ┌──────────────┐
                              │CircuitBreaker│          │ clustering.py│
                              └──────┬───────┘          │ (Topic DAG)  │
┌──────────────┐              ┌──────┴───────┐          └──────┬───────┘
│  provider.py │              │  provider.py │                 │
│  store.py    │    ──→       │  store.py    │      ──→        │  provider.py
│  schemas.py  │              │  schemas.py  │          ┌─────┴───────┐
│  config.py   │              │  config.py   │          │  provider.py │
│  embeddings  │              │  embeddings  │          │  store.py    │
│  consolidation│             │  consolidation│         │  schemas.py  │
│  indexer.py  │              │  indexer.py  │          │  config.py   │
│  cli.py      │              │  cli.py      │          └──────────────┘
└──────────────┘              └──────────────┘
Flat vector store             + Quality guard          + Topic clusters
                              + Evolution tracking     + Auto-extract
                              + Backfill tool          + Smart prefetch
```
