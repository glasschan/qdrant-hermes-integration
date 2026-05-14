# Hermes Qdrant Memory — Integration Pack

將 Hermes 嘅 built-in memory 升級做 **Qdrant vector memory**，支援 semantic search、auto-prefetch、跨 session recall。

## ⚠️ IMMUTABLE SAFETY RULES ⚠️

**🚫 NEVER touch another agent's collection** — plugin hard-scoped to `self._collection`
**🚫 NEVER delete any collection** — zero `delete_collection()` calls in codebase
**✅ Each agent auto-namespaces** — `hermes_memories_<hostname>_<profile>`

## 結構

```
hermes-qdrant-integration/
├── README.md          ← 呢份
├── SKILL.md           ← 完整 setup guide + troubleshooting
├── setup.sh           ← 一鍵 setup script
└── plugin/
    ├── __init__.py    ← MemoryProvider 實作 (~620 lines)
    └── plugin.yaml    ← Plugin metadata
```

## 快速安裝

```bash
# 1. 抄 plugin 去 Hermes
cp -r plugin ~/.hermes/hermes-agent/plugins/memory/hermes-memory-qdrant

# 2. 裝 dependency
cd ~/.hermes/hermes-agent && uv pip install qdrant-client

# 3. 開 config
hermes config set memory.provider hermes-memory-qdrant

# 4. 加 env vars 去 ~/.hermes/.env
cat >> ~/.hermes/.env << 'EOF'
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-key
EMBEDDING_BASE_URL=https://your-endpoint/v1
EMBEDDING_API_KEY=your-key
EMBEDDING_MODEL=your-model
QDRANT_COLLECTION=hermes_memories_your_project_name
EOF

# 5. 驗證
hermes doctor --fix
hermes chat -q "qdrant_remember 記低：Test memory"
hermes chat -q "qdrant_search 搜尋 'Test memory'"
```

或者一鍵 run `bash setup.sh`。

## Env Vars

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `QDRANT_URL` | No | http://localhost:6333 | Qdrant server URL |
| `QDRANT_API_KEY` | No | — | Qdrant API key |
| `QDRANT_COLLECTION` | No | auto-generated | **Per-agent namespace. Plugin never touches other collections.** |
| `EMBEDDING_BASE_URL` | **Yes** | — | OpenAI-compatible embeddings endpoint |
| `EMBEDDING_API_KEY` | **Yes** | — | Embedding API key |
| `EMBEDDING_MODEL` | No | doubao-embedding-vision | Embedding model name |

## Tools

| Tool | Description |
|------|-------------|
| `qdrant_search` | Semantic search within own collection |
| `qdrant_remember` | Store a fact within own collection |
| `qdrant_profile` | Get all memories within own collection |
| `qdrant_forget` | Delete a single point by ID only — **never drops collections** |

詳見 `SKILL.md`。
