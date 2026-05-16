"""OpenAI-compatible embedding client for Qdrant memory plugin."""

from __future__ import annotations

from urllib.parse import urljoin


def embed(texts: list[str], config: dict) -> list[list[float]]:
    """Get embeddings from an OpenAI-compatible API."""
    import requests

    base = config["embedding_base_url"].rstrip("/")
    url = urljoin(base + "/", "embeddings")

    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {config['embedding_api_key']}",
            "Content-Type": "application/json",
        },
        json={
            "model": config["embedding_model"],
            "input": texts,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    items = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in items]
