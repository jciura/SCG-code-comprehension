import os
from pathlib import Path
from typing import Dict, Any
import chromadb
from chromadb.config import Settings

default_chroma_path = "embeddings/chroma_storage"
default_collection_name = "scg_embeddings"
default_allow_reset = False


_chroma_clients: Dict[str, chromadb.PersistentClient] = {}


def get_chroma_client(storage_path: str = None,
                      allow_reset: bool = default_allow_reset) -> chromadb.PersistentClient:
    storage_path = storage_path or default_chroma_path
    if not os.path.isabs(storage_path):
        storage_path = os.path.abspath(storage_path)
    cache_key = f"{storage_path}_{allow_reset}"
    if cache_key in _chroma_clients:
        return _chroma_clients[cache_key]
    Path(storage_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=storage_path,
        settings=Settings(allow_reset=allow_reset)
    )
    _chroma_clients[cache_key] = client
    return client


def get_collection(collection_name: str = default_collection_name, storage_path: str = None, allow_reset: bool = default_allow_reset):
    client = get_chroma_client(storage_path, allow_reset)
    try:
        return client.get_collection(name=collection_name)
    except Exception as e:
        raise ValueError(f"Collection '{collection_name}' not found: {e}")


def get_or_create_collection(collection_name: str = default_collection_name, storage_path: str = None,
                             allow_reset: bool = default_allow_reset,
                             metadata: Dict[str, Any] = None):

    client = get_chroma_client(storage_path, allow_reset)
    return client.get_or_create_collection(name=collection_name, metadata=metadata or {})

