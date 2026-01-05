import os
from pathlib import Path
from typing import Any, Dict

import chromadb
from chromadb.config import Settings

from src.core.config import default_chroma_path, default_collection_name

default_allow_reset = False


_chroma_clients: Dict[str, chromadb.PersistentClient] = {}


def get_chroma_client(
    storage_path: str = None, allow_reset: bool = default_allow_reset
) -> chromadb.PersistentClient:
    """
    Returns a cached or newly initialized Chroma persistent client.

    Creates the client at the given storage path (defaulting to `default_chroma_path`).
    Uses an internal cache to avoid repeated initializations.

    Args:
        storage_path (str, optional): Filesystem path for Chroma storage.
            Defaults to `default_chroma_path`.
        allow_reset (bool, optional): Whether the Chroma client can reset collections.
            Defaults to `False`.

    Returns:
        chromadb.PersistentClient: Initialized or cached Chroma client instance.
    """
    storage_path = storage_path or default_chroma_path
    if not os.path.isabs(storage_path):
        storage_path = os.path.abspath(storage_path)
    cache_key = f"{storage_path}_{allow_reset}"
    if cache_key in _chroma_clients:
        return _chroma_clients[cache_key]
    Path(storage_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=storage_path, settings=Settings(allow_reset=allow_reset)
    )
    _chroma_clients[cache_key] = client
    return client


def get_collection(
    collection_name: str = default_collection_name,
    storage_path: str = None,
    allow_reset: bool = default_allow_reset,
):
    """
    Retrieves an existing Chroma collection by name.

    Args:
        collection_name (str, optional): Name of the collection to retrieve.
            Defaults to `default_collection_name`.
        storage_path (str, optional): Path to the Chroma storage directory.
            Defaults to `default_chroma_path`.
        allow_reset (bool, optional): Whether to allow resetting the client.
            Defaults to `False`.

    Returns:
        chromadb.Collection: The requested Chroma collection.

    Raises:
        ValueError: If the collection does not exist or cannot be retrieved.
    """
    client = get_chroma_client(storage_path, allow_reset)
    try:
        return client.get_collection(name=collection_name)
    except Exception as e:
        raise ValueError(f"Collection '{collection_name}' not found: {e}")


def get_or_create_collection(
    collection_name: str = default_collection_name,
    storage_path: str = None,
    allow_reset: bool = default_allow_reset,
    metadata: Dict[str, Any] = None,
):
    """
    Retrieves or creates a Chroma collection with the given name.

    Ensures the collection exists, creating it if necessary, and attaches
    optional metadata.

    Args:
        collection_name (str, optional): Name of the collection to get or create.
            Defaults to `default_collection_name`.
        storage_path (str, optional): Path to the Chroma storage directory.
            Defaults to `default_chroma_path`.
        allow_reset (bool, optional): Whether to allow resetting the client.
            Defaults to `False`.
        metadata (Dict[str, Any], optional): Additional metadata for the collection.
            Defaults to an empty dict.

    Returns:
        chromadb.Collection: The created or existing Chroma collection.
    """
    client = get_chroma_client(storage_path, allow_reset)
    if not metadata:
        metadata = {"description": "Code embeddings collection"}
    return client.get_or_create_collection(name=collection_name, metadata=metadata or {})
