"""HuggingFace Hub integration for Robodata.

This module provides functionality to download and work with datasets
from the HuggingFace Hub.

Usage:
    from robodata.hub import download_dataset, parse_hf_url

    # Parse hf:// URLs
    repo_id = parse_hf_url("hf://lerobot/pusht")

    # Download a dataset
    local_path = download_dataset("lerobot/pusht")
"""

from robodata.hub.download import download_dataset, get_cache_dir
from robodata.hub.url import is_hf_url, parse_hf_url

__all__ = [
    "download_dataset",
    "get_cache_dir",
    "is_hf_url",
    "parse_hf_url",
]
