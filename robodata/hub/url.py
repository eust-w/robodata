"""URL parsing utilities for HuggingFace Hub datasets.

Supports various URL formats:
    hf://lerobot/pusht
    hf://openvla/modified_libero_rlds
    huggingface://lerobot/pusht
    https://huggingface.co/datasets/lerobot/pusht
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class HFDatasetRef:
    """Reference to a HuggingFace dataset."""

    repo_id: str  # e.g., "lerobot/pusht"
    revision: str | None = None  # Branch/tag/commit
    subset: str | None = None  # Dataset subset/config


def is_hf_url(path: str) -> bool:
    """Check if the path is a HuggingFace URL.

    Args:
        path: Path or URL to check.

    Returns:
        True if this is a HuggingFace URL.
    """
    if not isinstance(path, str):
        return False

    # Check for hf:// or huggingface:// scheme
    if path.startswith(("hf://", "huggingface://")):
        return True

    # Check for huggingface.co URLs
    if "huggingface.co/datasets/" in path:
        return True

    return False


def parse_hf_url(url: str) -> HFDatasetRef:
    """Parse a HuggingFace dataset URL into components.

    Supported formats:
        hf://org/dataset
        hf://org/dataset@revision
        hf://org/dataset:subset
        hf://org/dataset:subset@revision
        huggingface://org/dataset
        https://huggingface.co/datasets/org/dataset

    Args:
        url: HuggingFace dataset URL.

    Returns:
        HFDatasetRef with parsed components.

    Raises:
        ValueError: If URL format is invalid.
    """
    original_url = url

    # Handle https://huggingface.co/datasets/org/repo URLs
    if "huggingface.co/datasets/" in url:
        match = re.search(r"huggingface\.co/datasets/([^/]+/[^/?#]+)", url)
        if match:
            repo_id = match.group(1)
            # Strip any trailing slashes or query params
            repo_id = repo_id.rstrip("/")
            return HFDatasetRef(repo_id=repo_id)
        raise ValueError(f"Invalid HuggingFace URL: {original_url}")

    # Handle hf:// and huggingface:// schemes
    if url.startswith("hf://"):
        url = url[5:]
    elif url.startswith("huggingface://"):
        url = url[14:]
    else:
        raise ValueError(f"Invalid HuggingFace URL scheme: {original_url}")

    # Parse revision (@branch or @commit)
    revision = None
    if "@" in url:
        url, revision = url.rsplit("@", 1)

    # Parse subset (:subset)
    subset = None
    if ":" in url and "/" in url:
        # Only split on : if it comes after the repo_id (after /)
        parts = url.split("/", 1)
        if len(parts) == 2 and ":" in parts[1]:
            org = parts[0]
            repo_and_subset = parts[1]
            repo, subset = repo_and_subset.split(":", 1)
            url = f"{org}/{repo}"

    # Validate repo_id format (org/repo)
    if "/" not in url:
        raise ValueError(
            f"Invalid repo_id format: {url}. Expected 'org/dataset' format."
        )

    # Clean up
    repo_id = url.strip("/")

    return HFDatasetRef(repo_id=repo_id, revision=revision, subset=subset)
