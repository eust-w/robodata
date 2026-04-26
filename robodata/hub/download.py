"""Download datasets from HuggingFace Hub.

Uses huggingface_hub for efficient downloading with:
- Automatic caching
- Resume support for interrupted downloads
- Progress tracking
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from robodata.core.exceptions import MissingDependencyError
from robodata.hub.url import HFDatasetRef, is_hf_url, parse_hf_url

if TYPE_CHECKING:
    from collections.abc import Callable


def _check_huggingface_hub() -> None:
    """Check if huggingface_hub is available."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        raise MissingDependencyError(
            dependency="huggingface_hub",
            feature="HuggingFace Hub integration",
            install_hint="pip install huggingface_hub",
        )


def get_cache_dir() -> Path:
    """Get the cache directory for downloaded datasets.

    Uses ROBODATA_CACHE_DIR environment variable if set,
    otherwise falls back to ~/.cache/robodata/datasets.

    Returns:
        Path to cache directory.
    """
    cache_dir = os.environ.get("ROBODATA_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir) / "datasets"

    return Path.home() / ".cache" / "robodata" / "datasets"


def download_dataset(
    source: str | HFDatasetRef,
    *,
    revision: str | None = None,
    cache_dir: Path | str | None = None,
    force_download: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Download a dataset from HuggingFace Hub.

    Args:
        source: Dataset repo_id (e.g., "lerobot/pusht") or HFDatasetRef or hf:// URL.
        revision: Git revision (branch, tag, or commit hash).
        cache_dir: Where to cache downloaded files. Defaults to ~/.cache/robodata/datasets.
        force_download: Re-download even if cached.
        progress_callback: Optional callback(downloaded_bytes, total_bytes).

    Returns:
        Path to the downloaded dataset directory.

    Raises:
        MissingDependencyError: If huggingface_hub is not installed.
        ValueError: If the source format is invalid.
    """
    _check_huggingface_hub()
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    # Parse the source
    if isinstance(source, HFDatasetRef):
        ref = source
    elif is_hf_url(source):
        ref = parse_hf_url(source)
    else:
        # Assume it's a repo_id
        ref = HFDatasetRef(repo_id=source)

    # Override revision if provided
    if revision:
        ref = HFDatasetRef(
            repo_id=ref.repo_id,
            revision=revision,
            subset=ref.subset,
        )

    # Set up cache directory
    if cache_dir is None:
        cache_dir = get_cache_dir()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download the dataset
    try:
        local_dir = snapshot_download(
            repo_id=ref.repo_id,
            repo_type="dataset",
            revision=ref.revision,
            cache_dir=str(cache_dir),
            force_download=force_download,
            # Use symlinks to save disk space
            local_dir_use_symlinks=True,
        )
        return Path(local_dir)

    except RepositoryNotFoundError:
        raise ValueError(
            f"Dataset not found: {ref.repo_id}\n"
            f"Check that the dataset exists at: https://huggingface.co/datasets/{ref.repo_id}"
        )
    except GatedRepoError:
        raise ValueError(
            f"Dataset '{ref.repo_id}' requires authentication.\n"
            f"Please run: huggingface-cli login\n"
            f"And ensure you have access to: https://huggingface.co/datasets/{ref.repo_id}"
        )


def resolve_path(path: str | Path) -> Path:
    """Resolve a path that may be a local path or HuggingFace URL.

    If the path is an hf:// URL, downloads the dataset and returns the local path.
    Otherwise, returns the path as-is.

    Args:
        path: Local path or HuggingFace URL.

    Returns:
        Resolved local path.
    """
    path_str = str(path)

    if is_hf_url(path_str):
        return download_dataset(path_str)

    return Path(path)
