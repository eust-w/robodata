#!/usr/bin/env python3
"""Generate a static HTML page for the dataset registry (for GitHub Pages)."""

from pathlib import Path

from robodata.registry import DatasetRegistry
from robodata.registry.html import generate_registry_html


def main() -> None:
    entries = list(DatasetRegistry.list())
    html = generate_registry_html(entries)

    out = Path(__file__).resolve().parent.parent / "docs" / "registry.html"
    out.write_text(html)
    print(f"Generated {out} with {len(entries)} datasets")


if __name__ == "__main__":
    main()
