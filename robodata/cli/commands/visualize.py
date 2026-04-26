"""Visualize command for Robodata CLI."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

console = Console()


def _resolve_dataset_path(path_str: str, demo: bool = False):
    """Resolve a dataset path, downloading from HuggingFace Hub if needed."""
    from robodata.hub import is_hf_url

    if is_hf_url(path_str):
        from robodata.hub import download_dataset, parse_hf_url

        ref = parse_hf_url(path_str)
        console.print(f"[cyan]Downloading from HuggingFace Hub:[/cyan] {ref.repo_id}")

        with console.status("[bold green]Downloading dataset..."):
            local_path = download_dataset(path_str)

        console.print(f"[green]Downloaded to:[/green] {local_path}")
        return local_path

    from pathlib import Path
    path = Path(path_str)
    if path.exists():
        return path

    if "/" not in path_str and not path_str.startswith("."):
        try:
            from robodata.registry import DatasetRegistry

            entry = DatasetRegistry.get(path_str)
            source = DatasetRegistry.get_source(path_str, demo=demo)
            console.print(
                f"[cyan]Resolved from registry:[/cyan] {entry.name} "
                f"({entry.format})"
            )

            if source.type == "hf_hub":
                return _resolve_dataset_path(f"hf://{source.uri}")
            elif source.type == "gcs":
                console.print(f"[yellow]GCS source:[/yellow] {source.uri}")
                if source.notes:
                    console.print(f"[dim]{source.notes}[/dim]")
                console.print("Download manually with: gsutil cp -r ...")
                raise typer.Exit(1)
            else:
                console.print(f"[yellow]Source:[/yellow] {source.uri}")
                if source.notes:
                    console.print(f"[dim]{source.notes}[/dim]")
                raise typer.Exit(1)
        except Exception:
            pass

    return path


def register(app: typer.Typer) -> None:
    """Register the visualize command with the Typer app."""
    @app.command("visualize")
    def visualize_cmd(
        path: str = typer.Argument(..., help="Path to dataset (local path, hf://org/repo, or registry ID)"),
        compare: Optional[str] = typer.Option(
            None, "--compare", "-c", help="Second dataset for side-by-side comparison"
        ),
        episode: int = typer.Option(0, "--episode", "-e", help="Starting episode index"),
        backend: str = typer.Option(
            "web", "--backend", "-b", help="Viewer backend: web (default), matplotlib, opencv"
        ),
        samples: int = typer.Option(10, "--samples", "-s", help="Max episodes to load"),
        segment: bool = typer.Option(False, "--segment", help="Run segmentation and show phase labels"),
        port: int = typer.Option(0, "--port", help="Port for web viewer (0 = auto)"),
    ) -> None:
        """Visualize a dataset interactively.

        Supports all formats (RLDS, LeRobot v2/v3, Zarr) through the unified viewer.
        Use --compare to show two datasets side-by-side for comparison.
        Supports HuggingFace Hub URLs (hf://org/repo) and registry IDs.

        Backends:
            web: Browser-based viewer (default). Segment overlay, keyboard controls.
            matplotlib: Interactive with sliders. Slower playback.
            opencv: Fast playback with keyboard controls. No comparison mode.

        Examples:
            robodata visualize pusht
            robodata visualize pusht --segment
            robodata visualize hf://lerobot/pusht --backend matplotlib
            robodata visualize original/ --compare converted/
            robodata visualize dataset/ --backend opencv
        """
        from robodata.core.exceptions import RobodataError

        # Resolve paths
        try:
            resolved_path = _resolve_dataset_path(path)
        except Exception as e:
            console.print(f"[red]Error resolving dataset:[/red] {e}")
            raise typer.Exit(1)

        compare_path = None
        if compare:
            try:
                compare_path = _resolve_dataset_path(compare)
            except Exception as e:
                console.print(f"[red]Error resolving comparison dataset:[/red] {e}")
                raise typer.Exit(1)

        try:
            from robodata.visualize import UnifiedViewer

            console.print()
            console.print("[cyan]Opening viewer...[/cyan]")
            console.print(f"  Dataset: {resolved_path}")
            if compare_path:
                console.print(f"  Compare: {compare_path}")
            console.print(f"  Backend: {backend}")
            console.print(f"  Episode: {episode}")
            if segment:
                console.print("  Segmentation: enabled")
            console.print()

            viewer = UnifiedViewer(resolved_path, compare_path, backend=backend, max_episodes=samples)
            viewer.show(episode=episode, enable_segmentation=segment, port=port)
        except RobodataError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Viewer error:[/red] {e}")
            raise typer.Exit(1)
