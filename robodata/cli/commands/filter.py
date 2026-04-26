"""Filter command for Robodata CLI."""

from __future__ import annotations

import json
from pathlib import Path

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
    """Register the filter command with the Typer app."""
    @app.command("filter")
    def filter_cmd(
        input_path: str = typer.Argument(..., help="Input dataset path"),
        output_path: Optional[Path] = typer.Argument(None, help="Output dataset path (dry-run if omitted)"),
        min_quality: Optional[float] = typer.Option(None, "--min-quality", help="Minimum quality score (0-10)"),
        exclude_flags: Optional[str] = typer.Option(None, "--exclude-flags", help="Comma-separated flags to exclude"),
        include_episodes: Optional[str] = typer.Option(None, "--include-episodes", help="Comma-separated episode IDs to include"),
        exclude_episodes: Optional[str] = typer.Option(None, "--exclude-episodes", help="Comma-separated episode IDs to exclude"),
        from_report: Optional[Path] = typer.Option(None, "--from-report", help="Use pre-computed quality report JSON"),
    ) -> None:
        """Filter dataset episodes based on quality scores or explicit lists.

        Writes the same format as input — no format conversion.

        Examples:
            robodata filter ./my_dataset --min-quality 6.0                          # Dry-run preview
            robodata filter ./my_dataset ./filtered --min-quality 6.0               # Write filtered dataset
            robodata filter ./my_dataset ./filtered --exclude-flags jerky,mostly_static
            robodata filter ./my_dataset ./filtered --from-report report.json --min-quality 7.0
            robodata filter ./my_dataset ./filtered --include-episodes ep_000,ep_001,ep_005
            robodata filter ./my_dataset ./filtered --exclude-episodes ep_003,ep_010
        """
        from robodata.core.exceptions import RobodataError
        from robodata.filter import FilterEngine, FilterConfig

        try:
            resolved_input = _resolve_dataset_path(input_path)
        except Exception as e:
            console.print(f"[red]Error resolving input dataset:[/red] {e}")
            raise typer.Exit(1)

        # Build filter config
        config_kwargs = {}
        if min_quality is not None:
            config_kwargs["min_quality"] = min_quality
        if exclude_flags:
            config_kwargs["exclude_flags"] = [f.strip() for f in exclude_flags.split(",")]
        if include_episodes:
            config_kwargs["include_episodes"] = [f.strip() for f in include_episodes.split(",")]
        if exclude_episodes:
            config_kwargs["exclude_episodes"] = [f.strip() for f in exclude_episodes.split(",")]

        config = FilterConfig(**config_kwargs)

        # Load quality report if provided
        report = None
        if from_report:
            if not from_report.exists():
                console.print(f"[red]Report file not found:[/red] {from_report}")
                raise typer.Exit(1)
            report_data = json.loads(from_report.read_text(encoding="utf-8"))
            from robodata.quality import QualityReport
            report = QualityReport.from_dict(report_data)

        engine = FilterEngine(config)

        with console.status("[bold green]Filtering episodes..."):
            try:
                result = engine.filter(resolved_input, output_path, quality_report=report)
            except RobodataError as e:
                console.print(f"[red]Error:[/red] {e}")
                raise typer.Exit(1)

        # Show results
        console.print()
        console.print(f"[bold]Input:[/bold] {resolved_input}")
        console.print(f"  Episodes: {result.input_episodes}")
        console.print(f"  Frames: {result.input_frames}")

        console.print(f"[bold]Output:[/bold] {result.output_path or 'dry-run'}")
        console.print(f"  Episodes: {result.output_episodes}")
        console.print(f"  Frames: {result.output_frames}")
        console.print(f"  Filtered: {result.input_episodes - result.output_episodes} episodes")

        if result.written:
            console.print()
            console.print("[green]Filtered dataset written successfully![/green]")
        else:
            console.print()
            console.print("[yellow]Dry-run mode - no files written[/yellow]")
            console.print("Provide output path to write filtered dataset.")
