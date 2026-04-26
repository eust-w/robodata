"""Quality command for Robodata CLI."""

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
    """Register the quality command with the Typer app."""
    @app.command("quality")
    def quality_cmd(
        path: str = typer.Argument(..., help="Path to dataset (local path, hf://org/repo, or registry ID)"),
        export: Optional[Path] = typer.Option(None, "--export", "-e", help="Export report to JSON file"),
        export_flagged: Optional[Path] = typer.Option(None, "--export-flagged", help="Export flagged episodes to JSON file"),
        sample: Optional[int] = typer.Option(None, "--sample", "-s", help="Analyze only N episodes (randomly sampled)"),
        quick: bool = typer.Option(False, "--quick", "-q", help="Quick mode: fewer metrics, no plots"),
        gripper_dim: int = typer.Option(-1, "--gripper-dim", help="Gripper dimension in action vector (-1=auto)"),
        fps: float = typer.Option(30.0, "--fps", help="Frames per second for time-based metrics"),
    ) -> None:
        """Analyze dataset quality with research-backed metrics.

        Scores each episode 0-10 based on 8 metrics from proprioception data alone.
        No video processing required.

        Metrics:
            - Smoothness (LDLJ) — jerk-based smoothness
            - Dead actions — zero/constant action detection
            - Gripper chatter — rapid open/close transitions
            - Static detection — idle periods
            - Timestamp regularity — dropped frames and jitter
            - Action saturation — time at hardware limits
            - Action entropy — diversity vs repetitiveness
            - Action bounds — saturation detection

        Examples:
            robodata quality ./my_dataset
            robodata quality hf://lerobot/aloha_sim_cube --export report.json
            robodata quality ./my_dataset --sample 50 --quick
            robodata quality ./my_dataset --export-flagged flagged.json
        """
        from robodata.core.exceptions import RobodataError
        from robodata.quality import QualityAnalyzer, QualityConfig

        try:
            resolved_path = _resolve_dataset_path(path)
        except Exception as e:
            console.print(f"[red]Error resolving dataset:[/red] {e}")
            raise typer.Exit(1)

        config = QualityConfig(
            gripper_dim=gripper_dim,
            fps=fps,
            quick_mode=quick,
        )

        analyzer = QualityAnalyzer(config)

        with console.status("[bold green]Analyzing dataset quality..."):
            try:
                report = analyzer.analyze(resolved_path, sample_episodes=sample)
            except RobodataError as e:
                console.print(f"[red]Error:[/red] {e}")
                raise typer.Exit(1)

        # Summary
        console.print()
        console.print(f"[bold]Dataset:[/bold] {resolved_path}")
        console.print(f"[bold]Episodes analyzed:[/bold] {len(report.episodes)}")
        console.print(f"[bold]Overall score:[/bold] {report.overall_score:.2f}/10")
        console.print(f"[bold]FPS:[/bold] {config.fps}")

        # Metric breakdown
        console.print()
        console.print("[bold]Metrics (0-10):[/bold]")
        for metric, score in report.metric_scores.items():
            console.print(f"  {metric}: {score:.2f}")

        # Flagged episodes
        flagged = [ep for ep in report.episodes if ep.flags]
        if flagged:
            console.print()
            console.print(f"[bold]Flagged episodes:[/bold] {len(flagged)}")
            console.print(f"  Threshold: <6.0 score or any critical flag")
            for ep in flagged[:10]:  # Show first 10
                flags_str = ", ".join(ep.flags)
                console.print(f"  - {ep.episode_id}: {ep.score:.2f} ({flags_str})")
            if len(flagged) > 10:
                console.print(f"  ... and {len(flagged) - 10} more")

        # Export reports
        if export:
            data = {
                "dataset_path": str(resolved_path),
                "overall_score": report.overall_score,
                "metric_scores": report.metric_scores,
                "episodes": [
                    {
                        "episode_id": ep.episode_id,
                        "score": ep.score,
                        "flags": ep.flags,
                        "metric_scores": ep.metric_scores,
                    }
                    for ep in report.episodes
                ],
            }
            export.write_text(json.dumps(data, indent=2), encoding="utf-8")
            console.print()
            console.print(f"[green]Report exported to:[/green] {export}")

        if export_flagged and flagged:
            flagged_data = [
                {
                    "episode_id": ep.episode_id,
                    "score": ep.score,
                    "flags": ep.flags,
                    "metric_scores": ep.metric_scores,
                }
                for ep in flagged
            ]
            export_flagged.write_text(json.dumps(flagged_data, indent=2), encoding="utf-8")
            console.print(f"[green]Flagged episodes exported to:[/green] {export_flagged}")
