"""Segment command for Robodata CLI."""

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
    """Register the segment command with the Typer app."""
    @app.command("segment")
    def segment_cmd(
        path: str = typer.Argument(..., help="Path to dataset (local path, hf://org/repo, or registry ID)"),
        signal: str = typer.Option("action", "--signal", help="Signal to segment: action, state, or proprioception"),
        penalty: str = typer.Option("bic", "--penalty", help="PELT penalty (bic, aic, or numeric)"),
        cost_model: str = typer.Option("rbf", "--cost-model", help="Cost model: linear, rbf, ar, or l2"),
        min_segment_length: int = typer.Option(15, "--min-segment-length", help="Minimum segment length in frames"),
        no_normalize: bool = typer.Option(False, "--no-normalize", help="Disable per-dimension normalization"),
        sample: Optional[int] = typer.Option(None, "--sample", "-s", help="Analyze only N episodes (randomly sampled)"),
        export: Optional[Path] = typer.Option(None, "--export", "-e", help="Export segments to JSON file"),
        plot: Optional[Path] = typer.Option(None, "--plot", "-p", help="Save timeline plot to image file"),
    ) -> None:
        """Segment episodes into sub-skills using PELT changepoint detection.

        Uses proprioception signals only — no video or image processing.
        Splits episodes into contiguous phases (sub-skills, regime changes, idle periods).

        Examples:
            robodata segment ./my_dataset
            robodata segment hf://lerobot/aloha_sim_cube
            robodata segment ./my_dataset --export segments.json --plot timeline.png
            robodata segment ./my_dataset --signal action --penalty 5.0 --cost-model l2
            robodata segment ./my_dataset --sample 20
            robodata segment ./my_dataset --no-normalize
            robodata segment ./my_dataset --penalty aic --min-segment-length 15
        """
        from robodata.core.exceptions import RobodataError
        from robodata.segment import SegmentAnalyzer, SegmentConfig

        try:
            resolved_path = _resolve_dataset_path(path)
        except Exception as e:
            console.print(f"[red]Error resolving dataset:[/red] {e}")
            raise typer.Exit(1)

        # Parse penalty
        if isinstance(penalty, str):
            if penalty.lower() == "bic":
                penalty_val = "bic"
            elif penalty.lower() == "aic":
                penalty_val = "aic"
            else:
                try:
                    penalty_val = float(penalty)
                except ValueError:
                    console.print(f"[red]Invalid penalty:[/red] {penalty}")
                    raise typer.Exit(1)
        else:
            penalty_val = penalty

        config = SegmentConfig(
            signal=signal,
            penalty=penalty_val,
            cost_model=cost_model,
            min_segment_length=min_segment_length,
            normalize=not no_normalize,
        )

        analyzer = SegmentAnalyzer(config)

        with console.status("[bold green]Segmenting episodes..."):
            try:
                results = analyzer.analyze(resolved_path, sample_episodes=sample)
            except RobodataError as e:
                console.print(f"[red]Error:[/red] {e}")
                raise typer.Exit(1)

        # Summary
        console.print()
        console.print(f"[bold]Dataset:[/bold] {resolved_path}")
        console.print(f"[bold]Episodes analyzed:[/bold] {len(results)}")
        console.print(f"[bold]Signal:[/bold] {config.signal}")
        console.print(f"[bold]Penalty:[/bold] {config.penalty}")
        console.print(f"[bold]Cost model:[/bold] {config.cost_model}")

        # Show per-episode stats
        total_segments = sum(len(r.segments) for r in results)
        avg_segments = total_segments / len(results) if results else 0
        console.print(f"[bold]Total segments:[/bold] {total_segments}")
        console.print(f"[bold]Avg segments/episode:[/bold] {avg_segments:.1f}")

        # Show first few episodes
        console.print()
        console.print("[bold]Sample episodes:[/bold]")
        for result_idx, result in enumerate(results[:5]):
            segment_count = len(result.segments)
            if segment_count > 0:
                segment_lengths = [segment.end_frame - segment.start_frame for segment in result.segments]
                avg_length = sum(segment_lengths) / len(segment_lengths)
                console.print(f"  {result.episode_id}: {segment_count} segments, avg length {avg_length:.0f} frames")
            else:
                console.print(f"  {result.episode_id}: no segments detected")

        # Export results
        if export:
            data = {
                "dataset_path": str(resolved_path),
                "config": {
                    "signal": config.signal,
                    "penalty": config.penalty,
                    "cost_model": config.cost_model,
                    "min_segment_length": config.min_segment_length,
                    "normalize": config.normalize,
                },
                "episodes": [
                    {
                        "episode_id": r.episode_id,
                        "segments": [
                            {
                                "start_frame": s.start_frame,
                                "end_frame": s.end_frame,
                                "length": s.end_frame - s.start_frame,
                                "mean": s.mean,
                                "score": s.score,
                            }
                            for s in r.segments
                        ],
                    }
                    for r in results
                ],
            }
            export.write_text(json.dumps(data, indent=2), encoding="utf-8")
            console.print()
            console.print(f"[green]Segments exported to:[/green] {export}")

        # Plot timeline
        if plot:
            try:
                import matplotlib.pyplot as plt
                import matplotlib.dates as mdates
                from datetime import datetime, timedelta

                # Create timeline plot for first episode with segments
                for result in results:
                    if result.segments:
                        fig, ax = plt.subplots(figsize=(12, 3))
                        for i, seg in enumerate(result.segments):
                            start_time = timedelta(seconds=seg.start_frame / 30.0)  # Assuming 30 FPS
                            end_time = timedelta(seconds=seg.end_frame / 30.0)
                            ax.barh(0, (end_time - start_time).total_seconds(), left=start_time.total_seconds(), 
                                   height=0.5, align='center', label=f'Segment {i+1}')
                        ax.set_xlabel('Time (seconds)')
                        ax.set_title(f'Episode {result.episode_id} Segments')
                        ax.set_yticks([])
                        ax.legend(loc='upper right')
                        plt.tight_layout()
                        plt.savefig(plot, dpi=150, bbox_inches='tight')
                        plt.close()
                        console.print()
                        console.print(f"[green]Timeline plot saved to:[/green] {plot}")
                        break
            except ImportError:
                console.print("[yellow]Warning:[/yellow] matplotlib not available for plotting")
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Could not save plot: {e}")
