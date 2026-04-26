"""Stats command for Robodata CLI."""

from __future__ import annotations

import json
from pathlib import Path

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

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
    """Register the stats command with the Typer app."""
    @app.command("stats")
    def stats_cmd(
        path: str = typer.Argument(..., help="Path to dataset (local path, hf://org/repo, or registry ID)"),
        sample: Optional[int] = typer.Option(None, "--sample", "-s", help="Analyze only N episodes (randomly sampled)"),
        output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save statistics to JSON file"),
        plot: bool = typer.Option(False, "--plot", help="Generate distribution plots"),
        quality: bool = typer.Option(False, "--quality", help="Include quality metrics"),
    ) -> None:
        """Compute dataset statistics and distributions.

        Analyzes action/state distributions, episode lengths, and optionally quality metrics.
        Can generate plots and export detailed statistics to JSON.

        Examples:
            robodata stats ./my_dataset
            robodata stats ./my_dataset --sample 100 --output stats.json
            robodata stats ./my_dataset --plot
            robodata stats ./my_dataset --quality --plot
        """
        from robodata.core.exceptions import RobodataError
        from robodata.stats import DatasetStats, StatsConfig

        try:
            resolved_path = _resolve_dataset_path(path)
        except Exception as e:
            console.print(f"[red]Error resolving dataset:[/red] {e}")
            raise typer.Exit(1)

        config = StatsConfig(
            sample_episodes=sample,
            include_quality=quality,
        )

        stats = DatasetStats(config)

        with console.status("[bold green]Computing statistics..."):
            try:
                result = stats.compute(resolved_path)
            except RobodataError as e:
                console.print(f"[red]Error:[/red] {e}")
                raise typer.Exit(1)

        # Basic info
        console.print()
        console.print(f"[bold]Dataset:[/bold] {resolved_path}")
        console.print(f"[bold]Episodes:[/bold] {result.num_episodes}")
        console.print(f"[bold]Total frames:[/bold] {result.total_frames}")
        console.print(f"[bold]FPS:[/bold] {result.fps or 'unknown'}")

        # Episode length stats
        if result.episode_lengths:
            import numpy as np
            lengths = np.array(result.episode_lengths)
            console.print()
            console.print("[bold]Episode Length (frames):[/bold]")
            console.print(f"  Mean: {lengths.mean():.1f}")
            console.print(f"  Std: {lengths.std():.1f}")
            console.print(f"  Min: {lengths.min()}")
            console.print(f"  Max: {lengths.max()}")
            console.print(f"  Median: {np.median(lengths):.1f}")

        # Action stats
        if result.action_stats:
            console.print()
            console.print("[bold]Action Dimensions:[/bold]")
            for dim, stats in result.action_stats.items():
                console.print(f"  Dim {dim}:")
                console.print(f"    Mean: {stats['mean']:.4f}")
                console.print(f"    Std: {stats['std']:.4f}")
                console.print(f"    Min: {stats['min']:.4f}")
                console.print(f"    Max: {stats['max']:.4f}")

        # State stats
        if result.state_stats:
            console.print()
            console.print("[bold]State Dimensions:[/bold]")
            for dim, stats in result.state_stats.items():
                console.print(f"  Dim {dim}:")
                console.print(f"    Mean: {stats['mean']:.4f}")
                console.print(f"    Std: {stats['std']:.4f}")
                console.print(f"    Min: {stats['min']:.4f}")
                console.print(f"    Max: {stats['max']:.4f}")

        # Quality stats
        if quality and result.quality_stats:
            console.print()
            console.print("[bold]Quality Metrics:[/bold]")
            console.print(f"  Overall score: {result.quality_stats['overall_score']:.2f}/10")
            for metric, score in result.quality_stats['metric_scores'].items():
                console.print(f"  {metric}: {score:.2f}")

        # Export to JSON
        if output:
            data = {
                "dataset_path": str(resolved_path),
                "num_episodes": result.num_episodes,
                "total_frames": result.total_frames,
                "fps": result.fps,
                "episode_lengths": result.episode_lengths,
                "action_stats": result.action_stats,
                "state_stats": result.state_stats,
                "quality_stats": result.quality_stats,
            }
            output.write_text(json.dumps(data, indent=2), encoding="utf-8")
            console.print()
            console.print(f"[green]Statistics exported to:[/green] {output}")

        # Generate plots
        if plot:
            try:
                import matplotlib.pyplot as plt
                import numpy as np

                # Create subplots
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                fig.suptitle(f"Dataset Statistics: {Path(resolved_path).name}", fontsize=14)

                # Episode length histogram
                if result.episode_lengths:
                    lengths = np.array(result.episode_lengths)
                    axes[0, 0].hist(lengths, bins=30, alpha=0.7, edgecolor='black')
                    axes[0, 0].set_title("Episode Length Distribution")
                    axes[0, 0].set_xlabel("Frames")
                    axes[0, 0].set_ylabel("Count")
                    axes[0, 0].axvline(lengths.mean(), color='red', linestyle='--', label=f"Mean: {lengths.mean():.1f}")
                    axes[0, 0].legend()

                # Action distribution (first dimension)
                if result.action_stats and 0 in result.action_stats:
                    actions = [ep[f"action_{0}"] for ep in result.action_data if f"action_{0}" in ep]
                    if actions:
                        axes[0, 1].hist(actions, bins=30, alpha=0.7, edgecolor='black')
                        axes[0, 1].set_title("Action Distribution (Dim 0)")
                        axes[0, 1].set_xlabel("Action Value")
                        axes[0, 1].set_ylabel("Count")

                # State distribution (first dimension)
                if result.state_stats and 0 in result.state_stats:
                    states = [ep[f"state_{0}"] for ep in result.state_data if f"state_{0}" in ep]
                    if states:
                        axes[1, 0].hist(states, bins=30, alpha=0.7, edgecolor='black')
                        axes[1, 0].set_title("State Distribution (Dim 0)")
                        axes[1, 0].set_xlabel("State Value")
                        axes[1, 0].set_ylabel("Count")

                # Quality scores
                if quality and result.quality_stats:
                    metrics = list(result.quality_stats['metric_scores'].keys())
                    scores = list(result.quality_stats['metric_scores'].values())
                    axes[1, 1].bar(metrics, scores)
                    axes[1, 1].set_title("Quality Scores")
                    axes[1, 1].set_ylabel("Score (0-10)")
                    axes[1, 1].tick_params(axis='x', rotation=45)

                plt.tight_layout()
                plot_path = Path("dataset_stats.png")
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                console.print()
                console.print(f"[green]Plots saved to:[/green] {plot_path}")
            except ImportError:
                console.print("[yellow]Warning:[/yellow] matplotlib not available for plotting")
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Could not generate plots: {e}")
