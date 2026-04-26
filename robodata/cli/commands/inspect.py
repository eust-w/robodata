"""Inspect command for Robodata CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def _resolve_dataset_path(path_str: str, demo: bool = False) -> Path:
    """Resolve a dataset path, downloading from HuggingFace Hub if needed.

    Resolution order:
    1. HuggingFace URL (hf://org/repo) — download via hub module
    2. Existing filesystem path — return as-is
    3. Registry dataset ID — lookup and resolve source URI
    4. Fall through as filesystem path (for error reporting by caller)

    Args:
        path_str: Local path, HuggingFace URL, or registry dataset ID.
        demo: If True and resolving via registry, use demo-suitable source.

    Returns:
        Resolved local path.
    """
    from robodata.hub import is_hf_url

    if is_hf_url(path_str):
        from robodata.hub import download_dataset, parse_hf_url

        ref = parse_hf_url(path_str)
        console.print(f"[cyan]Downloading from HuggingFace Hub:[/cyan] {ref.repo_id}")

        with console.status("[bold green]Downloading dataset..."):
            local_path = download_dataset(path_str)

        console.print(f"[green]Downloaded to:[/green] {local_path}")
        return local_path

    # Check filesystem first
    path = Path(path_str)
    if path.exists():
        return path

    # Try registry lookup for bare identifiers (no /, no . prefix)
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
        except ImportError:
            pass
        except Exception as e:
            from robodata.core.exceptions import DatasetNotFoundError

            if isinstance(e, DatasetNotFoundError):
                # Don't swallow — let it fall through to filesystem path
                pass
            elif isinstance(e, (typer.Exit, SystemExit)):
                raise
            # Other errors: fall through silently

    return path


def _quick_inspect_hub(path: str, output: str = "text") -> None:
    """Quick inspect a HuggingFace Hub dataset without downloading.

    Fetches metadata from Hub API and analyzes file structure.
    """
    from robodata.hub import parse_hf_url

    try:
        from huggingface_hub import HfApi, hf_hub_url
    except ImportError:
        console.print("[red]Error:[/red] huggingface_hub is required.")
        console.print("Install with: pip install huggingface_hub")
        raise typer.Exit(1)

    ref = parse_hf_url(path)
    api = HfApi()

    with console.status(f"[bold green]Fetching metadata for {ref.repo_id}..."):
        info = api.dataset_info(ref.repo_id, revision=ref.revision, files_metadata=True)

    # Analyze file types
    parquet_files = []
    hdf5_files = []
    tfrecord_files = []
    video_files = []
    file_stats = {}

    if info.siblings:
        for sibling in info.siblings:
            filename = sibling.rfilename
            # Count by extension
            if filename.endswith(".parquet"):
                parquet_files.append(filename)
            elif filename.endswith(".hdf5") or filename.endswith(".h5"):
                hdf5_files.append(filename)
            elif filename.endswith(".tfrecord"):
                tfrecord_files.append(filename)
            elif any(filename.endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv"]):
                video_files.append(filename)

            ext = Path(filename).suffix.lower()
            file_stats[ext] = file_stats.get(ext, 0) + 1

    # Summary
    console.print()
    console.print(f"[bold]Dataset:[/bold] {ref.repo_id}")
    console.print(f"[bold]Revision:[/bold] {ref.revision or 'main'}")
    console.print(f"[bold]Files:[/bold] {len(info.siblings) if info.siblings else 0}")

    # File type breakdown
    if file_stats:
        console.print()
        table = Table(title="File Types")
        table.add_column("Extension", style="cyan")
        table.add_column("Count", justify="right")
        for ext, count in sorted(file_stats.items(), key=lambda x: -x[1]):
            table.add_row(ext, str(count))
        console.print(table)

    # Key counts
    console.print()
    if parquet_files:
        console.print(f"[bold]Parquet files:[/bold] {len(parquet_files)}")
    if hdf5_files:
        console.print(f"[bold]HDF5 files:[/bold] {len(hdf5_files)}")
    if tfrecord_files:
        console.print(f"[bold]TFRecord files:[/bold] {len(tfrecord_files)}")
    if video_files:
        console.print(f"[bold]Video files:[/bold] {len(video_files)}")

    # Hub stats
    console.print()
    console.print(f"[dim]Downloads: {info.downloads or 0} | Likes: {info.likes or 0}[/dim]")

    if info.tags:
        console.print(f"[dim]Tags: {', '.join(info.tags[:5])}{'...' if len(info.tags) > 5 else ''}[/dim]")

    console.print()
    console.print("[yellow]Note:[/yellow] Use without --quick to download and get full schema details")


def _generate_config_template(info: "DatasetInfo", output_path: Path) -> None:
    """Generate a YAML config template based on dataset inspection.

    Args:
        info: Dataset inspection info.
        output_path: Path to save the generated config.
    """
    lines = [
        "# Robodata conversion configuration",
        f"# Generated from: {info.path}",
        f"# Source format: {info.format}",
        "",
        "# Target format for conversion",
        "target_format: lerobot-v3",
        "",
    ]

    # Add FPS and robot type
    if info.inferred_fps:
        lines.append(f"fps: {info.inferred_fps}")
    else:
        lines.append("# fps: 30  # Specify FPS (required if not in source)")

    if info.inferred_robot_type:
        lines.append(f"robot_type: {info.inferred_robot_type}")
    else:
        lines.append("# robot_type: franka  # Specify robot type")

    lines.append("")

    # Camera mappings
    if info.cameras:
        lines.append("# Camera name mapping (source → target)")
        lines.append("# Modify target names as needed for your use case")
        lines.append("cameras:")
        for cam_name in info.cameras:
            normalized = cam_name
            for prefix in ["steps/observation/", "observation.images.", "observation/", "steps/"]:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
            lines.append(f"  {normalized}: {normalized}")
        lines.append("")

    # Field mappings - try to identify action and state fields
    if info.observation_schema:
        lines.append("# Field mapping (source → target)")
        lines.append("# Uncomment and modify as needed")
        lines.append("fields:")

        action_candidates = []
        state_candidates = []
        for field_name in info.observation_schema:
            name_lower = field_name.lower()
            if "action" in name_lower:
                action_candidates.append(field_name)
            elif any(kw in name_lower for kw in ["state", "proprio", "joint", "ee_pos", "qpos"]):
                state_candidates.append(field_name)

        if action_candidates:
            lines.append("  # Action field mapping")
            for field in action_candidates[:3]:  # Limit to first few
                lines.append(f"  # {field}: action")
        if state_candidates:
            lines.append("  # State field mapping")
            for field in state_candidates[:3]:  # Limit to first few
                lines.append(f"  # {field}: state")

        lines.append("")
        lines.append("# Example field mappings (uncomment and modify):")
        lines.append("#  wrist_image: ext_camera")
        lines.append("#  qpos: state")
        lines.append("")

    # Add language instruction if present
    if info.has_language:
        lines.append("# Language instruction")
        lines.append("language_instruction:")
        lines.append("  enabled: true")
        lines.append("  # field: language_instruction  # Override field name if needed")
        lines.append("")

    # Write file
    output_path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]Config template written to:[/green] {output_path}")


def register(app: typer.Typer) -> None:
    """Register the inspect command with the Typer app."""
    @app.command("inspect")
    def inspect_cmd(
        path: str = typer.Argument(..., help="Path to dataset (local path or hf://org/repo)"),
        format: Optional[str] = typer.Option(
            None, "--format", "-f", help="Format hint (auto-detected if not provided)"
        ),
        output: str = typer.Option("text", "--output", "-o", help="Output format: text, json"),
        deep: bool = typer.Option(
            False, "--deep", "-d", help="Deep scan all episodes (slower but more accurate)"
        ),
        samples: int = typer.Option(5, "--samples", "-s", help="Number of episodes to sample"),
        generate_config: Optional[Path] = typer.Option(
            None, "--generate-config", "-g", help="Generate a YAML config template and save to this path"
        ),
        quick: bool = typer.Option(
            False, "--quick", "-q", help="Quick inspect for Hub datasets (metadata only, no download)"
        ),
    ) -> None:
        """Inspect a dataset and show its structure.

        Supports both local paths and HuggingFace Hub datasets.
        Use --generate-config to create a YAML config template based on the detected schema.
        Use --quick for Hub datasets to see metadata without downloading.

        Examples:
            robodata inspect my_dataset/
            robodata inspect hf://lerobot/pusht
            robodata inspect hf://lerobot/pusht --quick
            robodata inspect my_dataset/ --generate-config config.yaml
        """
        from robodata.core.exceptions import RobodataError
        from robodata.hub import is_hf_url

        # Quick inspect for Hub datasets (metadata only, no download)
        if quick and is_hf_url(path):
            _quick_inspect_hub(path, output)
            return

        # Check dataset size for HuggingFace URLs and offer --quick for large datasets
        if is_hf_url(path) and not quick:
            try:
                from huggingface_hub import HfApi
                from robodata.hub import parse_hf_url

                ref = parse_hf_url(path)
                api = HfApi()

                with console.status("[dim]Checking dataset size...[/dim]"):
                    # Use files_metadata=True to get file sizes
                    info = api.dataset_info(ref.repo_id, revision=ref.revision, files_metadata=True)
                    # Calculate total size from siblings (file list)
                    total_size = 0
                    num_files = 0
                    if info.siblings:
                        total_size = sum(s.size or 0 for s in info.siblings if s.size)
                        num_files = len(info.siblings)

                # Warn if dataset is large (> 500MB)
                if total_size > 500_000_000:
                    size_gb = total_size / 1_000_000_000
                    console.print(f"[yellow]Warning:[/yellow] Dataset is large ({size_gb:.2f} GB, {num_files} files)")
                    use_quick = typer.confirm("Use --quick mode (metadata only, no download)?", default=True)
                    if use_quick:
                        _quick_inspect_hub(path, output)
                        return
            except Exception:
                pass  # Continue with normal flow if size check fails

        from robodata.inspect import InspectionOptions, Inspector

        # Resolve HuggingFace URLs to local paths
        try:
            resolved_path = _resolve_dataset_path(path)
        except Exception as e:
            console.print(f"[red]Error downloading dataset:[/red] {e}")
            raise typer.Exit(1)

        options = InspectionOptions(
            sample_episodes=samples,
            deep_scan=deep,
        )
        inspector = Inspector(options)

        try:
            info = inspector.inspect(resolved_path, format)
        except RobodataError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

        if output == "json":
            # Convert to JSON-serializable dict
            data = {
                "path": str(info.path),
                "format": info.format,
                "format_version": info.format_version,
                "num_episodes": info.num_episodes,
                "total_frames": info.total_frames,
                "cameras": {
                    name: {
                        "height": cam.height,
                        "width": cam.width,
                        "channels": cam.channels,
                        "encoding": cam.encoding,
                    }
                    for name, cam in info.cameras.items()
                },
                "observation_schema": {
                    name: {
                        "shape": list(schema.shape),
                        "dtype": schema.dtype.value,
                    }
                    for name, schema in info.observation_schema.items()
                },
                "action_schema": (
                    {
                        "shape": list(info.action_schema.shape),
                        "dtype": info.action_schema.dtype.value,
                    }
                    if info.action_schema
                    else None
                ),
                "has_timestamps": info.has_timestamps,
                "has_language": info.has_language,
                "language_coverage": info.language_coverage,
                "has_rewards": info.has_rewards,
                "inferred_fps": info.inferred_fps,
                "inferred_gripper_index": info.inferred_gripper_index,
                "missing_required": info.missing_required,
                "sample_episode_id": info.sample_episode_id,
                "sample_num_frames": info.sample_num_frames,
                "sample_language": info.sample_language,
            }
            console.print(json.dumps(data, indent=2))
            return

        # Rich formatted output
        console.print()
        console.print(f"[bold]Dataset:[/bold] {info.path}")
        console.print(
            f"[bold]Format:[/bold] {info.format}"
            + (f" (v{info.format_version})" if info.format_version else "")
        )
        console.print(f"[bold]Episodes:[/bold] {info.num_episodes}")
        if info.total_frames:
            console.print(f"[bold]Total frames:[/bold] {info.total_frames}")

        # Schema table
        if info.observation_schema:
            console.print()
            table = Table(title="Observation Schema")
            table.add_column("Field", style="cyan")
            table.add_column("Type")
            table.add_column("Shape")

            for name, schema in info.observation_schema.items():
                table.add_row(name, schema.dtype.value, str(schema.shape))

            console.print(table)

        # Action schema
        if info.action_schema:
            console.print()
            console.print(
                f"[bold]Action:[/bold] {info.action_schema.dtype.value} {info.action_schema.shape}"
            )

        # Cameras
        if info.cameras:
            console.print()
            console.print("[bold]Cameras:[/bold]")
            for name, cam in info.cameras.items():
                console.print(f"  {name}: {cam.width}x{cam.height} ({cam.encoding})")

        # Inferred properties
        console.print()
        console.print("[bold]Inferred Properties:[/bold]")
        console.print(f"  FPS: {info.inferred_fps or '[dim]unknown[/dim]'}")
        console.print(
            f"  Gripper index: {info.inferred_gripper_index if info.inferred_gripper_index is not None else '[dim]unknown[/dim]'}"
        )
        console.print(f"  Timestamps: {'yes' if info.has_timestamps else 'no'}")
        console.print(f"  Language: {'yes' if info.has_language else 'no'}")
        if info.has_language:
            console.print(f"  Language coverage: {info.language_coverage:.0%}")
        console.print(f"  Rewards: {'yes' if info.has_rewards else 'no'}")

        # Sample
        if info.sample_episode_id:
            console.print()
            console.print("[bold]Sample Episode:[/bold]")
            console.print(f"  ID: {info.sample_episode_id}")
            if info.sample_num_frames:
                console.print(f"  Frames: {info.sample_num_frames}")
            if info.sample_language:
                console.print(f'  Language: "{info.sample_language}"')

        # Missing requirements
        if info.missing_required:
            console.print()
            console.print(
                f"[yellow]Missing for conversion:[/yellow] {', '.join(info.missing_required)}"
            )
            console.print("\n[dim]Provide these values in a config file for conversion.[/dim]")
        else:
            console.print()
            console.print("[green]Ready for conversion[/green]")

        # Generate config template if requested
        if generate_config:
            _generate_config_template(info, generate_config)
