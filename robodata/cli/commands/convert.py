"""Convert command for Robodata CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

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


def register(app: typer.Typer) -> None:
    """Register the convert command with the Typer app."""
    @app.command("convert")
    def convert_cmd(
        source: str = typer.Argument(..., help="Path to source dataset (local path or hf://org/repo)"),
        output: Path = typer.Argument(..., help="Path for output dataset"),
        config_file: Optional[Path] = typer.Option(
            None, "--config", help="YAML config file for conversion settings"
        ),
        target_format: Optional[str] = typer.Option(
            None, "--format", "-f", help="Target format (default: lerobot-v3)"
        ),
        source_format: Optional[str] = typer.Option(
            None, "--source-format", "-s", help="Source format (auto-detected if not provided)"
        ),
        fps: Optional[float] = typer.Option(None, "--fps", help="Override frames per second"),
        robot_type: Optional[str] = typer.Option(None, "--robot-type", "-r", help="Robot type"),
        camera: Optional[list[str]] = typer.Option(
            None, "--camera", "-c", help="Camera name mapping (format: source=target). Can be repeated."
        ),
        fail_on_error: bool = typer.Option(
            False, "--fail-on-error", help="Stop on first error instead of continuing"
        ),
        visualize: bool = typer.Option(
            False, "--visualize", "-v", help="Open visualizer after conversion to compare source and output"
        ),
        dry_run: bool = typer.Option(
            False, "--dry-run", "-n", help="Show what would be converted without writing any files"
        ),
        workers: int = typer.Option(
            1, "--workers", "-w", help="Number of parallel workers for episode processing (default: 1)"
        ),
        executor: str = typer.Option(
            "local",
            "--executor",
            help="Execution backend: local or ray",
        ),
        ray_address: Optional[str] = typer.Option(
            None,
            "--ray-address",
            help="Ray cluster address (e.g. auto, ray://host:10001). Only used when --executor ray.",
        ),
    ) -> None:
        """Convert a dataset from one format to another.

        Supports both local paths and HuggingFace Hub datasets.

        Example:
            robodata convert ./rlds_dataset ./output --format lerobot-v3 --fps 30
            robodata convert hf://lerobot/pusht ./output --format lerobot-v3
            robodata convert ./data.zarr ./output --format lerobot-v3 --visualize
            robodata convert ./data.zarr ./output -c front_cam=observation.images.front -c side_cam=observation.images.side
            robodata convert ./data.zarr ./output --dry-run  # Preview without writing
            robodata convert ./dataset ./output --config conversion.yaml
        """
        from robodata.convert import ConversionConfig, Converter
        from robodata.core.exceptions import RobodataError
        from robodata.formats.registry import FormatRegistry

        # Resolve HuggingFace URLs to local paths
        try:
            resolved_source = _resolve_dataset_path(source)
        except Exception as e:
            console.print(f"[red]Error downloading dataset:[/red] {e}")
            raise typer.Exit(1)

        # Load config from YAML file if provided
        if config_file:
            if not config_file.exists():
                console.print(f"[red]Error:[/red] Config file not found: {config_file}")
                raise typer.Exit(1)
            try:
                config = ConversionConfig.from_yaml(config_file)
                console.print(f"[dim]Loaded config from {config_file}[/dim]")
            except Exception as e:
                console.print(f"[red]Error loading config:[/red] {e}")
                raise typer.Exit(1)
        else:
            config = ConversionConfig()

        # CLI arguments override config file settings
        if target_format:
            config.target_format = target_format
        elif not config.target_format:
            config.target_format = "lerobot-v3"  # Default

        if fps is not None:
            config.fps = fps
        if robot_type:
            config.robot_type = robot_type
        if fail_on_error:
            config.fail_on_error = fail_on_error
        if workers > 1:
            config.num_workers = workers

        config.executor = executor
        config.ray_address = ray_address

        # Parse and merge camera mappings (format: source=target)
        # CLI camera mappings override config file mappings
        if camera:
            for mapping in camera:
                if "=" not in mapping:
                    console.print(f"[red]Invalid camera mapping '{mapping}'. Use format: source=target[/red]")
                    raise typer.Exit(1)
                src, tgt = mapping.split("=", 1)
                config.camera_mapping[src.strip()] = tgt.strip()

        # Dry run mode - just inspect and show what would happen
        if dry_run:
            console.print("[cyan]Dry run mode - no files will be written[/cyan]")
            console.print()

            # Detect source format
            detected_format = source_format or FormatRegistry.detect_format(resolved_source)
            if not detected_format:
                console.print(f"[red]Error:[/red] Could not detect format for {resolved_source}")
                raise typer.Exit(1)

            # Get reader and inspect
            reader = FormatRegistry.get_reader(detected_format)
            info = reader.inspect(resolved_source)

            console.print(f"[bold]Source:[/bold] {source}")
            console.print(f"  Format: {detected_format}")
            console.print(f"  Episodes: {info.num_episodes}")
            console.print(f"  Total frames: {info.total_frames}")
            console.print(f"  FPS: {config.fps or info.inferred_fps or 'unknown'}")

            if info.cameras:
                console.print(f"  Cameras: {len(info.cameras)}")
                for cam_name, cam_info in info.cameras.items():
                    mapped_name = config.get_camera_target(cam_name)
                    if cam_name != mapped_name:
                        console.print(f"    - {cam_name} → {mapped_name} ({cam_info.width}x{cam_info.height})")
                    else:
                        console.print(f"    - {cam_name} ({cam_info.width}x{cam_info.height})")

            if info.action_schema:
                console.print(f"  Action: {info.action_schema.shape}")
            if info.observation_schema:
                console.print(f"  Observations: {len(info.observation_schema)} fields")

            # Show field mappings if configured
            if config.field_mapping:
                console.print(f"  Field mappings: {len(config.field_mapping)}")
                for key, mapping in config.field_mapping.items():
                    target = mapping.get_target()
                    if mapping.transform:
                        console.print(f"    - {mapping.source} → {target} (transform: {mapping.transform})")
                    else:
                        console.print(f"    - {mapping.source} → {target}")

            console.print()
            console.print(f"[bold]Output:[/bold] {output}")
            console.print(f"  Format: {config.target_format}")
            console.print(f"  Robot type: {config.robot_type or info.inferred_robot_type or 'unknown'}")

            console.print()
            console.print("[green]Ready to convert.[/green] Run without --dry-run to proceed.")
            return

        converter = Converter(config)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Starting conversion...", total=None)

            def progress_callback(stage: str, current: int, total: int) -> None:
                if stage == "inspect":
                    progress.update(task, description="Inspecting source dataset...")
                elif stage == "episode":
                    progress.update(
                        task, description=f"Converting episode {current + 1}/{total or '?'}..."
                    )
                elif stage == "finalize":
                    progress.update(task, description="Writing metadata...")

            try:
                result = converter.convert(
                    resolved_source,
                    output,
                    target_format=config.target_format,
                    source_format=source_format,
                    progress_callback=progress_callback,
                )
            except RobodataError as e:
                console.print(f"[red]Error:[/red] {e}")
                raise typer.Exit(1)

        # Show results
        console.print()
        if result.success:
            console.print("[green]Conversion successful![/green]")
        else:
            console.print("[yellow]Conversion completed with errors[/yellow]")

        console.print(f"  Source format: {result.source_format}")
        console.print(f"  Target format: {result.target_format}")
        console.print(f"  Episodes converted: {result.episodes_converted}")
        if result.episodes_failed > 0:
            console.print(f"  Episodes failed: [red]{result.episodes_failed}[/red]")
        console.print(f"  Total frames: {result.total_frames}")
        console.print(f"  Output: {result.output_path}")

        if result.errors:
            console.print()
            console.print("[yellow]Errors:[/yellow]")
            for error in result.errors[:10]:  # Show first 10 errors
                console.print(f"  - {error}")
            if len(result.errors) > 10:
                console.print(f"  ... and {len(result.errors) - 10} more errors")

        if not result.success:
            raise typer.Exit(1)

        # Open visualizer if requested
        if visualize and result.success:
            try:
                from robodata.visualize import UnifiedViewer

                console.print()
                console.print("[cyan]Opening comparison viewer...[/cyan]")
                console.print(f"  Source: {resolved_source}")
                console.print(f"  Output: {output}")
                console.print("[dim]Controls: Episode/Frame sliders, Play/Pause button[/dim]")
                console.print()

                viewer = UnifiedViewer(resolved_source, output)
                viewer.show()
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Could not open visualizer: {e}")
