"""Hub command for Robodata CLI."""

from __future__ import annotations

from pathlib import Path

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def register(app: typer.Typer) -> None:
    """Register the hub command with the Typer app."""
    @app.command("hub")
    def hub_cmd(
        query: Optional[str] = typer.Argument(None, help="Search query (optional)"),
        author: Optional[str] = typer.Option(None, "--author", help="Filter by author/organization"),
        format: Optional[str] = typer.Option(None, "--format", help="Filter by dataset format"),
        limit: int = typer.Option(20, "--limit", "-l", help="Maximum number of results"),
        download: bool = typer.Option(False, "--download", "-d", help="Download the first matching dataset"),
    ) -> None:
        """Search and download datasets from HuggingFace Hub.

        Searches for robotics datasets and provides download links.
        Use --download to automatically download the first match.

        Examples:
            robodata hub "robot manipulation"
            robodata hub --author lerobot
            robodata hub --format rlds --limit 10
            robodata hub pusht --download
        """
        try:
            from huggingface_hub import HfApi
        except ImportError:
            console.print("[red]Error:[/red] huggingface_hub is required.")
            console.print("Install with: pip install huggingface_hub")
            raise typer.Exit(1)

        api = HfApi()

        # Build search filters
        filters = {}
        if author:
            filters["author"] = author

        # Search datasets
        with console.status("[bold green]Searching HuggingFace Hub..."):
            try:
                datasets = list(api.list_datasets(
                    search=query,
                    limit=limit,
                    sort="downloads",
                    direction=-1,
                    **filters
                ))
            except Exception as e:
                console.print(f"[red]Error searching Hub:[/red] {e}")
                raise typer.Exit(1)

        if not datasets:
            console.print("[yellow]No datasets found.[/yellow]")
            raise typer.Exit(1)

        # Filter for robotics-related datasets
        robotics_keywords = [
            "robot", "manipulation", "grasp", "arm", "hand", "gripper",
            "lerobot", "rlds", "robomimic", "aloha", "bridge", "franka"
        ]

        filtered_datasets = []
        for ds in datasets:
            text = f"{ds.id} {ds.tags or []} {ds.description or ''}".lower()
            if any(keyword in text for keyword in robotics_keywords):
                filtered_datasets.append(ds)

        if not filtered_datasets:
            console.print("[yellow]No robotics datasets found.[/yellow]")
            raise typer.Exit(1)

        # Display results
        table = Table(title=f"Robotics Datasets ({len(filtered_datasets)} found)")
        table.add_column("Dataset", style="cyan")
        table.add_column("Author")
        table.add_column("Downloads")
        table.add_column("Likes")
        table.add_column("Tags")

        for ds in filtered_datasets:
            tags_str = ", ".join(ds.tags[:3]) if ds.tags else ""
            if ds.tags and len(ds.tags) > 3:
                tags_str += "..."
            table.add_row(
                ds.id,
                ds.author or "Unknown",
                str(ds.downloads or 0),
                str(ds.likes or 0),
                tags_str,
            )

        console.print(table)

        # Download first match if requested
        if download:
            if not filtered_datasets:
                console.print("[red]No datasets to download.[/red]")
                raise typer.Exit(1)

            first_ds = filtered_datasets[0]
            console.print()
            console.print(f"[cyan]Downloading:[/cyan] {first_ds.id}")

            try:
                from robodata.hub import download_dataset
                local_path = download_dataset(f"hf://{first_ds.id}")
                console.print(f"[green]Downloaded to:[/green] {local_path}")
            except Exception as e:
                console.print(f"[red]Download failed:[/red] {e}")
                raise typer.Exit(1)
