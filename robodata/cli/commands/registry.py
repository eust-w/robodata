"""Registry command for Robodata CLI."""

from __future__ import annotations

import json
from pathlib import Path

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def register(app: typer.Typer) -> None:
    """Register the registry command with the Typer app."""
    registry_app = typer.Typer(name="registry", help="Dataset registry - browse and search known robotics datasets.")
    app.add_typer(registry_app, name="registry")

    @registry_app.command("list")
    def list_cmd(
        format: Optional[str] = typer.Option(None, "--format", help="Filter by format"),
        embodiment: Optional[str] = typer.Option(None, "--embodiment", help="Filter by embodiment"),
        tag: Optional[list[str]] = typer.Option(None, "--tag", help="Filter by tag (can repeat)"),
        html: bool = typer.Option(False, "--html", help="Generate interactive HTML viewer"),
        output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save list to JSON file"),
    ) -> None:
        """List and filter datasets from the registry.

        Examples:
            robodata registry list
            robodata registry list --format rlds
            robodata registry list --embodiment franka --tag real_world
            robodata registry list --html
            robodata registry list --output datasets.json
        """
        from robodata.registry import DatasetRegistry

        entries = DatasetRegistry.list_all()

        # Apply filters
        if format:
            entries = [e for e in entries if e.format == format]
        if embodiment:
            entries = [e for e in entries if embodiment in e.embodiment]
        if tag:
            entries = [e for e in entries if any(t in e.tags for t in tag)]

        if not entries:
            console.print("[yellow]No datasets found matching filters.[/yellow]")
            raise typer.Exit(1)

        if html:
            from robodata.registry.html import generate_html

            html_path = Path("registry.html")
            generate_html(entries, html_path)
            console.print(f"[green]Interactive HTML viewer written to:[/green] {html_path}")
            console.print("[dim]Open in a browser to search and filter.[/dim]")
            return

        # Rich table output
        table = Table(title=f"Datasets ({len(entries)} found)")
        table.add_column("ID", style="cyan")
        table.add_column("Name")
        table.add_column("Format")
        table.add_column("Embodiment")
        table.add_column("Tags")
        table.add_column("Verified", justify="center")

        for entry in entries:
            tags_str = ", ".join(entry.tags[:3])
            if len(entry.tags) > 3:
                tags_str += "..."
            table.add_row(
                entry.id,
                entry.name,
                entry.format,
                ", ".join(entry.embodiment),
                tags_str,
                "✓" if entry.verified else "✗",
            )

        console.print(table)

        if output:
            data = [
                {
                    "id": e.id,
                    "name": e.name,
                    "description": e.description,
                    "format": e.format,
                    "embodiment": e.embodiment,
                    "tags": e.tags,
                    "verified": e.verified,
                    "sources": [
                        {"type": s.type, "uri": s.uri, "notes": s.notes}
                        for s in e.sources
                    ],
                }
                for e in entries
            ]
            output.write_text(json.dumps(data, indent=2), encoding="utf-8")
            console.print(f"[green]Dataset list saved to:[/green] {output}")

    @registry_app.command("get")
    def get_cmd(
        dataset_id: str = typer.Argument(..., help="Dataset ID from registry"),
        demo: bool = typer.Option(False, "--demo", help="Use demo-suitable source if available"),
    ) -> None:
        """Get detailed information about a dataset.

        Shows all sources, metadata, and download instructions.

        Example:
            robodata registry get pusht
            robodata registry get pusht --demo
        """
        from robodata.registry import DatasetRegistry

        try:
            entry = DatasetRegistry.get(dataset_id)
        except Exception as e:
            console.print(f"[red]Dataset not found:[/red] {dataset_id}")
            raise typer.Exit(1)

        console.print()
        console.print(f"[bold]{entry.name}[/bold] ({entry.id})")
        console.print(f"[dim]{entry.description}[/dim]")
        console.print()
        console.print(f"[bold]Format:[/bold] {entry.format}")
        console.print(f"[bold]Embodiment:[/bold] {', '.join(entry.embodiment)}")
        console.print(f"[bold]Tags:[/bold] {', '.join(entry.tags)}")
        console.print(f"[bold]Verified:[/bold] {'Yes' if entry.verified else 'No'}")

        if entry.paper_url:
            console.print(f"[bold]Paper:[/bold] {entry.paper_url}")
        if entry.license:
            console.print(f"[bold]License:[/bold] {entry.license}")

        console.print()
        console.print("[bold]Sources:[/bold]")
        for i, source in enumerate(entry.sources, 1):
            console.print(f"  {i}. {source.type.upper()}: {source.uri}")
            if source.notes:
                console.print(f"     [dim]{source.notes}[/dim]")

        # Show demo source if available
        if demo and entry.demo_suitable and entry.demo_source_index is not None:
            demo_source = entry.sources[entry.demo_source_index]
            console.print()
            console.print("[bold]Demo source:[/bold]")
            console.print(f"  {demo_source.type.upper()}: {demo_source.uri}")
            if entry.demo_episodes:
                console.print(f"  Episodes: {entry.demo_episodes}")

    @registry_app.command("validate")
    def validate_cmd(
        path: Optional[Path] = typer.Option(None, "--path", help="Path to datasets.json file (uses default if not provided)"),
        probe: bool = typer.Option(False, "--probe", help="Check that source URIs are reachable"),
    ) -> None:
        """Validate the registry dataset entries.

        Checks for required fields, valid tags, and optionally tests source URIs.

        Example:
            robodata registry validate
            robodata registry validate --probe
            robodata registry validate --path custom_datasets.json
        """
        from robodata.registry import DatasetRegistry

        try:
            DatasetRegistry.validate(path, probe)
            console.print("[green]Registry validation passed![/green]")
        except Exception as e:
            console.print(f"[red]Validation failed:[/red] {e}")
            raise typer.Exit(1)
