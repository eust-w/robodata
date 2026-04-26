"""Info commands for Robodata CLI."""

from __future__ import annotations

import typer
from rich.console import Console

console = Console()


def register(app: typer.Typer) -> None:
    """Register the info commands with the Typer app."""
    @app.command("version")
    def version_cmd() -> None:
        """Show Robodata version information."""
        from robodata import __version__

        console.print(f"robodata version {__version__}")

    @app.command("formats")
    def formats_cmd() -> None:
        """List supported dataset formats."""
        from robodata.formats.registry import FormatRegistry

        formats = FormatRegistry.list_formats()
        
        console.print("[bold]Supported formats:[/bold]")
        for format_name in sorted(formats):
            console.print(f"  - {format_name}")
