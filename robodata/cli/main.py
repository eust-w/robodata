"""Command-line interface for Robodata.

Provides a thin wrapper around the Robodata API for command-line usage.
"""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    name="robodata",
    help="Robodata - Robotics dataset format converter",
    add_completion=False,
)
console = Console()

# Import all command modules to register their commands
from robodata.cli.commands import (
    convert,
    filter,
    hub,
    info,
    inspect,
    quality,
    registry,
    segment,
    stats,
    visualize,
)

# Register each command module with the main app
convert.register(app)
filter.register(app)
hub.register(app)
info.register(app)
inspect.register(app)
quality.register(app)
registry.register(app)
segment.register(app)
stats.register(app)
visualize.register(app)


if __name__ == "__main__":
    app()
