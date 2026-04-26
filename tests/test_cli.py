"""Tests for the CLI."""

from pathlib import Path

from typer.testing import CliRunner

from robodata.cli import app

runner = CliRunner()


class TestCLI:
    """Tests for CLI commands."""

    def test_version_command(self):
        """Test the version command."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.stdout

    def test_formats_command(self):
        """Test the formats command."""
        result = runner.invoke(app, ["formats"])
        assert result.exit_code == 0
        assert "zarr" in result.stdout
        assert "lerobot-v3" in result.stdout
        assert "rosbag" in result.stdout

    def test_inspect_zarr(self, temp_zarr_dataset: Path):
        """Test inspecting a Zarr dataset via CLI."""
        result = runner.invoke(app, ["inspect", str(temp_zarr_dataset)])
        assert result.exit_code == 0
        assert "zarr" in result.stdout.lower()
        assert "Episodes:" in result.stdout or "episodes" in result.stdout.lower()

    def test_inspect_with_format_flag(self, temp_zarr_dataset: Path):
        """Test inspect with --format flag."""
        result = runner.invoke(app, ["inspect", str(temp_zarr_dataset), "--format", "zarr"])
        assert result.exit_code == 0

    def test_inspect_nonexistent_path(self):
        """Test inspect with non-existent path."""
        result = runner.invoke(app, ["inspect", "/nonexistent/path"])
        assert result.exit_code != 0

    def test_help_command(self):
        """Test help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "inspect" in result.stdout
        assert "formats" in result.stdout
        assert "version" in result.stdout
