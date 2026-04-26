"""Developer tools for Robodata code quality and maintenance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robodata.core.quality import print_quality_report, run_quality_check


def main() -> None:
    """Main entry point for developer tools."""
    parser = argparse.ArgumentParser(description="Robodata Developer Tools")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Quality check command
    quality_parser = subparsers.add_parser(
        "quality", 
        help="Run code quality checks"
    )
    quality_parser.add_argument(
        "path", 
        nargs="?", 
        default="robodata",
        help="Path to analyze (default: robodata)"
    )
    quality_parser.add_argument(
        "--severity",
        choices=["error", "warning", "info"],
        default="info",
        help="Minimum severity level to report"
    )
    
    # Format check command
    format_parser = subparsers.add_parser(
        "format",
        help="Check code formatting"
    )
    format_parser.add_argument(
        "path",
        nargs="?",
        default="robodata", 
        help="Path to check (default: robodata)"
    )
    
    # Dependencies check command
    deps_parser = subparsers.add_parser(
        "deps",
        help="Check for unused dependencies"
    )
    deps_parser.add_argument(
        "path",
        nargs="?",
        default="robodata",
        help="Path to check (default: robodata)"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "quality":
        run_quality_command(args)
    elif args.command == "format":
        run_format_command(args)
    elif args.command == "deps":
        run_deps_command(args)


def run_quality_command(args: argparse.Namespace) -> None:
    """Run code quality checks."""
    print("🔍 Running code quality checks...")
    
    path = Path(args.path)
    if not path.exists():
        print(f"❌ Path does not exist: {path}")
        sys.exit(1)
    
    report = run_quality_check(path)
    
    # Filter by severity if requested
    if args.severity != "info":
        severity_order = {"error": 0, "warning": 1, "info": 2}
        min_level = severity_order[args.severity]
        
        filtered_issues = [
            issue for issue in report.issues
            if severity_order[issue.severity] <= min_level
        ]
        report.issues = filtered_issues
    
    print_quality_report(report)
    
    # Exit with error code if there are errors
    if report.errors:
        sys.exit(1)


def run_format_command(args: argparse.Namespace) -> None:
    """Run code formatting checks."""
    print("📝 Checking code formatting...")
    
    # Try to run black if available
    try:
        import subprocess
        
        result = subprocess.run([
            "black", "--check", "--diff", args.path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Code formatting is correct!")
        else:
            print("❌ Code formatting issues found:")
            print(result.stdout)
            print("\nTo fix formatting, run:")
            print(f"  black {args.path}")
            sys.exit(1)
            
    except FileNotFoundError:
        print("⚠️  Black is not installed. Install with: pip install black")
        print("For basic formatting check, you can also use:")
        print(f"  python -m robodata.dev_tools format --basic {args.path}")


def run_deps_command(args: argparse.Namespace) -> None:
    """Run dependency checks."""
    print("🔗 Checking for unused dependencies...")
    
    # Try to run pipdeptree if available
    try:
        import subprocess
        
        result = subprocess.run([
            "pipdeptree", "--packages", "robodata"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies check completed")
            print(result.stdout)
        else:
            print("⚠️  Could not analyze dependencies")
            
    except FileNotFoundError:
        print("⚠️  pipdeptree is not installed. Install with: pip install pipdeptree")
        print("This tool helps identify unused dependencies and potential conflicts.")


if __name__ == "__main__":
    main()
