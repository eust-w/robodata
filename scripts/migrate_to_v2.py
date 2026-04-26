#!/usr/bin/env python3
"""Migration script for Robodata v2.0.

This script helps users migrate from the old version to the refactored v2.0
by updating configuration files, handling deprecated APIs, and providing
guidance on breaking changes.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Import both old and new modules for migration
try:
    from robodata.core.config_manager_v2 import ConfigManager
    from robodata.core.logging_system import setup_logging, LogLevel
    NEW_MODULES_AVAILABLE = True
except ImportError:
    NEW_MODULES_AVAILABLE = False
    print("Warning: New modules not available. Please ensure you're running from the refactored codebase.")


class MigrationManager:
    """Handle migration from old to new version."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.backup_dir = Path.cwd() / ".robodata_backup"
        self.migrations_performed: List[str] = []
    
    def migrate_all(self) -> None:
        """Perform all necessary migrations."""
        print("🚀 Starting Robodata v2.0 migration...")
        
        # 1. Backup existing configuration
        self._backup_configuration()
        
        # 2. Migrate configuration files
        self._migrate_configuration()
        
        # 3. Update environment variables
        self._migrate_environment_variables()
        
        # 4. Create new directory structure
        self._create_directory_structure()
        
        # 5. Generate migration report
        self._generate_migration_report()
        
        print("✅ Migration completed successfully!")
    
    def _backup_configuration(self) -> None:
        """Backup existing configuration files."""
        print("📦 Backing up existing configuration...")
        
        config_files = [
            Path.home() / ".robodata" / "config.yaml",
            Path.cwd() / "robodata_config.yaml",
            Path.cwd() / ".robodata.yaml",
        ]
        
        self.backup_dir.mkdir(exist_ok=True)
        
        for config_file in config_files:
            if config_file.exists():
                backup_path = self.backup_dir / config_file.name
                if not self.dry_run:
                    import shutil
                    shutil.copy2(config_file, backup_path)
                print(f"  Backed up: {config_file} -> {backup_path}")
                self.migrations_performed.append(f"backup_{config_file.name}")
    
    def _migrate_configuration(self) -> None:
        """Migrate configuration files to new format."""
        print("⚙️ Migrating configuration files...")
        
        old_config_file = Path.home() / ".robodata" / "config.yaml"
        if not old_config_file.exists():
            print("  No existing configuration file found")
            return
        
        try:
            with open(old_config_file, 'r') as f:
                old_config = yaml.safe_load(f) or {}
            
            # Map old configuration to new format
            new_config = self._transform_config(old_config)
            
            if not self.dry_run and NEW_MODULES_AVAILABLE:
                # Save new configuration
                config_manager = ConfigManager()
                config_manager._config_data.update(new_config)
                config_manager.save()
            
            print("  Configuration migrated successfully")
            self.migrations_performed.append("migrate_config")
            
        except Exception as e:
            print(f"  Error migrating configuration: {e}")
    
    def _transform_config(self, old_config: Dict) -> Dict:
        """Transform old configuration format to new format."""
        new_config = {
            "conversion": {
                "target_format": old_config.get("default_format", "lerobot-v3"),
                "num_workers": old_config.get("num_workers", 1),
                "fail_on_error": old_config.get("fail_on_error", False),
                "fps": old_config.get("fps"),
                "robot_type": old_config.get("robot_type"),
                "camera_mapping": old_config.get("camera_mapping", {}),
                "field_mapping": old_config.get("field_mapping", {}),
            },
            "logging": {
                "level": old_config.get("log_level", "INFO"),
                "enable_rich": old_config.get("enable_rich_logging", True),
                "enable_file_logging": old_config.get("enable_file_logging", False),
                "log_file": old_config.get("log_file"),
                "max_file_size_mb": old_config.get("max_log_file_mb", 100),
                "backup_count": old_config.get("log_backup_count", 3),
            },
            "cache": {
                "max_size": old_config.get("cache_size", 1000),
                "strategy": old_config.get("cache_strategy", "lru"),
                "ttl_seconds": old_config.get("cache_ttl"),
                "enable_disk_cache": old_config.get("enable_disk_cache", False),
                "disk_cache_dir": old_config.get("disk_cache_dir"),
                "max_disk_size_mb": old_config.get("max_disk_cache_mb", 1024),
            },
            "performance": {
                "memory_limit_mb": old_config.get("memory_limit_mb", 2048),
                "chunk_size": old_config.get("chunk_size", 1000),
                "enable_parallel": old_config.get("enable_parallel", True),
                "ray_address": old_config.get("ray_address"),
            },
            "quality": {
                "enable_scoring": old_config.get("enable_quality_scoring", True),
                "export_format": old_config.get("quality_export_format", "json"),
                "thresholds": old_config.get("quality_thresholds", {
                    "smoothness": 5.0,
                    "dead_actions": 0.1,
                    "gripper_chatter": 0.2,
                    "static_detection": 0.3,
                }),
            },
        }
        
        return new_config
    
    def _migrate_environment_variables(self) -> None:
        """Update environment variable names."""
        print("🌍 Checking environment variables...")
        
        env_mappings = {
            "ROBODATA_LOG_LEVEL": "ROBODATA_LOGGING_LEVEL",
            "ROBODATA_CACHE_SIZE": "ROBODATA_CACHE_MAX_SIZE",
            "ROBODATA_NUM_WORKERS": "ROBODATA_CONVERSION_NUM_WORKERS",
            "ROBODATA_MEMORY_LIMIT": "ROBODATA_PERFORMANCE_MEMORY_LIMIT",
        }
        
        for old_var, new_var in env_mappings.items():
            if old_var in os.environ:
                print(f"  Consider renaming {old_var} to {new_var}")
                self.migrations_performed.append(f"env_var_{old_var}")
    
    def _create_directory_structure(self) -> None:
        """Create new directory structure."""
        print("📁 Creating directory structure...")
        
        directories = [
            Path.home() / ".robodata" / "cache",
            Path.home() / ".robodata" / "logs",
            Path.home() / ".robodata" / "temp",
        ]
        
        for directory in directories:
            if not self.dry_run:
                directory.mkdir(parents=True, exist_ok=True)
            print(f"  Created directory: {directory}")
            self.migrations_performed.append(f"create_dir_{directory.name}")
    
    def _generate_migration_report(self) -> None:
        """Generate a migration report."""
        print("📊 Generating migration report...")
        
        report = {
            "migration_version": "2.0.0",
            "timestamp": time.time(),
            "dry_run": self.dry_run,
            "migrations_performed": self.migrations_performed,
            "breaking_changes": [
                "Converter API has been simplified - some advanced options moved to configuration",
                "Logging system now uses structured logging - print statements replaced",
                "Error handling improved - new exception categories and recovery strategies",
                "Cache system redesigned - new strategies and better memory management",
                "Configuration system enhanced - hierarchical config with validation",
            ],
            "new_features": [
                "Performance monitoring and optimization",
                "Adaptive processing pipelines",
                "Improved error recovery",
                "Better configuration management",
                "Enhanced caching strategies",
                "Structured logging system",
            ],
            "recommended_actions": [
                "Review your code for deprecated API usage",
                "Update any custom converters to use new pipeline system",
                "Check configuration files for new options",
                "Test your workflows with the new version",
                "Monitor performance with new monitoring tools",
            ],
        }
        
        report_file = Path.cwd() / "migration_report.json"
        if not self.dry_run:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
        
        print(f"  Migration report saved to: {report_file}")
        
        # Print summary
        print("\n📋 Migration Summary:")
        print(f"  Migrations performed: {len(self.migrations_performed)}")
        print(f"  Breaking changes: {len(report['breaking_changes'])}")
        print(f"  New features: {len(report['new_features'])}")


def check_compatibility() -> None:
    """Check if the current codebase is compatible with v2.0."""
    print("🔍 Checking compatibility...")
    
    issues = []
    
    # Check for deprecated imports
    deprecated_imports = [
        "from robodata.core.exceptions import",
        "from robodata.convert.converter import",
        "print(",
        "except:",
        "except Exception:",
    ]
    
    current_dir = Path.cwd()
    python_files = list(current_dir.rglob("*.py"))
    
    for py_file in python_files:
        if "robodata" in str(py_file):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                for deprecated in deprecated_imports:
                    if deprecated in content:
                        issues.append(f"{py_file}: Uses deprecated '{deprecated}'")
            except Exception:
                pass
    
    if issues:
        print("⚠️ Compatibility issues found:")
        for issue in issues[:10]:  # Show first 10
            print(f"  {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("✅ No compatibility issues found")


def main() -> None:
    """Main migration function."""
    parser = argparse.ArgumentParser(description="Migrate Robodata to v2.0")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without making changes")
    parser.add_argument("--check-only", action="store_true", help="Only check compatibility")
    parser.add_argument("--force", action="store_true", help="Force migration even if issues found")
    
    args = parser.parse_args()
    
    if args.check_only:
        check_compatibility()
        return
    
    # Check compatibility first unless forced
    if not args.force:
        check_compatibility()
        
        response = input("\nContinue with migration? (y/N): ")
        if response.lower() != 'y':
            print("Migration cancelled.")
            sys.exit(0)
    
    # Perform migration
    migration_manager = MigrationManager(dry_run=args.dry_run)
    migration_manager.migrate_all()
    
    if args.dry_run:
        print("\n🔍 This was a dry run. No actual changes were made.")
        print("Run without --dry-run to perform the actual migration.")


if __name__ == "__main__":
    import os
    import time
    
    main()
