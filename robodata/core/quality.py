"""Code quality and validation utilities for Robodata."""

from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Set

from robodata.core.constants import MAX_FILE_SIZE_MB, SUPPORTED_FORMATS


@dataclass
class QualityIssue:
    """Represents a code quality issue."""
    
    severity: str  # "error", "warning", "info"
    message: str
    file_path: str
    line_number: int
    rule_name: str
    
    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.file_path}:{self.line_number} - {self.message}"


@dataclass
class QualityReport:
    """Report of code quality analysis."""
    
    issues: List[QualityIssue]
    total_files: int
    analyzed_files: int
    
    @property
    def errors(self) -> List[QualityIssue]:
        return [i for i in self.issues if i.severity == "error"]
    
    @property
    def warnings(self) -> List[QualityIssue]:
        return [i for i in self.issues if i.severity == "warning"]
    
    @property
    def info_count(self) -> int:
        return len([i for i in self.issues if i.severity == "info"])


class CodeQualityChecker:
    """Code quality checker for Robodata codebase."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.issues: List[QualityIssue] = []
        
        # Define quality rules
        self.rules = {
            'no_hardcoded_paths': self._check_hardcoded_paths,
            'no_magic_numbers': self._check_magic_numbers,
            'proper_docstrings': self._check_docstrings,
            'type_annotations': self._check_type_annotations,
            'error_handling': self._check_error_handling,
            'naming_conventions': self._check_naming_conventions,
            'import_organization': self._check_import_organization,
        }
    
    def analyze_directory(self, directory: Path) -> QualityReport:
        """Analyze all Python files in a directory.
        
        Args:
            directory: Directory to analyze
            
        Returns:
            QualityReport with all issues found
        """
        self.issues.clear()
        
        python_files = list(directory.rglob("*.py"))
        total_files = len(python_files)
        analyzed_files = 0
        
        for file_path in python_files:
            try:
                self._analyze_file(file_path)
                analyzed_files += 1
            except Exception as e:
                self.logger.error(f"Error analyzing {file_path}: {e}")
        
        return QualityReport(
            issues=self.issues.copy(),
            total_files=total_files,
            analyzed_files=analyzed_files
        )
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            self.logger.error(f"Could not read {file_path}: {e}")
            return
        
        # Apply all rules
        for rule_name, rule_func in self.rules.items():
            try:
                rule_func(file_path, lines)
            except Exception as e:
                self.logger.error(f"Error in rule {rule_name} for {file_path}: {e}")
    
    def _check_hardcoded_paths(self, file_path: Path, lines: List[str]) -> None:
        """Check for hardcoded absolute paths."""
        path_patterns = [
            r'["\']/[/\\]',  # Unix/Windows absolute paths
            r'["\'][A-Za-z]:[\\/]',  # Windows drive paths
            r'["\']home/',  # Unix home directory
            r'["\']Users/',  # macOS user directory
        ]
        
        for line_num, line in enumerate(lines, 1):
            for pattern in path_patterns:
                if re.search(pattern, line) and not line.strip().startswith('#'):
                    self.issues.append(QualityIssue(
                        severity="warning",
                        message=f"Potential hardcoded path: {line.strip()}",
                        file_path=str(file_path),
                        line_number=line_num,
                        rule_name="no_hardcoded_paths"
                    ))
    
    def _check_magic_numbers(self, file_path: Path, lines: List[str]) -> None:
        """Check for magic numbers (except common small values)."""
        # Allow small numbers (0-10) and common values
        allowed_numbers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000}
        
        for line_num, line in enumerate(lines, 1):
            # Skip comments and docstrings
            if line.strip().startswith('#') or '"""' in line or "'''" in line:
                continue
            
            # Find numbers
            numbers = re.findall(r'\b([1-9]\d{2,})\b', line)
            for num_str in numbers:
                num = int(num_str)
                if num not in allowed_numbers:
                    self.issues.append(QualityIssue(
                        severity="info",
                        message=f"Magic number {num} should be named constant",
                        file_path=str(file_path),
                        line_number=line_num,
                        rule_name="no_magic_numbers"
                    ))
    
    def _check_docstrings(self, file_path: Path, lines: List[str]) -> None:
        """Check for proper docstrings."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            return
        
        try:
            tree = ast.parse(content)
        except:
            return
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    self.issues.append(QualityIssue(
                        severity="warning",
                        message=f"Missing docstring for {node.name}",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        rule_name="proper_docstrings"
                    ))
    
    def _check_type_annotations(self, file_path: Path, lines: List[str]) -> None:
        """Check for type annotations in function signatures."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            return
        
        try:
            tree = ast.parse(content)
        except:
            return
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private/internal functions
                if node.name.startswith('_'):
                    continue
                
                has_annotations = (
                    node.returns is not None or
                    any(arg.annotation is not None for arg in node.args.args)
                )
                
                if not has_annotations:
                    self.issues.append(QualityIssue(
                        severity="info",
                        message=f"Function {node.name} lacks type annotations",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        rule_name="type_annotations"
                    ))
    
    def _check_error_handling(self, file_path: Path, lines: List[str]) -> None:
        """Check for proper error handling."""
        for line_num, line in enumerate(lines, 1):
            # Check for bare except clauses
            if re.search(r'except\s*:', line):
                self.issues.append(QualityIssue(
                    severity="warning",
                    message="Bare except clause - should specify exception type",
                    file_path=str(file_path),
                    line_number=line_num,
                    rule_name="error_handling"
                ))
            
            # Check for except Exception without logging
            if re.search(r'except\s+Exception', line):
                # Look ahead for logging in the next few lines
                has_logging = False
                for i in range(3):
                    if line_num + i < len(lines):
                        next_line = lines[line_num + i]
                        if 'log' in next_line.lower() or 'print' in next_line.lower():
                            has_logging = True
                            break
                
                if not has_logging:
                    self.issues.append(QualityIssue(
                        severity="info",
                        message="Exception caught without logging/printing",
                        file_path=str(file_path),
                        line_number=line_num,
                        rule_name="error_handling"
                    ))
    
    def _check_naming_conventions(self, file_path: Path, lines: List[str]) -> None:
        """Check for naming convention violations."""
        for line_num, line in enumerate(lines, 1):
            # Check class names (PascalCase)
            class_match = re.search(r'class\s+([a-z][a-zA-Z0-9_]*)\s*:', line)
            if class_match:
                class_name = class_match.group(1)
                self.issues.append(QualityIssue(
                    severity="warning",
                    message=f"Class name '{class_name}' should use PascalCase",
                    file_path=str(file_path),
                    line_number=line_num,
                    rule_name="naming_conventions"
                ))
            
            # Check function/variable names (snake_case)
            func_match = re.search(r'def\s+([A-Z][a-zA-Z0-9_]*)\s*\(', line)
            if func_match:
                func_name = func_match.group(1)
                self.issues.append(QualityIssue(
                    severity="warning",
                    message=f"Function name '{func_name}' should use snake_case",
                    file_path=str(file_path),
                    line_number=line_num,
                    rule_name="naming_conventions"
                ))
    
    def _check_import_organization(self, file_path: Path, lines: List[str]) -> None:
        """Check for organized imports."""
        import_lines = []
        import_indices = []
        
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')):
                import_lines.append(line.strip())
                import_indices.append(i)
        
        if len(import_lines) < 2:
            return
        
        # Check if imports are grouped properly
        stdlib_imports = []
        third_party_imports = []
        local_imports = []
        
        for line in import_lines:
            if line.startswith('from robodata'):
                local_imports.append(line)
            elif any(imp in line for imp in ['os', 'sys', 'pathlib', 'typing', 'dataclasses']):
                stdlib_imports.append(line)
            else:
                third_party_imports.append(line)
        
        # Check if imports are in the wrong order
        expected_order = stdlib_imports + third_party_imports + local_imports
        if import_lines != expected_order:
            self.issues.append(QualityIssue(
                severity="info",
                message="Imports should be organized: stdlib, third-party, local",
                file_path=str(file_path),
                line_number=import_indices[0],
                rule_name="import_organization"
            ))


def run_quality_check(directory: Path) -> QualityReport:
    """Run code quality check on directory.
    
    Args:
        directory: Directory to check
        
    Returns:
        QualityReport with analysis results
    """
    checker = CodeQualityChecker()
    return checker.analyze_directory(directory)


def print_quality_report(report: QualityReport) -> None:
    """Print quality report to console.
    
    Args:
        report: Quality report to print
    """
    print(f"\n=== Code Quality Report ===")
    print(f"Files analyzed: {report.analyzed_files}/{report.total_files}")
    print(f"Issues found: {len(report.issues)}")
    print(f"  Errors: {len(report.errors)}")
    print(f"  Warnings: {len(report.warnings)}")
    print(f"  Info: {report.info_count}")
    
    if report.issues:
        print("\n=== Issues ===")
        for issue in sorted(report.issues, key=lambda x: (x.severity, x.file_path, x.line_number)):
            print(issue)
    else:
        print("\n✅ No issues found!")


# Import ast for docstring and type annotation checking
import ast
