"""
Repository Scanner Agent

Scans repositories to identify files needing migration.
Applies filters and estimates complexity.
"""

import time
from pathlib import Path
from typing import List, Optional, Set

from src.agent_orchestration.schemas.orchestration_models import (
    FileInfo,
    RepositoryScanResult,
)
from src.ast_engine.parsers.python_parser import PythonParser
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RepositoryScanner:
    """Scans repositories for files to migrate"""

    def __init__(
        self,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize scanner

        Args:
            include_patterns: File patterns to include (e.g., ['*.py'])
            exclude_patterns: File patterns to exclude (e.g., ['test_*', '*_test.py'])
        """
        self.include_patterns = include_patterns or ["*.py"]
        self.exclude_patterns = exclude_patterns or [
            "test_*",
            "*_test.py",
            "venv/*",
            ".venv/*",
            "env/*",
            ".env/*",
            "node_modules/*",
            "__pycache__/*",
            "*.pyc",
            ".git/*",
            "build/*",
            "dist/*",
        ]
        # Parser will be created per-file during analysis

    def scan_repository(
        self,
        repo_path: str,
        source_framework: str = "tensorflow==1.15.0",
        target_framework: str = "pytorch",
    ) -> RepositoryScanResult:
        """
        Scan repository for files to migrate

        Args:
            repo_path: Path to repository
            source_framework: Source framework to detect
            target_framework: Target framework

        Returns:
            RepositoryScanResult with discovered files
        """
        logger.info(f"Scanning repository: {repo_path}")
        start_time = time.time()

        repo = Path(repo_path)
        if not repo.exists():
            raise FileNotFoundError(f"Repository not found: {repo_path}")

        files_to_migrate: List[FileInfo] = []
        ignored_count = 0

        # Scan for Python files
        for pattern in self.include_patterns:
            for file_path in repo.rglob(pattern):
                # Skip if matches exclude patterns
                if self._should_exclude(file_path, repo):
                    ignored_count += 1
                    continue

                # Analyze file
                file_info = self._analyze_file(
                    file_path, repo, source_framework, target_framework
                )

                if file_info:
                    files_to_migrate.append(file_info)

        duration_ms = (time.time() - start_time) * 1000

        result = RepositoryScanResult(
            repository_path=str(repo),
            total_files=len(files_to_migrate) + ignored_count,
            files_to_migrate=files_to_migrate,
            ignored_files=ignored_count,
            scan_duration_ms=duration_ms,
        )

        logger.info(
            f"Scan complete: {len(files_to_migrate)} files to migrate, "
            f"{ignored_count} ignored, {duration_ms:.0f}ms"
        )

        return result

    def _should_exclude(self, file_path: Path, repo_root: Path) -> bool:
        """Check if file should be excluded"""
        relative_path = file_path.relative_to(repo_root)
        relative_str = str(relative_path)

        for pattern in self.exclude_patterns:
            # Simple pattern matching
            if pattern.startswith("*") and pattern.endswith("*"):
                # Contains pattern
                if pattern[1:-1] in relative_str:
                    return True
            elif pattern.startswith("*"):
                # Ends with pattern
                if relative_str.endswith(pattern[1:]):
                    return True
            elif pattern.endswith("*"):
                # Starts with pattern
                if relative_str.startswith(pattern[:-1]):
                    return True
            elif pattern in relative_str:
                # Exact match or contains
                return True

        return False

    def _analyze_file(
        self,
        file_path: Path,
        repo_root: Path,
        source_framework: str,
        target_framework: str,
    ) -> Optional[FileInfo]:
        """
        Analyze a file to determine if it needs migration

        Args:
            file_path: Path to file
            repo_root: Repository root
            source_framework: Source framework
            target_framework: Target framework

        Returns:
            FileInfo if file needs migration, None otherwise
        """
        try:
            # Read file
            code = file_path.read_text(encoding="utf-8")
            lines_of_code = len(code.splitlines())

            # Parse with AST (create parser per file)
            parser = PythonParser(str(file_path))
            analysis = parser.parse()

            # Check if file uses source framework
            framework_name = source_framework.split("==")[0]
            uses_framework = any(
                framework_name in imp.module
                for imp in analysis.imports
            )

            if not uses_framework:
                # File doesn't use source framework, skip
                return None

            # Estimate complexity
            complexity = self._estimate_complexity(analysis.functions, analysis.classes)

            # Get dependencies (other Python files it imports)
            dependencies = self._extract_dependencies(analysis.imports, repo_root)

            relative_path = str(file_path.relative_to(repo_root))

            return FileInfo(
                path=relative_path,
                language="python",
                framework=source_framework,
                target_framework=target_framework,
                lines_of_code=lines_of_code,
                complexity=complexity,
                dependencies=dependencies,
            )

        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return None

    def _estimate_complexity(self, functions: list, classes: list) -> str:
        """Estimate complexity based on code structure"""
        total_entities = len(functions) + len(classes)

        if total_entities == 0:
            return "LOW"
        elif total_entities < 5:
            return "LOW"
        elif total_entities < 15:
            return "MEDIUM"
        elif total_entities < 30:
            return "HIGH"
        else:
            return "CRITICAL"

    def _extract_dependencies(self, imports: list, repo_root: Path) -> List[str]:
        """Extract local file dependencies"""
        dependencies = []

        for imp in imports:
            # Check if it's a local import (not a package)
            if "." in imp.module:
                # Relative import
                module_parts = imp.module.split(".")
                potential_file = repo_root / Path("/".join(module_parts)).with_suffix(".py")

                if potential_file.exists():
                    try:
                        rel_path = str(potential_file.relative_to(repo_root))
                        dependencies.append(rel_path)
                    except ValueError:
                        pass

        return dependencies
