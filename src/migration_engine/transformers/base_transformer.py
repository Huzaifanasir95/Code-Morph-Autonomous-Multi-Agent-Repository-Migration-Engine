"""
Base Transformer Interface

Abstract base class for all code transformers.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from src.ast_engine.schemas.migration_models import CodeTransformation, MigrationPlan


class BaseTransformer(ABC):
    """Abstract base class for code transformers"""

    def __init__(self, source_framework: str, target_framework: str):
        """
        Initialize transformer

        Args:
            source_framework: Source framework name
            target_framework: Target framework name
        """
        self.source_framework = source_framework
        self.target_framework = target_framework

    @abstractmethod
    def transform_file(
        self, file_path: str, migration_plan: MigrationPlan, output_path: Optional[str] = None
    ) -> str:
        """
        Transform a single file according to migration plan

        Args:
            file_path: Path to source file
            migration_plan: Migration plan with transformations
            output_path: Optional output path (None for in-place)

        Returns:
            Path to transformed file
        """
        pass

    @abstractmethod
    def apply_transformation(
        self, code: str, transformation: CodeTransformation
    ) -> str:
        """
        Apply a single transformation to code

        Args:
            code: Source code
            transformation: Transformation to apply

        Returns:
            Transformed code
        """
        pass

    def read_file(self, file_path: str) -> str:
        """
        Read source file

        Args:
            file_path: Path to file

        Returns:
            File contents
        """
        return Path(file_path).read_text(encoding="utf-8")

    def write_file(self, file_path: str, content: str) -> None:
        """
        Write transformed code to file

        Args:
            file_path: Path to output file
            content: Transformed code
        """
        output = Path(file_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")

    def backup_file(self, file_path: str) -> str:
        """
        Create backup of original file

        Args:
            file_path: Path to file

        Returns:
            Path to backup file
        """
        source = Path(file_path)
        backup = source.with_suffix(source.suffix + ".backup")
        backup.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        return str(backup)
