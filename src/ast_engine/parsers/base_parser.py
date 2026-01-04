"""
Base parser interface

All language-specific parsers inherit from this abstract base class
to ensure consistent API across different languages.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from src.ast_engine.schemas.ast_models import CodeAnalysisResult


class BaseParser(ABC):
    """Abstract base class for AST parsers"""

    def __init__(self, file_path: str):
        """
        Initialize parser with file path

        Args:
            file_path: Path to the source code file to parse
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    @abstractmethod
    def parse(self) -> CodeAnalysisResult:
        """
        Parse the source file and return analysis result

        Returns:
            CodeAnalysisResult containing all extracted information
        """
        pass

    @abstractmethod
    def extract_imports(self) -> List:
        """Extract import statements from the source file"""
        pass

    @abstractmethod
    def extract_functions(self) -> List:
        """Extract function definitions from the source file"""
        pass

    @abstractmethod
    def extract_classes(self) -> List:
        """Extract class definitions from the source file"""
        pass

    def read_source(self) -> str:
        """
        Read source file contents

        Returns:
            Source code as string
        """
        return self.file_path.read_text(encoding="utf-8")

    def count_lines(self) -> int:
        """
        Count total lines in source file

        Returns:
            Number of lines
        """
        return len(self.read_source().splitlines())
