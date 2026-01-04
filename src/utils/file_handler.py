"""
File handling utilities for Code-Morph

Provides helpers for reading, writing, and managing files.
"""

import shutil
from pathlib import Path
from typing import List, Optional


class FileHandler:
    """Utilities for file operations"""

    @staticmethod
    def read_file(file_path: str, encoding: str = "utf-8") -> str:
        """
        Read file contents

        Args:
            file_path: Path to file
            encoding: File encoding

        Returns:
            File contents as string
        """
        return Path(file_path).read_text(encoding=encoding)

    @staticmethod
    def write_file(file_path: str, content: str, encoding: str = "utf-8") -> None:
        """
        Write content to file

        Args:
            file_path: Path to file
            content: Content to write
            encoding: File encoding
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)

    @staticmethod
    def copy_file(source: str, destination: str) -> None:
        """
        Copy file from source to destination

        Args:
            source: Source file path
            destination: Destination file path
        """
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    @staticmethod
    def find_python_files(directory: str, recursive: bool = True) -> List[Path]:
        """
        Find all Python files in directory

        Args:
            directory: Directory to search
            recursive: Whether to search recursively

        Returns:
            List of Python file paths
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        pattern = "**/*.py" if recursive else "*.py"
        return sorted(dir_path.glob(pattern))

    @staticmethod
    def get_project_structure(directory: str, max_depth: int = 3) -> str:
        """
        Generate text representation of project structure

        Args:
            directory: Root directory
            max_depth: Maximum depth to traverse

        Returns:
            String representation of directory tree
        """
        lines = []
        dir_path = Path(directory)

        def walk_directory(path: Path, prefix: str = "", depth: int = 0) -> None:
            if depth > max_depth:
                return

            try:
                items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            except PermissionError:
                return

            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                lines.append(f"{prefix}{current_prefix}{item.name}")

                if item.is_dir() and not item.name.startswith("."):
                    extension = "    " if is_last else "│   "
                    walk_directory(item, prefix + extension, depth + 1)

        lines.append(dir_path.name)
        walk_directory(dir_path)
        return "\n".join(lines)

    @staticmethod
    def backup_file(file_path: str, backup_dir: Optional[str] = None) -> str:
        """
        Create backup of file

        Args:
            file_path: Path to file to backup
            backup_dir: Directory for backups (None for same directory)

        Returns:
            Path to backup file
        """
        source_path = Path(file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if backup_dir:
            backup_path = Path(backup_dir) / f"{source_path.name}.backup"
            backup_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            backup_path = source_path.with_suffix(source_path.suffix + ".backup")

        shutil.copy2(source_path, backup_path)
        return str(backup_path)

    @staticmethod
    def clean_directory(directory: str, pattern: str = "*.backup") -> int:
        """
        Clean files matching pattern from directory

        Args:
            directory: Directory to clean
            pattern: File pattern to match

        Returns:
            Number of files deleted
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            return 0

        files = list(dir_path.glob(pattern))
        for file in files:
            file.unlink()

        return len(files)
