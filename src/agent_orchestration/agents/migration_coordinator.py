"""
Migration Coordinator Agent

Coordinates parallel migrations with rate limiting and error handling.
"""

import asyncio
import time
from pathlib import Path
from typing import List, Optional

from src.agent_orchestration.schemas.orchestration_models import (
    FileInfo,
    MigrationStatus,
)
from src.migration_engine.transformers.python_transformer import PythonTransformer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MigrationCoordinator:
    """Coordinates file migrations with parallel execution"""

    def __init__(
        self,
        output_dir: str = "outputs/migrated",
        max_parallel: int = 3,
        rate_limit_delay: float = 5.0,
    ):
        """
        Initialize coordinator

        Args:
            output_dir: Directory for migrated files
            max_parallel: Maximum parallel migrations
            rate_limit_delay: Delay between batches (seconds) for rate limiting
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_parallel = max_parallel
        self.rate_limit_delay = rate_limit_delay
        self.transformer = PythonTransformer()

    async def migrate_batch(
        self, files: List[FileInfo], repo_path: str
    ) -> List[FileInfo]:
        """
        Migrate a batch of files in parallel

        Args:
            files: Files to migrate
            repo_path: Repository root path

        Returns:
            Updated file info with migration results
        """
        logger.info(f"Migrating batch of {len(files)} files")

        # Create tasks for parallel execution
        tasks = [
            self._migrate_single_file(file_info, repo_path) for file_info in files
        ]

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Update file status
        updated_files = []
        for file_info, result in zip(files, results):
            if isinstance(result, Exception):
                file_info.status = MigrationStatus.FAILED
                file_info.error = str(result)
                logger.error(f"Migration failed for {file_info.path}: {result}")
            else:
                file_info = result
                logger.info(
                    f"Migration completed for {file_info.path}: {file_info.status}"
                )

            updated_files.append(file_info)

        # Rate limiting delay
        if self.rate_limit_delay > 0:
            logger.info(f"Rate limit delay: {self.rate_limit_delay}s")
            await asyncio.sleep(self.rate_limit_delay)

        return updated_files

    async def _migrate_single_file(
        self, file_info: FileInfo, repo_path: str
    ) -> FileInfo:
        """
        Migrate a single file

        Args:
            file_info: File to migrate
            repo_path: Repository root

        Returns:
            Updated file info
        """
        try:
            file_info.status = MigrationStatus.IN_PROGRESS
            logger.info(f"Starting migration: {file_info.path}")

            # Read source file
            source_path = Path(repo_path) / file_info.path
            source_code = source_path.read_text(encoding="utf-8")

            # Create output path (preserve directory structure)
            output_path = self.output_dir / file_info.path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate migration plan first
            from src.ast_engine.analyzers.migration_plan_generator import MigrationPlanGenerator
            from src.utils.config import get_rules_file
            
            rules_file = get_rules_file("python")
            planner = MigrationPlanGenerator(rules_file)
            
            loop = asyncio.get_event_loop()
            migration_plan = await loop.run_in_executor(
                None,
                planner.generate_plan,
                str(source_path),
                file_info.target_framework,
                file_info.framework,
            )

            # Transform code with migration plan
            migrated_code = await loop.run_in_executor(
                None,
                self.transformer.transform_file,
                str(source_path),
                migration_plan,
                str(output_path),
            )

            if migrated_code:
                file_info.status = MigrationStatus.COMPLETED
                file_info.output_path = str(output_path)
                logger.info(f"Migration successful: {file_info.path}")
            else:
                file_info.status = MigrationStatus.FAILED
                file_info.error = "Transformation returned empty result"
                logger.error(f"Migration failed: {file_info.path}")

        except Exception as e:
            file_info.status = MigrationStatus.FAILED
            file_info.error = str(e)
            logger.exception(f"Migration error for {file_info.path}: {e}")

        return file_info

    def migrate_batch_sync(
        self, files: List[FileInfo], repo_path: str
    ) -> List[FileInfo]:
        """
        Synchronous wrapper for migrate_batch

        Args:
            files: Files to migrate
            repo_path: Repository root

        Returns:
            Updated file info
        """
        return asyncio.run(self.migrate_batch(files, repo_path))
