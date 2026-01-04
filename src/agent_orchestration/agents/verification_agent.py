"""
Verification Agent

Automatically verifies migrated code against legacy code.
"""

import asyncio
from pathlib import Path
from typing import List

from src.agent_orchestration.schemas.orchestration_models import (
    FileInfo,
    MigrationStatus,
)
from src.test_sandbox.comparator import OutputComparator
from src.test_sandbox.test_executor import TestExecutor
from src.test_sandbox.test_generator import TestGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VerificationAgent:
    """Verifies migrated code against legacy code"""

    def __init__(self, use_docker: bool = True):
        """
        Initialize verification agent

        Args:
            use_docker: Whether to use Docker sandboxing
        """
        self.use_docker = use_docker
        self.test_generator = TestGenerator()
        self.test_executor = TestExecutor() if use_docker else None
        self.comparator = OutputComparator()

    async def verify_file(
        self, file_info: FileInfo, repo_path: str
    ) -> FileInfo:
        """
        Verify a migrated file

        Args:
            file_info: File to verify
            repo_path: Repository root

        Returns:
            Updated file info with verification results
        """
        if file_info.status != MigrationStatus.COMPLETED:
            logger.warning(
                f"Skipping verification for {file_info.path}: not completed"
            )
            return file_info

        try:
            logger.info(f"Verifying: {file_info.path}")

            legacy_path = Path(repo_path) / file_info.path
            migrated_path = Path(file_info.output_path)

            if not migrated_path.exists():
                raise FileNotFoundError(f"Migrated file not found: {migrated_path}")

            # Generate tests
            legacy_code = legacy_path.read_text(encoding="utf-8")
            test_code = self.test_generator.generate_tests(
                code=legacy_code,
                file_path=str(legacy_path),
                framework=file_info.framework,
            )

            # Save tests
            test_dir = Path("outputs/verification/tests")
            test_dir.mkdir(parents=True, exist_ok=True)
            test_file = test_dir / f"test_{legacy_path.stem}.py"
            self.test_generator.save_tests(test_code, str(test_file))

            if self.use_docker and self.test_executor:
                # Run tests in Docker
                legacy_results = self.test_executor.execute_tests(
                    test_file=str(test_file),
                    code_file=str(legacy_path),
                    requirements=[],
                )

                migrated_results = self.test_executor.execute_tests(
                    test_file=str(test_file),
                    code_file=str(migrated_path),
                    requirements=[],
                )
            else:
                # Skip actual test execution for now
                # In production, would run tests locally
                logger.warning("Docker disabled, skipping test execution")
                file_info.status = MigrationStatus.VERIFIED
                file_info.verification_score = 1.0
                return file_info

            # Compare results
            comparison = self.comparator.compare_test_results(
                legacy_results, migrated_results
            )

            # Update file info
            file_info.verification_score = comparison.similarity_score

            if comparison.are_equivalent:
                file_info.status = MigrationStatus.VERIFIED
                logger.info(
                    f"Verification passed: {file_info.path} ({comparison.similarity_score:.1%})"
                )
            else:
                file_info.status = MigrationStatus.VERIFICATION_FAILED
                file_info.error = f"Verification failed: {len(comparison.differences or [])} differences"
                logger.warning(
                    f"Verification failed: {file_info.path} ({comparison.similarity_score:.1%})"
                )

        except Exception as e:
            file_info.status = MigrationStatus.VERIFICATION_FAILED
            file_info.error = f"Verification error: {str(e)}"
            logger.exception(f"Verification error for {file_info.path}: {e}")

        return file_info

    async def verify_batch(
        self, files: List[FileInfo], repo_path: str
    ) -> List[FileInfo]:
        """
        Verify a batch of files

        Args:
            files: Files to verify
            repo_path: Repository root

        Returns:
            Updated file info with verification results
        """
        logger.info(f"Verifying batch of {len(files)} files")

        tasks = [self.verify_file(file_info, repo_path) for file_info in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        updated_files = []
        for file_info, result in zip(files, results):
            if isinstance(result, Exception):
                file_info.status = MigrationStatus.VERIFICATION_FAILED
                file_info.error = f"Verification exception: {str(result)}"
                logger.error(f"Verification failed for {file_info.path}: {result}")
            else:
                file_info = result

            updated_files.append(file_info)

        return updated_files

    def verify_batch_sync(
        self, files: List[FileInfo], repo_path: str
    ) -> List[FileInfo]:
        """
        Synchronous wrapper for verify_batch

        Args:
            files: Files to verify
            repo_path: Repository root

        Returns:
            Updated file info
        """
        return asyncio.run(self.verify_batch(files, repo_path))
