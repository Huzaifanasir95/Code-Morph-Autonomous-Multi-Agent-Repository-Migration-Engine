"""
Repository Migration Orchestrator

Coordinates all agents to perform autonomous repository-wide migrations.
"""

import time
from datetime import datetime
from pathlib import Path

from src.agent_orchestration.agents.dependency_resolver import DependencyResolver
from src.agent_orchestration.agents.migration_coordinator import MigrationCoordinator
from src.agent_orchestration.agents.report_generator import ReportGenerator
from src.agent_orchestration.agents.repository_scanner import RepositoryScanner
from src.agent_orchestration.agents.verification_agent import VerificationAgent
from src.agent_orchestration.schemas.orchestration_models import (
    MigrationBatch,
    MigrationReport,
    MigrationStatus,
    OrchestrationState,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RepositoryOrchestrator:
    """Orchestrates repository-wide migrations"""

    def __init__(
        self,
        output_dir: str = "outputs/migrated",
        include_patterns: list = None,
        exclude_patterns: list = None,
        use_docker: bool = False,
        max_parallel: int = 3,
        rate_limit_delay: float = 5.0,
    ):
        """
        Initialize orchestrator

        Args:
            output_dir: Output directory for migrated files
            include_patterns: File patterns to include
            exclude_patterns: File patterns to exclude
            use_docker: Whether to use Docker for verification
            max_parallel: Maximum parallel migrations
            rate_limit_delay: Delay between batches (seconds)
        """
        self.scanner = RepositoryScanner(
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        self.resolver = DependencyResolver()
        self.coordinator = MigrationCoordinator(
            output_dir=output_dir,
            max_parallel=max_parallel,
            rate_limit_delay=rate_limit_delay,
        )
        self.verifier = VerificationAgent(use_docker=use_docker)
        self.report_generator = ReportGenerator()

    def migrate_repository(
        self,
        repo_path: str,
        source_framework: str = "tensorflow==1.15.0",
        target_framework: str = "pytorch",
        verify: bool = True,
    ) -> MigrationReport:
        """
        Migrate entire repository

        Args:
            repo_path: Path to repository
            source_framework: Source framework
            target_framework: Target framework
            verify: Whether to verify migrations

        Returns:
            MigrationReport with results
        """
        logger.info(f"Starting repository migration: {repo_path}")
        start_time = time.time()

        # Initialize state
        state = OrchestrationState(
            repository_path=repo_path,
            target_framework=target_framework,
        )

        try:
            # Step 1: Scan repository
            logger.info("[Step 1/5] Scanning repository...")
            scan_result = self.scanner.scan_repository(
                repo_path, source_framework, target_framework
            )
            state.scan_result = scan_result

            if not scan_result.files_to_migrate:
                logger.warning("No files found to migrate")
                state.completed_at = datetime.now().isoformat()
                return self.report_generator.generate_report(state)

            logger.info(f"Found {len(scan_result.files_to_migrate)} files to migrate")

            # Step 2: Resolve dependencies
            logger.info("[Step 2/5] Resolving dependencies...")
            dependency_graph = self.resolver.resolve_dependencies(
                scan_result.files_to_migrate
            )
            state.dependency_graph = dependency_graph

            if dependency_graph.circular_dependencies:
                logger.warning(
                    f"Found {len(dependency_graph.circular_dependencies)} circular dependencies"
                )
                for cycle in dependency_graph.circular_dependencies:
                    state.errors.append(f"Circular dependency: {' -> '.join(cycle)}")

            # Step 3: Create batches
            logger.info("[Step 3/5] Creating migration batches...")
            file_batches = self.resolver.create_batches(
                scan_result.files_to_migrate,
                dependency_graph,
                batch_size=self.coordinator.max_parallel,
            )

            # Convert to MigrationBatch objects
            for i, batch_files in enumerate(file_batches):
                batch = MigrationBatch(
                    batch_id=i + 1,
                    files=batch_files,
                    max_parallel=self.coordinator.max_parallel,
                )
                state.batches.append(batch)

            logger.info(f"Created {len(state.batches)} batches")

            # Step 4: Migrate batches
            logger.info("[Step 4/5] Migrating files...")
            for i, batch in enumerate(state.batches):
                logger.info(
                    f"Processing batch {i + 1}/{len(state.batches)} ({len(batch.files)} files)"
                )
                batch.status = MigrationStatus.IN_PROGRESS
                batch.started_at = datetime.now().isoformat()

                # Migrate batch
                updated_files = self.coordinator.migrate_batch_sync(
                    batch.files, repo_path
                )
                batch.files = updated_files

                # Update counts
                for file in updated_files:
                    if file.status == MigrationStatus.COMPLETED:
                        state.completed_files += 1
                    elif file.status == MigrationStatus.FAILED:
                        state.failed_files += 1
                        state.errors.append(f"{file.path}: {file.error}")

                batch.status = MigrationStatus.COMPLETED
                batch.completed_at = datetime.now().isoformat()

                logger.info(
                    f"Batch {i + 1} complete: "
                    f"{state.completed_files} completed, {state.failed_files} failed"
                )

            # Step 5: Verify (if enabled)
            if verify:
                logger.info("[Step 5/5] Verifying migrations...")
                for i, batch in enumerate(state.batches):
                    logger.info(f"Verifying batch {i + 1}/{len(state.batches)}")

                    # Only verify completed files
                    files_to_verify = [
                        f
                        for f in batch.files
                        if f.status == MigrationStatus.COMPLETED
                    ]

                    if files_to_verify:
                        verified_files = self.verifier.verify_batch_sync(
                            files_to_verify, repo_path
                        )

                        # Update batch files
                        for file in verified_files:
                            if file.status == MigrationStatus.VERIFIED:
                                state.verified_files += 1

                        # Update batch.files with verified results
                        file_map = {f.path: f for f in verified_files}
                        batch.files = [
                            file_map.get(f.path, f) for f in batch.files
                        ]

                logger.info(f"Verification complete: {state.verified_files} verified")
            else:
                logger.info("[Step 5/5] Skipping verification")

            # Calculate total duration
            state.total_duration_ms = (time.time() - start_time) * 1000
            state.completed_at = datetime.now().isoformat()

            # Generate report
            report = self.report_generator.generate_report(state)

            logger.info(
                f"Migration complete: {report.successful}/{report.total_files} successful, "
                f"{report.verified} verified, {report.failed} failed"
            )

            return report

        except Exception as e:
            logger.exception(f"Orchestration failed: {e}")
            state.errors.append(f"Orchestration error: {str(e)}")
            state.completed_at = datetime.now().isoformat()
            state.total_duration_ms = (time.time() - start_time) * 1000
            return self.report_generator.generate_report(state)
