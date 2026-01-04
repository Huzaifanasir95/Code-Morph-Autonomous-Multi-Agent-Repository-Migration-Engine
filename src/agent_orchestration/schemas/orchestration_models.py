"""
Agent Orchestration Schemas

Pydantic models for agent state management and communication.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MigrationStatus(str, Enum):
    """Status of a file migration"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"
    VERIFICATION_FAILED = "verification_failed"


class FileInfo(BaseModel):
    """Information about a file to migrate"""

    path: str = Field(..., description="Relative path from repository root")
    language: str = Field(..., description="Programming language")
    framework: str = Field(..., description="Source framework")
    target_framework: str = Field(..., description="Target framework")
    lines_of_code: int = Field(0, description="Number of lines")
    complexity: str = Field("MEDIUM", description="Estimated complexity")
    dependencies: List[str] = Field(default_factory=list, description="File dependencies")
    status: MigrationStatus = Field(
        MigrationStatus.PENDING, description="Migration status"
    )
    output_path: Optional[str] = Field(None, description="Path to migrated file")
    error: Optional[str] = Field(None, description="Error message if failed")
    verification_score: Optional[float] = Field(
        None, description="Verification similarity score"
    )


class RepositoryScanResult(BaseModel):
    """Results from repository scan"""

    repository_path: str = Field(..., description="Path to repository")
    total_files: int = Field(..., description="Total files found")
    files_to_migrate: List[FileInfo] = Field(
        default_factory=list, description="Files needing migration"
    )
    ignored_files: int = Field(0, description="Files ignored by filters")
    scan_duration_ms: float = Field(..., description="Scan duration")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Scan timestamp",
    )


class DependencyGraph(BaseModel):
    """Dependency graph for migration order"""

    nodes: List[str] = Field(default_factory=list, description="File paths")
    edges: List[tuple[str, str]] = Field(
        default_factory=list, description="Dependency edges (from, to)"
    )
    migration_order: List[str] = Field(
        default_factory=list, description="Topologically sorted migration order"
    )
    circular_dependencies: List[List[str]] = Field(
        default_factory=list, description="Circular dependency cycles"
    )


class MigrationBatch(BaseModel):
    """Batch of files to migrate in parallel"""

    batch_id: int = Field(..., description="Batch identifier")
    files: List[FileInfo] = Field(..., description="Files in batch")
    max_parallel: int = Field(3, description="Max parallel migrations")
    status: MigrationStatus = Field(
        MigrationStatus.PENDING, description="Batch status"
    )
    started_at: Optional[str] = Field(None, description="Start timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")


class OrchestrationState(BaseModel):
    """Overall orchestration state"""

    repository_path: str = Field(..., description="Repository being migrated")
    target_framework: str = Field(..., description="Target framework")
    scan_result: Optional[RepositoryScanResult] = Field(
        None, description="Scan results"
    )
    dependency_graph: Optional[DependencyGraph] = Field(
        None, description="Dependency graph"
    )
    batches: List[MigrationBatch] = Field(
        default_factory=list, description="Migration batches"
    )
    completed_files: int = Field(0, description="Successfully migrated files")
    failed_files: int = Field(0, description="Failed migrations")
    verified_files: int = Field(0, description="Verified migrations")
    total_duration_ms: float = Field(0.0, description="Total duration")
    errors: List[str] = Field(default_factory=list, description="Error log")
    started_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Start timestamp",
    )
    completed_at: Optional[str] = Field(None, description="Completion timestamp")


class MigrationReport(BaseModel):
    """Final migration report"""

    repository: str = Field(..., description="Repository path")
    target_framework: str = Field(..., description="Target framework")
    total_files: int = Field(..., description="Total files processed")
    successful: int = Field(..., description="Successfully migrated")
    failed: int = Field(..., description="Failed migrations")
    verified: int = Field(..., description="Verified migrations")
    verification_failed: int = Field(..., description="Verification failures")
    success_rate: float = Field(..., description="Success rate percentage")
    verification_rate: float = Field(..., description="Verification rate percentage")
    total_duration_seconds: float = Field(..., description="Total duration")
    files: List[FileInfo] = Field(..., description="All processed files")
    errors: List[str] = Field(default_factory=list, description="Error summary")
    timestamp: str = Field(..., description="Report timestamp")

    @property
    def is_successful(self) -> bool:
        """Check if migration was successful"""
        return self.failed == 0 and self.verification_failed == 0
