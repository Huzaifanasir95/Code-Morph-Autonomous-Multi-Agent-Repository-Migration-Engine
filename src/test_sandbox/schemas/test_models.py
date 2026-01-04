"""
Type-safe Pydantic models for test execution and verification

These schemas define the structure of test results, comparisons,
and verification reports.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TestStatus(str, Enum):
    """Test execution status"""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class TestResult(BaseModel):
    """Result of a single test execution"""

    test_name: str = Field(..., description="Name of the test")
    status: TestStatus = Field(..., description="Test execution status")
    duration_ms: float = Field(..., description="Test execution time in milliseconds")
    output: str = Field(default="", description="Test output/stdout")
    error: Optional[str] = Field(None, description="Error message if failed")
    traceback: Optional[str] = Field(None, description="Stack trace if error")
    assertions: int = Field(0, description="Number of assertions")

    class Config:
        json_schema_extra = {
            "example": {
                "test_name": "test_train_model",
                "status": "passed",
                "duration_ms": 1250.5,
                "output": "Training completed successfully",
                "error": None,
                "traceback": None,
                "assertions": 5,
            }
        }


class TestSuiteResult(BaseModel):
    """Results from running a test suite"""

    file_path: str = Field(..., description="Path to file being tested")
    framework: str = Field(..., description="Framework (legacy or migrated)")
    total_tests: int = Field(..., description="Total number of tests")
    passed: int = Field(0, description="Number of passed tests")
    failed: int = Field(0, description="Number of failed tests")
    errors: int = Field(0, description="Number of errors")
    skipped: int = Field(0, description="Number of skipped tests")
    duration_ms: float = Field(..., description="Total execution time")
    tests: List[TestResult] = Field(default_factory=list, description="Individual test results")
    coverage: Optional[float] = Field(None, description="Code coverage percentage")

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100


class OutputComparison(BaseModel):
    """Comparison between legacy and migrated outputs"""

    test_name: str = Field(..., description="Test being compared")
    outputs_match: bool = Field(..., description="Whether outputs are identical")
    legacy_output: Any = Field(..., description="Output from legacy code")
    migrated_output: Any = Field(..., description="Output from migrated code")
    differences: List[str] = Field(default_factory=list, description="List of differences found")
    similarity_score: float = Field(
        1.0, ge=0.0, le=1.0, description="Similarity score (0-1)"
    )
    tolerance_used: Optional[float] = Field(None, description="Numerical tolerance if applicable")


class VerificationReport(BaseModel):
    """Complete verification report comparing legacy and migrated code"""

    legacy_file: str = Field(..., description="Path to legacy file")
    migrated_file: str = Field(..., description="Path to migrated file")
    legacy_results: TestSuiteResult = Field(..., description="Test results for legacy code")
    migrated_results: TestSuiteResult = Field(..., description="Test results for migrated code")
    comparisons: List[OutputComparison] = Field(
        default_factory=list, description="Output comparisons"
    )
    behavioral_equivalence: bool = Field(..., description="Overall behavioral equivalence")
    equivalence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall equivalence score"
    )
    warnings: List[str] = Field(default_factory=list, description="Verification warnings")
    timestamp: str = Field(..., description="Verification timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "legacy_file": "legacy_mnist.py",
                "migrated_file": "pytorch_mnist.py",
                "behavioral_equivalence": True,
                "equivalence_score": 0.98,
                "warnings": ["Minor numerical differences within tolerance"],
                "timestamp": "2026-01-04T13:00:00",
            }
        }


class DockerConfig(BaseModel):
    """Configuration for Docker sandbox"""

    image: str = Field(default="python:3.10-slim", description="Docker image to use")
    memory_limit: str = Field(default="512m", description="Memory limit")
    cpu_limit: float = Field(default=1.0, description="CPU limit (cores)")
    timeout_seconds: int = Field(default=300, description="Execution timeout")
    network_disabled: bool = Field(default=True, description="Disable network access")
    working_dir: str = Field(default="/workspace", description="Working directory in container")


class SandboxEnvironment(BaseModel):
    """Sandbox environment configuration"""

    container_id: Optional[str] = Field(None, description="Docker container ID")
    status: str = Field(default="created", description="Environment status")
    config: DockerConfig = Field(default_factory=DockerConfig, description="Docker configuration")
    volumes: Dict[str, str] = Field(default_factory=dict, description="Volume mappings")
    environment_vars: Dict[str, str] = Field(
        default_factory=dict, description="Environment variables"
    )
