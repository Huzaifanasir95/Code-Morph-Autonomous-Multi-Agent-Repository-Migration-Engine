"""
Test Executor

Runs generated tests in sandboxed Docker environments and collects results.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from src.test_sandbox.docker_manager import DockerManager
from src.test_sandbox.schemas.test_models import (
    DockerConfig,
    SandboxEnvironment,
    TestResult,
    TestStatus,
    TestSuiteResult,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TestExecutor:
    """Executes tests in sandboxed environments"""

    def __init__(self, docker_manager: Optional[DockerManager] = None):
        """
        Initialize test executor

        Args:
            docker_manager: Docker manager instance
        """
        self.docker = docker_manager or DockerManager()

    def execute_tests(
        self,
        test_file: str,
        code_file: str,
        requirements: Optional[List[str]] = None,
        config: Optional[DockerConfig] = None,
    ) -> TestSuiteResult:
        """
        Execute pytest test suite in sandbox

        Args:
            test_file: Path to test file
            code_file: Path to code being tested
            requirements: Additional packages to install
            config: Docker configuration

        Returns:
            TestSuiteResult with execution details
        """
        logger.info(f"Executing tests: {test_file}")

        sandbox = None
        try:
            # Create sandbox
            sandbox = self._create_test_sandbox(config)
            self.docker.start_sandbox(sandbox)

            # Setup environment
            self._setup_test_environment(
                sandbox, test_file, code_file, requirements or []
            )

            # Run tests with pytest
            test_results = self._run_pytest(sandbox)

            # Create suite result
            suite_result = TestSuiteResult(
                file_path=test_file,
                framework="python",
                total_tests=len(test_results),
                passed=sum(1 for t in test_results if t.status == TestStatus.PASSED),
                failed=sum(1 for t in test_results if t.status == TestStatus.FAILED),
                errors=sum(1 for t in test_results if t.status == TestStatus.ERROR),
                duration_ms=sum(t.duration_ms for t in test_results),
                tests=test_results,
            )

            logger.info(
                f"Tests completed: {suite_result.passed}/{suite_result.total_tests} passed"
            )
            return suite_result

        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return TestSuiteResult(
                file_path=test_file,
                framework="python",
                total_tests=0,
                passed=0,
                failed=0,
                errors=1,
                duration_ms=0.0,
                tests=[
                    TestResult(
                        test_name="execution_error",
                        status=TestStatus.ERROR,
                        duration_ms=0.0,
                        error=str(e),
                    )
                ],
            )

        finally:
            if sandbox:
                self._cleanup_sandbox(sandbox)

    def _create_test_sandbox(
        self, config: Optional[DockerConfig] = None
    ) -> SandboxEnvironment:
        """
        Create sandbox for test execution

        Args:
            config: Docker configuration

        Returns:
            Created sandbox environment
        """
        if config is None:
            # Allow network for pip install
            config = DockerConfig(image="python:3.10-slim", network_disabled=False)

        sandbox = self.docker.create_sandbox(
            config=config,
            environment={
                "PYTHONPATH": "/workspace",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )

        return sandbox

    def _setup_test_environment(
        self,
        sandbox: SandboxEnvironment,
        test_file: str,
        code_file: str,
        requirements: List[str],
    ) -> None:
        """
        Setup test environment in sandbox

        Args:
            sandbox: Sandbox environment
            test_file: Test file to copy
            code_file: Code file to copy
            requirements: Packages to install
        """
        logger.info("Setting up test environment")

        # Create workspace directory
        self.docker.execute_command(sandbox, "mkdir -p /workspace")

        # Copy test file
        test_dest = f"/workspace/{Path(test_file).name}"
        self.docker.copy_to_sandbox(sandbox, test_file, test_dest)

        # Copy code file
        code_dest = f"/workspace/{Path(code_file).name}"
        self.docker.copy_to_sandbox(sandbox, code_file, code_dest)

        # Install pytest
        logger.info("Installing pytest")
        exit_code, stdout, stderr = self.docker.execute_command(
            sandbox, "pip install -q pytest pytest-json-report"
        )

        if exit_code != 0:
            logger.error(f"Failed to install pytest: {stderr}")
            raise RuntimeError(f"pytest installation failed: {stderr}")

        # Install requirements
        if requirements:
            logger.info(f"Installing requirements: {requirements}")
            packages = " ".join(requirements)
            exit_code, stdout, stderr = self.docker.execute_command(
                sandbox, f"pip install -q {packages}"
            )

            if exit_code != 0:
                logger.warning(f"Some packages failed to install: {stderr}")

    def _run_pytest(self, sandbox: SandboxEnvironment) -> List[TestResult]:
        """
        Run pytest and collect results

        Args:
            sandbox: Sandbox environment

        Returns:
            List of test results
        """
        logger.info("Running pytest")

        # Run pytest with JSON reporter
        exit_code, stdout, stderr = self.docker.execute_command(
            sandbox,
            "pytest /workspace -v --json-report --json-report-file=/workspace/report.json",
        )

        # Get report file
        report_path = "/workspace/report.json"
        exit_code, report_json, stderr = self.docker.execute_command(
            sandbox, f"cat {report_path}"
        )

        if exit_code != 0 or not report_json:
            logger.error("Failed to get test report")
            return self._parse_pytest_output(stdout)

        # Parse JSON report
        try:
            report = json.loads(report_json)
            return self._parse_json_report(report)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON report: {e}")
            return self._parse_pytest_output(stdout)

    def _parse_json_report(self, report: dict) -> List[TestResult]:
        """
        Parse pytest JSON report

        Args:
            report: JSON report dictionary

        Returns:
            List of test results
        """
        results = []

        for test in report.get("tests", []):
            outcome = test.get("outcome", "unknown")
            status_map = {
                "passed": TestStatus.PASSED,
                "failed": TestStatus.FAILED,
                "error": TestStatus.ERROR,
                "skipped": TestStatus.SKIPPED,
            }
            result = TestResult(
                test_name=test.get("nodeid", "unknown"),
                status=status_map.get(outcome, TestStatus.ERROR),
                duration_ms=test.get("call", {}).get("duration", 0.0) * 1000,
                error=test.get("call", {}).get("longrepr") if outcome != "passed" else None,
                output=test.get("call", {}).get("stdout", ""),
            )
            results.append(result)

        return results

    def _parse_pytest_output(self, output: str) -> List[TestResult]:
        """
        Parse pytest text output as fallback

        Args:
            output: Pytest stdout

        Returns:
            List of test results
        """
        results = []
        lines = output.split("\n")

        for line in lines:
            if "::" in line and ("PASSED" in line or "FAILED" in line):
                parts = line.split("::")
                if len(parts) >= 2:
                    test_name = "::".join(parts[:2])
                    passed = "PASSED" in line

                    result = TestResult(
                        test_name=test_name,
                        status=TestStatus.PASSED if passed else TestStatus.FAILED,
                        duration_ms=0.0,
                    )
                    results.append(result)

        return results

    def _cleanup_sandbox(self, sandbox: SandboxEnvironment) -> None:
        """
        Cleanup sandbox environment

        Args:
            sandbox: Sandbox to cleanup
        """
        try:
            self.docker.stop_sandbox(sandbox)
            self.docker.destroy_sandbox(sandbox)
        except Exception as e:
            logger.error(f"Failed to cleanup sandbox: {e}")
