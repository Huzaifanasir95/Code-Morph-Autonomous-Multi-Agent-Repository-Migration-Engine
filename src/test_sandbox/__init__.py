"""
Test Sandbox Module

Provides automated verification of behavioral equivalence through:
- Test generation using LLM
- Docker-based sandboxed execution
- Deep output comparison
- Verification reporting
"""

from src.test_sandbox.comparator import OutputComparator
from src.test_sandbox.docker_manager import DockerManager
from src.test_sandbox.test_executor import TestExecutor
from src.test_sandbox.test_generator import TestGenerator

__all__ = [
    "TestGenerator",
    "DockerManager",
    "TestExecutor",
    "OutputComparator",
]
