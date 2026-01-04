"""
Output Comparator

Compares outputs between legacy and migrated code to verify behavioral equivalence.
Uses deepdiff for deep comparison with numerical tolerance.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from deepdiff import DeepDiff

from src.test_sandbox.schemas.test_models import (
    TestStatus,
    TestSuiteResult,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SimpleComparison:
    """Simple comparison result for CLI"""
    are_equivalent: bool
    similarity_score: float
    differences: Optional[List[str]] = None
    legacy_output: str = ""
    migrated_output: str = ""


class OutputComparator:
    """Compares outputs to verify behavioral equivalence"""

    def __init__(
        self,
        numerical_tolerance: float = 1e-5,
        ignore_order: bool = False,
    ):
        """
        Initialize comparator

        Args:
            numerical_tolerance: Tolerance for numerical comparisons
            ignore_order: Whether to ignore order in sequences
        """
        self.numerical_tolerance = numerical_tolerance
        self.ignore_order = ignore_order

    def compare_test_results(
        self,
        legacy_results: TestSuiteResult,
        migrated_results: TestSuiteResult,
    ) -> SimpleComparison:
        """
        Compare test suite results

        Args:
            legacy_results: Results from legacy code
            migrated_results: Results from migrated code

        Returns:
            SimpleComparison with detailed analysis
        """
        logger.info("Comparing test results")

        # Check if both suites passed
        legacy_passed = legacy_results.passed == legacy_results.total_tests
        migrated_passed = migrated_results.passed == migrated_results.total_tests

        if not legacy_passed:
            return SimpleComparison(
                are_equivalent=False,
                differences=["Legacy code tests failed"],
                similarity_score=0.0,
                legacy_output=self._summarize_results(legacy_results),
                migrated_output=self._summarize_results(migrated_results),
            )

        if not migrated_passed:
            return SimpleComparison(
                are_equivalent=False,
                differences=["Migrated code tests failed"],
                similarity_score=0.0,
                legacy_output=self._summarize_results(legacy_results),
                migrated_output=self._summarize_results(migrated_results),
            )

        # Both passed - compare outputs
        differences = []
        total_comparisons = 0
        matching_comparisons = 0

        for legacy_test in legacy_results.tests:
            # Find corresponding migrated test
            migrated_test = next(
                (
                    t
                    for t in migrated_results.tests
                    if t.test_name == legacy_test.test_name
                ),
                None,
            )

            if not migrated_test:
                differences.append(
                    f"Test {legacy_test.test_name} not found in migrated results"
                )
                continue

            total_comparisons += 1

            # Compare outputs
            if legacy_test.output and migrated_test.output:
                output_diff = self._compare_outputs(
                    legacy_test.output, migrated_test.output
                )

                if output_diff:
                    differences.append(
                        f"{legacy_test.test_name}: {output_diff}"
                    )
                else:
                    matching_comparisons += 1

        # Calculate similarity score
        similarity_score = (
            matching_comparisons / total_comparisons if total_comparisons > 0 else 1.0
        )

        are_equivalent = len(differences) == 0

        return SimpleComparison(
            are_equivalent=are_equivalent,
            differences=differences if differences else None,
            similarity_score=similarity_score,
            legacy_output=self._summarize_results(legacy_results),
            migrated_output=self._summarize_results(migrated_results),
        )

    def compare_outputs(
        self,
        legacy_output: Any,
        migrated_output: Any,
    ) -> SimpleComparison:
        """
        Compare raw outputs

        Args:
            legacy_output: Output from legacy code
            migrated_output: Output from migrated code

        Returns:
            SimpleComparison with detailed analysis
        """
        logger.info("Comparing raw outputs")

        differences_list = self._compare_values(legacy_output, migrated_output)

        are_equivalent = len(differences_list) == 0

        # Calculate similarity score
        if are_equivalent:
            similarity_score = 1.0
        else:
            # Simple heuristic: fewer differences = higher similarity
            max_diffs = max(
                self._count_elements(legacy_output),
                self._count_elements(migrated_output),
                1,
            )
            similarity_score = max(0.0, 1.0 - (len(differences_list) / max_diffs))

        return SimpleComparison(
            are_equivalent=are_equivalent,
            differences=differences_list if differences_list else None,
            similarity_score=similarity_score,
            legacy_output=str(legacy_output)[:1000],  # Truncate for display
            migrated_output=str(migrated_output)[:1000],
        )

    def _compare_outputs(self, legacy: str, migrated: str) -> Optional[str]:
        """
        Compare test outputs

        Args:
            legacy: Legacy output
            migrated: Migrated output

        Returns:
            Difference description or None if equivalent
        """
        # Try parsing as JSON
        try:
            legacy_json = json.loads(legacy)
            migrated_json = json.loads(migrated)
            diffs = self._compare_values(legacy_json, migrated_json)
            return "; ".join(diffs) if diffs else None
        except json.JSONDecodeError:
            pass

        # Try parsing as numbers
        try:
            legacy_num = float(legacy.strip())
            migrated_num = float(migrated.strip())

            if abs(legacy_num - migrated_num) > self.numerical_tolerance:
                return f"Numerical difference: {legacy_num} vs {migrated_num}"
            return None
        except ValueError:
            pass

        # String comparison
        if legacy.strip() != migrated.strip():
            return f"String mismatch"

        return None

    def _compare_values(self, legacy: Any, migrated: Any) -> List[str]:
        """
        Deep compare values using deepdiff

        Args:
            legacy: Legacy value
            migrated: Migrated value

        Returns:
            List of differences
        """
        # Handle numpy arrays
        if isinstance(legacy, np.ndarray) and isinstance(migrated, np.ndarray):
            if legacy.shape != migrated.shape:
                return [f"Shape mismatch: {legacy.shape} vs {migrated.shape}"]

            if not np.allclose(legacy, migrated, rtol=self.numerical_tolerance):
                max_diff = np.max(np.abs(legacy - migrated))
                return [f"Array values differ by up to {max_diff}"]

            return []

        # Use deepdiff for complex structures
        diff = DeepDiff(
            legacy,
            migrated,
            ignore_order=self.ignore_order,
            significant_digits=int(-np.log10(self.numerical_tolerance)),
        )

        if not diff:
            return []

        differences = []

        # Parse deepdiff results
        if "values_changed" in diff:
            for key, change in diff["values_changed"].items():
                differences.append(
                    f"{key}: {change['old_value']} -> {change['new_value']}"
                )

        if "type_changes" in diff:
            for key, change in diff["type_changes"].items():
                differences.append(
                    f"{key}: type changed from {change['old_type']} to {change['new_type']}"
                )

        if "dictionary_item_added" in diff:
            differences.append(
                f"Added items: {list(diff['dictionary_item_added'])}"
            )

        if "dictionary_item_removed" in diff:
            differences.append(
                f"Removed items: {list(diff['dictionary_item_removed'])}"
            )

        if "iterable_item_added" in diff:
            differences.append(f"Added items in sequences")

        if "iterable_item_removed" in diff:
            differences.append(f"Removed items in sequences")

        return differences

    def _summarize_results(self, results: TestSuiteResult) -> str:
        """
        Summarize test results

        Args:
            results: Test suite results

        Returns:
            Summary string
        """
        return f"{results.passed}/{results.total_tests} tests passed"

    def _count_elements(self, obj: Any) -> int:
        """
        Count elements in object

        Args:
            obj: Object to count

        Returns:
            Element count
        """
        if isinstance(obj, (list, tuple)):
            return len(obj)
        elif isinstance(obj, dict):
            return len(obj)
        elif isinstance(obj, np.ndarray):
            return obj.size
        else:
            return 1
