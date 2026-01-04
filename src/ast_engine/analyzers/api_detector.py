"""
Deprecated API Detector

Identifies deprecated APIs and patterns in code based on
configurable rules. Useful for finding TensorFlow 1.x, old React patterns, etc.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.ast_engine.schemas.ast_models import CodeAnalysisResult, DeprecatedAPIUsage


class APIDetector:
    """Detects deprecated APIs and patterns in code"""

    def __init__(self, rules_file: Optional[str] = None):
        """
        Initialize API detector with rules

        Args:
            rules_file: Path to YAML file with detection rules
        """
        self.rules: Dict = {}
        if rules_file:
            self.load_rules(rules_file)

    def load_rules(self, rules_file: str) -> None:
        """
        Load detection rules from YAML file

        Args:
            rules_file: Path to YAML rules file
        """
        rules_path = Path(rules_file)
        if not rules_path.exists():
            raise FileNotFoundError(f"Rules file not found: {rules_file}")

        with open(rules_path, "r", encoding="utf-8") as f:
            self.rules = yaml.safe_load(f)

    def detect(
        self, analysis: CodeAnalysisResult, framework: str = "tensorflow_v1"
    ) -> List[DeprecatedAPIUsage]:
        """
        Detect deprecated API usages in analyzed code

        Args:
            analysis: CodeAnalysisResult from parsing
            framework: Framework to check for (e.g., 'tensorflow_v1')

        Returns:
            List of DeprecatedAPIUsage objects
        """
        if not self.rules or "deprecated_apis" not in self.rules:
            return []

        deprecated_apis: List[DeprecatedAPIUsage] = []
        source_code = Path(analysis.file_path).read_text(encoding="utf-8")
        lines = source_code.splitlines()

        # Get rules for specified framework
        framework_rules = self.rules.get("deprecated_apis", {}).get(framework, [])

        for rule in framework_rules:
            pattern = rule.get("pattern", "")
            replacement = rule.get("replacement", "")
            severity = rule.get("severity", "medium")

            # Find all occurrences in the code
            for line_num, line in enumerate(lines, start=1):
                if pattern in line:
                    # Extract context (current line with some surrounding context)
                    context_start = max(0, line_num - 2)
                    context_end = min(len(lines), line_num + 1)
                    context = "\n".join(lines[context_start:context_end])

                    deprecated_apis.append(
                        DeprecatedAPIUsage(
                            api_name=pattern,
                            pattern=pattern,
                            line_number=line_num,
                            severity=severity,
                            suggestion=replacement,
                            context=context,
                        )
                    )

        # Also check function calls for deprecated patterns
        for func in analysis.functions:
            for call in func.calls:
                deprecated_apis.extend(self._check_call_pattern(call, func.line_start, framework_rules))

        return deprecated_apis

    def _check_call_pattern(
        self, call: str, line_number: int, rules: List[Dict]
    ) -> List[DeprecatedAPIUsage]:
        """
        Check if a function call matches deprecated patterns

        Args:
            call: Function call string
            line_number: Line number of the call
            rules: List of rule dictionaries

        Returns:
            List of DeprecatedAPIUsage objects
        """
        deprecated = []
        for rule in rules:
            pattern = rule.get("pattern", "")
            if pattern in call:
                deprecated.append(
                    DeprecatedAPIUsage(
                        api_name=pattern,
                        pattern=pattern,
                        line_number=line_number,
                        severity=rule.get("severity", "medium"),
                        suggestion=rule.get("replacement", ""),
                        context=f"Function call: {call}",
                    )
                )
        return deprecated

    def detect_patterns(
        self, source_code: str, pattern_type: str = "anti_patterns"
    ) -> List[Dict]:
        """
        Detect general code patterns (anti-patterns, etc.)

        Args:
            source_code: Source code string
            pattern_type: Type of patterns to detect

        Returns:
            List of detected patterns with details
        """
        if not self.rules or "code_patterns" not in self.rules:
            return []

        patterns = self.rules.get("code_patterns", {}).get(pattern_type, [])
        detected = []

        lines = source_code.splitlines()
        for line_num, line in enumerate(lines, start=1):
            for pattern_rule in patterns:
                pattern = pattern_rule.get("pattern", "")
                if pattern in line:
                    detected.append(
                        {
                            "pattern": pattern,
                            "line_number": line_num,
                            "suggestion": pattern_rule.get("suggestion", ""),
                            "severity": pattern_rule.get("severity", "low"),
                            "context": line.strip(),
                        }
                    )

        return detected

    def get_framework_info(self, dependencies: List[str]) -> Optional[Dict]:
        """
        Identify framework and version from dependencies

        Args:
            dependencies: List of dependency names

        Returns:
            Dictionary with framework info or None
        """
        framework_info = {}

        # Check for TensorFlow
        if "tensorflow" in dependencies:
            framework_info["name"] = "tensorflow"
            framework_info["legacy"] = True  # Assume legacy until proven otherwise
            framework_info["detection_key"] = "tensorflow_v1"

        # Check for PyTorch
        if "torch" in dependencies:
            framework_info["name"] = "pytorch"
            framework_info["legacy"] = False

        # Check for React (would need package.json analysis)
        # This is simplified for Python focus

        return framework_info if framework_info else None

    def generate_summary(self, deprecated_apis: List[DeprecatedAPIUsage]) -> Dict:
        """
        Generate summary of deprecated API usage

        Args:
            deprecated_apis: List of detected deprecated APIs

        Returns:
            Summary dictionary with counts and severity breakdown
        """
        summary = {
            "total_count": len(deprecated_apis),
            "by_severity": {"low": 0, "medium": 0, "high": 0, "critical": 0},
            "unique_apis": set(),
        }

        for api in deprecated_apis:
            summary["by_severity"][api.severity] += 1
            summary["unique_apis"].add(api.api_name)

        summary["unique_apis"] = list(summary["unique_apis"])
        summary["unique_count"] = len(summary["unique_apis"])

        return summary
