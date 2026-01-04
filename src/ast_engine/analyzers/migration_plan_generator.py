"""
Migration Plan Generator

Orchestrates AST analysis, dependency analysis, and API detection
to generate comprehensive migration plans.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.ast_engine.analyzers.api_detector import APIDetector
from src.ast_engine.analyzers.dependency_analyzer import DependencyAnalyzer
from src.ast_engine.parsers.python_parser import PythonParser
from src.ast_engine.schemas.ast_models import CodeAnalysisResult
from src.ast_engine.schemas.migration_models import (
    CodeTransformation,
    MigrationComplexity,
    MigrationPlan,
    TransformationType,
)


class MigrationPlanGenerator:
    """Generates migration plans by combining multiple analysis steps"""

    def __init__(self, rules_file: Optional[str] = None):
        """
        Initialize migration plan generator

        Args:
            rules_file: Path to YAML rules file for API detection
        """
        self.api_detector = APIDetector(rules_file) if rules_file else APIDetector()
        self.dependency_analyzer = DependencyAnalyzer()

    def generate_plan(
        self,
        file_path: str,
        target_framework: str = "pytorch",
        source_framework: str = "tensorflow==1.15.0",
    ) -> MigrationPlan:
        """
        Generate complete migration plan for a file

        Args:
            file_path: Path to file to analyze
            target_framework: Target framework (e.g., 'pytorch')
            source_framework: Source framework (e.g., 'tensorflow==1.15.0')

        Returns:
            MigrationPlan with all transformations
        """
        # Step 1: Parse the file
        parser = PythonParser(file_path)
        analysis = parser.parse()

        # Step 2: Detect deprecated APIs
        framework_key = self._get_framework_key(source_framework)
        deprecated_apis = self.api_detector.detect(analysis, framework=framework_key)

        # Update analysis with deprecated APIs
        analysis.deprecated_apis = deprecated_apis

        # Step 3: Generate transformations
        transformations = self._generate_transformations(analysis, target_framework)

        # Step 4: Calculate complexity and confidence
        complexity = self._calculate_complexity(analysis, deprecated_apis)
        confidence = self._calculate_confidence(deprecated_apis, transformations)

        # Step 5: Generate warnings
        warnings = self._generate_warnings(analysis, deprecated_apis, target_framework)

        # Step 6: Estimate time
        estimated_time = self._estimate_time(transformations, complexity)

        # Step 7: Determine if manual review needed
        manual_review = self._requires_manual_review(complexity, confidence, deprecated_apis)

        return MigrationPlan(
            file_path=file_path,
            source_framework=source_framework,
            target_framework=target_framework,
            deprecated_apis=[api.api_name for api in deprecated_apis],
            transformations=transformations,
            estimated_complexity=complexity,
            confidence_score=confidence,
            warnings=warnings,
            manual_review_required=manual_review,
            estimated_time_minutes=estimated_time,
            dependencies=analysis.dependencies,
            metadata={
                "total_lines": str(analysis.total_lines),
                "num_functions": str(len(analysis.functions)),
                "num_classes": str(len(analysis.classes)),
                "deprecated_api_count": str(len(deprecated_apis)),
            },
        )

    def _get_framework_key(self, framework: str) -> str:
        """Convert framework string to detection key"""
        if "tensorflow" in framework.lower() and "1." in framework:
            return "tensorflow_v1"
        return framework.split("==")[0].lower()

    def _generate_transformations(
        self, analysis: CodeAnalysisResult, target_framework: str
    ) -> List[CodeTransformation]:
        """Generate transformation steps from analysis"""
        transformations = []

        # For each deprecated API, create a transformation
        for api in analysis.deprecated_apis:
            transformations.append(
                CodeTransformation(
                    transformation_type=TransformationType.API_REPLACEMENT,
                    description=f"Replace {api.api_name} with {target_framework} equivalent",
                    target_location=analysis.file_path,
                    line_range=(api.line_number, api.line_number),
                    old_pattern=api.pattern,
                    new_pattern=api.suggestion,
                    requires_llm=True,  # Complex transformations need LLM
                    confidence_score=0.7 if api.severity in ["high", "critical"] else 0.85,
                    dependencies=[],
                )
            )

        # Add import transformation
        if transformations:
            transformations.insert(
                0,
                CodeTransformation(
                    transformation_type=TransformationType.DEPENDENCY_UPDATE,
                    description=f"Update imports from {analysis.dependencies[0]} to {target_framework}",
                    target_location=analysis.file_path,
                    line_range=(1, 10),
                    old_pattern=f"import {analysis.dependencies[0]}",
                    new_pattern=f"import {target_framework}",
                    requires_llm=False,
                    confidence_score=0.95,
                    dependencies=[],
                ),
            )

        return transformations

    def _calculate_complexity(
        self, analysis: CodeAnalysisResult, deprecated_apis: List
    ) -> MigrationComplexity:
        """Calculate migration complexity"""
        score = 0

        # Factor 1: Number of deprecated APIs
        score += len(deprecated_apis) * 2

        # Factor 2: High severity APIs
        high_severity = sum(1 for api in deprecated_apis if api.severity in ["high", "critical"])
        score += high_severity * 3

        # Factor 3: Code size
        score += analysis.total_lines / 50

        # Factor 4: Number of classes (more complex)
        score += len(analysis.classes) * 2

        # Determine complexity level
        if score < 5:
            return MigrationComplexity.LOW
        elif score < 15:
            return MigrationComplexity.MEDIUM
        elif score < 30:
            return MigrationComplexity.HIGH
        else:
            return MigrationComplexity.CRITICAL

    def _calculate_confidence(
        self, deprecated_apis: List, transformations: List[CodeTransformation]
    ) -> float:
        """Calculate confidence score"""
        if not transformations:
            return 1.0

        # Average confidence of all transformations
        avg_confidence = sum(t.confidence_score for t in transformations) / len(transformations)

        # Reduce confidence for many deprecated APIs
        penalty = max(0, len(deprecated_apis) - 5) * 0.05
        confidence = max(0.3, avg_confidence - penalty)

        return round(confidence, 2)

    def _generate_warnings(
        self, analysis: CodeAnalysisResult, deprecated_apis: List, target_framework: str
    ) -> List[str]:
        """Generate warnings for potential issues"""
        warnings = []

        # Warn about high complexity
        if analysis.complexity_score > 7:
            warnings.append("High code complexity detected - thorough testing recommended")

        # Warn about many deprecated APIs
        if len(deprecated_apis) > 10:
            warnings.append(
                f"Found {len(deprecated_apis)} deprecated APIs - expect significant changes"
            )

        # Framework-specific warnings
        if "tensorflow" in analysis.dependencies and "pytorch" in target_framework.lower():
            warnings.append("TensorFlow to PyTorch migration requires architecture changes")

        # Warn about critical severity APIs
        critical = [api for api in deprecated_apis if api.severity == "critical"]
        if critical:
            warnings.append(f"Found {len(critical)} critical deprecated APIs requiring immediate attention")

        return warnings

    def _estimate_time(
        self, transformations: List[CodeTransformation], complexity: MigrationComplexity
    ) -> int:
        """Estimate time in minutes"""
        # Base time per transformation
        time_per_transform = 5

        # Additional time for LLM-required transformations
        llm_transforms = sum(1 for t in transformations if t.requires_llm)
        time = len(transformations) * time_per_transform
        time += llm_transforms * 10  # LLM transforms take longer

        # Complexity multiplier
        complexity_multiplier = {
            MigrationComplexity.LOW: 1.0,
            MigrationComplexity.MEDIUM: 1.5,
            MigrationComplexity.HIGH: 2.0,
            MigrationComplexity.CRITICAL: 3.0,
        }

        time = int(time * complexity_multiplier[complexity])
        return max(5, time)  # Minimum 5 minutes

    def _requires_manual_review(
        self, complexity: MigrationComplexity, confidence: float, deprecated_apis: List
    ) -> bool:
        """Determine if manual review is needed"""
        # High/critical complexity always needs review
        if complexity in [MigrationComplexity.HIGH, MigrationComplexity.CRITICAL]:
            return True

        # Low confidence needs review
        if confidence < 0.7:
            return True

        # Many deprecated APIs need review
        if len(deprecated_apis) > 15:
            return True

        return False

    def save_plan(self, plan: MigrationPlan, output_path: str) -> None:
        """
        Save migration plan to JSON file

        Args:
            plan: MigrationPlan to save
            output_path: Path to output JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(plan.model_dump(), f, indent=2)

    def load_plan(self, plan_path: str) -> MigrationPlan:
        """
        Load migration plan from JSON file

        Args:
            plan_path: Path to JSON file

        Returns:
            MigrationPlan object
        """
        with open(plan_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return MigrationPlan(**data)
