"""
Type-safe Pydantic models for Migration Plans

These schemas define the structure of migration plans generated
by the Analyst Agent and consumed by the Migration Agent.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class MigrationComplexity(str, Enum):
    """Complexity levels for migrations"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TransformationType(str, Enum):
    """Types of code transformations"""

    API_REPLACEMENT = "api_replacement"
    REFACTOR = "refactor"
    MODERNIZATION = "modernization"
    OPTIMIZATION = "optimization"
    DEPENDENCY_UPDATE = "dependency_update"


class CodeTransformation(BaseModel):
    """Individual code transformation to apply"""

    transformation_type: TransformationType = Field(..., description="Type of transformation")
    description: str = Field(..., description="Human-readable description")
    target_location: str = Field(..., description="File path or function/class name")
    line_range: Optional[tuple[int, int]] = Field(None, description="Line range to transform")
    old_pattern: str = Field(..., description="Pattern to replace (can be regex)")
    new_pattern: str = Field(..., description="New code pattern")
    requires_llm: bool = Field(False, description="Whether LLM assistance is needed")
    confidence_score: float = Field(
        1.0, ge=0.0, le=1.0, description="Confidence in transformation"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="Transformations that must happen first"
    )


class MigrationRule(BaseModel):
    """Rule for migration from one framework/pattern to another"""

    rule_id: str = Field(..., description="Unique identifier for rule")
    name: str = Field(..., description="Rule name")
    from_framework: str = Field(..., description="Source framework/pattern")
    to_framework: str = Field(..., description="Target framework/pattern")
    patterns: List[str] = Field(..., description="Patterns this rule matches")
    example_before: str = Field(..., description="Example code before transformation")
    example_after: str = Field(..., description="Example code after transformation")
    complexity: MigrationComplexity = Field(..., description="Complexity of applying rule")


class MigrationPlan(BaseModel):
    """Complete migration plan for a file or project"""

    file_path: str = Field(..., description="Path to file being migrated")
    source_framework: str = Field(..., description="Current framework/version")
    target_framework: str = Field(..., description="Target framework/version")
    deprecated_apis: List[str] = Field(
        default_factory=list, description="Deprecated APIs detected"
    )
    transformations: List[CodeTransformation] = Field(
        default_factory=list, description="Ordered list of transformations to apply"
    )
    estimated_complexity: MigrationComplexity = Field(..., description="Overall complexity")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    warnings: List[str] = Field(
        default_factory=list, description="Warnings about potential issues"
    )
    manual_review_required: bool = Field(
        False, description="Whether manual review is recommended"
    )
    estimated_time_minutes: int = Field(..., gt=0, description="Estimated migration time")
    dependencies: List[str] = Field(
        default_factory=list, description="External dependencies needed"
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict, description="Additional metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "file_path": "/path/to/legacy_model.py",
                "source_framework": "tensorflow==1.15.0",
                "target_framework": "pytorch==2.5.0",
                "deprecated_apis": ["tf.Session", "tf.placeholder", "tf.global_variables_initializer"],
                "transformations": [
                    {
                        "transformation_type": "api_replacement",
                        "description": "Replace tf.Session with eager execution",
                        "target_location": "train_model",
                        "line_range": (15, 20),
                        "old_pattern": "with tf.Session() as sess:",
                        "new_pattern": "# Remove session context, use eager execution",
                        "requires_llm": True,
                        "confidence_score": 0.85,
                        "dependencies": [],
                    }
                ],
                "estimated_complexity": "high",
                "confidence_score": 0.8,
                "warnings": [
                    "TensorFlow 1.x to PyTorch requires significant architecture changes"
                ],
                "manual_review_required": True,
                "estimated_time_minutes": 120,
                "dependencies": ["torch>=2.5.0", "torchvision>=0.20.0"],
                "metadata": {"original_lines": "100", "framework_version_gap": "major"},
            }
        }


class ProjectMigrationPlan(BaseModel):
    """Migration plan for an entire project"""

    project_name: str = Field(..., description="Project name")
    project_path: str = Field(..., description="Root path of project")
    file_plans: List[MigrationPlan] = Field(
        default_factory=list, description="Migration plans for each file"
    )
    global_transformations: List[CodeTransformation] = Field(
        default_factory=list, description="Transformations that affect multiple files"
    )
    migration_order: List[str] = Field(
        default_factory=list, description="Recommended order to migrate files"
    )
    total_estimated_time_minutes: int = Field(..., gt=0, description="Total estimated time")
    overall_complexity: MigrationComplexity = Field(..., description="Overall project complexity")
    critical_warnings: List[str] = Field(
        default_factory=list, description="Critical issues to address"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "project_name": "legacy-tf-classifier",
                "project_path": "/path/to/project",
                "file_plans": [],
                "global_transformations": [],
                "migration_order": ["utils.py", "model.py", "train.py", "evaluate.py"],
                "total_estimated_time_minutes": 360,
                "overall_complexity": "high",
                "critical_warnings": ["Requires PyTorch 2.5+ for all features"],
            }
        }
