"""
Type-safe Pydantic models for AST data structures

These schemas ensure type safety across the entire system and
provide validation for data passed between agents.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """AST node types"""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    IMPORT = "import"
    VARIABLE = "variable"
    MODULE = "module"


class ImportInfo(BaseModel):
    """Information about an import statement"""

    module: str = Field(..., description="Module being imported")
    names: List[str] = Field(default_factory=list, description="Names imported from module")
    alias: Optional[str] = Field(None, description="Alias for the import")
    line_number: int = Field(..., description="Line number where import occurs")
    is_from_import: bool = Field(False, description="Whether this is a 'from X import Y' statement")


class FunctionInfo(BaseModel):
    """Information about a function definition"""

    name: str = Field(..., description="Function name")
    parameters: List[str] = Field(default_factory=list, description="Parameter names")
    return_type: Optional[str] = Field(None, description="Return type annotation")
    decorators: List[str] = Field(default_factory=list, description="Decorator names")
    docstring: Optional[str] = Field(None, description="Function docstring")
    line_start: int = Field(..., description="Starting line number")
    line_end: int = Field(..., description="Ending line number")
    is_async: bool = Field(False, description="Whether function is async")
    calls: List[str] = Field(
        default_factory=list, description="Names of functions/methods called within"
    )


class ClassInfo(BaseModel):
    """Information about a class definition"""

    name: str = Field(..., description="Class name")
    bases: List[str] = Field(default_factory=list, description="Base class names")
    methods: List[FunctionInfo] = Field(default_factory=list, description="Method definitions")
    decorators: List[str] = Field(default_factory=list, description="Class decorators")
    docstring: Optional[str] = Field(None, description="Class docstring")
    line_start: int = Field(..., description="Starting line number")
    line_end: int = Field(..., description="Ending line number")


class DeprecatedAPIUsage(BaseModel):
    """Information about deprecated API usage"""

    api_name: str = Field(..., description="Name of deprecated API")
    pattern: str = Field(..., description="Pattern that matched (e.g., 'tf.Session')")
    line_number: int = Field(..., description="Line number where API is used")
    severity: str = Field(..., pattern="^(low|medium|high|critical)$")
    suggestion: str = Field(..., description="Suggested replacement or alternative")
    context: str = Field(..., description="Code snippet showing usage")


class DependencyNode(BaseModel):
    """Node in the dependency graph"""

    name: str = Field(..., description="Module or file name")
    node_type: NodeType = Field(..., description="Type of node")
    imports: List[str] = Field(default_factory=list, description="Modules this node imports")
    imported_by: List[str] = Field(
        default_factory=list, description="Modules that import this node"
    )
    file_path: Optional[str] = Field(None, description="File path if local module")


class CodeAnalysisResult(BaseModel):
    """Complete analysis result for a Python file"""

    file_path: str = Field(..., description="Path to analyzed file")
    imports: List[ImportInfo] = Field(default_factory=list, description="All import statements")
    functions: List[FunctionInfo] = Field(default_factory=list, description="Function definitions")
    classes: List[ClassInfo] = Field(default_factory=list, description="Class definitions")
    deprecated_apis: List[DeprecatedAPIUsage] = Field(
        default_factory=list, description="Deprecated API usages found"
    )
    total_lines: int = Field(..., description="Total lines of code")
    dependencies: List[str] = Field(
        default_factory=list, description="External dependencies detected"
    )
    complexity_score: float = Field(
        0.0, ge=0.0, le=10.0, description="Estimated complexity (0-10)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "file_path": "/path/to/legacy_model.py",
                "imports": [
                    {
                        "module": "tensorflow",
                        "names": ["Session", "placeholder"],
                        "alias": "tf",
                        "line_number": 1,
                        "is_from_import": False,
                    }
                ],
                "functions": [
                    {
                        "name": "train_model",
                        "parameters": ["data", "labels"],
                        "return_type": None,
                        "decorators": [],
                        "docstring": "Trains the TensorFlow model",
                        "line_start": 10,
                        "line_end": 25,
                        "is_async": False,
                        "calls": ["tf.Session", "tf.placeholder"],
                    }
                ],
                "classes": [],
                "deprecated_apis": [
                    {
                        "api_name": "tf.Session",
                        "pattern": "tf.Session",
                        "line_number": 15,
                        "severity": "high",
                        "suggestion": "Use eager execution or tf.function",
                        "context": "with tf.Session() as sess:",
                    }
                ],
                "total_lines": 100,
                "dependencies": ["tensorflow", "numpy"],
                "complexity_score": 6.5,
            }
        }
