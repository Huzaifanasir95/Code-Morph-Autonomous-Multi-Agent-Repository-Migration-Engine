"""
Python AST Parser using libcst

This parser provides deep analysis of Python source code including:
- Import statements and dependencies
- Function and class definitions
- Method calls and API usage
- Code structure and complexity
"""

from typing import List, Set

import libcst as cst
from libcst import metadata

from src.ast_engine.parsers.base_parser import BaseParser
from src.ast_engine.schemas.ast_models import (
    ClassInfo,
    CodeAnalysisResult,
    FunctionInfo,
    ImportInfo,
)


class PythonParser(BaseParser):
    """Parser for Python source code using libcst"""

    def __init__(self, file_path: str):
        """
        Initialize Python parser

        Args:
            file_path: Path to Python source file
        """
        super().__init__(file_path)
        self.source_code = self.read_source()
        self.module: cst.Module = cst.parse_module(self.source_code)

    def parse(self) -> CodeAnalysisResult:
        """
        Parse Python file and extract all relevant information

        Returns:
            CodeAnalysisResult with complete analysis
        """
        imports = self.extract_imports()
        functions = self.extract_functions()
        classes = self.extract_classes()
        dependencies = self._extract_dependencies(imports)
        complexity_score = self._calculate_complexity(functions, classes)

        return CodeAnalysisResult(
            file_path=str(self.file_path),
            imports=imports,
            functions=functions,
            classes=classes,
            deprecated_apis=[],  # Filled by API detector
            total_lines=self.count_lines(),
            dependencies=dependencies,
            complexity_score=complexity_score,
        )

    def extract_imports(self) -> List[ImportInfo]:
        """
        Extract all import statements

        Returns:
            List of ImportInfo objects
        """
        visitor = ImportCollector()
        self.module.visit(visitor)
        return visitor.imports

    def extract_functions(self) -> List[FunctionInfo]:
        """
        Extract function definitions (excluding methods)

        Returns:
            List of FunctionInfo objects
        """
        visitor = FunctionCollector()
        self.module.visit(visitor)
        return visitor.functions

    def extract_classes(self) -> List[ClassInfo]:
        """
        Extract class definitions including methods

        Returns:
            List of ClassInfo objects
        """
        visitor = ClassCollector()
        self.module.visit(visitor)
        return visitor.classes

    def _extract_dependencies(self, imports: List[ImportInfo]) -> List[str]:
        """
        Extract unique external dependencies from imports

        Args:
            imports: List of import statements

        Returns:
            List of unique module names
        """
        dependencies: Set[str] = set()
        for imp in imports:
            # Get top-level module name (e.g., 'tensorflow' from 'tensorflow.keras')
            top_level = imp.module.split(".")[0]
            dependencies.add(top_level)
        return sorted(list(dependencies))

    def _calculate_complexity(
        self, functions: List[FunctionInfo], classes: List[ClassInfo]
    ) -> float:
        """
        Calculate estimated code complexity score

        Args:
            functions: List of functions
            classes: List of classes

        Returns:
            Complexity score (0-10)
        """
        # Simple heuristic: based on number of functions, classes, and their sizes
        num_functions = len(functions)
        num_classes = len(classes)
        num_methods = sum(len(cls.methods) for cls in classes)

        total_lines = self.count_lines()

        # Complexity factors
        complexity = 0.0
        complexity += min(num_functions * 0.1, 2.0)  # Functions contribute
        complexity += min(num_classes * 0.3, 3.0)  # Classes contribute more
        complexity += min(num_methods * 0.1, 2.0)  # Methods contribute
        complexity += min(total_lines / 100, 3.0)  # Lines of code

        return min(complexity, 10.0)


class ImportCollector(cst.CSTVisitor):
    """Visitor to collect import statements"""

    def __init__(self) -> None:
        self.imports: List[ImportInfo] = []
        self._current_line = 1

    def visit_Import(self, node: cst.Import) -> None:
        """Visit import statement: import X"""
        for name in node.names:
            if isinstance(name, cst.ImportAlias):
                module_name = self._get_dotted_name(name.name)
                alias = name.asname.name.value if name.asname else None
                self.imports.append(
                    ImportInfo(
                        module=module_name,
                        names=[],
                        alias=alias,
                        line_number=self._get_line_number(node),
                        is_from_import=False,
                    )
                )

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        """Visit from-import statement: from X import Y"""
        if node.module:
            module_name = self._get_dotted_name(node.module)
        else:
            module_name = "."  # Relative import

        if isinstance(node.names, cst.ImportStar):
            imported_names = ["*"]
        else:
            imported_names = []
            for name in node.names:
                if isinstance(name, cst.ImportAlias):
                    imported_names.append(name.name.value)

        self.imports.append(
            ImportInfo(
                module=module_name,
                names=imported_names,
                alias=None,
                line_number=self._get_line_number(node),
                is_from_import=True,
            )
        )

    def _get_dotted_name(self, node: cst.BaseExpression) -> str:
        """Convert dotted name node to string"""
        if isinstance(node, cst.Name):
            return node.value
        elif isinstance(node, cst.Attribute):
            return f"{self._get_dotted_name(node.value)}.{node.attr.value}"
        return ""

    def _get_line_number(self, node: cst.CSTNode) -> int:
        """Get line number for a node (approximate)"""
        # libcst doesn't provide direct line numbers without metadata
        # For simplicity, we return approximate values
        return getattr(node, "_line", 0) or self._current_line


class FunctionCollector(cst.CSTVisitor):
    """Visitor to collect function definitions"""

    def __init__(self) -> None:
        self.functions: List[FunctionInfo] = []
        self._in_class = False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        """Mark that we're inside a class (to skip methods)"""
        self._in_class = True
        return True  # Continue visiting children

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        """Mark that we've left the class"""
        self._in_class = False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Visit function definition"""
        # Skip methods (functions inside classes)
        if self._in_class:
            return

        func_info = self._extract_function_info(node)
        self.functions.append(func_info)

    def _extract_function_info(self, node: cst.FunctionDef) -> FunctionInfo:
        """Extract information from a function definition"""
        name = node.name.value
        parameters = [param.name.value for param in node.params.params]

        # Extract decorators
        decorators = []
        for decorator in node.decorators:
            if isinstance(decorator.decorator, cst.Name):
                decorators.append(decorator.decorator.value)

        # Extract docstring
        docstring = None
        if node.body.body and isinstance(node.body.body[0], cst.SimpleStatementLine):
            first_stmt = node.body.body[0].body[0]
            if isinstance(first_stmt, cst.Expr) and isinstance(first_stmt.value, cst.SimpleString):
                docstring = first_stmt.value.value.strip('"""').strip("'''").strip()

        # Extract function calls
        call_collector = CallCollector()
        node.visit(call_collector)

        return FunctionInfo(
            name=name,
            parameters=parameters,
            return_type=None,  # Would need type annotation parsing
            decorators=decorators,
            docstring=docstring,
            line_start=0,  # Simplified
            line_end=0,
            is_async=isinstance(node, cst.FunctionDef) and node.asynchronous is not None,
            calls=call_collector.calls,
        )


class ClassCollector(cst.CSTVisitor):
    """Visitor to collect class definitions"""

    def __init__(self) -> None:
        self.classes: List[ClassInfo] = []

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """Visit class definition"""
        name = node.name.value

        # Extract base classes
        bases = []
        for arg in node.bases:
            if isinstance(arg.value, cst.Name):
                bases.append(arg.value.value)

        # Extract decorators
        decorators = []
        for decorator in node.decorators:
            if isinstance(decorator.decorator, cst.Name):
                decorators.append(decorator.decorator.value)

        # Extract docstring
        docstring = None
        if node.body.body and isinstance(node.body.body[0], cst.SimpleStatementLine):
            first_stmt = node.body.body[0].body[0]
            if isinstance(first_stmt, cst.Expr) and isinstance(first_stmt.value, cst.SimpleString):
                docstring = first_stmt.value.value.strip('"""').strip("'''").strip()

        # Extract methods
        methods = []
        for item in node.body.body:
            if isinstance(item, cst.FunctionDef):
                func_collector = FunctionCollector()
                func_collector._in_class = False  # Extract as function
                method_info = func_collector._extract_function_info(item)
                methods.append(method_info)

        self.classes.append(
            ClassInfo(
                name=name,
                bases=bases,
                methods=methods,
                decorators=decorators,
                docstring=docstring,
                line_start=0,
                line_end=0,
            )
        )


class CallCollector(cst.CSTVisitor):
    """Visitor to collect function/method calls"""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def visit_Call(self, node: cst.Call) -> None:
        """Visit function call"""
        call_name = self._get_call_name(node.func)
        if call_name:
            self.calls.append(call_name)

    def _get_call_name(self, node: cst.BaseExpression) -> str:
        """Extract the name of the called function/method"""
        if isinstance(node, cst.Name):
            return node.value
        elif isinstance(node, cst.Attribute):
            # For chained calls like obj.method()
            base = self._get_call_name(node.value)
            return f"{base}.{node.attr.value}" if base else node.attr.value
        return ""
