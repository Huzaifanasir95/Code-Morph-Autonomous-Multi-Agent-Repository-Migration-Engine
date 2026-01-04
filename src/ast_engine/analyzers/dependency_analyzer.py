"""
Dependency Analyzer using NetworkX

Builds and analyzes dependency graphs for codebases to understand
relationships between modules and identify migration order.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set

import networkx as nx

from src.ast_engine.schemas.ast_models import CodeAnalysisResult, DependencyNode, NodeType


class DependencyAnalyzer:
    """Analyzes dependencies between modules using graph structures"""

    def __init__(self) -> None:
        """Initialize dependency analyzer with empty graph"""
        self.graph: nx.DiGraph = nx.DiGraph()
        self.nodes: Dict[str, DependencyNode] = {}

    def add_file_analysis(self, analysis: CodeAnalysisResult) -> None:
        """
        Add a file's analysis to the dependency graph

        Args:
            analysis: CodeAnalysisResult from parsing a file
        """
        file_name = Path(analysis.file_path).stem

        # Extract imports (dependencies)
        imports = [imp.module for imp in analysis.imports]

        # Create or update node
        node = DependencyNode(
            name=file_name,
            node_type=NodeType.MODULE,
            imports=imports,
            imported_by=[],
            file_path=analysis.file_path,
        )

        self.nodes[file_name] = node
        self.graph.add_node(file_name, **node.model_dump())

        # Add edges for imports
        for import_name in imports:
            # Add edge from this file to imported module
            if not self.graph.has_node(import_name):
                self.graph.add_node(import_name, name=import_name, node_type=NodeType.MODULE)
            self.graph.add_edge(file_name, import_name)

    def get_dependencies(self, module_name: str) -> List[str]:
        """
        Get all dependencies for a module

        Args:
            module_name: Name of the module

        Returns:
            List of module names this module depends on
        """
        if module_name not in self.graph:
            return []
        return list(self.graph.successors(module_name))

    def get_dependents(self, module_name: str) -> List[str]:
        """
        Get all modules that depend on this module

        Args:
            module_name: Name of the module

        Returns:
            List of module names that import this module
        """
        if module_name not in self.graph:
            return []
        return list(self.graph.predecessors(module_name))

    def find_circular_dependencies(self) -> List[List[str]]:
        """
        Find circular dependencies in the codebase

        Returns:
            List of cycles, where each cycle is a list of module names
        """
        try:
            cycles = list(nx.simple_cycles(self.graph))
            return cycles
        except nx.NetworkXNoCycle:
            return []

    def get_migration_order(self) -> List[str]:
        """
        Get recommended order to migrate files (topological sort)

        Returns:
            List of module names in migration order (dependencies first)
        """
        try:
            # Reverse graph to get dependencies before dependents
            return list(reversed(list(nx.topological_sort(self.graph))))
        except nx.NetworkXError:
            # Graph has cycles, use alternative approach
            # Return strongly connected components
            components = list(nx.strongly_connected_components(self.graph))
            # Flatten and return
            result = []
            for component in components:
                result.extend(sorted(component))
            return result

    def get_root_modules(self) -> List[str]:
        """
        Get modules with no dependencies (good starting points)

        Returns:
            List of root module names
        """
        return [node for node in self.graph.nodes() if self.graph.out_degree(node) == 0]

    def get_leaf_modules(self) -> List[str]:
        """
        Get modules that nothing depends on (can migrate last)

        Returns:
            List of leaf module names
        """
        return [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]

    def analyze_complexity(self) -> Dict[str, float]:
        """
        Analyze complexity of each module based on connections

        Returns:
            Dict mapping module names to complexity scores
        """
        complexity = {}
        for node in self.graph.nodes():
            # Complexity based on in-degree and out-degree
            in_degree = self.graph.in_degree(node)
            out_degree = self.graph.out_degree(node)
            # Modules with many connections are more complex
            complexity[node] = (in_degree + out_degree) / 2.0
        return complexity

    def get_external_dependencies(self) -> Set[str]:
        """
        Get all external dependencies (not part of the codebase)

        Returns:
            Set of external package names
        """
        external_deps = set()
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            # If no file_path, it's an external dependency
            if "file_path" not in node_data or node_data["file_path"] is None:
                external_deps.add(node)
        return external_deps

    def visualize_graph(self, output_path: Optional[str] = None) -> str:
        """
        Generate a text representation of the dependency graph

        Args:
            output_path: Optional path to save visualization

        Returns:
            String representation of the graph
        """
        lines = ["Dependency Graph:", "=" * 50, ""]

        # Show each node and its dependencies
        for node in sorted(self.graph.nodes()):
            dependencies = self.get_dependencies(node)
            dependents = self.get_dependents(node)

            lines.append(f"📦 {node}")
            if dependencies:
                lines.append(f"  ├─ Depends on: {', '.join(dependencies)}")
            if dependents:
                lines.append(f"  └─ Used by: {', '.join(dependents)}")
            lines.append("")

        # Show cycles if any
        cycles = self.find_circular_dependencies()
        if cycles:
            lines.append("⚠️  Circular Dependencies Detected:")
            for i, cycle in enumerate(cycles, 1):
                lines.append(f"  {i}. {' → '.join(cycle)} → {cycle[0]}")
            lines.append("")

        # Show migration order
        migration_order = self.get_migration_order()
        lines.append("📋 Recommended Migration Order:")
        for i, module in enumerate(migration_order, 1):
            lines.append(f"  {i}. {module}")

        result = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(result, encoding="utf-8")

        return result

    def to_dict(self) -> Dict:
        """
        Convert dependency graph to dictionary format

        Returns:
            Dictionary representation of the graph
        """
        return {
            "nodes": [
                {
                    "name": node,
                    "dependencies": self.get_dependencies(node),
                    "dependents": self.get_dependents(node),
                    **self.graph.nodes[node],
                }
                for node in self.graph.nodes()
            ],
            "edges": list(self.graph.edges()),
            "circular_dependencies": self.find_circular_dependencies(),
            "migration_order": self.get_migration_order(),
            "external_dependencies": list(self.get_external_dependencies()),
        }
