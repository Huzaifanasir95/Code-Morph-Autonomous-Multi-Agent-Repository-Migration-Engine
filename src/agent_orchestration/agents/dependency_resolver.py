"""
Dependency Resolver Agent

Analyzes import dependencies and determines safe migration order.
Uses topological sorting to avoid breaking dependencies.
"""

from typing import List, Set

import networkx as nx

from src.agent_orchestration.schemas.orchestration_models import (
    DependencyGraph,
    FileInfo,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DependencyResolver:
    """Resolves file dependencies and determines migration order"""

    def resolve_dependencies(self, files: List[FileInfo]) -> DependencyGraph:
        """
        Resolve dependencies and compute migration order

        Args:
            files: List of files to migrate

        Returns:
            DependencyGraph with migration order
        """
        logger.info(f"Resolving dependencies for {len(files)} files")

        # Build dependency graph
        graph = nx.DiGraph()

        # Add all files as nodes
        file_map = {f.path: f for f in files}
        for file in files:
            graph.add_node(file.path)

        # Add dependency edges
        edges = []
        for file in files:
            for dep in file.dependencies:
                if dep in file_map:
                    # Add edge: dependent -> dependency
                    # (we need to migrate dependency first)
                    graph.add_edge(file.path, dep)
                    edges.append((file.path, dep))

        # Detect circular dependencies
        circular_deps = []
        try:
            cycles = list(nx.simple_cycles(graph))
            if cycles:
                logger.warning(f"Found {len(cycles)} circular dependencies")
                circular_deps = cycles
        except Exception as e:
            logger.error(f"Error detecting cycles: {e}")

        # Compute topological sort (migration order)
        # Files with no dependencies come first
        migration_order = []
        try:
            # Reverse graph so dependencies come before dependents
            migration_order = list(nx.topological_sort(graph.reverse()))
            logger.info(f"Computed migration order for {len(migration_order)} files")
        except nx.NetworkXError as e:
            logger.error(f"Topological sort failed (circular dependencies?): {e}")
            # Fallback: use simple ordering
            migration_order = [f.path for f in files]

        result = DependencyGraph(
            nodes=list(graph.nodes()),
            edges=edges,
            migration_order=migration_order,
            circular_dependencies=circular_deps,
        )

        return result

    def create_batches(
        self, files: List[FileInfo], dependency_graph: DependencyGraph, batch_size: int = 3
    ) -> List[List[FileInfo]]:
        """
        Create batches of files that can be migrated in parallel

        Args:
            files: List of files to migrate
            dependency_graph: Dependency graph
            batch_size: Maximum files per batch

        Returns:
            List of file batches
        """
        logger.info(f"Creating batches with max size {batch_size}")

        file_map = {f.path: f for f in files}
        migration_order = dependency_graph.migration_order

        batches: List[List[FileInfo]] = []
        current_batch: List[FileInfo] = []
        migrated: Set[str] = set()

        for file_path in migration_order:
            file_info = file_map.get(file_path)
            if not file_info:
                continue

            # Check if all dependencies are migrated
            deps_satisfied = all(dep in migrated for dep in file_info.dependencies)

            if deps_satisfied:
                current_batch.append(file_info)

                # Start new batch if current is full
                if len(current_batch) >= batch_size:
                    batches.append(current_batch)
                    migrated.update(f.path for f in current_batch)
                    current_batch = []
            else:
                # Dependencies not satisfied, finish current batch
                if current_batch:
                    batches.append(current_batch)
                    migrated.update(f.path for f in current_batch)
                    current_batch = []

                # Start new batch with this file
                current_batch.append(file_info)

        # Add remaining batch
        if current_batch:
            batches.append(current_batch)

        logger.info(f"Created {len(batches)} batches")
        return batches
