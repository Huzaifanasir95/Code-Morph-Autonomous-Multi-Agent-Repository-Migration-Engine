"""
Unit tests for Dependency Analyzer

Tests the DependencyAnalyzer class for building and analyzing
dependency graphs.
"""

import pytest

from src.ast_engine.analyzers.dependency_analyzer import DependencyAnalyzer
from src.ast_engine.schemas.ast_models import CodeAnalysisResult, ImportInfo


@pytest.fixture
def sample_analyses():
    """Create sample analysis results for testing"""
    # File A imports numpy and pandas
    analysis_a = CodeAnalysisResult(
        file_path="project/module_a.py",
        imports=[
            ImportInfo(module="numpy", names=[], alias="np", line_number=1, is_from_import=False),
            ImportInfo(module="pandas", names=[], alias="pd", line_number=2, is_from_import=False),
        ],
        functions=[],
        classes=[],
        deprecated_apis=[],
        total_lines=50,
        dependencies=["numpy", "pandas"],
        complexity_score=3.5,
    )
    
    # File B imports numpy and module_a
    analysis_b = CodeAnalysisResult(
        file_path="project/module_b.py",
        imports=[
            ImportInfo(module="numpy", names=[], alias="np", line_number=1, is_from_import=False),
            ImportInfo(module="module_a", names=[], alias=None, line_number=2, is_from_import=False),
        ],
        functions=[],
        classes=[],
        deprecated_apis=[],
        total_lines=30,
        dependencies=["numpy", "module_a"],
        complexity_score=2.0,
    )
    
    # File C imports module_b
    analysis_c = CodeAnalysisResult(
        file_path="project/module_c.py",
        imports=[
            ImportInfo(module="module_b", names=[], alias=None, line_number=1, is_from_import=False),
        ],
        functions=[],
        classes=[],
        deprecated_apis=[],
        total_lines=20,
        dependencies=["module_b"],
        complexity_score=1.5,
    )
    
    return [analysis_a, analysis_b, analysis_c]


def test_analyzer_initialization():
    """Test analyzer can be initialized"""
    analyzer = DependencyAnalyzer()
    assert analyzer.graph is not None
    assert len(analyzer.nodes) == 0


def test_add_file_analysis(sample_analyses):
    """Test adding file analyses to graph"""
    analyzer = DependencyAnalyzer()
    
    for analysis in sample_analyses:
        analyzer.add_file_analysis(analysis)
    
    # Should have nodes for modules
    assert len(analyzer.graph.nodes()) > 0


def test_get_dependencies(sample_analyses):
    """Test getting dependencies for a module"""
    analyzer = DependencyAnalyzer()
    
    for analysis in sample_analyses:
        analyzer.add_file_analysis(analysis)
    
    # module_a depends on numpy and pandas
    deps_a = analyzer.get_dependencies("module_a")
    assert "numpy" in deps_a
    assert "pandas" in deps_a
    
    # module_b depends on numpy and module_a
    deps_b = analyzer.get_dependencies("module_b")
    assert "numpy" in deps_b
    assert "module_a" in deps_b


def test_get_dependents(sample_analyses):
    """Test getting modules that depend on a module"""
    analyzer = DependencyAnalyzer()
    
    for analysis in sample_analyses:
        analyzer.add_file_analysis(analysis)
    
    # module_a is used by module_b
    dependents_a = analyzer.get_dependents("module_a")
    assert "module_b" in dependents_a
    
    # module_b is used by module_c
    dependents_b = analyzer.get_dependents("module_b")
    assert "module_c" in dependents_b


def test_find_circular_dependencies():
    """Test detecting circular dependencies"""
    analyzer = DependencyAnalyzer()
    
    # Create circular dependency: A -> B -> A
    analysis_a = CodeAnalysisResult(
        file_path="a.py",
        imports=[ImportInfo(module="module_b", names=[], alias=None, line_number=1, is_from_import=False)],
        functions=[],
        classes=[],
        deprecated_apis=[],
        total_lines=10,
        dependencies=["module_b"],
        complexity_score=1.0,
    )
    
    analysis_b = CodeAnalysisResult(
        file_path="b.py",
        imports=[ImportInfo(module="module_a", names=[], alias=None, line_number=1, is_from_import=False)],
        functions=[],
        classes=[],
        deprecated_apis=[],
        total_lines=10,
        dependencies=["module_a"],
        complexity_score=1.0,
    )
    
    analyzer.add_file_analysis(analysis_a)
    analyzer.add_file_analysis(analysis_b)
    
    cycles = analyzer.find_circular_dependencies()
    # Should detect at least one cycle
    assert len(cycles) > 0


def test_get_migration_order(sample_analyses):
    """Test getting recommended migration order"""
    analyzer = DependencyAnalyzer()
    
    for analysis in sample_analyses:
        analyzer.add_file_analysis(analysis)
    
    order = analyzer.get_migration_order()
    
    # Should return a list of modules
    assert len(order) > 0
    assert isinstance(order, list)


def test_get_root_modules(sample_analyses):
    """Test finding root modules (no dependencies)"""
    analyzer = DependencyAnalyzer()
    
    for analysis in sample_analyses:
        analyzer.add_file_analysis(analysis)
    
    roots = analyzer.get_root_modules()
    
    # External modules like numpy, pandas should be roots
    assert "numpy" in roots or "pandas" in roots


def test_get_external_dependencies(sample_analyses):
    """Test identifying external dependencies"""
    analyzer = DependencyAnalyzer()
    
    for analysis in sample_analyses:
        analyzer.add_file_analysis(analysis)
    
    external = analyzer.get_external_dependencies()
    
    # numpy and pandas are external
    assert "numpy" in external
    assert "pandas" in external


def test_analyze_complexity(sample_analyses):
    """Test complexity analysis"""
    analyzer = DependencyAnalyzer()
    
    for analysis in sample_analyses:
        analyzer.add_file_analysis(analysis)
    
    complexity = analyzer.analyze_complexity()
    
    # Should return complexity scores for modules
    assert len(complexity) > 0
    assert all(isinstance(score, (int, float)) for score in complexity.values())


def test_visualize_graph(sample_analyses):
    """Test graph visualization"""
    analyzer = DependencyAnalyzer()
    
    for analysis in sample_analyses:
        analyzer.add_file_analysis(analysis)
    
    visualization = analyzer.visualize_graph()
    
    # Should return a string
    assert isinstance(visualization, str)
    assert len(visualization) > 0
    assert "Dependency Graph" in visualization


def test_to_dict(sample_analyses):
    """Test converting graph to dictionary"""
    analyzer = DependencyAnalyzer()
    
    for analysis in sample_analyses:
        analyzer.add_file_analysis(analysis)
    
    graph_dict = analyzer.to_dict()
    
    # Should have expected keys
    assert "nodes" in graph_dict
    assert "edges" in graph_dict
    assert "migration_order" in graph_dict
    assert "external_dependencies" in graph_dict
