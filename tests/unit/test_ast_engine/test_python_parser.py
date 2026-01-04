"""
Unit tests for Python AST Parser

Tests the PythonParser class for extracting imports, functions,
classes, and other code structures.
"""

import pytest
from pathlib import Path

from src.ast_engine.parsers.python_parser import PythonParser
from src.ast_engine.schemas.ast_models import CodeAnalysisResult


@pytest.fixture
def sample_code_file(tmp_path):
    """Create a temporary Python file for testing"""
    code = '''
"""Sample module for testing"""
import numpy as np
from typing import List, Dict
import tensorflow as tf

def simple_function(x, y):
    """A simple function"""
    return x + y

async def async_function(data: List[int]) -> int:
    """An async function"""
    result = sum(data)
    return result

@property
def decorated_function():
    """A decorated function"""
    return 42

class SampleClass:
    """A sample class"""
    
    def __init__(self, value):
        self.value = value
    
    def method(self):
        """A method"""
        return self.value * 2
    
    def call_functions(self):
        """Method that calls other functions"""
        result = simple_function(1, 2)
        return result
'''
    
    file_path = tmp_path / "sample.py"
    file_path.write_text(code)
    return str(file_path)


def test_parser_initialization(sample_code_file):
    """Test parser can be initialized with a file"""
    parser = PythonParser(sample_code_file)
    assert parser.file_path.exists()
    assert parser.source_code is not None


def test_parser_nonexistent_file():
    """Test parser raises error for nonexistent file"""
    with pytest.raises(FileNotFoundError):
        PythonParser("nonexistent_file.py")


def test_extract_imports(sample_code_file):
    """Test extracting import statements"""
    parser = PythonParser(sample_code_file)
    imports = parser.extract_imports()
    
    assert len(imports) >= 3
    
    # Check for numpy import
    numpy_import = next((imp for imp in imports if imp.module == "numpy"), None)
    assert numpy_import is not None
    assert numpy_import.alias == "np"
    
    # Check for typing import
    typing_import = next((imp for imp in imports if imp.module == "typing"), None)
    assert typing_import is not None
    assert typing_import.is_from_import is True


def test_extract_functions(sample_code_file):
    """Test extracting function definitions"""
    parser = PythonParser(sample_code_file)
    functions = parser.extract_functions()
    
    # Should find functions but not methods
    assert len(functions) >= 2
    
    # Check simple_function
    simple_func = next((f for f in functions if f.name == "simple_function"), None)
    assert simple_func is not None
    assert "x" in simple_func.parameters
    assert "y" in simple_func.parameters
    
    # Check async function
    async_func = next((f for f in functions if f.name == "async_function"), None)
    assert async_func is not None
    assert async_func.is_async is True


def test_extract_classes(sample_code_file):
    """Test extracting class definitions"""
    parser = PythonParser(sample_code_file)
    classes = parser.extract_classes()
    
    assert len(classes) >= 1
    
    # Check SampleClass
    sample_class = next((c for c in classes if c.name == "SampleClass"), None)
    assert sample_class is not None
    assert len(sample_class.methods) >= 3
    
    # Check methods
    method_names = [m.name for m in sample_class.methods]
    assert "__init__" in method_names
    assert "method" in method_names


def test_full_parse(sample_code_file):
    """Test complete parsing of file"""
    parser = PythonParser(sample_code_file)
    analysis = parser.parse()
    
    assert isinstance(analysis, CodeAnalysisResult)
    assert analysis.file_path == sample_code_file
    assert len(analysis.imports) >= 3
    assert len(analysis.functions) >= 2
    assert len(analysis.classes) >= 1
    assert len(analysis.dependencies) >= 3
    assert analysis.total_lines > 0
    assert 0 <= analysis.complexity_score <= 10


def test_count_lines(sample_code_file):
    """Test line counting"""
    parser = PythonParser(sample_code_file)
    lines = parser.count_lines()
    assert lines > 30  # Our sample has more than 30 lines


def test_extract_dependencies(sample_code_file):
    """Test dependency extraction"""
    parser = PythonParser(sample_code_file)
    analysis = parser.parse()
    
    dependencies = analysis.dependencies
    assert "numpy" in dependencies
    assert "typing" in dependencies
    assert "tensorflow" in dependencies


def test_complexity_calculation(sample_code_file):
    """Test complexity score calculation"""
    parser = PythonParser(sample_code_file)
    analysis = parser.parse()
    
    # Complexity should be reasonable
    assert analysis.complexity_score > 0
    assert analysis.complexity_score < 10


def test_empty_file(tmp_path):
    """Test parsing an empty file"""
    empty_file = tmp_path / "empty.py"
    empty_file.write_text("")
    
    parser = PythonParser(str(empty_file))
    analysis = parser.parse()
    
    assert len(analysis.imports) == 0
    assert len(analysis.functions) == 0
    assert len(analysis.classes) == 0
    assert analysis.total_lines == 0
