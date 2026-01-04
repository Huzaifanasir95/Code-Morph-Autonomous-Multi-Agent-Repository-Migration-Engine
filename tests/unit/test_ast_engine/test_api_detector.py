"""
Unit tests for API Detector

Tests the APIDetector class for identifying deprecated APIs.
"""

import pytest
from pathlib import Path

from src.ast_engine.analyzers.api_detector import APIDetector
from src.ast_engine.parsers.python_parser import PythonParser


@pytest.fixture
def rules_file(tmp_path):
    """Create a temporary rules file for testing"""
    rules_content = """
deprecated_apis:
  tensorflow_v1:
    - pattern: "tf.Session"
      replacement: "Use eager execution or tf.function"
      severity: "high"
    - pattern: "tf.placeholder"
      replacement: "Use tf.Variable or function arguments"
      severity: "high"
    - pattern: "tf.global_variables_initializer"
      replacement: "Variables are initialized automatically in TF2"
      severity: "medium"

code_patterns:
  anti_patterns:
    - pattern: "except:"
      suggestion: "Use specific exception types"
      severity: "low"
"""
    
    rules_path = tmp_path / "test_rules.yaml"
    rules_path.write_text(rules_content)
    return str(rules_path)


@pytest.fixture
def legacy_tf_file(tmp_path):
    """Create a legacy TensorFlow file for testing"""
    code = '''
import tensorflow as tf
import numpy as np

def train_model():
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    
    logits = tf.matmul(x, W) + b
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        # Training code here
        pass

if __name__ == "__main__":
    train_model()
'''
    
    file_path = tmp_path / "legacy_tf.py"
    file_path.write_text(code)
    return str(file_path)


def test_detector_initialization_without_rules():
    """Test detector can be initialized without rules"""
    detector = APIDetector()
    assert detector.rules == {}


def test_detector_initialization_with_rules(rules_file):
    """Test detector can be initialized with rules file"""
    detector = APIDetector(rules_file)
    assert detector.rules is not None
    assert "deprecated_apis" in detector.rules


def test_load_rules(rules_file):
    """Test loading rules from file"""
    detector = APIDetector()
    detector.load_rules(rules_file)
    
    assert "deprecated_apis" in detector.rules
    assert "tensorflow_v1" in detector.rules["deprecated_apis"]


def test_load_rules_nonexistent_file():
    """Test loading rules from nonexistent file raises error"""
    detector = APIDetector()
    with pytest.raises(FileNotFoundError):
        detector.load_rules("nonexistent_rules.yaml")


def test_detect_deprecated_apis(rules_file, legacy_tf_file):
    """Test detecting deprecated APIs in code"""
    detector = APIDetector(rules_file)
    parser = PythonParser(legacy_tf_file)
    analysis = parser.parse()
    
    deprecated = detector.detect(analysis, framework="tensorflow_v1")
    
    # Should find deprecated APIs
    assert len(deprecated) > 0
    
    # Check for specific patterns
    patterns = [api.pattern for api in deprecated]
    assert "tf.Session" in patterns or "tf.placeholder" in patterns


def test_get_framework_info():
    """Test identifying framework from dependencies"""
    detector = APIDetector()
    
    # Test TensorFlow
    tf_info = detector.get_framework_info(["tensorflow", "numpy"])
    assert tf_info is not None
    assert tf_info["name"] == "tensorflow"
    
    # Test PyTorch
    pytorch_info = detector.get_framework_info(["torch", "numpy"])
    assert pytorch_info is not None
    assert pytorch_info["name"] == "pytorch"
    
    # Test no framework
    no_info = detector.get_framework_info(["numpy", "pandas"])
    assert no_info is None or len(no_info) == 0


def test_generate_summary(rules_file, legacy_tf_file):
    """Test generating summary of deprecated API usage"""
    detector = APIDetector(rules_file)
    parser = PythonParser(legacy_tf_file)
    analysis = parser.parse()
    
    deprecated = detector.detect(analysis, framework="tensorflow_v1")
    summary = detector.generate_summary(deprecated)
    
    # Check summary structure
    assert "total_count" in summary
    assert "by_severity" in summary
    assert "unique_apis" in summary
    assert "unique_count" in summary
    
    # Verify counts
    assert summary["total_count"] == len(deprecated)
    assert summary["unique_count"] > 0


def test_detect_patterns(rules_file):
    """Test detecting code patterns"""
    detector = APIDetector(rules_file)
    
    code_with_pattern = '''
try:
    risky_operation()
except:
    pass
'''
    
    patterns = detector.detect_patterns(code_with_pattern, pattern_type="anti_patterns")
    
    # Should detect bare except
    assert len(patterns) > 0
    assert any("except:" in p["pattern"] for p in patterns)


def test_detect_no_deprecated_apis(rules_file, tmp_path):
    """Test detection with no deprecated APIs"""
    # Create clean modern code
    clean_file = tmp_path / "clean.py"
    clean_file.write_text("""
import numpy as np

def add(a, b):
    return a + b
""")
    
    detector = APIDetector(rules_file)
    parser = PythonParser(str(clean_file))
    analysis = parser.parse()
    
    deprecated = detector.detect(analysis, framework="tensorflow_v1")
    
    # Should find no deprecated APIs
    assert len(deprecated) == 0


def test_severity_levels(rules_file, legacy_tf_file):
    """Test that severity levels are correctly assigned"""
    detector = APIDetector(rules_file)
    parser = PythonParser(legacy_tf_file)
    analysis = parser.parse()
    
    deprecated = detector.detect(analysis, framework="tensorflow_v1")
    
    # Check that severities are valid
    valid_severities = ["low", "medium", "high", "critical"]
    for api in deprecated:
        assert api.severity in valid_severities
