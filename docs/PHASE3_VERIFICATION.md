# Phase 3: Verification Sandbox - Complete! 🎉

Phase 3 of Code-Morph implements **automated behavioral equivalence verification** to prove zero logical drift between legacy and migrated code.

## Architecture

### Components

1. **Test Generator** (`test_generator.py`)
   - Auto-generates pytest test suites using Groq LLM
   - Creates unit tests and comparison tests
   - Fallback test generation when LLM fails
   - Saves tests to configurable output directory

2. **Docker Manager** (`docker_manager.py`)
   - Manages Docker container lifecycle
   - Creates isolated sandbox environments
   - Handles file copying and command execution
   - Resource limits (CPU, memory)
   - Container cleanup

3. **Test Executor** (`test_executor.py`)
   - Executes pytest in sandboxed containers
   - Installs dependencies automatically
   - Collects test results with JSON reporter
   - Parses execution output
   - Handles timeouts gracefully

4. **Output Comparator** (`comparator.py`)
   - Deep comparison using `deepdiff`
   - Numerical tolerance for floating-point
   - Array comparison with numpy
   - Similarity scoring
   - Detailed difference reporting

## Usage

### CLI Command

```bash
python -m src.main verify <legacy_file> <migrated_file> [OPTIONS]
```

### Options

- `--output, -o`: Save verification report as JSON
- `--requirements, -r`: Comma-separated dependencies (e.g., "torch,numpy")
- `--no-docker`: Skip Docker sandboxing, run tests locally

### Example

```bash
# Verify with Docker sandboxing
python -m src.main verify \
  examples/tensorflow_to_pytorch/input/legacy_mnist_classifier.py \
  examples/tensorflow_to_pytorch/output/pytorch_mnist_classifier.py \
  --requirements "torch,tensorflow==1.15.0" \
  --output outputs/verification_report.json

# Verify without Docker (local execution)
python -m src.main verify \
  legacy.py \
  migrated.py \
  --no-docker
```

## Workflow

1. **Test Generation**
   - LLM analyzes code structure
   - Generates comprehensive pytest tests
   - Includes unit tests and equivalence checks

2. **Sandboxed Execution**
   - Spins up isolated Docker containers
   - Installs dependencies
   - Copies code files
   - Runs pytest with JSON reporter

3. **Result Collection**
   - Parses pytest JSON output
   - Extracts test results, outputs, execution time
   - Handles errors gracefully

4. **Comparison**
   - Deep comparison of outputs
   - Numerical tolerance for floats
   - Array comparison for numpy/torch tensors
   - Calculates similarity score

5. **Reporting**
   - Rich terminal output with tables
   - Similarity percentage
   - List of differences
   - Optional JSON report

## Features

### Docker Sandboxing

- **Isolation**: Each test runs in a clean container
- **Resource Limits**: CPU and memory constraints
- **Network Disabled**: Security by default
- **Auto-Cleanup**: Containers destroyed after execution

### Intelligent Comparison

- **Type-Aware**: Handles strings, numbers, arrays, dicts
- **Numerical Tolerance**: Configurable epsilon (default: 1e-5)
- **Deep Diff**: Recursive comparison of nested structures
- **Ignore Order**: Optional sequence order independence

### LLM-Powered Test Generation

- **Deterministic Tests**: Seeded for reproducibility
- **Comprehensive Coverage**: Units, integration, edge cases
- **Mocking Support**: Auto-generates mocks for external deps
- **Fallback Templates**: Works even if LLM fails

## Configuration

### Docker Config (defaults)

```python
DockerConfig(
    image="python:3.10-slim",
    memory_limit="2g",
    cpu_limit=2.0,
    timeout_seconds=300,
    network_disabled=True,
)
```

### Comparator Config

```python
OutputComparator(
    numerical_tolerance=1e-5,
    ignore_order=False,
)
```

## Output Format

### Terminal Display

```
🔬 Code-Morph Verification Engine
Proving zero logical drift...

✓ Generated tests: outputs/verification/tests/test_legacy.py
✓ Executing tests in sandbox...

📊 Verification Results:

┏━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ Metric        ┃ Legacy ┃ Migrated ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ Tests Passed  │ 10/10  │ 10/10    │
│ Execution Time│ 2.45s  │ 2.38s    │
└───────────────┴────────┴──────────┘

🔍 Behavioral Equivalence:
  Similarity Score: 100.0%

✅ VERIFIED: Zero logical drift confirmed!
```

### JSON Report

```json
{
  "legacy_results": {
    "file": "legacy.py",
    "passed": 10,
    "total": 10,
    "execution_time": 2.45
  },
  "migrated_results": {
    "file": "migrated.py",
    "passed": 10,
    "total": 10,
    "execution_time": 2.38
  },
  "comparison": {
    "are_equivalent": true,
    "similarity_score": 1.0,
    "differences": null
  }
}
```

## Dependencies

```
docker>=7.0.0
pytest>=7.4.0
pytest-docker>=2.0.0
deepdiff>=6.7.0
numpy>=1.24.0
groq>=0.4.0  # For test generation
```

## Next Steps

- [ ] **Phase 4**: Agent Orchestration with LangGraph
  - Multi-agent collaboration
  - Automated workflow orchestration
  - Dependency resolution across repos
  
- [ ] **Phase 5**: TypeScript Support
  - ts-morph AST parser
  - JavaScript/TypeScript transformer
  - React/Next.js migration rules

- [ ] **Phase 6**: Polish & Portfolio Demo
  - Video walkthrough
  - Interactive documentation
  - Performance benchmarks
  - Case studies

## Troubleshooting

### Docker Not Available

If Docker is not installed or not running:
```bash
# Use --no-docker flag
python -m src.main verify legacy.py migrated.py --no-docker
```

### Rate Limiting

If you hit Groq rate limits during test generation:
- The system will automatically retry with exponential backoff
- Fallback template tests will be used if all retries fail

### Test Failures

If tests fail unexpectedly:
- Check that dependencies are specified correctly with `-r`
- Verify code is syntactically correct
- Review test file in `outputs/verification/tests/`

---

**Status**: ✅ Phase 3 Complete  
**Next**: Phase 4 - Agent Orchestration
