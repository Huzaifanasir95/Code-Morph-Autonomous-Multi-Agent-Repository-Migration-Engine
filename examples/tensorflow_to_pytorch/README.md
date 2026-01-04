# TensorFlow 1.x to PyTorch Migration Example

This directory contains a sample legacy TensorFlow 1.x codebase for demonstration purposes.

## Input Directory

The `input/` folder contains:
- **legacy_mnist_classifier.py**: A simple MNIST classifier using deprecated TensorFlow 1.x APIs

### Deprecated APIs Used

This example demonstrates common TensorFlow 1.x patterns that need migration:

1. **Session API**: `tf.Session()` - No longer needed in TF2/PyTorch
2. **Placeholders**: `tf.placeholder()` - Replaced by function arguments or tf.Variable
3. **Global Variables**: `tf.global_variables_initializer()` - Auto-initialized in TF2
4. **Feed Dictionaries**: `feed_dict={}` - Replaced by direct function calls
5. **Variable Scopes**: `tf.variable_scope()` - Replaced by Keras layers or PyTorch modules
6. **Old Optimizers**: `tf.train.AdamOptimizer` - Now `tf.keras.optimizers.Adam` or PyTorch optimizers
7. **Deprecated Loss**: `tf.nn.softmax_cross_entropy_with_logits` - Updated in TF2/PyTorch

## Running the Analysis

Use Code-Morph to analyze this legacy code:

```bash
# Analyze the legacy file
code-morph analyze examples/tensorflow_to_pytorch/input/legacy_mnist_classifier.py

# Generate detailed migration plan
code-morph analyze examples/tensorflow_to_pytorch/input/legacy_mnist_classifier.py \
    --output outputs/migration_plans/mnist_plan.json \
    --target pytorch \
    --framework tensorflow==1.15.0
```

## Expected Output

Code-Morph will:
1. Parse the AST and extract all functions, classes, and imports
2. Detect all deprecated TensorFlow 1.x APIs
3. Generate a comprehensive migration plan with transformations
4. Estimate complexity, time, and confidence scores
5. Provide warnings about potential issues

## Output Directory

The `output/` folder will contain the migrated PyTorch code (Phase 2).
