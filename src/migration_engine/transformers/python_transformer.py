"""
Python Transformer

Transforms Python code from one framework to another using Groq LLM.
Specializes in TensorFlow → PyTorch migrations.
"""

import re
from typing import List, Optional

from src.ast_engine.schemas.migration_models import CodeTransformation, MigrationPlan
from src.migration_engine.llm_integration.groq_client import get_groq_client
from src.migration_engine.transformers.base_transformer import BaseTransformer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PythonTransformer(BaseTransformer):
    """Transformer for Python code migrations"""

    def __init__(
        self,
        source_framework: str = "tensorflow",
        target_framework: str = "pytorch",
        use_llm: bool = True,
    ):
        """
        Initialize Python transformer

        Args:
            source_framework: Source framework
            target_framework: Target framework
            use_llm: Whether to use LLM for complex transformations
        """
        super().__init__(source_framework, target_framework)
        self.use_llm = use_llm
        if use_llm:
            self.llm_client = get_groq_client()

    def transform_file(
        self, file_path: str, migration_plan: MigrationPlan, output_path: Optional[str] = None
    ) -> str:
        """
        Transform a Python file according to migration plan

        Args:
            file_path: Path to source file
            migration_plan: Migration plan with transformations
            output_path: Optional output path

        Returns:
            Path to transformed file
        """
        logger.info(f"Starting transformation: {file_path}")

        # Read original code
        original_code = self.read_file(file_path)

        # Apply transformations sequentially
        transformed_code = original_code
        for i, transformation in enumerate(migration_plan.transformations, 1):
            logger.info(f"Applying transformation {i}/{len(migration_plan.transformations)}: {transformation.description}")
            
            try:
                transformed_code = self.apply_transformation(transformed_code, transformation)
            except Exception as e:
                logger.error(f"Error applying transformation: {e}")
                # Continue with next transformation

        # If using LLM and transformations require it, do a final pass
        if self.use_llm and any(t.requires_llm for t in migration_plan.transformations):
            logger.info("Performing final LLM-based refinement")
            transformed_code = self._llm_final_pass(transformed_code, migration_plan)

        # Write to output
        if output_path is None:
            output_path = file_path.replace(".py", "_migrated.py")

        self.write_file(output_path, transformed_code)
        logger.info(f"Transformation complete: {output_path}")

        return output_path

    def apply_transformation(
        self, code: str, transformation: CodeTransformation
    ) -> str:
        """
        Apply a single transformation to code

        Args:
            code: Source code
            transformation: Transformation to apply

        Returns:
            Transformed code
        """
        if transformation.requires_llm and self.use_llm:
            return self._apply_llm_transformation(code, transformation)
        else:
            return self._apply_pattern_transformation(code, transformation)

    def _apply_pattern_transformation(
        self, code: str, transformation: CodeTransformation
    ) -> str:
        """
        Apply pattern-based transformation (simple find-replace)

        Args:
            code: Source code
            transformation: Transformation to apply

        Returns:
            Transformed code
        """
        # Simple pattern replacement
        old_pattern = transformation.old_pattern
        new_pattern = transformation.new_pattern

        # Replace all occurrences
        transformed = code.replace(old_pattern, new_pattern)

        return transformed

    def _apply_llm_transformation(
        self, code: str, transformation: CodeTransformation
    ) -> str:
        """
        Apply LLM-based transformation for complex changes

        Args:
            code: Source code
            transformation: Transformation to apply

        Returns:
            Transformed code
        """
        logger.info(f"Using LLM for transformation: {transformation.description}")

        try:
            # Use LLM to transform the specific section
            transformed = self.llm_client.generate_code_transformation(
                legacy_code=code,
                source_framework=self.source_framework,
                target_framework=self.target_framework,
                transformation_description=transformation.description,
            )
            return transformed
        except Exception as e:
            logger.error(f"LLM transformation failed: {e}")
            # Fallback to pattern-based
            return self._apply_pattern_transformation(code, transformation)

    def _llm_final_pass(self, code: str, migration_plan: MigrationPlan) -> str:
        """
        Final LLM pass to clean up and optimize transformed code

        Args:
            code: Partially transformed code
            migration_plan: Migration plan with context

        Returns:
            Fully transformed and optimized code
        """
        system_prompt = f"""You are an expert Python developer specializing in {self.target_framework}.

Your task is to review and finalize a code migration from {self.source_framework} to {self.target_framework}.

Requirements:
1. Ensure all {self.source_framework} APIs are replaced with {self.target_framework} equivalents
2. Use modern {self.target_framework} best practices
3. Maintain the same functionality and logic
4. Fix any syntax errors or issues
5. Optimize for performance where possible
6. Keep code clean and readable
7. Preserve docstrings and comments

Return ONLY the final code, no explanations."""

        user_prompt = f"""Finalize this code migration from {self.source_framework} to {self.target_framework}:

```python
{code}
```

Ensure the code:
- Uses {self.target_framework} APIs correctly
- Has no {self.source_framework} remnants
- Is syntactically correct and follows best practices
- Maintains the original functionality

Return ONLY the complete finalized code."""

        try:
            finalized_code = self.llm_client.generate_completion(user_prompt, system_prompt)
            # Clean markdown formatting
            finalized_code = self.llm_client._clean_code_output(finalized_code)
            return finalized_code
        except Exception as e:
            logger.error(f"Final LLM pass failed: {e}")
            return code  # Return partially transformed code

    def update_imports(self, code: str) -> str:
        """
        Update import statements for target framework

        Args:
            code: Source code

        Returns:
            Code with updated imports
        """
        if "tensorflow" in self.source_framework.lower() and "pytorch" in self.target_framework.lower():
            # TensorFlow to PyTorch import mappings
            code = re.sub(r"import tensorflow as tf", "import torch\nimport torch.nn as nn\nimport torch.optim as optim", code)
            code = re.sub(r"import tensorflow", "import torch", code)
            code = re.sub(r"from tensorflow[.\w]* import", "from torch import", code)

        return code
