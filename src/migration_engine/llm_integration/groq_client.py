"""
Groq LLM Client

Provides interface to Groq API for code generation and transformation.
Uses models like llama-3.3-70b-versatile for high-quality code generation.
"""

from typing import Dict, List, Optional

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GroqLLMClient:
    """Client for interacting with Groq API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ):
        """
        Initialize Groq client

        Args:
            api_key: Groq API key (uses settings if None)
            model: Model name (uses settings if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key or settings.groq_api_key
        self.model = model or settings.default_model
        self.temperature = temperature or settings.temperature
        self.max_tokens = max_tokens or settings.max_tokens

        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY in .env file")

        self.client = Groq(api_key=self.api_key)
        logger.info(f"Initialized Groq client with model: {self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def generate_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate completion from prompt

        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Generated text completion
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )

            completion = response.choices[0].message.content
            logger.info(f"Generated completion: {len(completion)} characters")
            return completion

        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise

    def generate_code_transformation(
        self,
        legacy_code: str,
        source_framework: str,
        target_framework: str,
        transformation_description: str,
    ) -> str:
        """
        Generate code transformation using Groq

        Args:
            legacy_code: Original code to transform
            source_framework: Source framework name
            target_framework: Target framework name
            transformation_description: Description of transformation needed

        Returns:
            Transformed code
        """
        system_prompt = f"""You are an expert code migration assistant specializing in transforming code from {source_framework} to {target_framework}.

Your task is to:
1. Analyze the provided legacy code
2. Apply the requested transformation
3. Return ONLY the transformed code without explanations
4. Maintain code functionality and logic
5. Use modern idioms and best practices for {target_framework}
6. Preserve variable names and structure where possible"""

        user_prompt = f"""Transform this {source_framework} code to {target_framework}:

**Transformation Required:**
{transformation_description}

**Legacy Code:**
```python
{legacy_code}
```

**Instructions:**
- Return ONLY the transformed code
- No explanations or markdown
- Ensure the code is syntactically correct
- Maintain the same functionality"""

        logger.info(f"Generating transformation: {source_framework} → {target_framework}")
        transformed_code = self.generate_completion(user_prompt, system_prompt)

        # Clean up any markdown formatting
        transformed_code = self._clean_code_output(transformed_code)

        return transformed_code

    def generate_tests(self, code: str, language: str = "python") -> str:
        """
        Generate unit tests for code

        Args:
            code: Code to generate tests for
            language: Programming language

        Returns:
            Generated test code
        """
        system_prompt = f"""You are an expert test engineer specializing in {language} testing.

Your task is to:
1. Analyze the provided code
2. Generate comprehensive unit tests
3. Return ONLY the test code
4. Use pytest framework for Python
5. Cover edge cases and common scenarios"""

        user_prompt = f"""Generate unit tests for this code:

```{language}
{code}
```

**Requirements:**
- Use pytest
- Test all functions and methods
- Include edge cases
- Return ONLY test code, no explanations"""

        logger.info(f"Generating tests for {language} code")
        test_code = self.generate_completion(user_prompt, system_prompt)

        return self._clean_code_output(test_code)

    def explain_code(self, code: str, detail_level: str = "medium") -> str:
        """
        Generate explanation of code

        Args:
            code: Code to explain
            detail_level: Level of detail (low, medium, high)

        Returns:
            Code explanation
        """
        system_prompt = """You are an expert code educator who explains code clearly and concisely."""

        detail_instructions = {
            "low": "Provide a brief 2-3 sentence summary",
            "medium": "Provide a clear explanation with key points",
            "high": "Provide detailed line-by-line analysis",
        }

        user_prompt = f"""Explain this code. {detail_instructions.get(detail_level, detail_instructions['medium'])}.

```python
{code}
```"""

        return self.generate_completion(user_prompt, system_prompt)

    def _clean_code_output(self, code: str) -> str:
        """
        Clean code output by removing markdown formatting

        Args:
            code: Code string that might contain markdown

        Returns:
            Cleaned code
        """
        # Remove markdown code blocks
        if "```" in code:
            lines = code.split("\n")
            cleaned_lines = []
            in_code_block = False

            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block or not line.strip().startswith("```"):
                    cleaned_lines.append(line)

            code = "\n".join(cleaned_lines)

        return code.strip()

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token
        return len(text) // 4


# Global client instance
_groq_client: Optional[GroqLLMClient] = None


def get_groq_client() -> GroqLLMClient:
    """
    Get global Groq client instance

    Returns:
        GroqLLMClient instance
    """
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqLLMClient()
    return _groq_client
