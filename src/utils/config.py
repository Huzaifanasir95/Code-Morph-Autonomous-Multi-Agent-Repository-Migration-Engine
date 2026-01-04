"""
Configuration management for Code-Morph

Handles loading configuration from environment variables and files.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # LLM Configuration
    groq_api_key: str = ""
    default_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.2
    max_tokens: int = 4096

    # Logging
    log_level: str = "INFO"
    log_file: str = "outputs/code-morph.log"

    # Sandbox Configuration
    docker_timeout: int = 300
    max_retries: int = 3

    # State Management
    redis_url: str = "redis://localhost:6379"
    state_db: str = "outputs/state.db"

    # Paths
    project_root: Path = Path(__file__).parent.parent.parent
    config_dir: Path = project_root / "config"
    outputs_dir: Path = project_root / "outputs"
    examples_dir: Path = project_root / "examples"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


def get_rules_file(language: str = "python") -> Optional[str]:
    """
    Get path to migration rules file for a language

    Args:
        language: Programming language

    Returns:
        Path to rules file or None if not found
    """
    rules_path = settings.config_dir / "migration_rules" / f"{language}.yaml"
    if rules_path.exists():
        return str(rules_path)
    return None


def ensure_output_dirs() -> None:
    """Ensure all output directories exist"""
    dirs = [
        settings.outputs_dir,
        settings.outputs_dir / "migration_plans",
        settings.outputs_dir / "migrated_code",
        settings.outputs_dir / "test_reports",
        settings.outputs_dir / "videos",
    ]

    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
