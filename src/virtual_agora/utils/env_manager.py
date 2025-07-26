"""Environment variable management for Virtual Agora.

This module provides secure loading, validation, and management of
environment variables, particularly API keys for LLM providers.
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Set

from dotenv import load_dotenv, find_dotenv

from virtual_agora.utils.exceptions import ConfigurationError
from virtual_agora.utils.logging import get_logger


logger = get_logger(__name__)


class EnvironmentManager:
    """Manages environment variables and API keys securely."""

    # Known API key environment variables
    API_KEY_VARS = {
        "GOOGLE_API_KEY": "Google Gemini",
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic",
        "GROK_API_KEY": "Grok",
    }

    # Optional environment variables
    OPTIONAL_VARS = {
        "VIRTUAL_AGORA_LOG_LEVEL": "INFO",
        "VIRTUAL_AGORA_LOG_DIR": "logs",
        "VIRTUAL_AGORA_REPORT_DIR": "reports",
        "VIRTUAL_AGORA_SESSION_TIMEOUT": "3600",  # seconds
    }

    def __init__(self, env_file: Optional[Path] = None, override: bool = True):
        """Initialize the environment manager.

        Args:
            env_file: Path to .env file. If None, searches for .env file.
            override: Whether to override existing environment variables.
        """
        self.env_file = env_file
        self.override = override
        self._loaded = False
        self._missing_keys: Set[str] = set()

    def load(self) -> None:
        """Load environment variables from file and system environment.

        Raises:
            ConfigurationError: If .env file is specified but not found.
        """
        if self._loaded:
            return

        # Load from .env file
        if self.env_file:
            if not self.env_file.exists():
                raise ConfigurationError(
                    f"Environment file not found: {self.env_file}",
                    details={"path": str(self.env_file.absolute())},
                )
            load_dotenv(self.env_file, override=self.override)
            logger.info(f"Loaded environment from: {self.env_file}")
        else:
            # Try to find .env file
            env_path = find_dotenv()
            if env_path:
                load_dotenv(env_path, override=self.override)
                logger.info(f"Loaded environment from: {env_path}")
                self.env_file = Path(env_path)
            else:
                logger.info("No .env file found, using system environment only")

        # Check file permissions if .env exists
        if self.env_file and self.env_file.exists():
            self._check_env_file_security()

        self._loaded = True

    def _check_env_file_security(self) -> None:
        """Check if .env file has secure permissions."""
        if not self.env_file or not self.env_file.exists():
            return

        # Check if file is world-readable (Unix-like systems)
        try:
            import stat

            file_stat = self.env_file.stat()
            mode = file_stat.st_mode

            # Check if others have read permission
            if mode & stat.S_IROTH:
                warnings.warn(
                    f"Warning: {self.env_file} is world-readable. "
                    "Consider restricting permissions with: chmod 600 "
                    + str(self.env_file),
                    UserWarning,
                )

            # Check if .env is in .gitignore
            gitignore_path = self.env_file.parent / ".gitignore"
            if gitignore_path.exists():
                gitignore_content = gitignore_path.read_text()
                if (
                    ".env" not in gitignore_content
                    and str(self.env_file.name) not in gitignore_content
                ):
                    warnings.warn(
                        f"Warning: {self.env_file.name} may not be in .gitignore. "
                        "Ensure it's excluded from version control.",
                        UserWarning,
                    )
        except (ImportError, OSError) as e:
            logger.debug(f"Could not check file permissions: {e}")

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider.

        Args:
            provider: Provider name (e.g., "Google", "OpenAI").

        Returns:
            API key if found, None otherwise.
        """
        self.load()

        # Map provider name to environment variable
        env_var_map = {
            "Google": "GOOGLE_API_KEY",
            "OpenAI": "OPENAI_API_KEY",
            "Anthropic": "ANTHROPIC_API_KEY",
            "Grok": "GROK_API_KEY",
        }

        env_var = env_var_map.get(provider)
        if not env_var:
            logger.warning(f"Unknown provider: {provider}")
            return None

        return os.getenv(env_var)

    def get_all_api_keys(self) -> Dict[str, Optional[str]]:
        """Get all API keys.

        Returns:
            Dictionary mapping environment variable names to values.
        """
        self.load()
        return {var: os.getenv(var) for var in self.API_KEY_VARS}

    def validate_api_keys(self, required_providers: Set[str]) -> Dict[str, bool]:
        """Validate that required API keys are present.

        Args:
            required_providers: Set of provider names that need API keys.

        Returns:
            Dictionary mapping provider names to availability status.
        """
        self.load()

        results = {}
        self._missing_keys.clear()

        for provider in required_providers:
            key = self.get_api_key(provider)
            is_present = key is not None and len(key.strip()) > 0
            results[provider] = is_present

            if not is_present:
                env_var_map = {
                    "Google": "GOOGLE_API_KEY",
                    "OpenAI": "OPENAI_API_KEY",
                    "Anthropic": "ANTHROPIC_API_KEY",
                    "Grok": "GROK_API_KEY",
                }
                if provider in env_var_map:
                    self._missing_keys.add(env_var_map[provider])

        return results

    def get_missing_keys(self) -> Set[str]:
        """Get set of missing API key environment variable names.

        Returns:
            Set of environment variable names that are missing.
        """
        return self._missing_keys.copy()

    def get_missing_providers(self, required_providers: Set[str]) -> Set[str]:
        """Get set of providers with missing API keys.

        Args:
            required_providers: Set of provider names to check.

        Returns:
            Set of provider names with missing keys.
        """
        validation = self.validate_api_keys(required_providers)
        return {provider for provider, available in validation.items() if not available}

    def mask_api_key(self, key: Optional[str]) -> str:
        """Mask an API key for safe display.

        Args:
            key: API key to mask.

        Returns:
            Masked key showing only last 4 characters.
        """
        if not key:
            return "<not set>"

        if len(key) <= 8:
            return "****"

        return f"{'*' * (len(key) - 4)}{key[-4:]}"

    def get_env_var(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Get an environment variable with optional default.

        Args:
            name: Environment variable name.
            default: Default value if not found.

        Returns:
            Environment variable value or default.
        """
        self.load()
        return os.getenv(name, default)

    def get_int_env_var(self, name: str, default: int = 0) -> int:
        """Get an environment variable as integer.

        Args:
            name: Environment variable name.
            default: Default value if not found or invalid.

        Returns:
            Integer value or default.
        """
        value = self.get_env_var(name)
        if not value:
            return default

        try:
            return int(value)
        except ValueError:
            logger.warning(f"Invalid integer value for {name}: {value}")
            return default

    def get_bool_env_var(self, name: str, default: bool = False) -> bool:
        """Get an environment variable as boolean.

        Args:
            name: Environment variable name.
            default: Default value if not found.

        Returns:
            Boolean value or default.
        """
        value = self.get_env_var(name)
        if not value:
            return default

        return value.lower() in ("true", "1", "yes", "on")

    def get_status_report(self) -> Dict[str, Any]:
        """Get a status report of environment configuration.

        Returns:
            Dictionary with environment status information.
        """
        self.load()

        report = {
            "env_file": str(self.env_file) if self.env_file else None,
            "env_file_exists": self.env_file.exists() if self.env_file else False,
            "api_keys": {},
            "optional_vars": {},
        }

        # Check API keys (masked)
        for var, provider in self.API_KEY_VARS.items():
            value = os.getenv(var)
            report["api_keys"][provider] = {
                "variable": var,
                "is_set": value is not None,
                "masked_value": self.mask_api_key(value),
            }

        # Check optional variables
        for var, default in self.OPTIONAL_VARS.items():
            value = os.getenv(var, default)
            report["optional_vars"][var] = value

        return report
