"""Configuration loader for Virtual Agora.

This module handles loading and parsing YAML configuration files
with proper error handling and validation.
"""

import os
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import ValidationError

from virtual_agora.config.models import Config
from virtual_agora.utils.exceptions import ConfigurationError
from virtual_agora.utils.logging import get_logger


logger = get_logger(__name__)


class ConfigLoader:
    """Loads and validates configuration from YAML files."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize the configuration loader.

        Args:
            config_path: Path to the configuration file.
                        Defaults to 'config.yml' in current directory.
        """
        if config_path is None:
            config_path = Path("config.yml")

        self.config_path = Path(config_path)
        self._config: Optional[Config] = None

    def load(self) -> Config:
        """Load and validate the configuration file.

        Returns:
            Validated configuration object.

        Raises:
            ConfigurationError: If the configuration is invalid or cannot be loaded.
        """
        if self._config is not None:
            return self._config

        try:
            # Check if file exists
            if not self.config_path.exists():
                raise ConfigurationError(
                    f"Configuration file not found: {self.config_path}",
                    details={"path": str(self.config_path.absolute())},
                )

            # Load YAML file
            logger.info(f"Loading configuration from: {self.config_path}")
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)

            if raw_config is None:
                raise ConfigurationError(
                    "Configuration file is empty",
                    details={"path": str(self.config_path)},
                )

            # Validate with Pydantic
            self._config = Config(**raw_config)

            # Log successful load
            logger.info(
                f"Configuration loaded successfully: "
                f"1 moderator, {self._config.get_total_agent_count()} agents"
            )

            return self._config

        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML syntax in configuration file: {e}",
                details={
                    "path": str(self.config_path),
                    "error": str(e),
                },
            )
        except ValidationError as e:
            # Format Pydantic validation errors nicely
            error_messages = []
            for error in e.errors():
                loc = " -> ".join(str(x) for x in error["loc"])
                msg = error["msg"]
                error_messages.append(f"{loc}: {msg}")

            raise ConfigurationError(
                "Configuration validation failed:\n" + "\n".join(error_messages),
                details={
                    "path": str(self.config_path),
                    "errors": e.errors(),
                },
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration: {e}",
                details={
                    "path": str(self.config_path),
                    "error_type": type(e).__name__,
                },
            )

    def reload(self) -> Config:
        """Reload the configuration file.

        This forces a fresh load of the configuration, useful for
        picking up changes during development.

        Returns:
            Validated configuration object.
        """
        self._config = None
        return self.load()

    @property
    def config(self) -> Config:
        """Get the loaded configuration, loading it if necessary.

        Returns:
            Validated configuration object.
        """
        if self._config is None:
            self.load()
        return self._config

    def validate_api_keys(self) -> dict[str, bool]:
        """Check if required API keys are present in environment.

        Returns:
            Dictionary mapping provider names to availability status.

        Note:
            This only checks if the environment variables exist,
            not if the keys are valid.
        """
        if self._config is None:
            self.load()

        required_providers = set()

        # Add moderator provider
        required_providers.add(self._config.moderator.provider.value)

        # Add agent providers
        for agent in self._config.agents:
            required_providers.add(agent.provider.value)

        # Check environment variables
        key_mapping = {
            "Google": "GOOGLE_API_KEY",
            "OpenAI": "OPENAI_API_KEY",
            "Anthropic": "ANTHROPIC_API_KEY",
            "Grok": "GROK_API_KEY",
        }

        results = {}
        for provider in required_providers:
            env_var = key_mapping.get(provider)
            if env_var:
                results[provider] = bool(os.getenv(env_var))
            else:
                results[provider] = False

        return results

    def get_missing_api_keys(self) -> list[str]:
        """Get list of missing API keys.

        Returns:
            List of provider names with missing API keys.

        .. deprecated:: 0.2.0
            Use EnvironmentManager.get_missing_providers() instead.
        """
        key_status = self.validate_api_keys()
        return [provider for provider, available in key_status.items() if not available]
