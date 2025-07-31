"""LangSmith configuration and initialization module.

This module handles the configuration and setup of LangSmith for
observability and tracing of LangChain/LangGraph operations.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
import warnings

from pydantic import Field
from pydantic_settings import BaseSettings

from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class LangSmithConfig(BaseSettings):
    """LangSmith configuration settings.

    This configuration is loaded from environment variables with the
    LANGSMITH_ prefix.
    """

    tracing: bool = Field(
        default=False, description="Enable or disable LangSmith tracing"
    )

    api_key: Optional[str] = Field(
        default=None, description="LangSmith API key for authentication"
    )

    project: Optional[str] = Field(
        default="virtual-agora",
        description="LangSmith project name for organizing traces",
    )

    endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API endpoint URL",
    )

    class Config:
        env_prefix = "LANGSMITH_"
        case_sensitive = False
        extra = "ignore"


@dataclass
class LangSmithStatus:
    """Status of LangSmith configuration and initialization."""

    enabled: bool
    configured: bool
    project: Optional[str] = None
    endpoint: Optional[str] = None
    error: Optional[str] = None


def initialize_langsmith() -> LangSmithStatus:
    """Initialize LangSmith tracing if configured.

    This function:
    1. Loads LangSmith configuration from environment
    2. Validates the configuration
    3. Sets up tracing if enabled
    4. Returns status information

    Returns:
        LangSmithStatus: Status of LangSmith initialization
    """
    try:
        # Load configuration
        config = LangSmithConfig()

        logger.debug(
            f"LangSmith config loaded: tracing={config.tracing}, "
            f"api_key={'set' if config.api_key else 'not set'}, "
            f"project={config.project}, endpoint={config.endpoint}"
        )

        # Check if tracing is disabled
        if not config.tracing:
            logger.info("LangSmith tracing is disabled (LANGSMITH_TRACING != 'true')")
            return LangSmithStatus(enabled=False, configured=False)

        # Validate API key if tracing is enabled
        if not config.api_key:
            logger.warning(
                "LangSmith tracing is enabled but LANGSMITH_API_KEY is not set. "
                "Tracing will be disabled. Please set LANGSMITH_API_KEY environment variable."
            )
            return LangSmithStatus(
                enabled=False, configured=False, error="API key not configured"
            )

        # Validate API key format
        if not config.api_key.startswith("lsv2_") and not config.api_key.startswith(
            "ls__"
        ):
            logger.warning(
                f"LangSmith API key may be invalid. Expected format: 'lsv2_...' or 'ls__...', "
                f"got: '{config.api_key[:10]}...'"
            )

        # Set environment variables for LangChain/LangGraph
        # These need to be set before any LangChain imports
        logger.debug("Setting LangSmith environment variables...")
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = config.api_key

        if config.project:
            os.environ["LANGSMITH_PROJECT"] = config.project
            logger.debug(f"Set LANGSMITH_PROJECT={config.project}")

        if config.endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = config.endpoint
            logger.debug(f"Set LANGSMITH_ENDPOINT={config.endpoint}")

        # Optionally set callback behavior for better performance
        # Default to true for local development
        if "LANGCHAIN_CALLBACKS_BACKGROUND" not in os.environ:
            os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "true"
            logger.debug("Set LANGCHAIN_CALLBACKS_BACKGROUND=true")

        # Verify environment variables are actually set
        verification_errors = []
        if os.environ.get("LANGSMITH_TRACING") != "true":
            verification_errors.append("LANGSMITH_TRACING not set to 'true'")
        if not os.environ.get("LANGSMITH_API_KEY"):
            verification_errors.append("LANGSMITH_API_KEY not set in environment")

        if verification_errors:
            error_msg = f"Environment variable verification failed: {', '.join(verification_errors)}"
            logger.error(error_msg)
            return LangSmithStatus(enabled=False, configured=False, error=error_msg)

        logger.info(
            f"LangSmith tracing initialized successfully: "
            f"project={config.project}, endpoint={config.endpoint}"
        )

        # Log all LangSmith-related environment variables for debugging
        logger.debug("Current LangSmith environment variables:")
        for key, value in os.environ.items():
            if key.startswith("LANGSMITH_") or key.startswith("LANGCHAIN_"):
                masked_value = value[:10] + "..." if "KEY" in key else value
                logger.debug(f"  {key}={masked_value}")

        return LangSmithStatus(
            enabled=True,
            configured=True,
            project=config.project,
            endpoint=config.endpoint,
        )

    except Exception as e:
        logger.error(f"Failed to initialize LangSmith: {e}", exc_info=True)
        return LangSmithStatus(enabled=False, configured=False, error=str(e))


def get_langsmith_url(run_id: Optional[str] = None) -> Optional[str]:
    """Get the LangSmith URL for viewing traces.

    Args:
        run_id: Optional specific run ID to link to

    Returns:
        URL to LangSmith dashboard or None if not configured
    """
    config = LangSmithConfig()

    if not config.api_key or not config.tracing:
        return None

    base_url = config.endpoint.replace("/api/v1", "").rstrip("/")
    project = config.project or "default"

    if run_id:
        return f"{base_url}/projects/p/{project}/runs/{run_id}"
    else:
        return f"{base_url}/projects/p/{project}"


def validate_langsmith_config() -> Dict[str, Any]:
    """Validate LangSmith configuration and return diagnostics.

    Returns:
        Dictionary containing validation results and diagnostics
    """
    config = LangSmithConfig()

    diagnostics = {
        "tracing_enabled": config.tracing,
        "api_key_configured": bool(config.api_key),
        "api_key_valid_format": False,
        "project": config.project,
        "endpoint": config.endpoint,
        "endpoint_reachable": False,
        "env_vars_set": {},
        "errors": [],
        "warnings": [],
    }

    # Check environment variable settings
    env_vars_to_check = [
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
        "LANGCHAIN_CALLBACKS_BACKGROUND",
    ]

    for var in env_vars_to_check:
        value = os.environ.get(var)
        diagnostics["env_vars_set"][var] = bool(value)
        if value and "KEY" in var:
            # Mask sensitive values
            diagnostics["env_vars_set"][f"{var}_value"] = value[:10] + "..."

    if not config.tracing:
        diagnostics["errors"].append(
            "Tracing is disabled (LANGSMITH_TRACING != 'true')"
        )

    if config.tracing and not config.api_key:
        diagnostics["errors"].append("API key not configured")

    # Validate API key format
    if config.api_key:
        if config.api_key.startswith("lsv2_") or config.api_key.startswith("ls__"):
            diagnostics["api_key_valid_format"] = True
        else:
            diagnostics["warnings"].append(
                f"API key format may be invalid. Expected 'lsv2_...' or 'ls__...', "
                f"got '{config.api_key[:10]}...'"
            )

    # Check if LangChain has already been imported
    import sys

    langchain_modules = [mod for mod in sys.modules if mod.startswith("langchain")]
    if langchain_modules and config.tracing:
        diagnostics["warnings"].append(
            f"LangChain modules already imported before LangSmith initialization: "
            f"{', '.join(langchain_modules[:5])}{'...' if len(langchain_modules) > 5 else ''}"
        )

    # Optionally test endpoint connectivity
    if config.api_key and config.tracing:
        try:
            import requests

            logger.debug(f"Testing LangSmith endpoint: {config.endpoint}/info")
            response = requests.get(
                f"{config.endpoint}/info",
                headers={"x-api-key": config.api_key},
                timeout=5,
            )
            diagnostics["endpoint_reachable"] = response.status_code == 200
            if response.status_code != 200:
                diagnostics["errors"].append(
                    f"Endpoint test returned status {response.status_code}"
                )
        except requests.exceptions.Timeout:
            diagnostics["errors"].append("Endpoint test timed out after 5 seconds")
        except Exception as e:
            diagnostics["errors"].append(f"Endpoint test failed: {e}")

    return diagnostics
