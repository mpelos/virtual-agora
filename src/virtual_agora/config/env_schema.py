"""Environment variable schema definitions for Virtual Agora.

This module defines Pydantic models for validating environment
variables and provides structured access to configuration.
"""

import re
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

from virtual_agora.utils.exceptions import ConfigurationError


class APIKeyConfig(BaseModel):
    """Configuration for a single API key."""
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    key: Optional[str] = Field(None, min_length=10)
    provider: str
    
    @field_validator("key")
    @classmethod
    def validate_key_format(cls, v: Optional[str], info) -> Optional[str]:
        """Validate API key format based on provider."""
        if v is None:
            return v
            
        provider = info.data.get("provider")
        
        # Basic validation patterns for known providers
        patterns = {
            "Google": r"^AIza[0-9A-Za-z\-_]{35}$",  # Google API keys start with AIza
            "OpenAI": r"^sk-[A-Za-z0-9]{48}$",      # OpenAI keys start with sk-
            "Anthropic": r"^sk-ant-[A-Za-z0-9\-_]+$",  # Anthropic keys start with sk-ant-
            # Grok pattern unknown
        }
        
        pattern = patterns.get(provider)
        if pattern and not re.match(pattern, v):
            # Just warn, don't fail - patterns might change
            import warnings
            warnings.warn(
                f"API key for {provider} may have incorrect format. "
                f"Please verify it's correct.",
                UserWarning
            )
            
        return v


class EnvironmentConfig(BaseSettings):
    """Environment configuration using Pydantic settings.
    
    This provides validated access to environment variables with
    type conversion and default values.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # Ignore extra environment variables
    )
    
    # API Keys
    google_api_key: Optional[str] = Field(
        None,
        alias="GOOGLE_API_KEY",
        description="Google Gemini API key"
    )
    openai_api_key: Optional[str] = Field(
        None,
        alias="OPENAI_API_KEY",
        description="OpenAI API key"
    )
    anthropic_api_key: Optional[str] = Field(
        None,
        alias="ANTHROPIC_API_KEY",
        description="Anthropic API key"
    )
    grok_api_key: Optional[str] = Field(
        None,
        alias="GROK_API_KEY",
        description="Grok API key"
    )
    
    # Application settings
    log_level: str = Field(
        "INFO",
        alias="VIRTUAL_AGORA_LOG_LEVEL",
        description="Logging level",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
    )
    log_dir: str = Field(
        "logs",
        alias="VIRTUAL_AGORA_LOG_DIR",
        description="Directory for log files"
    )
    report_dir: str = Field(
        "reports",
        alias="VIRTUAL_AGORA_REPORT_DIR",
        description="Directory for generated reports"
    )
    session_timeout: int = Field(
        3600,
        alias="VIRTUAL_AGORA_SESSION_TIMEOUT",
        description="Session timeout in seconds",
        gt=0,
        le=86400,  # Max 24 hours
    )
    
    # Development settings
    debug_mode: bool = Field(
        False,
        alias="VIRTUAL_AGORA_DEBUG",
        description="Enable debug mode"
    )
    disable_color: bool = Field(
        False,
        alias="VIRTUAL_AGORA_NO_COLOR",
        description="Disable colored terminal output"
    )
    
    def get_api_keys(self) -> Dict[str, Optional[APIKeyConfig]]:
        """Get all API keys as validated config objects.
        
        Returns:
            Dictionary mapping provider names to API key configs.
        """
        return {
            "Google": APIKeyConfig(
                key=self.google_api_key,
                provider="Google"
            ) if self.google_api_key else None,
            "OpenAI": APIKeyConfig(
                key=self.openai_api_key,
                provider="OpenAI"
            ) if self.openai_api_key else None,
            "Anthropic": APIKeyConfig(
                key=self.anthropic_api_key,
                provider="Anthropic"
            ) if self.anthropic_api_key else None,
            "Grok": APIKeyConfig(
                key=self.grok_api_key,
                provider="Grok"
            ) if self.grok_api_key else None,
        }
        
    def get_api_key_for_provider(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider.
        
        Args:
            provider: Provider name.
            
        Returns:
            API key if available, None otherwise.
        """
        provider_map = {
            "Google": self.google_api_key,
            "OpenAI": self.openai_api_key,
            "Anthropic": self.anthropic_api_key,
            "Grok": self.grok_api_key,
        }
        return provider_map.get(provider)
        
    def validate_required_keys(self, required_providers: set[str]) -> None:
        """Validate that required API keys are present.
        
        Args:
            required_providers: Set of provider names that need keys.
            
        Raises:
            ConfigurationError: If any required keys are missing.
        """
        missing_providers = []
        
        for provider in required_providers:
            key = self.get_api_key_for_provider(provider)
            if not key:
                missing_providers.append(provider)
                
        if missing_providers:
            provider_to_env = {
                "Google": "GOOGLE_API_KEY",
                "OpenAI": "OPENAI_API_KEY",
                "Anthropic": "ANTHROPIC_API_KEY",
                "Grok": "GROK_API_KEY",
            }
            
            missing_vars = [
                provider_to_env.get(p, f"{p.upper()}_API_KEY")
                for p in missing_providers
            ]
            
            raise ConfigurationError(
                f"Missing API keys for providers: {', '.join(missing_providers)}",
                details={
                    "missing_providers": missing_providers,
                    "missing_variables": missing_vars,
                    "help": "Set the required environment variables in your .env file or system environment",
                }
            )
            
    def mask_sensitive_values(self) -> Dict[str, Any]:
        """Get configuration with masked sensitive values.
        
        Returns:
            Dictionary with configuration values, API keys masked.
        """
        config = self.model_dump()
        
        # Mask API keys
        for key in ["google_api_key", "openai_api_key", "anthropic_api_key", "grok_api_key"]:
            if config.get(key):
                value = config[key]
                if len(value) > 8:
                    config[key] = f"{'*' * (len(value) - 4)}{value[-4:]}"
                else:
                    config[key] = "****"
                    
        return config