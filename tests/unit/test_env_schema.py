"""Tests for environment variable schema."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from virtual_agora.config.env_schema import APIKeyConfig, EnvironmentConfig
from virtual_agora.utils.exceptions import ConfigurationError


class TestAPIKeyConfig:
    """Test the APIKeyConfig model."""
    
    def test_valid_api_key(self):
        """Test creating valid API key config."""
        config = APIKeyConfig(
            key="AIzaSyD-1234567890abcdefghijklmnopqrstuv",
            provider="Google"
        )
        assert config.key == "AIzaSyD-1234567890abcdefghijklmnopqrstuv"
        assert config.provider == "Google"
        
    def test_none_api_key(self):
        """Test API key can be None."""
        config = APIKeyConfig(key=None, provider="Google")
        assert config.key is None
        
    def test_short_api_key(self):
        """Test API key minimum length validation."""
        with pytest.raises(ValidationError) as exc_info:
            APIKeyConfig(key="short", provider="Google")
        assert "at least 10 characters" in str(exc_info.value)
        
    def test_whitespace_stripping(self):
        """Test that whitespace is stripped from keys."""
        config = APIKeyConfig(
            key="  AIzaSyD-1234567890abcdefghijklmnopqrstuv  ",
            provider="Google"
        )
        assert config.key == "AIzaSyD-1234567890abcdefghijklmnopqrstuv"
        
    @patch("warnings.warn")
    def test_google_key_format_warning(self, mock_warn):
        """Test Google API key format validation warning."""
        # Valid format - no warning
        config = APIKeyConfig(
            key="AIzaSyD-1234567890abcdefghijklmnopqrstuv",
            provider="Google"
        )
        assert not mock_warn.called
        
        # Invalid format - should warn
        mock_warn.reset_mock()
        config = APIKeyConfig(
            key="invalid-google-key-format-1234567890",
            provider="Google"
        )
        assert mock_warn.called
        assert "incorrect format" in str(mock_warn.call_args[0][0])
        
    @patch("warnings.warn")
    def test_openai_key_format_warning(self, mock_warn):
        """Test OpenAI API key format validation warning."""
        # Valid format
        config = APIKeyConfig(
            key="sk-" + "a" * 48,
            provider="OpenAI"
        )
        assert not mock_warn.called
        
        # Invalid format
        mock_warn.reset_mock()
        config = APIKeyConfig(
            key="openai-invalid-key",
            provider="OpenAI"
        )
        assert mock_warn.called
        
    def test_unknown_provider_no_validation(self):
        """Test that unknown providers don't trigger format validation."""
        # Should not raise or warn for unknown provider
        config = APIKeyConfig(
            key="any-format-key-123",
            provider="UnknownProvider"
        )
        assert config.key == "any-format-key-123"


class TestEnvironmentConfig:
    """Test the EnvironmentConfig model."""
    
    def test_load_from_env_vars(self, monkeypatch):
        """Test loading configuration from environment variables."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test_google_key")
        monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
        monkeypatch.setenv("VIRTUAL_AGORA_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("VIRTUAL_AGORA_LOG_DIR", "custom_logs")
        monkeypatch.setenv("VIRTUAL_AGORA_SESSION_TIMEOUT", "7200")
        monkeypatch.setenv("VIRTUAL_AGORA_DEBUG", "true")
        
        config = EnvironmentConfig()
        
        assert config.google_api_key == "test_google_key"
        assert config.openai_api_key == "test_openai_key"
        assert config.anthropic_api_key is None
        assert config.log_level == "DEBUG"
        assert config.log_dir == "custom_logs"
        assert config.session_timeout == 7200
        assert config.debug_mode is True
        
    def test_default_values(self):
        """Test default values when environment variables are not set."""
        # Clear any existing env vars
        for key in ["GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", 
                   "VIRTUAL_AGORA_LOG_LEVEL", "VIRTUAL_AGORA_LOG_DIR"]:
            os.environ.pop(key, None)
            
        config = EnvironmentConfig()
        
        assert config.google_api_key is None
        assert config.log_level == "INFO"
        assert config.log_dir == "logs"
        assert config.report_dir == "reports"
        assert config.session_timeout == 3600
        assert config.debug_mode is False
        assert config.disable_color is False
        
    def test_log_level_validation(self, monkeypatch):
        """Test log level validation."""
        # Valid log level
        monkeypatch.setenv("VIRTUAL_AGORA_LOG_LEVEL", "WARNING")
        config = EnvironmentConfig()
        assert config.log_level == "WARNING"
        
        # Invalid log level
        monkeypatch.setenv("VIRTUAL_AGORA_LOG_LEVEL", "INVALID")
        with pytest.raises(ValidationError) as exc_info:
            EnvironmentConfig()
        assert "string does not match regex" in str(exc_info.value)
        
    def test_session_timeout_validation(self, monkeypatch):
        """Test session timeout validation."""
        # Valid timeout
        monkeypatch.setenv("VIRTUAL_AGORA_SESSION_TIMEOUT", "3600")
        config = EnvironmentConfig()
        assert config.session_timeout == 3600
        
        # Invalid - too small
        monkeypatch.setenv("VIRTUAL_AGORA_SESSION_TIMEOUT", "0")
        with pytest.raises(ValidationError) as exc_info:
            EnvironmentConfig()
        assert "greater than 0" in str(exc_info.value)
        
        # Invalid - too large
        monkeypatch.setenv("VIRTUAL_AGORA_SESSION_TIMEOUT", "100000")
        with pytest.raises(ValidationError) as exc_info:
            EnvironmentConfig()
        assert "less than or equal to 86400" in str(exc_info.value)
        
    def test_get_api_keys(self, monkeypatch):
        """Test getting all API keys as config objects."""
        monkeypatch.setenv("GOOGLE_API_KEY", "google_key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic_key")
        
        config = EnvironmentConfig()
        api_keys = config.get_api_keys()
        
        assert api_keys["Google"] is not None
        assert api_keys["Google"].key == "google_key"
        assert api_keys["Google"].provider == "Google"
        
        assert api_keys["OpenAI"] is None  # Not set
        
        assert api_keys["Anthropic"] is not None
        assert api_keys["Anthropic"].key == "anthropic_key"
        
    def test_get_api_key_for_provider(self, monkeypatch):
        """Test getting API key for specific provider."""
        monkeypatch.setenv("GOOGLE_API_KEY", "google_key")
        monkeypatch.setenv("OPENAI_API_KEY", "openai_key")
        
        config = EnvironmentConfig()
        
        assert config.get_api_key_for_provider("Google") == "google_key"
        assert config.get_api_key_for_provider("OpenAI") == "openai_key"
        assert config.get_api_key_for_provider("Anthropic") is None
        assert config.get_api_key_for_provider("Unknown") is None
        
    def test_validate_required_keys_success(self, monkeypatch):
        """Test successful validation of required keys."""
        monkeypatch.setenv("GOOGLE_API_KEY", "google_key")
        monkeypatch.setenv("OPENAI_API_KEY", "openai_key")
        
        config = EnvironmentConfig()
        
        # Should not raise
        config.validate_required_keys({"Google", "OpenAI"})
        
    def test_validate_required_keys_missing(self, monkeypatch):
        """Test validation failure when keys are missing."""
        monkeypatch.setenv("GOOGLE_API_KEY", "google_key")
        
        config = EnvironmentConfig()
        
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate_required_keys({"Google", "OpenAI", "Anthropic"})
            
        error = exc_info.value
        assert "OpenAI" in str(error)
        assert "Anthropic" in str(error)
        assert error.details["missing_providers"] == ["OpenAI", "Anthropic"]
        assert "OPENAI_API_KEY" in error.details["missing_variables"]
        assert "ANTHROPIC_API_KEY" in error.details["missing_variables"]
        
    def test_mask_sensitive_values(self, monkeypatch):
        """Test masking sensitive configuration values."""
        monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSyD-1234567890abcdefghijklmnopqrstuv")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-abcdefghijklmnop")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "short")
        
        config = EnvironmentConfig()
        masked = config.mask_sensitive_values()
        
        # API keys should be masked
        assert masked["google_api_key"] == "************************************stuv"
        assert masked["openai_api_key"] == "**************mnop"
        assert masked["anthropic_api_key"] == "****"  # Too short
        
        # Other values should not be masked
        assert masked["log_level"] == config.log_level
        assert masked["log_dir"] == config.log_dir
        
    def test_load_from_env_file(self, tmp_path, monkeypatch):
        """Test loading from .env file using Pydantic settings."""
        env_file = tmp_path / ".env"
        env_file.write_text("""
GOOGLE_API_KEY=file_google_key
OPENAI_API_KEY=file_openai_key
VIRTUAL_AGORA_LOG_LEVEL=ERROR
VIRTUAL_AGORA_DEBUG=1
""")
        
        # Change to temp directory so .env is found
        monkeypatch.chdir(tmp_path)
        
        config = EnvironmentConfig()
        
        assert config.google_api_key == "file_google_key"
        assert config.openai_api_key == "file_openai_key"
        assert config.log_level == "ERROR"
        assert config.debug_mode is True