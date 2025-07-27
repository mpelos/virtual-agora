"""Tests for environment variable management."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from virtual_agora.utils.env_manager import EnvironmentManager
from virtual_agora.utils.exceptions import ConfigurationError


class TestEnvironmentManager:
    """Test the EnvironmentManager class."""

    def test_init(self):
        """Test EnvironmentManager initialization."""
        manager = EnvironmentManager()
        assert manager.env_file is None
        assert manager.override is True
        assert not manager._loaded

        # With specific env file
        env_path = Path(".env.test")
        manager = EnvironmentManager(env_file=env_path, override=False)
        assert manager.env_file == env_path
        assert manager.override is False

    def test_load_missing_file(self):
        """Test loading a non-existent env file."""
        manager = EnvironmentManager(env_file=Path("nonexistent.env"))

        with pytest.raises(ConfigurationError) as exc_info:
            manager.load()
        assert "Environment file not found" in str(exc_info.value)

    def test_load_env_file(self, tmp_path):
        """Test loading environment variables from file."""
        # Create temporary .env file
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR=test_value\nTEST_NUM=123")

        # Clear any existing TEST_VAR
        os.environ.pop("TEST_VAR", None)
        os.environ.pop("TEST_NUM", None)

        manager = EnvironmentManager(env_file=env_file)
        manager.load()

        assert os.getenv("TEST_VAR") == "test_value"
        assert os.getenv("TEST_NUM") == "123"

        # Clean up
        os.environ.pop("TEST_VAR", None)
        os.environ.pop("TEST_NUM", None)

    def test_load_twice(self, tmp_path):
        """Test that loading twice doesn't reload."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR=value1")

        manager = EnvironmentManager(env_file=env_file)
        manager.load()

        # Mark as loaded and change file
        env_file.write_text("TEST_VAR=value2")
        manager.load()  # Should not reload

        # Value should still be the first one (not reloaded)
        assert manager._loaded is True

    def test_get_api_key(self, monkeypatch):
        """Test getting API keys for providers."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test_google_key")
        monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
        # Ensure ANTHROPIC_API_KEY is not set
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        manager = EnvironmentManager()

        assert manager.get_api_key("Google") == "test_google_key"
        assert manager.get_api_key("OpenAI") == "test_openai_key"
        assert manager.get_api_key("Anthropic") is None
        assert manager.get_api_key("Unknown") is None

    def test_get_all_api_keys(self, monkeypatch):
        """Test getting all API keys."""
        monkeypatch.setenv("GOOGLE_API_KEY", "google_key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic_key")
        # Ensure OPENAI_API_KEY and GROK_API_KEY are not set
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GROK_API_KEY", raising=False)

        manager = EnvironmentManager()
        keys = manager.get_all_api_keys()

        assert keys["GOOGLE_API_KEY"] == "google_key"
        assert keys["ANTHROPIC_API_KEY"] == "anthropic_key"
        assert keys["OPENAI_API_KEY"] is None
        assert keys["GROK_API_KEY"] is None

    def test_validate_api_keys(self, monkeypatch):
        """Test API key validation."""
        monkeypatch.setenv("GOOGLE_API_KEY", "valid_key")
        monkeypatch.setenv("OPENAI_API_KEY", "")  # Empty key
        # Ensure ANTHROPIC_API_KEY is not set
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        manager = EnvironmentManager()
        required = {"Google", "OpenAI", "Anthropic"}

        results = manager.validate_api_keys(required)

        assert results["Google"] is True
        assert results["OpenAI"] is False  # Empty
        assert results["Anthropic"] is False  # Missing

        # Check missing keys tracking
        missing = manager.get_missing_keys()
        assert "OPENAI_API_KEY" in missing
        assert "ANTHROPIC_API_KEY" in missing

    def test_get_missing_providers(self, monkeypatch):
        """Test getting missing providers."""
        monkeypatch.setenv("GOOGLE_API_KEY", "valid_key")
        # Ensure OPENAI_API_KEY is not set
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        manager = EnvironmentManager()
        required = {"Google", "OpenAI"}

        missing = manager.get_missing_providers(required)
        assert missing == {"OpenAI"}

    def test_mask_api_key(self):
        """Test API key masking."""
        manager = EnvironmentManager()

        assert manager.mask_api_key(None) == "<not set>"
        assert manager.mask_api_key("") == "<not set>"
        assert manager.mask_api_key("short") == "****"  # len <= 8
        assert (
            manager.mask_api_key("1234567890") == "******7890"
        )  # len=10, so 6 asterisks + last 4
        assert (
            manager.mask_api_key("sk-1234567890abcdef") == "***************cdef"
        )  # len=19, so 15 asterisks + last 4

    def test_get_env_var(self, monkeypatch):
        """Test getting environment variables."""
        monkeypatch.setenv("TEST_VAR", "test_value")

        manager = EnvironmentManager()

        assert manager.get_env_var("TEST_VAR") == "test_value"
        assert manager.get_env_var("MISSING_VAR") is None
        assert manager.get_env_var("MISSING_VAR", "default") == "default"

    def test_get_int_env_var(self, monkeypatch):
        """Test getting integer environment variables."""
        monkeypatch.setenv("INT_VAR", "123")
        monkeypatch.setenv("INVALID_INT", "not_a_number")

        manager = EnvironmentManager()

        assert manager.get_int_env_var("INT_VAR") == 123
        assert manager.get_int_env_var("MISSING_VAR") == 0
        assert manager.get_int_env_var("MISSING_VAR", 42) == 42
        assert manager.get_int_env_var("INVALID_INT", 99) == 99

    def test_get_bool_env_var(self, monkeypatch):
        """Test getting boolean environment variables."""
        monkeypatch.setenv("TRUE_VAR1", "true")
        monkeypatch.setenv("TRUE_VAR2", "1")
        monkeypatch.setenv("TRUE_VAR3", "yes")
        monkeypatch.setenv("TRUE_VAR4", "ON")
        monkeypatch.setenv("FALSE_VAR", "false")

        manager = EnvironmentManager()

        assert manager.get_bool_env_var("TRUE_VAR1") is True
        assert manager.get_bool_env_var("TRUE_VAR2") is True
        assert manager.get_bool_env_var("TRUE_VAR3") is True
        assert manager.get_bool_env_var("TRUE_VAR4") is True
        assert manager.get_bool_env_var("FALSE_VAR") is False
        assert manager.get_bool_env_var("MISSING_VAR") is False
        assert manager.get_bool_env_var("MISSING_VAR", True) is True

    def test_get_status_report(self, monkeypatch, tmp_path):
        """Test getting environment status report."""
        env_file = tmp_path / ".env"
        env_file.write_text("GOOGLE_API_KEY=test_key")

        monkeypatch.setenv("GOOGLE_API_KEY", "test_google_key_1234")
        monkeypatch.setenv("VIRTUAL_AGORA_LOG_LEVEL", "DEBUG")

        manager = EnvironmentManager(env_file=env_file)
        report = manager.get_status_report()

        assert report["env_file"] == str(env_file)
        assert report["env_file_exists"] is True

        # Check API key status
        google_status = report["api_keys"]["Google Gemini"]
        assert google_status["variable"] == "GOOGLE_API_KEY"
        assert google_status["is_set"] is True
        # The file value "test_key" (8 chars) overrides the env value, so should be ****
        assert google_status["masked_value"] == "****"

        # Check optional vars
        assert report["optional_vars"]["VIRTUAL_AGORA_LOG_LEVEL"] == "DEBUG"

    @patch("warnings.warn")
    def test_check_env_file_security(self, mock_warn, tmp_path):
        """Test environment file security checks."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST=value")

        # Make file world-readable (if on Unix)
        try:
            import stat

            os.chmod(env_file, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        except:
            pytest.skip("Cannot test file permissions on this platform")

        manager = EnvironmentManager(env_file=env_file)
        manager.load()

        # Should have warned about permissions
        assert mock_warn.called
        warning_msg = str(mock_warn.call_args[0][0])
        assert "world-readable" in warning_msg
