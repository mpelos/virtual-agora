"""Tests for security utilities."""

import os
import tempfile
from pathlib import Path

import pytest

from virtual_agora.utils.security import (
    mask_api_key,
    mask_dict_values,
    secure_compare,
    validate_api_key_format,
    sanitize_for_logging,
    generate_session_id,
    hash_api_key,
    create_provider_help_message,
    check_env_file_security,
)


class TestSecurityUtilities:
    """Test security utility functions."""

    def test_mask_api_key(self):
        """Test API key masking."""
        # Test None and empty cases
        assert mask_api_key(None) == "<not set>"
        assert mask_api_key("") == "<empty>"
        assert mask_api_key("   ") == "<empty>"

        # Test short keys
        assert mask_api_key("1234") == "****"
        assert mask_api_key("12345678") == "********"

        # Test normal keys
        assert mask_api_key("1234567890abcdef") == "12**********cdef"
        assert (
            mask_api_key("sk-1234567890abcdefghijklmnop")
            == "sk***********************mnop"
        )

        # Test custom show_chars
        assert mask_api_key("1234567890", show_chars=2) == "12******90"
        assert mask_api_key("1234567890", show_chars=0) == "**********"

    def test_mask_dict_values(self):
        """Test masking dictionary values."""
        data = {
            "api_key": "sk-1234567890abcdef",
            "secret": "my_secret_value",
            "normal": "normal_value",
            "empty": "",
            "none": None,
        }

        masked = mask_dict_values(
            data, ["api_key", "secret", "missing", "empty", "none"]
        )

        assert masked["api_key"] == "sk*************cdef"
        assert masked["secret"] == "my*********alue"
        assert masked["normal"] == "normal_value"  # Not masked
        assert masked["empty"] == "<empty>"
        assert masked["none"] == "<not set>"

        # Original dict should not be modified
        assert data["api_key"] == "sk-1234567890abcdef"

    def test_secure_compare(self):
        """Test secure string comparison."""
        assert secure_compare("hello", "hello") is True
        assert secure_compare("hello", "world") is False
        assert secure_compare("", "") is True
        assert secure_compare("a", "A") is False

        # Should work with Unicode
        assert secure_compare("héllo", "héllo") is True
        assert secure_compare("😊", "😊") is True

    def test_validate_api_key_format(self):
        """Test API key format validation."""
        # Google keys
        valid, msg = validate_api_key_format(
            "AIzaSyD-1234567890abcdefghijklmnopqrstuvw", "Google"
        )
        assert valid is True
        assert msg is None

        valid, msg = validate_api_key_format("invalid_google_key", "Google")
        assert valid is False
        assert "AIza" in msg

        # OpenAI keys
        valid, msg = validate_api_key_format("sk-" + "a" * 48, "OpenAI")
        assert valid is True

        valid, msg = validate_api_key_format("invalid_openai_key", "OpenAI")
        assert valid is False
        assert "sk-" in msg

        # Anthropic keys
        valid, msg = validate_api_key_format("sk-ant-1234567890abcdef", "Anthropic")
        assert valid is True

        # Unknown provider
        valid, msg = validate_api_key_format("any_key", "Unknown")
        assert valid is True  # Can't validate unknown providers

        # Empty key
        valid, msg = validate_api_key_format("", "Google")
        assert valid is False
        assert "empty" in msg

    def test_sanitize_for_logging(self):
        """Test log sanitization."""
        # Test with various API keys
        text = "My Google key is AIzaSyD-1234567890abcdefghijklmnopqrstuv and OpenAI key is sk-abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJK"
        sanitized = sanitize_for_logging(text)

        assert "AIzaSyD-1234567890abcdefghijklmnopqrstuv" not in sanitized
        assert "sk-abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJK" not in sanitized
        assert "****" in sanitized

        # Test with custom patterns
        text = "Password: secret123"
        sanitized = sanitize_for_logging(text, [r"secret\d+"])
        assert "secret123" not in sanitized

        # Test with base64 and hex strings
        text = "Token: YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY3ODkw and hash: abcdef1234567890abcdef1234567890"
        sanitized = sanitize_for_logging(text)
        assert "YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY3ODkw" not in sanitized
        assert "abcdef1234567890abcdef1234567890" not in sanitized

    def test_generate_session_id(self):
        """Test session ID generation."""
        # Generate multiple IDs
        ids = [generate_session_id() for _ in range(10)]

        # All should be unique
        assert len(set(ids)) == 10

        # All should be 32 characters (16 bytes as hex)
        for session_id in ids:
            assert len(session_id) == 32
            assert all(c in "0123456789abcdef" for c in session_id)

    def test_hash_api_key(self):
        """Test API key hashing."""
        key1 = "sk-1234567890abcdef"
        key2 = "sk-1234567890abcdef"
        key3 = "sk-different-key"

        hash1 = hash_api_key(key1)
        hash2 = hash_api_key(key2)
        hash3 = hash_api_key(key3)

        # Same key should produce same hash
        assert hash1 == hash2

        # Different keys should produce different hashes
        assert hash1 != hash3

        # Hash should be 64 characters (SHA-256 as hex)
        assert len(hash1) == 64

    def test_create_provider_help_message(self):
        """Test provider help message generation."""
        # Known providers
        google_help = create_provider_help_message("Google")
        assert "makersuite.google.com" in google_help
        assert "GOOGLE_API_KEY" in google_help

        openai_help = create_provider_help_message("OpenAI")
        assert "platform.openai.com" in openai_help
        assert "OPENAI_API_KEY" in openai_help

        anthropic_help = create_provider_help_message("Anthropic")
        assert "console.anthropic.com" in anthropic_help
        assert "ANTHROPIC_API_KEY" in anthropic_help

        grok_help = create_provider_help_message("Grok")
        assert "limited" in grok_help
        assert "GROK_API_KEY" in grok_help

        # Unknown provider
        unknown_help = create_provider_help_message("Unknown")
        assert "obtain an API key" in unknown_help

    def test_check_env_file_security(self, tmp_path):
        """Test environment file security checking."""
        env_file = tmp_path / ".env"
        env_file.write_text("API_KEY=secret")

        # Try to test on Unix-like systems
        try:
            import stat

            # Make file world-readable
            os.chmod(
                env_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
            )

            warnings = check_env_file_security(str(env_file))
            assert any("world-readable" in w for w in warnings)

            # Make file world-writable
            os.chmod(
                env_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWOTH
            )

            warnings = check_env_file_security(str(env_file))
            assert any("world-writable" in w for w in warnings)

            # Make file secure
            os.chmod(env_file, stat.S_IRUSR | stat.S_IWUSR)

            warnings = check_env_file_security(str(env_file))
            assert len(warnings) == 0

        except (ImportError, AttributeError):
            # Can't test on this platform
            warnings = check_env_file_security(str(env_file))
            assert isinstance(warnings, list)
