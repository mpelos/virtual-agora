"""Tests for configuration loading and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from virtual_agora.config.models import Config, ModeratorConfig, AgentConfig, Provider
from virtual_agora.config.loader import ConfigLoader
from virtual_agora.config.validators import ConfigValidator
from virtual_agora.utils.exceptions import ConfigurationError


class TestConfigSchema:
    """Test configuration schema and Pydantic models."""

    def test_provider_enum_case_insensitive(self):
        """Test that Provider enum handles case variations."""
        assert Provider("Google") == Provider.GOOGLE
        assert Provider("google") == Provider.GOOGLE
        assert Provider("GOOGLE") == Provider.GOOGLE
        assert Provider("OpenAI") == Provider.OPENAI
        assert Provider("openai") == Provider.OPENAI

    def test_moderator_config_valid(self):
        """Test valid moderator configuration."""
        config = ModeratorConfig(provider="Google", model="gemini-2.5-pro")
        assert config.provider == Provider.GOOGLE
        assert config.model == "gemini-2.5-pro"

    def test_agent_config_valid(self):
        """Test valid agent configuration."""
        config = AgentConfig(provider="OpenAI", model="gpt-4o", count=2)
        assert config.provider == Provider.OPENAI
        assert config.model == "gpt-4o"
        assert config.count == 2

    def test_agent_config_default_count(self):
        """Test agent configuration with default count."""
        config = AgentConfig(provider="Anthropic", model="claude-3-opus-20240229")
        assert config.count == 1

    def test_agent_config_invalid_count(self):
        """Test agent configuration with invalid count."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(provider="OpenAI", model="gpt-4o", count=0)
        assert "greater than or equal to 1" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(provider="OpenAI", model="gpt-4o", count=15)
        assert "less than or equal to 10" in str(exc_info.value)

    def test_config_valid(self):
        """Test valid complete configuration."""
        config = Config(
            moderator=ModeratorConfig(provider="Google", model="gemini-2.5-pro"),
            agents=[
                AgentConfig(provider="OpenAI", model="gpt-4o", count=2),
                AgentConfig(provider="Anthropic", model="claude-3-opus-20240229"),
            ],
        )
        assert config.get_total_agent_count() == 3
        assert len(config.get_agent_names()) == 3

    def test_config_too_many_agents(self):
        """Test configuration with too many agents."""
        with pytest.raises(ValidationError) as exc_info:
            Config(
                moderator=ModeratorConfig(provider="Google", model="gemini-2.5-pro"),
                agents=[
                    AgentConfig(provider="OpenAI", model="gpt-4o", count=10),
                    AgentConfig(provider="OpenAI", model="gpt-4o", count=10),
                    AgentConfig(provider="OpenAI", model="gpt-4o", count=5),
                ],
            )
        assert "Too many agents" in str(exc_info.value)

    def test_config_too_few_agents(self):
        """Test configuration with too few agents."""
        with pytest.raises(ValidationError) as exc_info:
            Config(
                moderator=ModeratorConfig(provider="Google", model="gemini-2.5-pro"),
                agents=[
                    AgentConfig(provider="OpenAI", model="gpt-4o", count=1),
                ],
            )
        assert "At least 2 agents required" in str(exc_info.value)

    def test_config_agent_names(self):
        """Test agent name generation."""
        config = Config(
            moderator=ModeratorConfig(provider="Google", model="gemini-2.5-pro"),
            agents=[
                AgentConfig(provider="OpenAI", model="gpt-4o", count=2),
                AgentConfig(
                    provider="Anthropic", model="claude-3-opus-20240229", count=1
                ),
            ],
        )
        names = config.get_agent_names()
        assert names == ["gpt-4o-1", "gpt-4o-2", "claude-3-opus-20240229-1"]


class TestConfigLoader:
    """Test configuration loading from YAML files."""

    def test_load_valid_config(self, temp_config_file):
        """Test loading a valid configuration file."""
        loader = ConfigLoader(temp_config_file)
        config = loader.load()

        assert config.moderator.provider == Provider.GOOGLE
        assert config.moderator.model == "gemini-2.5-pro"
        assert len(config.agents) == 1
        assert config.agents[0].provider == Provider.OPENAI
        assert config.agents[0].count == 2

    def test_load_missing_file(self):
        """Test loading a non-existent configuration file."""
        loader = ConfigLoader("nonexistent.yml")

        with pytest.raises(ConfigurationError) as exc_info:
            loader.load()
        assert "Configuration file not found" in str(exc_info.value)

    def test_load_empty_file(self, tmp_path):
        """Test loading an empty configuration file."""
        empty_file = tmp_path / "empty.yml"
        empty_file.write_text("")

        loader = ConfigLoader(empty_file)
        with pytest.raises(ConfigurationError) as exc_info:
            loader.load()
        assert "Configuration file is empty" in str(exc_info.value)

    def test_load_invalid_yaml(self, tmp_path):
        """Test loading a file with invalid YAML syntax."""
        invalid_file = tmp_path / "invalid.yml"
        invalid_file.write_text("invalid: yaml: syntax:")

        loader = ConfigLoader(invalid_file)
        with pytest.raises(ConfigurationError) as exc_info:
            loader.load()
        assert "Invalid YAML syntax" in str(exc_info.value)

    def test_load_missing_required_fields(self, tmp_path):
        """Test loading a configuration with missing required fields."""
        incomplete_file = tmp_path / "incomplete.yml"
        incomplete_file.write_text(
            """
moderator:
  provider: Google
  # missing model
agents:
  - provider: OpenAI
    model: gpt-4o
"""
        )

        loader = ConfigLoader(incomplete_file)
        with pytest.raises(ConfigurationError) as exc_info:
            loader.load()
        assert "Configuration validation failed" in str(exc_info.value)
        assert "model" in str(exc_info.value)

    def test_reload_config(self, temp_config_file):
        """Test reloading configuration."""
        loader = ConfigLoader(temp_config_file)
        config1 = loader.load()
        config2 = loader.reload()

        # Should be different objects but same content
        assert config1 is not config2
        assert config1.moderator.model == config2.moderator.model

    def test_validate_api_keys(self, temp_config_file, mock_env_vars):
        """Test API key validation."""
        loader = ConfigLoader(temp_config_file)
        loader.load()

        key_status = loader.validate_api_keys()
        assert key_status["Google"] is True
        assert key_status["OpenAI"] is True

    def test_get_missing_api_keys(self, temp_config_file, monkeypatch):
        """Test getting missing API keys."""
        # Remove some environment variables
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        loader = ConfigLoader(temp_config_file)
        loader.load()

        missing = loader.get_missing_api_keys()
        assert "Google" in missing
        assert "OpenAI" not in missing


class TestConfigValidator:
    """Test additional configuration validation."""

    def test_validate_all_valid_config(self):
        """Test validation of a valid configuration."""
        config = Config(
            moderator=ModeratorConfig(provider="Google", model="gemini-2.5-pro"),
            agents=[
                AgentConfig(provider="OpenAI", model="gpt-4o", count=2),
                AgentConfig(provider="Anthropic", model="claude-3-opus-20240229"),
            ],
        )

        validator = ConfigValidator(config)
        # Should not raise any exceptions
        validator.validate_all()

    def test_validate_provider_diversity_warning(self, caplog):
        """Test provider diversity validation with warning."""
        config = Config(
            moderator=ModeratorConfig(provider="OpenAI", model="gpt-4o"),
            agents=[
                AgentConfig(provider="OpenAI", model="gpt-4o", count=3),
                AgentConfig(provider="OpenAI", model="gpt-3.5-turbo"),
            ],
        )

        validator = ConfigValidator(config)
        validator.validate_provider_diversity()

        # Check for warning in logs
        assert "Provider 'OpenAI' represents" in caplog.text
        assert "Consider adding more provider diversity" in caplog.text

    def test_validate_model_compatibility_warning(self, caplog):
        """Test model compatibility validation with warning."""
        config = Config(
            moderator=ModeratorConfig(
                provider="OpenAI", model="gpt-3.5-turbo"  # Less capable model
            ),
            agents=[
                AgentConfig(provider="OpenAI", model="gpt-4o", count=2),
            ],
        )

        validator = ConfigValidator(config)
        validator.validate_model_compatibility()

        # Check for warning in logs
        assert "may have limited capabilities" in caplog.text

    def test_validation_report(self):
        """Test generation of validation report."""
        config = Config(
            moderator=ModeratorConfig(provider="Google", model="gemini-2.5-pro"),
            agents=[
                AgentConfig(provider="OpenAI", model="gpt-4o", count=2),
                AgentConfig(provider="Anthropic", model="claude-3-opus-20240229"),
                AgentConfig(provider="Google", model="gemini-2.5-pro"),
            ],
        )

        validator = ConfigValidator(config)
        report = validator.get_validation_report()

        assert report["valid"] is True
        assert report["stats"]["total_agents"] == 4
        assert report["stats"]["providers"]["OpenAI"] == 2
        assert report["stats"]["providers"]["Anthropic"] == 1
        assert report["stats"]["providers"]["Google"] == 1
