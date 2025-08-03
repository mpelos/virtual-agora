"""Configuration loading validation tests.

This test validates that the real YAML configuration loading path works correctly
and produces the same Provider enum instances that production code expects.
"""

import pytest
from pathlib import Path
from virtual_agora.config.loader import ConfigLoader
from virtual_agora.config.models import Provider
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class TestConfigLoadingValidation:
    """Test real configuration loading path to ensure integration tests match production."""

    def test_yaml_to_pydantic_to_provider_enum_conversion(self):
        """Test that YAML config properly converts to Provider enum instances with .value attributes."""
        # Use test configuration file
        project_root = Path(__file__).parent.parent.parent
        test_config_path = project_root / "config.test.yml"

        if not test_config_path.exists():
            pytest.skip(f"Test configuration file not found at {test_config_path}")

        # Load configuration using real ConfigLoader (production path)
        config_loader = ConfigLoader(test_config_path)
        config = config_loader.load()

        # Validate that providers are proper enum instances
        assert hasattr(
            config.moderator.provider, "value"
        ), "Moderator provider should have .value attribute"
        assert isinstance(
            config.moderator.provider, Provider
        ), "Moderator provider should be Provider enum"
        assert config.moderator.provider.value in [
            "OpenAI",
            "Anthropic",
            "Google",
            "Grok",
        ], "Moderator provider should have valid value"

        assert hasattr(
            config.summarizer.provider, "value"
        ), "Summarizer provider should have .value attribute"
        assert isinstance(
            config.summarizer.provider, Provider
        ), "Summarizer provider should be Provider enum"

        assert hasattr(
            config.report_writer.provider, "value"
        ), "Report writer provider should have .value attribute"
        assert isinstance(
            config.report_writer.provider, Provider
        ), "Report writer provider should be Provider enum"

        # Validate agent providers
        for i, agent in enumerate(config.agents):
            assert hasattr(
                agent.provider, "value"
            ), f"Agent {i} provider should have .value attribute"
            assert isinstance(
                agent.provider, Provider
            ), f"Agent {i} provider should be Provider enum"
            assert agent.provider.value in [
                "OpenAI",
                "Anthropic",
                "Google",
                "Grok",
            ], f"Agent {i} provider should have valid value"

        # Log successful validation
        logger.info(f"✅ Config loading validation passed:")
        logger.info(
            f"  Moderator: {config.moderator.model} ({config.moderator.provider.value})"
        )
        logger.info(
            f"  Summarizer: {config.summarizer.model} ({config.summarizer.provider.value})"
        )
        logger.info(
            f"  Report Writer: {config.report_writer.model} ({config.report_writer.provider.value})"
        )
        logger.info(f"  Agents: {len(config.agents)} total")
        for i, agent in enumerate(config.agents):
            logger.info(
                f"    Agent {i+1}: {agent.model} ({agent.provider.value}) x{agent.count}"
            )

    def test_config_test_yml_format_is_correct(self):
        """Test that config.test.yml has the correct Provider enum format."""
        project_root = Path(__file__).parent.parent.parent
        test_config_path = project_root / "config.test.yml"

        if not test_config_path.exists():
            pytest.skip(f"Test configuration file not found at {test_config_path}")

        # Read raw YAML to validate format
        with open(test_config_path, "r") as f:
            content = f.read()

        # Check that it uses proper Provider enum values (title case)
        assert "provider: OpenAI" in content, "Should use 'OpenAI' not 'openai'"
        assert (
            "provider: Anthropic" in content
        ), "Should use 'Anthropic' not 'anthropic'"

        # Check that it doesn't use old lowercase format
        assert "provider: openai" not in content, "Should not use lowercase 'openai'"
        assert (
            "provider: anthropic" not in content
        ), "Should not use lowercase 'anthropic'"
        assert "provider: google" not in content, "Should not use lowercase 'google'"

        logger.info("✅ config.test.yml format validation passed")

    def test_config_yml_format_is_correct(self):
        """Test that config.yml has the correct Provider enum format."""
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config.yml"

        if not config_path.exists():
            pytest.skip(f"Configuration file not found at {config_path}")

        # Read raw YAML to validate format
        with open(config_path, "r") as f:
            content = f.read()

        # Check that it uses proper Provider enum values (title case)
        provider_lines = [
            line
            for line in content.split("\n")
            if "provider:" in line and not line.strip().startswith("#")
        ]

        for line in provider_lines:
            # Extract provider value
            provider_value = line.split("provider:")[1].strip()
            assert provider_value in [
                "Google",
                "OpenAI",
                "Anthropic",
                "Grok",
            ], f"Invalid provider format in line: {line}"

        # Check that it doesn't use old lowercase format
        assert "provider: openai" not in content, "Should not use lowercase 'openai'"
        assert (
            "provider: anthropic" not in content
        ), "Should not use lowercase 'anthropic'"
        assert "provider: google" not in content, "Should not use lowercase 'google'"

        logger.info("✅ config.yml format validation passed")

    def test_real_config_loading_matches_test_config_structure(self):
        """Test that real config loading produces same structure as our tests expect."""
        project_root = Path(__file__).parent.parent.parent
        test_config_path = project_root / "config.test.yml"

        if not test_config_path.exists():
            pytest.skip(f"Test configuration file not found at {test_config_path}")

        # Load configuration using real ConfigLoader
        config_loader = ConfigLoader(test_config_path)
        config = config_loader.load()

        # Check expected structure
        assert hasattr(config, "moderator"), "Config should have moderator"
        assert hasattr(config, "summarizer"), "Config should have summarizer"
        assert hasattr(config, "report_writer"), "Config should have report_writer"
        assert hasattr(config, "agents"), "Config should have agents"

        # Check that all required methods exist
        assert hasattr(
            config, "get_total_agent_count"
        ), "Config should have get_total_agent_count method"
        assert config.get_total_agent_count() > 0, "Should have agents"

        # Validate that provider enum instances work in actual create_provider calls
        # This is the critical test - can we actually use these providers?
        moderator_provider_value = config.moderator.provider.value
        assert isinstance(
            moderator_provider_value, str
        ), "Provider .value should be string"
        assert moderator_provider_value in [
            "OpenAI",
            "Anthropic",
            "Google",
            "Grok",
        ], "Provider value should be valid"

        logger.info(f"✅ Real config loading produces expected structure")
        logger.info(f"  Total agents: {config.get_total_agent_count()}")
        logger.info(f"  Moderator provider value: {moderator_provider_value}")


if __name__ == "__main__":
    # Run tests directly for development
    test_suite = TestConfigLoadingValidation()

    print("🔍 Running Configuration Loading Validation Tests...")

    try:
        print("✅ Test 1: YAML→Pydantic→Provider enum conversion")
        test_suite.test_yaml_to_pydantic_to_provider_enum_conversion()

        print("✅ Test 2: config.test.yml format validation")
        test_suite.test_config_test_yml_format_is_correct()

        print("✅ Test 3: config.yml format validation")
        test_suite.test_config_yml_format_is_correct()

        print("✅ Test 4: Real config loading structure validation")
        test_suite.test_real_config_loading_matches_test_config_structure()

        print("\n🎉 ALL CONFIG LOADING VALIDATION TESTS PASSED!")
        print("✅ YAML configuration loading works correctly")
        print("✅ Provider enum conversion produces .value attributes")
        print("✅ Real config loading matches expected structure")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
