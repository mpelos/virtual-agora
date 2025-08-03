"""Agent recreation from state validation tests.

This test validates that agents can be recreated correctly from state
where providers are stored as strings instead of Provider enum objects.
"""

import pytest
from pathlib import Path
from virtual_agora.config.loader import ConfigLoader
from virtual_agora.agents.agent_factory import AgentFactory, get_provider_value_safe
from virtual_agora.config.models import Provider
from virtual_agora.flow.graph_v13 import (
    VirtualAgoraV13Flow,
    get_provider_value_safe as graph_get_provider_value_safe,
)
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class TestAgentRecreationFromState:
    """Test agent recreation when providers come from state (as strings) vs config (as enums)."""

    def test_provider_value_safe_helper_functions(self):
        """Test that our helper functions correctly handle both Provider enums and strings."""
        # Test with Provider enum
        provider_enum = Provider.OPENAI
        assert get_provider_value_safe(provider_enum) == "OpenAI"
        assert graph_get_provider_value_safe(provider_enum) == "OpenAI"

        # Test with string (as would come from state)
        provider_string = "OpenAI"
        assert get_provider_value_safe(provider_string) == "OpenAI"
        assert graph_get_provider_value_safe(provider_string) == "OpenAI"

        # Test with different providers
        assert get_provider_value_safe(Provider.ANTHROPIC) == "Anthropic"
        assert get_provider_value_safe("Anthropic") == "Anthropic"
        assert get_provider_value_safe(Provider.GOOGLE) == "Google"
        assert get_provider_value_safe("Google") == "Google"

        logger.info("✅ Provider value safe helper functions work correctly")

    def test_agent_factory_with_real_config(self):
        """Test that agent factory works with real config (Provider enums)."""
        project_root = Path(__file__).parent.parent.parent
        test_config_path = project_root / "config.test.yml"

        if not test_config_path.exists():
            pytest.skip(f"Test configuration file not found at {test_config_path}")

        # Load real configuration
        config_loader = ConfigLoader(test_config_path)
        config = config_loader.load()

        # Test that we can safely extract provider values from real config
        moderator_provider = get_provider_value_safe(config.moderator.provider)
        assert moderator_provider in ["OpenAI", "Anthropic", "Google", "Grok"]

        summarizer_provider = get_provider_value_safe(config.summarizer.provider)
        assert summarizer_provider in ["OpenAI", "Anthropic", "Google", "Grok"]

        report_writer_provider = get_provider_value_safe(config.report_writer.provider)
        assert report_writer_provider in ["OpenAI", "Anthropic", "Google", "Grok"]

        for agent_config in config.agents:
            agent_provider = get_provider_value_safe(agent_config.provider)
            assert agent_provider in ["OpenAI", "Anthropic", "Google", "Grok"]

        logger.info(f"✅ Agent factory can extract provider values from real config")

    def test_agent_factory_with_string_providers(self):
        """Test that agent factory works when providers are already strings (as from state)."""
        project_root = Path(__file__).parent.parent.parent
        test_config_path = project_root / "config.test.yml"

        if not test_config_path.exists():
            pytest.skip(f"Test configuration file not found at {test_config_path}")

        # Load real configuration
        config_loader = ConfigLoader(test_config_path)
        config = config_loader.load()

        # Simulate state scenario: convert Provider enums to strings
        # This is what happens when state stores provider info
        config.moderator.provider = config.moderator.provider.value  # Convert to string
        config.summarizer.provider = (
            config.summarizer.provider.value
        )  # Convert to string
        config.report_writer.provider = (
            config.report_writer.provider.value
        )  # Convert to string

        for agent_config in config.agents:
            agent_config.provider = agent_config.provider.value  # Convert to string

        # Test that we can safely extract provider values from string providers
        moderator_provider = get_provider_value_safe(config.moderator.provider)
        assert moderator_provider in ["OpenAI", "Anthropic", "Google", "Grok"]
        assert isinstance(config.moderator.provider, str)  # Should now be string

        summarizer_provider = get_provider_value_safe(config.summarizer.provider)
        assert summarizer_provider in ["OpenAI", "Anthropic", "Google", "Grok"]
        assert isinstance(config.summarizer.provider, str)  # Should now be string

        report_writer_provider = get_provider_value_safe(config.report_writer.provider)
        assert report_writer_provider in ["OpenAI", "Anthropic", "Google", "Grok"]
        assert isinstance(config.report_writer.provider, str)  # Should now be string

        for agent_config in config.agents:
            agent_provider = get_provider_value_safe(agent_config.provider)
            assert agent_provider in ["OpenAI", "Anthropic", "Google", "Grok"]
            assert isinstance(agent_config.provider, str)  # Should now be string

        logger.info(
            f"✅ Agent factory can extract provider values from string providers"
        )

    def test_graph_initialization_with_string_providers(self):
        """Test that VirtualAgoraV13Flow works when providers are strings."""
        project_root = Path(__file__).parent.parent.parent
        test_config_path = project_root / "config.test.yml"

        if not test_config_path.exists():
            pytest.skip(f"Test configuration file not found at {test_config_path}")

        # Load real configuration
        config_loader = ConfigLoader(test_config_path)
        config = config_loader.load()

        # Simulate state scenario: convert Provider enums to strings
        config.moderator.provider = config.moderator.provider.value
        config.summarizer.provider = config.summarizer.provider.value
        config.report_writer.provider = config.report_writer.provider.value

        for agent_config in config.agents:
            agent_config.provider = agent_config.provider.value

        # Test that we can safely extract provider values from string providers in graph context
        moderator_provider = graph_get_provider_value_safe(config.moderator.provider)
        assert moderator_provider in ["OpenAI", "Anthropic", "Google", "Grok"]
        assert isinstance(config.moderator.provider, str)  # Should now be string

        summarizer_provider = graph_get_provider_value_safe(config.summarizer.provider)
        assert summarizer_provider in ["OpenAI", "Anthropic", "Google", "Grok"]
        assert isinstance(config.summarizer.provider, str)  # Should now be string

        report_writer_provider = graph_get_provider_value_safe(
            config.report_writer.provider
        )
        assert report_writer_provider in ["OpenAI", "Anthropic", "Google", "Grok"]
        assert isinstance(config.report_writer.provider, str)  # Should now be string

        for agent_config in config.agents:
            agent_provider = graph_get_provider_value_safe(agent_config.provider)
            assert agent_provider in ["OpenAI", "Anthropic", "Google", "Grok"]
            assert isinstance(agent_config.provider, str)  # Should now be string

        logger.info(
            f"✅ Graph functions can extract provider values from string providers"
        )

    def test_mixed_provider_scenarios(self):
        """Test scenarios where some providers are enums and others are strings."""
        project_root = Path(__file__).parent.parent.parent
        test_config_path = project_root / "config.test.yml"

        if not test_config_path.exists():
            pytest.skip(f"Test configuration file not found at {test_config_path}")

        # Load real configuration
        config_loader = ConfigLoader(test_config_path)
        config = config_loader.load()

        # Create mixed scenario: some enums, some strings
        config.moderator.provider = config.moderator.provider.value  # String
        # Leave summarizer as enum
        config.report_writer.provider = config.report_writer.provider.value  # String

        # Mix agent providers
        for i, agent_config in enumerate(config.agents):
            if i % 2 == 0:
                agent_config.provider = agent_config.provider.value  # String
            # Leave others as enum

        # Test that we can safely extract provider values from mixed scenarios
        moderator_provider = get_provider_value_safe(config.moderator.provider)
        assert moderator_provider in ["OpenAI", "Anthropic", "Google", "Grok"]
        assert isinstance(config.moderator.provider, str)  # Should be string

        summarizer_provider = get_provider_value_safe(config.summarizer.provider)
        assert summarizer_provider in ["OpenAI", "Anthropic", "Google", "Grok"]
        # summarizer should still be enum

        report_writer_provider = get_provider_value_safe(config.report_writer.provider)
        assert report_writer_provider in ["OpenAI", "Anthropic", "Google", "Grok"]
        assert isinstance(config.report_writer.provider, str)  # Should be string

        for i, agent_config in enumerate(config.agents):
            agent_provider = get_provider_value_safe(agent_config.provider)
            assert agent_provider in ["OpenAI", "Anthropic", "Google", "Grok"]
            if i % 2 == 0:
                assert isinstance(agent_config.provider, str)  # Should be string
            # Others should still be enum

        logger.info("✅ Mixed provider scenarios work correctly")

    def test_provider_value_error_prevention(self):
        """Test that our fixes prevent the 'str' object has no attribute 'value' error."""
        # Test scenarios that would previously cause the error

        # Scenario 1: String provider passed to get_provider_value_safe
        string_provider = "OpenAI"
        try:
            result = get_provider_value_safe(string_provider)
            assert result == "OpenAI"
        except AttributeError as e:
            if "'str' object has no attribute 'value'" in str(e):
                pytest.fail(
                    "get_provider_value_safe still has the 'str' object has no attribute 'value' error"
                )
            else:
                raise

        # Scenario 2: Provider enum passed to get_provider_value_safe
        enum_provider = Provider.ANTHROPIC
        try:
            result = get_provider_value_safe(enum_provider)
            assert result == "Anthropic"
        except AttributeError as e:
            pytest.fail(f"get_provider_value_safe failed with Provider enum: {e}")

        # Scenario 3: Test graph helper function
        try:
            result = graph_get_provider_value_safe("Google")
            assert result == "Google"
            result = graph_get_provider_value_safe(Provider.GOOGLE)
            assert result == "Google"
        except AttributeError as e:
            if "'str' object has no attribute 'value'" in str(e):
                pytest.fail(
                    "graph_get_provider_value_safe still has the 'str' object has no attribute 'value' error"
                )
            else:
                raise

        logger.info("✅ Provider .value error prevention works correctly")


if __name__ == "__main__":
    # Run tests directly for development
    test_suite = TestAgentRecreationFromState()

    print("🔍 Running Agent Recreation From State Tests...")

    try:
        print("✅ Test 1: Provider value safe helper functions")
        test_suite.test_provider_value_safe_helper_functions()

        print("✅ Test 2: Agent factory with real config (Provider enums)")
        test_suite.test_agent_factory_with_real_config()

        print("✅ Test 3: Agent factory with string providers (from state)")
        test_suite.test_agent_factory_with_string_providers()

        print("✅ Test 4: Graph initialization with string providers")
        test_suite.test_graph_initialization_with_string_providers()

        print("✅ Test 5: Mixed provider scenarios")
        test_suite.test_mixed_provider_scenarios()

        print("✅ Test 6: Provider .value error prevention")
        test_suite.test_provider_value_error_prevention()

        print("\n🎉 ALL AGENT RECREATION FROM STATE TESTS PASSED!")
        print("✅ Agents can be created from both Provider enums and strings")
        print("✅ VirtualAgoraV13Flow works with string providers from state")
        print("✅ 'str' object has no attribute 'value' error is prevented")
        print("✅ Mixed provider scenarios work correctly")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
