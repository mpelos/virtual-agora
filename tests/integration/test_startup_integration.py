"""Integration tests for application startup and configuration flows.

This module tests the complete startup sequence from CLI arguments through
to running discussion, including configuration validation, environment setup,
and error handling scenarios.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock, AsyncMock
from datetime import datetime

from virtual_agora.main import run_application, parse_arguments
from virtual_agora.config.loader import ConfigLoader
from virtual_agora.utils.env_manager import EnvironmentManager
from virtual_agora.utils.exceptions import ConfigurationError, VirtualAgoraError

from ..helpers.fake_llm import create_fake_llm_pool
from ..helpers.integration_utils import (
    IntegrationTestHelper,
    patch_ui_components,
    create_test_config_file,
    create_test_env_file,
)


class TestStartupIntegration:
    """Test complete application startup scenarios."""

    def setup_method(self):
        """Set up test method with temporary files."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yml"
        self.env_path = Path(self.temp_dir) / "test.env"

    def teardown_method(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_startup_flow_success(self):
        """Test successful complete startup from CLI to discussion initialization."""
        # Create valid config and env files
        create_test_config_file(self.config_path, num_agents=2)
        create_test_env_file(self.env_path)

        # Mock command line arguments
        mock_args = Mock()
        mock_args.config = self.config_path
        mock_args.env = self.env_path
        mock_args.log_level = "INFO"
        mock_args.no_color = False
        mock_args.dry_run = True  # Use dry run to avoid full execution

        with patch_ui_components():
            # This should complete without errors
            result = await run_application(mock_args)
            assert result == 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_missing_config_file_error(self):
        """Test error handling when config file is missing."""
        mock_args = Mock()
        mock_args.config = Path("nonexistent_config.yml")
        mock_args.env = self.env_path
        mock_args.log_level = "INFO"
        mock_args.no_color = False
        mock_args.dry_run = False

        create_test_env_file(self.env_path)

        with patch_ui_components():
            result = await run_application(mock_args)
            assert result == 1  # Configuration error exit code

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_missing_api_keys_error(self):
        """Test error handling when required API keys are missing."""
        # Create config requiring OpenAI but don't provide API key
        create_test_config_file(self.config_path, provider="openai", num_agents=2)

        # Create env file without required keys
        with open(self.env_path, "w") as f:
            f.write("# No API keys provided\n")

        mock_args = Mock()
        mock_args.config = self.config_path
        mock_args.env = self.env_path
        mock_args.log_level = "INFO"
        mock_args.no_color = False
        mock_args.dry_run = False

        with patch_ui_components():
            result = await run_application(mock_args)
            assert result == 1  # Missing API keys error

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_invalid_config_format_error(self):
        """Test error handling when config file has invalid YAML format."""
        # Create invalid YAML file
        with open(self.config_path, "w") as f:
            f.write(
                """
moderator:
  provider: openai
  model: gpt-4
agents:
  - provider: openai
    model: gpt-3.5-turbo
    count: 2
  - invalid_yaml: [unclosed bracket
"""
            )

        create_test_env_file(self.env_path)

        mock_args = Mock()
        mock_args.config = self.config_path
        mock_args.env = self.env_path
        mock_args.log_level = "INFO"
        mock_args.no_color = False
        mock_args.dry_run = False

        with patch_ui_components():
            result = await run_application(mock_args)
            assert result == 1  # Configuration error

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_invalid_config_validation_error(self):
        """Test error handling when config has invalid values."""
        # Create config with invalid values
        with open(self.config_path, "w") as f:
            f.write(
                """
moderator:
  provider: invalid_provider
  model: gpt-4
agents:
  - provider: openai
    model: gpt-3.5-turbo
    count: -1  # Invalid count
"""
            )

        create_test_env_file(self.env_path)

        mock_args = Mock()
        mock_args.config = self.config_path
        mock_args.env = self.env_path
        mock_args.log_level = "INFO"
        mock_args.no_color = False
        mock_args.dry_run = False

        with patch_ui_components():
            result = await run_application(mock_args)
            assert result == 1  # Configuration validation error

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_state_manager_initialization_success(self):
        """Test that StateManager is properly initialized and accessible."""
        create_test_config_file(self.config_path, num_agents=2)
        create_test_env_file(self.env_path)

        mock_args = Mock()
        mock_args.config = self.config_path
        mock_args.env = self.env_path
        mock_args.log_level = "INFO"
        mock_args.no_color = False
        mock_args.dry_run = True

        # Track state manager creation
        with patch_ui_components():
            result = await run_application(mock_args)
            assert result == 0
            # This test should pass if the StateManager.state property works correctly

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_flow_compilation_success(self):
        """Test that VirtualAgoraFlow compiles successfully during startup."""
        create_test_config_file(self.config_path, num_agents=3)
        create_test_env_file(self.env_path)

        mock_args = Mock()
        mock_args.config = self.config_path
        mock_args.env = self.env_path
        mock_args.log_level = "DEBUG"  # More detailed logging
        mock_args.no_color = False
        mock_args.dry_run = True

        with patch_ui_components():
            result = await run_application(mock_args)
            assert result == 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_environment_validation_with_mixed_providers(self):
        """Test environment validation with multiple different providers."""
        # Create config using multiple providers
        with open(self.config_path, "w") as f:
            f.write(
                """
moderator:
  provider: openai
  model: gpt-4
agents:
  - provider: anthropic
    model: claude-3-haiku
    count: 1
  - provider: openai
    model: gpt-3.5-turbo
    count: 2
"""
            )

        # Create env with all required keys
        with open(self.env_path, "w") as f:
            f.write(
                """
OPENAI_API_KEY=test_openai_key
ANTHROPIC_API_KEY=test_anthropic_key
"""
            )

        mock_args = Mock()
        mock_args.config = self.config_path
        mock_args.env = self.env_path
        mock_args.log_level = "INFO"
        mock_args.no_color = False
        mock_args.dry_run = True

        with patch_ui_components():
            result = await run_application(mock_args)
            assert result == 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_interrupt_handler_setup(self):
        """Test that interrupt handlers are properly set up during startup."""
        create_test_config_file(self.config_path, num_agents=2)
        create_test_env_file(self.env_path)

        mock_args = Mock()
        mock_args.config = self.config_path
        mock_args.env = self.env_path
        mock_args.log_level = "INFO"
        mock_args.no_color = False
        mock_args.dry_run = True

        with patch("virtual_agora.main.setup_interrupt_handlers") as mock_setup:
            mock_interrupt_handler = Mock()
            mock_interrupt_handler.teardown = Mock()
            mock_setup.return_value = mock_interrupt_handler

            with patch_ui_components():
                result = await run_application(mock_args)
                assert result == 0

                # Verify interrupt handlers were set up and torn down
                mock_setup.assert_called_once()
                mock_interrupt_handler.teardown.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cli_argument_parsing(self):
        """Test that CLI arguments are correctly parsed and used."""
        create_test_config_file(self.config_path, num_agents=2)
        create_test_env_file(self.env_path)

        # Test with custom log level
        mock_args = Mock()
        mock_args.config = self.config_path
        mock_args.env = self.env_path
        mock_args.log_level = "DEBUG"
        mock_args.no_color = True  # Test no-color option
        mock_args.dry_run = True

        with patch_ui_components():
            result = await run_application(mock_args)
            assert result == 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_session_creation_and_state_access(self):
        """Test that session is created and state can be accessed without errors."""
        create_test_config_file(self.config_path, num_agents=2)
        create_test_env_file(self.env_path)

        mock_args = Mock()
        mock_args.config = self.config_path
        mock_args.env = self.env_path
        mock_args.log_level = "INFO"
        mock_args.no_color = False
        mock_args.dry_run = False  # Run actual session creation

        # Mock user input for topic
        with patch(
            "virtual_agora.ui.human_in_the_loop.get_initial_topic"
        ) as mock_topic:
            mock_topic.return_value = "Test topic for startup integration"

            with patch_ui_components():
                # Mock the actual discussion flow to avoid long execution
                with patch(
                    "virtual_agora.flow.graph.VirtualAgoraFlow.stream"
                ) as mock_stream:
                    mock_stream.return_value = iter(
                        [{"current_phase": 1}]
                    )  # Single update

                    result = await run_application(mock_args)
                    # Should complete successfully without the AttributeError
                    assert result == 0


class TestStartupErrorScenarios:
    """Test startup error scenarios and recovery."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_keyboard_interrupt_during_startup(self):
        """Test graceful handling of keyboard interrupt during startup."""
        temp_dir = tempfile.mkdtemp()
        config_path = Path(temp_dir) / "test_config.yml"
        env_path = Path(temp_dir) / "test.env"

        try:
            create_test_config_file(config_path, num_agents=2)
            create_test_env_file(env_path)

            mock_args = Mock()
            mock_args.config = config_path
            mock_args.env = env_path
            mock_args.log_level = "INFO"
            mock_args.no_color = False
            mock_args.dry_run = False

            # Simulate KeyboardInterrupt during flow initialization
            with patch(
                "virtual_agora.flow.graph.VirtualAgoraFlow.__init__"
            ) as mock_init:
                mock_init.side_effect = KeyboardInterrupt("User interrupt")

                with patch_ui_components():
                    result = await run_application(mock_args)
                    assert result == 130  # Standard SIGINT exit code

        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_unexpected_exception_during_startup(self):
        """Test handling of unexpected exceptions during startup."""
        temp_dir = tempfile.mkdtemp()
        config_path = Path(temp_dir) / "test_config.yml"
        env_path = Path(temp_dir) / "test.env"

        try:
            create_test_config_file(config_path, num_agents=2)
            create_test_env_file(env_path)

            mock_args = Mock()
            mock_args.config = config_path
            mock_args.env = env_path
            mock_args.log_level = "INFO"
            mock_args.no_color = False
            mock_args.dry_run = False

            # Simulate unexpected error during config loading
            with patch("virtual_agora.config.loader.ConfigLoader.load") as mock_load:
                mock_load.side_effect = RuntimeError("Unexpected error")

                with patch_ui_components():
                    result = await run_application(mock_args)
                    assert result == 1  # Generic error exit code

        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


class TestConfigurationIntegration:
    """Test configuration loading and validation integration."""

    def setup_method(self):
        """Set up test method."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.integration
    def test_config_loader_with_env_integration(self):
        """Test that ConfigLoader works with EnvironmentManager integration."""
        config_path = Path(self.temp_dir) / "integration_config.yml"
        env_path = Path(self.temp_dir) / "integration.env"

        create_test_config_file(config_path, num_agents=3)
        create_test_env_file(env_path)

        # Load environment first
        env_manager = EnvironmentManager(env_file=env_path)
        env_manager.load()

        # Then load config
        config_loader = ConfigLoader(config_path)
        config = config_loader.load()

        # Verify config is valid and environment is accessible
        assert config.moderator.provider is not None
        assert len(config.agents) > 0
        assert env_manager.get_status_report()["total_variables"] > 0

    @pytest.mark.integration
    def test_config_validation_with_complex_setup(self):
        """Test configuration validation with complex multi-provider setup."""
        config_path = Path(self.temp_dir) / "complex_config.yml"

        # Create complex configuration
        with open(config_path, "w") as f:
            f.write(
                """
moderator:
  provider: openai
  model: gpt-4
  temperature: 0.7
  max_tokens: 4000

agents:
  - provider: anthropic
    model: claude-3-haiku
    count: 2
    temperature: 0.6
  - provider: openai  
    model: gpt-3.5-turbo
    count: 3
    temperature: 0.8
    max_tokens: 2000
  - provider: groq
    model: llama-3-8b
    count: 1
    temperature: 0.9

discussion:
  max_rounds_per_topic: 8
  consensus_threshold: 0.7
  context_window_limit: 6000

reporting:
  format: markdown
  include_metadata: true
  export_format: pdf
"""
            )

        from virtual_agora.config.validators import ConfigValidator

        config_loader = ConfigLoader(config_path)
        config = config_loader.load()

        validator = ConfigValidator(config)
        validation_report = validator.get_validation_report()

        # Should validate successfully despite complexity
        assert (
            "errors" not in validation_report or len(validation_report["errors"]) == 0
        )
        assert config.get_total_agent_count() == 6
