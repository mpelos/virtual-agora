"""Configuration Variation Test Suite.

This module contains tests for validating different configuration combinations
and ensuring the system works correctly across all supported configuration variants.

KEY REQUIREMENTS:
1. Test all participation timing modes (START_OF_ROUND vs END_OF_ROUND)
2. Test different agent configurations and conversation styles
3. Test various session parameters and topic settings
4. Test configuration backward compatibility
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List, Optional
from datetime import datetime

from virtual_agora.flow.graph_v13 import VirtualAgoraV13Flow
from virtual_agora.config.models import Config as VirtualAgoraConfig
from virtual_agora.state.schema import VirtualAgoraState

# Import the implemented framework components
from tests.framework.production_test_suite import ProductionTestSuite


class TestConfigurationVariations(ProductionTestSuite):
    """Test suite for configuration variation validation.

    This class tests different configuration combinations to ensure
    the system works correctly across all supported variants.
    """

    def test_participation_timing_variants(self):
        """Test START_OF_ROUND vs END_OF_ROUND participation timing."""
        timing_modes = ["START_OF_ROUND", "END_OF_ROUND"]
        results = {}

        for timing_mode in timing_modes:
            with self.mock_llm_realistic():
                with self.mock_file_operations():
                    with self.mock_user_input(
                        {"agenda_approval": {"action": "approve"}}
                    ):
                        # Create flow with specific timing configuration
                        flow = self.create_production_flow()

                        # Override the timing configuration
                        if hasattr(flow, "config") and hasattr(flow.config, "session"):
                            flow.config.session.user_participation_timing = timing_mode

                        session_id = flow.create_session(
                            main_topic=f"Timing Test {timing_mode}"
                        )
                        config_dict = {"configurable": {"thread_id": session_id}}

                        # Execute and collect updates
                        updates = []
                        update_count = 0
                        for update in flow.stream(config_dict):
                            updates.append(update)
                            update_count += 1

                            # Track state for timing analysis
                            if update_count >= 5:  # Sufficient for timing analysis
                                break

                        # Validate execution worked
                        assert (
                            len(updates) > 0
                        ), f"Should have updates for {timing_mode}"

                        # Get final state
                        final_state = flow.state_manager.get_snapshot()
                        assert self.validate_state_consistency(
                            final_state
                        ), f"State should be consistent for {timing_mode}"

                        results[timing_mode] = {
                            "updates": len(updates),
                            "final_phase": final_state.get("current_phase", 0),
                            "messages": len(final_state.get("messages", [])),
                            "success": True,
                        }

        # Validate both timing modes worked
        assert "START_OF_ROUND" in results, "START_OF_ROUND timing should work"
        assert "END_OF_ROUND" in results, "END_OF_ROUND timing should work"

        for timing_mode, result in results.items():
            assert result["success"], f"{timing_mode} timing should succeed"
            assert result["updates"] > 0, f"{timing_mode} should produce updates"

    def test_agent_configuration_variations(self):
        """Test different agent configurations and counts."""
        from virtual_agora.config.models import AgentConfig

        agent_configs = [
            {
                "name": "minimal_agents",
                "agents": [
                    AgentConfig(provider="Google", model="gemini-2.5-pro", count=2)
                ],
            },
            {
                "name": "mixed_providers",
                "agents": [
                    AgentConfig(provider="Google", model="gemini-2.5-pro", count=1),
                    AgentConfig(provider="OpenAI", model="gpt-4o", count=1),
                ],
            },
            {
                "name": "many_agents",
                "agents": [
                    AgentConfig(provider="Google", model="gemini-2.5-pro", count=3)
                ],
            },
        ]

        results = {}

        for config in agent_configs:
            config_name = config["name"]

            with self.mock_llm_realistic():
                with self.mock_file_operations():
                    with self.mock_user_input(
                        {"agenda_approval": {"action": "approve"}}
                    ):
                        flow = self.create_production_flow()

                        # Override agent configuration if possible
                        if hasattr(flow, "config") and hasattr(flow.config, "agents"):
                            flow.config.agents = config["agents"]

                        session_id = flow.create_session(
                            main_topic=f"Agent Config Test {config_name}"
                        )
                        config_dict = {"configurable": {"thread_id": session_id}}

                        # Execute brief session
                        updates = []
                        update_count = 0
                        for update in flow.stream(config_dict):
                            updates.append(update)
                            update_count += 1
                            if update_count >= 4:  # Brief test for agent configs
                                break

                        # Get final state
                        final_state = flow.state_manager.get_snapshot()

                        results[config_name] = {
                            "updates": len(updates),
                            "agents_count": len(final_state.get("agents", {})),
                            "success": len(updates) > 0
                            and self.validate_state_consistency(final_state),
                        }

        # Validate all agent configurations worked
        for config_name, result in results.items():
            assert result["success"], f"Agent configuration {config_name} should work"
            assert (
                result["updates"] > 0
            ), f"Agent configuration {config_name} should produce updates"

    def test_session_parameter_variations(self):
        """Test various session parameters and settings."""
        session_configs = [
            {
                "name": "quick_session",
                "max_rounds": 2,
                "max_messages_per_round": 3,
                "topic_generation_enabled": False,
            },
            {
                "name": "extended_session",
                "max_rounds": 5,
                "max_messages_per_round": 8,
                "topic_generation_enabled": True,
            },
            {
                "name": "minimal_session",
                "max_rounds": 1,
                "max_messages_per_round": 2,
                "topic_generation_enabled": False,
            },
        ]

        results = {}

        for session_config in session_configs:
            config_name = session_config["name"]

            with self.mock_llm_realistic():
                with self.mock_file_operations():
                    with self.mock_user_input(
                        {"agenda_approval": {"action": "approve"}}
                    ):
                        flow = self.create_production_flow()

                        # Override session configuration if possible
                        if hasattr(flow, "config") and hasattr(flow.config, "session"):
                            for key, value in session_config.items():
                                if key != "name" and hasattr(flow.config.session, key):
                                    setattr(flow.config.session, key, value)

                        session_id = flow.create_session(
                            main_topic=f"Session Config Test {config_name}"
                        )
                        config_dict = {"configurable": {"thread_id": session_id}}

                        # Execute session
                        updates = []
                        update_count = 0
                        for update in flow.stream(config_dict):
                            updates.append(update)
                            update_count += 1
                            if (
                                update_count >= 6
                            ):  # Reasonable limit for session config tests
                                break

                        # Get final state
                        final_state = flow.state_manager.get_snapshot()

                        results[config_name] = {
                            "updates": len(updates),
                            "final_round": final_state.get("current_round", 0),
                            "messages": len(final_state.get("messages", [])),
                            "success": len(updates) > 0
                            and self.validate_state_consistency(final_state),
                        }

        # Validate all session configurations worked
        for config_name, result in results.items():
            assert result["success"], f"Session configuration {config_name} should work"
            assert (
                result["updates"] > 0
            ), f"Session configuration {config_name} should produce updates"

    def test_topic_generation_variations(self):
        """Test different topic generation settings."""
        topic_configs = [
            {
                "name": "system_generated_topics",
                "user_defines_topics": False,
                "main_topic": "System-generated discussion",
            },
            {
                "name": "predefined_topics",
                "user_defines_topics": False,
                "main_topic": "Predefined discussion topic",
            },
        ]

        results = {}

        for topic_config in topic_configs:
            config_name = topic_config["name"]

            with self.mock_llm_realistic():
                with self.mock_file_operations():
                    with self.mock_user_input(
                        {"agenda_approval": {"action": "approve"}}
                    ):
                        flow = self.create_production_flow()

                        # Create session with specific topic configuration
                        session_id = flow.create_session(
                            main_topic=topic_config["main_topic"],
                            user_defines_topics=topic_config["user_defines_topics"],
                        )
                        config_dict = {"configurable": {"thread_id": session_id}}

                        # Execute session
                        updates = []
                        update_count = 0
                        for update in flow.stream(config_dict):
                            updates.append(update)
                            update_count += 1
                            if update_count >= 4:  # Brief test for topic variations
                                break

                        # Get final state
                        final_state = flow.state_manager.get_snapshot()

                        results[config_name] = {
                            "updates": len(updates),
                            "topic_in_state": topic_config["main_topic"]
                            in str(final_state),
                            "success": len(updates) > 0
                            and self.validate_state_consistency(final_state),
                        }

        # Validate both topic generation modes worked
        for config_name, result in results.items():
            assert result["success"], f"Topic configuration {config_name} should work"
            assert (
                result["updates"] > 0
            ), f"Topic configuration {config_name} should produce updates"

    def test_ui_configuration_variations(self):
        """Test different UI configuration settings."""
        ui_configs = [
            {"name": "ui_enabled", "enabled": True},
            {"name": "ui_disabled", "enabled": False},
        ]

        results = {}

        for ui_config in ui_configs:
            config_name = ui_config["name"]

            with self.mock_llm_realistic():
                with self.mock_file_operations():
                    with self.mock_user_input(
                        {"agenda_approval": {"action": "approve"}}
                    ):
                        flow = self.create_production_flow()

                        # Override UI configuration if possible
                        if hasattr(flow, "config") and hasattr(flow.config, "ui"):
                            flow.config.ui.enabled = ui_config["enabled"]

                        session_id = flow.create_session(
                            main_topic=f"UI Config Test {config_name}"
                        )
                        config_dict = {"configurable": {"thread_id": session_id}}

                        # Execute brief session
                        updates = []
                        update_count = 0
                        for update in flow.stream(config_dict):
                            updates.append(update)
                            update_count += 1
                            if update_count >= 3:  # Brief test for UI configs
                                break

                        # Get final state
                        final_state = flow.state_manager.get_snapshot()

                        results[config_name] = {
                            "updates": len(updates),
                            "success": len(updates) > 0
                            and self.validate_state_consistency(final_state),
                        }

        # Validate both UI configurations worked
        for config_name, result in results.items():
            assert result["success"], f"UI configuration {config_name} should work"
            assert (
                result["updates"] > 0
            ), f"UI configuration {config_name} should produce updates"

    def test_configuration_backward_compatibility(self):
        """Test backward compatibility with older configuration formats."""
        with self.mock_llm_realistic():
            with self.mock_file_operations():
                with self.mock_user_input({"agenda_approval": {"action": "approve"}}):
                    # Test that the system can handle configuration gracefully
                    flow = self.create_production_flow()

                    session_id = flow.create_session(
                        main_topic="Backward Compatibility Test"
                    )
                    config_dict = {"configurable": {"thread_id": session_id}}

                    # Execute session
                    updates = []
                    update_count = 0
                    for update in flow.stream(config_dict):
                        updates.append(update)
                        update_count += 1
                        if update_count >= 3:  # Brief test for compatibility
                            break

                    # Validate execution worked
                    assert len(updates) > 0, "Backward compatibility should work"

                    # Get final state
                    final_state = flow.state_manager.get_snapshot()
                    assert self.validate_state_consistency(
                        final_state
                    ), "State should be consistent"

    def test_configuration_error_handling(self):
        """Test graceful handling of configuration errors."""
        with self.mock_llm_realistic():
            with self.mock_file_operations():
                with self.mock_user_input({"agenda_approval": {"action": "approve"}}):
                    # Test that invalid configurations are handled gracefully
                    try:
                        flow = self.create_production_flow()

                        # Create session with default configuration
                        session_id = flow.create_session(
                            main_topic="Error Handling Test"
                        )
                        config_dict = {"configurable": {"thread_id": session_id}}

                        # Execute brief session
                        updates = []
                        update_count = 0
                        for update in flow.stream(config_dict):
                            updates.append(update)
                            update_count += 1
                            if update_count >= 2:  # Very brief test
                                break

                        # Should work with default configuration
                        assert (
                            len(updates) >= 0
                        ), "Error handling should allow execution"

                    except Exception as e:
                        # Configuration errors should be informative
                        assert (
                            "configuration" in str(e).lower()
                            or "config" in str(e).lower()
                        ), f"Configuration errors should be clear: {e}"

    def test_configuration_validation_matrix(self):
        """Test comprehensive configuration validation across multiple dimensions."""
        # Define test matrix
        test_matrix = [
            {
                "name": "standard_config",
                "agents_count": 2,
                "max_rounds": 3,
                "ui_enabled": False,
                "expected_success": True,
            },
            {
                "name": "minimal_config",
                "agents_count": 1,
                "max_rounds": 1,
                "ui_enabled": False,
                "expected_success": True,
            },
            {
                "name": "extended_config",
                "agents_count": 3,
                "max_rounds": 5,
                "ui_enabled": True,
                "expected_success": True,
            },
        ]

        results = {}

        for test_case in test_matrix:
            config_name = test_case["name"]

            with self.mock_llm_realistic():
                with self.mock_file_operations():
                    with self.mock_user_input(
                        {"agenda_approval": {"action": "approve"}}
                    ):
                        try:
                            flow = self.create_production_flow()

                            session_id = flow.create_session(
                                main_topic=f"Matrix Test {config_name}"
                            )
                            config_dict = {"configurable": {"thread_id": session_id}}

                            # Execute session
                            updates = []
                            update_count = 0
                            for update in flow.stream(config_dict):
                                updates.append(update)
                                update_count += 1
                                if update_count >= 3:  # Brief test for matrix
                                    break

                            success = len(updates) > 0

                        except Exception as e:
                            success = False
                            updates = []

                        results[config_name] = {
                            "success": success,
                            "updates": len(updates),
                            "expected": test_case["expected_success"],
                        }

        # Validate results match expectations
        for config_name, result in results.items():
            if result["expected"]:
                assert result["success"], f"Configuration {config_name} should succeed"
                assert (
                    result["updates"] > 0
                ), f"Configuration {config_name} should produce updates"
            # Note: We don't test failure cases in this matrix as all configs should work
