"""Tests for LangGraph UI integration."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

from virtual_agora.ui.langgraph_integration import (
    LangGraphUIIntegration,
    LangGraphUIMiddleware,
    get_ui_integration,
    initialize_ui_integration,
    update_ui_from_state_change,
    notify_agent_started,
    notify_agent_completed,
    notify_agent_error,
)
from virtual_agora.state.schema import VirtualAgoraState, UIState
from virtual_agora.ui.theme import ProviderType
from virtual_agora.ui.dashboard import PhaseType, AgentStatus


class TestLangGraphUIIntegration:
    """Test LangGraph UI integration functionality."""

    def test_initialization(self):
        """Test UI integration initialization."""
        integration = LangGraphUIIntegration()

        assert integration.console is not None
        assert integration.discussion_display is not None
        assert integration.dashboard is not None
        assert integration.dashboard_manager is not None
        assert integration._last_known_state is None
        assert len(integration._ui_callbacks) > 0  # Has default callbacks registered
        assert integration._state_listeners == []

    def test_register_callback(self):
        """Test registering UI callbacks."""
        integration = LangGraphUIIntegration()

        def test_callback(data):
            pass

        integration.register_callback("test_event", test_callback)

        assert "test_event" in integration._ui_callbacks
        assert test_callback in integration._ui_callbacks["test_event"]

    def test_trigger_callbacks(self):
        """Test triggering registered callbacks."""
        integration = LangGraphUIIntegration()

        callback_called = []

        def test_callback(data):
            callback_called.append(data)

        integration.register_callback("test_event", test_callback)
        integration._trigger_callbacks("test_event", {"test": "data"})

        assert len(callback_called) == 1
        assert callback_called[0] == {"test": "data"}

    def test_trigger_callbacks_error_handling(self):
        """Test callback error handling."""
        integration = LangGraphUIIntegration()

        def failing_callback(data):
            raise Exception("Test error")

        integration.register_callback("test_event", failing_callback)

        # Should not raise exception
        integration._trigger_callbacks("test_event", {"test": "data"})

    @patch("virtual_agora.ui.langgraph_integration.initialize_accessibility")
    def test_initialize_ui_for_session(self, mock_init_accessibility):
        """Test initializing UI for a session."""
        integration = LangGraphUIIntegration()

        # Create mock state
        state: VirtualAgoraState = {
            "session_id": "test_session",
            "start_time": datetime.now(),
            "config_hash": "test_hash",
            "current_phase": 0,
            "main_topic": "Test Topic",
            "agents": {
                "agent1": {
                    "id": "agent1",
                    "model": "gpt-4",
                    "provider": "openai",
                    "role": "participant",
                    "message_count": 0,
                    "created_at": datetime.now(),
                }
            },
            "moderator_id": "moderator",
        }

        with patch.object(integration.discussion_display, "display_session_header"):
            with patch.object(integration.dashboard_manager, "initialize_session"):
                result = integration.initialize_ui_for_session(state)

        # Should return updated state with UI state
        assert "ui_state" in result
        assert result["ui_state"]["console_initialized"] is True
        assert result["ui_state"]["theme_applied"] is True

        # Should have called accessibility initialization
        mock_init_accessibility.assert_called_once()

    def test_detect_state_changes_phase_change(self):
        """Test detecting phase changes."""
        integration = LangGraphUIIntegration()

        old_state = {"current_phase": 0}
        new_state = {"current_phase": 1}

        changes = integration._detect_state_changes(old_state, new_state)

        assert "phase_change" in changes
        assert changes["phase_change"]["from_phase"] == 0
        assert changes["phase_change"]["to_phase"] == 1

    def test_detect_state_changes_round_change(self):
        """Test detecting round changes."""
        integration = LangGraphUIIntegration()

        old_state = {"current_round": 1}
        new_state = {
            "current_round": 2,
            "active_topic": "New Topic",
            "agents": {"agent1": {}},
        }

        changes = integration._detect_state_changes(old_state, new_state)

        assert "round_started" in changes
        assert changes["round_started"]["round_number"] == 2
        assert changes["round_started"]["topic"] == "New Topic"

    def test_detect_state_changes_message_added(self):
        """Test detecting new messages."""
        integration = LangGraphUIIntegration()

        old_state = {"messages": [{"id": "1", "content": "First"}]}
        new_state = {
            "messages": [
                {"id": "1", "content": "First"},
                {"id": "2", "content": "Second"},
            ]
        }

        changes = integration._detect_state_changes(old_state, new_state)

        assert "message_added" in changes
        assert changes["message_added"]["id"] == "2"

    def test_detect_state_changes_vote_started(self):
        """Test detecting vote started."""
        integration = LangGraphUIIntegration()

        old_state = {"active_vote": None}
        new_state = {"active_vote": {"id": "vote1", "vote_type": "continue"}}

        changes = integration._detect_state_changes(old_state, new_state)

        assert "vote_started" in changes
        assert changes["vote_started"]["id"] == "vote1"

    def test_detect_state_changes_vote_completed(self):
        """Test detecting vote completion."""
        integration = LangGraphUIIntegration()

        old_state = {"active_vote": {"id": "vote1"}}
        new_state = {
            "active_vote": None,
            "vote_history": [{"id": "vote1", "status": "completed", "result": "yes"}],
        }

        changes = integration._detect_state_changes(old_state, new_state)

        assert "vote_completed" in changes
        assert changes["vote_completed"]["id"] == "vote1"

    def test_detect_state_changes_agenda_updated(self):
        """Test detecting agenda updates."""
        integration = LangGraphUIIntegration()

        old_state = {"agenda": {"topics": ["A", "B"]}}
        new_state = {"agenda": {"topics": ["A", "B", "C"]}}

        changes = integration._detect_state_changes(old_state, new_state)

        assert "agenda_updated" in changes
        assert changes["agenda_updated"]["topics"] == ["A", "B", "C"]

    def test_detect_state_changes_topic_changed(self):
        """Test detecting topic changes."""
        integration = LangGraphUIIntegration()

        old_state = {"active_topic": "Topic A"}
        new_state = {"active_topic": "Topic B"}

        changes = integration._detect_state_changes(old_state, new_state)

        assert "topic_changed" in changes
        assert changes["topic_changed"]["from_topic"] == "Topic A"
        assert changes["topic_changed"]["to_topic"] == "Topic B"

    def test_update_ui_from_state(self):
        """Test updating UI from state changes."""
        integration = LangGraphUIIntegration()

        # Set initial state
        old_state = {"current_phase": 0, "ui_state": {"last_ui_update": datetime.now()}}
        integration._last_known_state = old_state

        # Create new state with changes
        new_state = {"current_phase": 1, "ui_state": {"last_ui_update": datetime.now()}}

        # Mock the callback
        callback_called = []

        def mock_callback(data):
            callback_called.append(data)

        integration.register_callback("phase_change", mock_callback)

        integration.update_ui_from_state(new_state)

        # Should have triggered phase change callback
        assert len(callback_called) == 1
        assert callback_called[0]["from_phase"] == 0
        assert callback_called[0]["to_phase"] == 1

        # Should update last known state
        assert integration._last_known_state == new_state


class TestDefaultCallbacks:
    """Test default UI callback implementations."""

    def test_on_phase_change(self):
        """Test phase change callback."""
        integration = LangGraphUIIntegration()

        change_data = {"from_phase": 0, "to_phase": 1, "timestamp": datetime.now()}

        with patch.object(integration.discussion_display, "display_phase_transition"):
            with patch.object(integration.dashboard, "set_phase"):
                integration._on_phase_change(change_data)

                integration.discussion_display.display_phase_transition.assert_called_once_with(
                    "Initialization", "Agenda Setting"
                )
                integration.dashboard.set_phase.assert_called_once_with(
                    PhaseType.AGENDA_SETTING
                )

    def test_on_message_added(self):
        """Test message added callback."""
        integration = LangGraphUIIntegration()

        # Set up state with agent info
        integration._last_known_state = {
            "agents": {"agent1": {"provider": "openai"}},
            "current_round": 1,
        }

        message = {
            "speaker_id": "agent1",
            "speaker_role": "participant",
            "content": "Test message",
            "timestamp": datetime.now(),
            "topic": "Test Topic",
        }

        with patch("virtual_agora.ui.langgraph_integration.add_agent_message"):
            with patch.object(
                integration.dashboard_manager, "agent_completed_response"
            ):
                integration._on_message_added(message)

                # Should have called add_agent_message with correct parameters
                # and dashboard update
                integration.dashboard_manager.agent_completed_response.assert_called_once_with(
                    "agent1"
                )

    def test_on_round_started(self):
        """Test round started callback."""
        integration = LangGraphUIIntegration()

        round_data = {
            "round_number": 2,
            "topic": "Test Topic",
            "agents": ["agent1", "agent2"],
        }

        with patch("virtual_agora.ui.langgraph_integration.start_discussion_round"):
            with patch.object(integration.dashboard, "set_current_topic"):
                integration._on_round_started(round_data)

                integration.dashboard.set_current_topic.assert_called_once_with(
                    "Test Topic", 2
                )

    def test_on_vote_started(self):
        """Test vote started callback."""
        integration = LangGraphUIIntegration()

        vote_data = {"vote_type": "continue_discussion", "options": ["Yes", "No"]}

        with patch.object(integration.console, "print_system_message"):
            with patch.object(integration.dashboard_manager, "start_voting"):
                integration._on_vote_started(vote_data)

                integration.console.print_system_message.assert_called_once()
                integration.dashboard_manager.start_voting.assert_called_once()

    def test_on_agent_error(self):
        """Test agent error callback."""
        integration = LangGraphUIIntegration()

        error_data = {"agent_id": "agent1", "error": "Test error message"}

        with patch.object(integration.dashboard_manager, "agent_error"):
            with patch.object(integration.discussion_display, "display_error_message"):
                integration._on_agent_error(error_data)

                integration.dashboard_manager.agent_error.assert_called_once_with(
                    "agent1", "Test error message"
                )
                integration.discussion_display.display_error_message.assert_called_once_with(
                    "Test error message", "Agent: agent1"
                )


class TestLangGraphUIMiddleware:
    """Test LangGraph UI middleware."""

    def test_initialization(self):
        """Test middleware initialization."""
        mock_integration = Mock()
        middleware = LangGraphUIMiddleware(mock_integration)

        assert middleware.ui_integration is mock_integration

    @pytest.mark.asyncio
    async def test_wrap_node_async(self):
        """Test wrapping async node function."""
        mock_integration = Mock()
        middleware = LangGraphUIMiddleware(mock_integration)

        async def test_node(state):
            return {"updated": True}

        wrapped = middleware.wrap_node(test_node)

        state = {"test": "state"}

        with patch("virtual_agora.ui.langgraph_integration.operation_spinner"):
            result = await wrapped(state)

        assert result == {"updated": True}
        mock_integration.update_ui_from_state.assert_called_once_with({"updated": True})

    def test_wrap_node_sync(self):
        """Test wrapping sync node function."""
        mock_integration = Mock()
        middleware = LangGraphUIMiddleware(mock_integration)

        def test_node(state):
            return {"updated": True}

        wrapped = middleware.wrap_node(test_node)

        state = {"test": "state"}

        with patch("virtual_agora.ui.langgraph_integration.operation_spinner"):
            # Note: wrapped function is async even if original was sync
            import asyncio

            result = asyncio.run(wrapped(state))

        assert result == {"updated": True}
        mock_integration.update_ui_from_state.assert_called_once_with({"updated": True})


class TestGlobalFunctions:
    """Test global UI integration functions."""

    def test_get_ui_integration(self):
        """Test getting global UI integration."""
        integration1 = get_ui_integration()
        integration2 = get_ui_integration()

        assert integration1 is integration2
        assert isinstance(integration1, LangGraphUIIntegration)

    def test_initialize_ui_integration(self):
        """Test initializing UI integration."""
        integration = initialize_ui_integration()

        assert isinstance(integration, LangGraphUIIntegration)

    def test_notify_agent_started(self):
        """Test notifying agent started."""
        mock_integration = Mock()

        with patch(
            "virtual_agora.ui.langgraph_integration.get_ui_integration",
            return_value=mock_integration,
        ):
            state = {"test": "state"}
            result = notify_agent_started(state, "agent1")

            assert result is state
            mock_integration._trigger_callbacks.assert_called_once_with(
                "agent_responding", {"agent_id": "agent1"}
            )

    def test_notify_agent_completed(self):
        """Test notifying agent completed."""
        mock_integration = Mock()

        with patch(
            "virtual_agora.ui.langgraph_integration.get_ui_integration",
            return_value=mock_integration,
        ):
            state = {"test": "state"}
            result = notify_agent_completed(state, "agent1")

            assert result is state
            mock_integration._trigger_callbacks.assert_called_once_with(
                "agent_completed", {"agent_id": "agent1"}
            )

    def test_notify_agent_error(self):
        """Test notifying agent error."""
        mock_integration = Mock()

        with patch(
            "virtual_agora.ui.langgraph_integration.get_ui_integration",
            return_value=mock_integration,
        ):
            state = {"test": "state"}
            result = notify_agent_error(state, "agent1", "Test error")

            assert result is state
            mock_integration._trigger_callbacks.assert_called_once_with(
                "agent_error", {"agent_id": "agent1", "error": "Test error"}
            )

    def test_update_ui_from_state_change(self):
        """Test updating UI from state change."""
        mock_integration = Mock()

        with patch(
            "virtual_agora.ui.langgraph_integration.get_ui_integration",
            return_value=mock_integration,
        ):
            state = {"test": "state"}
            result = update_ui_from_state_change(state)

            assert result is state
            mock_integration.update_ui_from_state.assert_called_once_with(state)


class TestIntegrationEdgeCases:
    """Test edge cases and error conditions."""

    def test_detect_changes_with_none_states(self):
        """Test detecting changes with None states."""
        integration = LangGraphUIIntegration()

        changes = integration._detect_state_changes({}, {"current_phase": 1})

        assert "phase_change" in changes
        assert changes["phase_change"]["from_phase"] is None
        assert changes["phase_change"]["to_phase"] == 1

    def test_update_ui_with_no_previous_state(self):
        """Test updating UI with no previous state."""
        integration = LangGraphUIIntegration()

        new_state = {"current_phase": 1}

        # Should not crash
        integration.update_ui_from_state(new_state)

        assert integration._last_known_state == new_state

    def test_callback_registration_multiple_callbacks(self):
        """Test registering multiple callbacks for same event."""
        integration = LangGraphUIIntegration()

        def callback1(data):
            pass

        def callback2(data):
            pass

        integration.register_callback("test_event", callback1)
        integration.register_callback("test_event", callback2)

        assert len(integration._ui_callbacks["test_event"]) == 2
        assert callback1 in integration._ui_callbacks["test_event"]
        assert callback2 in integration._ui_callbacks["test_event"]

    def test_message_added_moderator_message(self):
        """Test handling moderator message."""
        integration = LangGraphUIIntegration()

        message = {
            "speaker_id": "moderator",
            "speaker_role": "moderator",
            "content": "Moderator message",
            "timestamp": datetime.now(),
        }

        with patch("virtual_agora.ui.langgraph_integration.add_moderator_message"):
            with patch.object(
                integration.dashboard_manager, "agent_completed_response"
            ):
                integration._on_message_added(message)

                # Should call moderator message function
                integration.dashboard_manager.agent_completed_response.assert_called_once_with(
                    "moderator"
                )


if __name__ == "__main__":
    pytest.main([__file__])
