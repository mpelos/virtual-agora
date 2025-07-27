"""Integration tests for HITL system in Virtual Agora v1.3."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

from virtual_agora.config.models import Config as VirtualAgoraConfig
from virtual_agora.app_v13 import VirtualAgoraApplicationV13
from virtual_agora.ui.hitl_state import HITLApprovalType, HITLInteraction, HITLResponse
from virtual_agora.flow.graph_v13 import VirtualAgoraV13Flow


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock(spec=VirtualAgoraConfig)
    config.moderator = Mock(provider=Mock(value="openai"), model="gpt-4")
    config.agents = []
    config.get_total_agent_count = Mock(return_value=5)
    return config


@pytest.fixture
def app_v13(mock_config):
    """Create v1.3 application instance."""
    with patch('virtual_agora.app_v13.get_console'):
        with patch('virtual_agora.app_v13.VirtualAgoraV13Flow'):
            app = VirtualAgoraApplicationV13(mock_config)
            # Mock the flow components
            app.flow = Mock()
            app.state_manager = Mock()
            app.state_manager.state = {}
            app.specialized_agents = {}
            app.discussing_agents = []
            return app


class TestHITLIntegration:
    """Test HITL integration with v1.3 application."""
    
    def test_hitl_gate_handling(self, app_v13):
        """Test complete HITL gate handling flow."""
        # Prepare state with HITL gate
        state = {
            "hitl_state": {
                "awaiting_approval": True,
                "approval_type": "periodic_stop",
                "prompt_message": "5-round checkpoint"
            },
            "current_round": 5,
            "active_topic": "AI Ethics",
            "completed_topics": ["Introduction"],
            "topic_queue": ["Future Implications"],
            "messages": ["msg1", "msg2"],
            "start_time": datetime.now()
        }
        
        # Mock HITL manager response
        mock_response = HITLResponse(
            approved=True,
            action="continue",
            metadata={"checkpoint_round": 5}
        )
        
        with patch.object(app_v13.hitl_manager, 'process_interaction', return_value=mock_response):
            result = app_v13.handle_hitl_gate(state)
        
        # Verify state updates
        assert result["hitl_state"]["awaiting_approval"] is False
        assert "approval_history" in result["hitl_state"]
        assert len(result["hitl_state"]["approval_history"]) == 1
        
        # Verify approval history entry
        history = result["hitl_state"]["approval_history"][0]
        assert history["type"] == "periodic_stop"
        assert history["result"] == "continue"
        assert history["approved"] is True
    
    def test_periodic_stop_force_end(self, app_v13):
        """Test periodic stop with force end action."""
        state = {
            "hitl_state": {
                "awaiting_approval": True,
                "approval_type": "periodic_stop",
                "prompt_message": "Checkpoint"
            },
            "current_round": 10,
            "active_topic": "Current Topic"
        }
        
        # Mock force end response
        mock_response = HITLResponse(
            approved=True,
            action="force_topic_end",
            reason="User wants to end topic"
        )
        
        with patch.object(app_v13.hitl_manager, 'process_interaction', return_value=mock_response):
            result = app_v13.handle_hitl_gate(state)
        
        # Verify force end updates
        assert result["user_forced_conclusion"] is True
        assert result["force_reason"] == "User wants to end topic"
        assert result["hitl_state"]["awaiting_approval"] is False
    
    def test_agenda_approval_flow(self, app_v13):
        """Test agenda approval flow."""
        proposed_agenda = ["Topic 1", "Topic 2", "Topic 3"]
        
        state = {
            "hitl_state": {
                "awaiting_approval": True,
                "approval_type": "agenda_approval",
                "prompt_message": "Approve agenda?"
            },
            "proposed_agenda": proposed_agenda
        }
        
        # Mock edited agenda response
        edited_agenda = ["Modified Topic 1", "Topic 2", "New Topic 3"]
        mock_response = HITLResponse(
            approved=True,
            action="edited",
            modified_data=edited_agenda
        )
        
        with patch.object(app_v13.hitl_manager, 'process_interaction', return_value=mock_response):
            result = app_v13.handle_hitl_gate(state)
        
        # Verify agenda updates
        assert result["agenda"]["topics"] == edited_agenda
        assert result["topic_queue"] == edited_agenda
        assert result["agenda_approved"] is True
        assert result["agenda_edited"] is True
    
    def test_session_continuation_override(self, app_v13):
        """Test session continuation with agent override."""
        state = {
            "hitl_state": {
                "awaiting_approval": True,
                "approval_type": "session_continuation",
                "prompt_message": "Continue session?"
            },
            "conclusion_vote": {
                "recommendation": "end",
                "vote_counts": {"continue": 2, "end": 3}
            }
        }
        
        # Mock user override
        mock_response = HITLResponse(
            approved=True,
            action="continue_session"
        )
        
        with patch.object(app_v13.hitl_manager, 'process_interaction', return_value=mock_response):
            result = app_v13.handle_hitl_gate(state)
        
        # Verify continuation override
        assert result["user_approves_continuation"] is True
    
    def test_interrupt_handling(self, app_v13):
        """Test interrupt callback handling."""
        # Test end topic interrupt
        app_v13._handle_interrupt({
            "action": "end_topic",
            "reason": "User interrupt"
        })
        
        app_v13.state_manager.update_state.assert_called_with({
            "user_forced_conclusion": True,
            "force_reason": "User interrupt"
        })
        
        # Test skip to report interrupt
        app_v13._handle_interrupt({
            "action": "skip_to_report",
            "reason": "User wants report"
        })
        
        app_v13.state_manager.update_state.assert_called_with({
            "topic_queue": [],
            "skip_to_report": True
        })
    
    def test_checkpoint_notification_display(self, app_v13):
        """Test checkpoint notification with session controller."""
        state = {
            "current_round": 5,
            "active_topic": "Test Topic",
            "messages": ["msg1", "msg2", "msg3"],
            "completed_topics": ["Topic 1"],
            "start_time": datetime.now()
        }
        
        with patch.object(app_v13.session_controller, 'check_periodic_control', return_value=True):
            with patch.object(app_v13.session_controller, 'display_checkpoint_notification') as mock_display:
                # Create HITL state
                state["hitl_state"] = {
                    "awaiting_approval": True,
                    "approval_type": "periodic_stop",
                    "prompt_message": "Checkpoint"
                }
                
                # Mock HITL response
                mock_response = HITLResponse(approved=True, action="continue")
                
                with patch.object(app_v13.hitl_manager, 'process_interaction', return_value=mock_response):
                    app_v13.handle_hitl_gate(state)
                
                # Verify checkpoint notification was displayed
                mock_display.assert_called_once_with(5, "Test Topic", state)
    
    def test_agent_display_integration(self, app_v13):
        """Test agent invocation and result display."""
        # Test agent invocation display
        app_v13.show_agent_invocation(
            agent_type="summarizer",
            task="Summarize discussion",
            context={"round": 5}
        )
        
        app_v13.dashboard.agent_display.show_agent_invocation.assert_called_once_with(
            "summarizer",
            "Summarize discussion",
            {"round": 5},
            status="active"
        )
        
        # Test agent result display
        app_v13.show_agent_result(
            agent_type="summarizer",
            result="Summary complete",
            execution_time=1.5
        )
        
        app_v13.dashboard.agent_display.show_agent_result.assert_called_once_with(
            "summarizer",
            "Summary complete",
            1.5
        )
    
    def test_dashboard_update_integration(self, app_v13):
        """Test dashboard update with state."""
        state = {
            "current_phase": 2,
            "active_topic": "Current Discussion",
            "messages": ["msg1", "msg2"],
            "completed_topics": ["Topic 1"]
        }
        
        app_v13.state_manager.state = state
        app_v13.update_dashboard()
        
        app_v13.dashboard.update_session_info.assert_called_once_with(state)
    
    @patch('virtual_agora.app_v13.LoadingSpinner')
    @patch('virtual_agora.app_v13.get_initial_topic')
    @patch('virtual_agora.app_v13.Live')
    def test_run_session_with_hitl(self, mock_live, mock_get_topic, mock_spinner, app_v13):
        """Test running a session with HITL interactions."""
        # Mock topic input
        mock_get_topic.return_value = "AI Safety"
        
        # Mock flow creation and compilation
        app_v13.flow.create_session = Mock(return_value="test_session_id")
        app_v13.flow.compile = Mock()
        
        # Mock flow stream with HITL gate
        flow_updates = [
            {"current_phase": 1},
            {"hitl_gate": True}  # Triggers HITL handling
        ]
        app_v13.flow.stream = Mock(return_value=iter(flow_updates))
        
        # Mock state with HITL gate
        app_v13.state_manager.state = {
            "hitl_state": {
                "awaiting_approval": True,
                "approval_type": "agenda_approval",
                "prompt_message": "Approve?"
            },
            "proposed_agenda": ["Topic 1", "Topic 2"]
        }
        
        # Mock HITL response
        mock_response = HITLResponse(
            approved=True,
            action="approved",
            modified_data=["Topic 1", "Topic 2"]
        )
        
        # Mock live context manager
        mock_live_instance = MagicMock()
        mock_live.return_value.__enter__.return_value = mock_live_instance
        
        with patch.object(app_v13.hitl_manager, 'process_interaction', return_value=mock_response):
            with patch.object(app_v13, '_show_final_summary'):
                result = app_v13.run_session()
        
        # Verify session was created
        app_v13.flow.create_session.assert_called_once()
        
        # Verify HITL was processed
        app_v13.hitl_manager.process_interaction.assert_called()
        
        # Verify state was updated
        app_v13.state_manager.update_state.assert_called()
        
        assert result == 0  # Success
    
    def test_error_handling_in_hitl(self, app_v13):
        """Test error handling during HITL processing."""
        state = {
            "hitl_state": {
                "awaiting_approval": True,
                "approval_type": "invalid_type",  # This will cause an error
                "prompt_message": "Test"
            }
        }
        
        # Should handle error gracefully
        with pytest.raises(ValueError):
            app_v13.handle_hitl_gate(state)
    
    def test_checkpoint_creation_during_phases(self, app_v13):
        """Test checkpoint creation at phase transitions."""
        # Test pause interrupt with checkpoint
        checkpoint_data = {
            "timestamp": datetime.now(),
            "state": {"test": "data"},
            "reason": "User pause"
        }
        
        app_v13._handle_interrupt({
            "action": "pause",
            "checkpoint": checkpoint_data
        })
        
        # Verify checkpoint was saved
        app_v13.checkpoint_manager.save_checkpoint.assert_called_once()
        call_args = app_v13.checkpoint_manager.save_checkpoint.call_args
        assert call_args[0][0] == app_v13.session_id
        assert call_args[0][1] == app_v13.state_manager.state
        assert call_args[1]["metadata"] == checkpoint_data