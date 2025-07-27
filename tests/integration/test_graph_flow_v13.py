"""Integration tests for v1.3 graph flow.

Tests the complete v1.3 node-centric architecture with
specialized agents and enhanced HITL features.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import BaseMessage

from virtual_agora.flow.graph_v13 import VirtualAgoraV13Flow
from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.config.models import Config as VirtualAgoraConfig, Provider


class TestV13GraphFlow:
    """Test v1.3 graph flow integration."""
    
    @pytest.fixture
    def test_config(self):
        """Create minimal test configuration."""
        config = VirtualAgoraConfig(
            moderator={
                "provider": Provider.GOOGLE,
                "model": "mock-model",
                "temperature": 0.7
            },
            summarizer={
                "provider": Provider.GOOGLE,
                "model": "mock-model",
                "temperature": 0.3
            },
            topic_report={
                "provider": Provider.GOOGLE,
                "model": "mock-model",
                "temperature": 0.5
            },
            ecclesia_report={
                "provider": Provider.GOOGLE,
                "model": "mock-model", 
                "temperature": 0.5
            },
            agents=[
                {
                    "provider": Provider.GOOGLE,
                    "model": "mock-agent",
                    "count": 3,
                    "temperature": 0.7
                }
            ]
        )
        return config
    
    @pytest.fixture
    def v13_flow(self, test_config):
        """Create v1.3 flow instance."""
        # Mock the entire _initialize_agents method to avoid provider creation
        with patch.object(VirtualAgoraV13Flow, '_initialize_agents') as mock_init:
            flow = VirtualAgoraV13Flow(test_config, enable_monitoring=False)
            
            # Manually set up mock agents
            flow.specialized_agents = {
                "moderator": Mock(agent_id="moderator"),
                "summarizer": Mock(agent_id="summarizer"),
                "topic_report": Mock(agent_id="topic_report"),
                "ecclesia_report": Mock(agent_id="ecclesia_report")
            }
            
            flow.discussing_agents = [
                Mock(agent_id="mock-agent_1"),
                Mock(agent_id="mock-agent_2"),
                Mock(agent_id="mock-agent_3")
            ]
            
            return flow
    
    def test_flow_initialization(self, v13_flow):
        """Test v1.3 flow initializes correctly."""
        assert v13_flow is not None
        assert len(v13_flow.specialized_agents) == 4  # moderator, summarizer, topic_report, ecclesia_report
        assert len(v13_flow.discussing_agents) == 3  # 3 discussion agents
        assert v13_flow.graph is None  # Not built yet
    
    def test_specialized_agents_created(self, v13_flow):
        """Test all specialized agents are created."""
        assert "moderator" in v13_flow.specialized_agents
        assert "summarizer" in v13_flow.specialized_agents
        assert "topic_report" in v13_flow.specialized_agents
        assert "ecclesia_report" in v13_flow.specialized_agents
        
        # Verify agent IDs
        assert v13_flow.specialized_agents["moderator"].agent_id == "moderator"
        assert v13_flow.specialized_agents["summarizer"].agent_id == "summarizer"
    
    def test_discussing_agents_created(self, v13_flow):
        """Test discussing agents are created with correct IDs."""
        assert len(v13_flow.discussing_agents) == 3
        
        # Check agent IDs
        agent_ids = [agent.agent_id for agent in v13_flow.discussing_agents]
        assert "mock-agent_1" in agent_ids
        assert "mock-agent_2" in agent_ids
        assert "mock-agent_3" in agent_ids
    
    def test_build_graph(self, v13_flow):
        """Test graph building with v1.3 structure."""
        graph = v13_flow.build_graph()
        
        assert graph is not None
        assert v13_flow.graph is not None
        
        # Verify key nodes exist
        nodes = graph.nodes
        assert "config_and_keys" in nodes
        assert "agent_instantiation" in nodes
        assert "get_theme" in nodes
        assert "agenda_proposal" in nodes
        assert "discussion_round" in nodes
        assert "periodic_user_stop" in nodes
        assert "final_report_generation" in nodes
    
    def test_compile_graph(self, v13_flow):
        """Test graph compilation."""
        compiled = v13_flow.compile()
        
        assert compiled is not None
        assert v13_flow.compiled_graph is not None
    
    def test_create_session(self, v13_flow):
        """Test session creation with v1.3 state."""
        session_id = v13_flow.create_session(main_topic="Test Topic")
        
        assert session_id is not None
        
        # Check state initialization
        state = v13_flow.state_manager.state
        assert state["session_id"] == session_id
        assert state["main_topic"] == "Test Topic"
        assert state["current_phase"] == 0
        assert state["current_round"] == 0
        assert "hitl_state" in state
        assert "flow_control" in state
        assert state["periodic_stop_counter"] == 0
        assert state["user_forced_conclusion"] is False
    
    def test_phase_0_flow(self, v13_flow):
        """Test Phase 0 initialization flow."""
        # Create session
        v13_flow.create_session()
        
        # Test config_and_keys node
        state = {}
        result = v13_flow.nodes.config_and_keys_node(state)
        
        assert result["config_loaded"] is True
        assert "initialization_timestamp" in result
        assert result["system_status"] == "ready"
    
    def test_phase_1_agenda_setting(self, v13_flow):
        """Test Phase 1 agenda setting nodes."""
        v13_flow.create_session(main_topic="AI Safety")
        
        # Mock agent responses for proposals
        for agent in v13_flow.discussing_agents:
            agent.return_value = {
                "messages": [Mock(content="Topic 1\nTopic 2")]
            }
        
        # Test agenda proposal node
        state = {"main_topic": "AI Safety"}
        result = v13_flow.nodes.agenda_proposal_node(state)
        
        assert "proposed_topics" in result
        assert len(result["proposed_topics"]) == 3  # 3 agents
        assert result["current_phase"] == 1
    
    @patch('virtual_agora.flow.nodes_v13.interrupt')
    def test_periodic_stop_mechanism(self, mock_interrupt, v13_flow):
        """Test 5-round periodic stop functionality."""
        v13_flow.create_session()
        
        # Set up state at round 5
        state = {
            "current_round": 5,
            "active_topic": "Test Topic",
            "periodic_stop_counter": 0
        }
        
        # Test periodic stop node
        result = v13_flow.nodes.periodic_user_stop_node(state)
        
        # Should trigger interrupt
        mock_interrupt.assert_called_once()
        
        # Check state updates
        assert result["hitl_state"]["awaiting_approval"] is True
        assert result["hitl_state"]["approval_type"] == "periodic_stop"
        assert "Do you wish to end" in result["hitl_state"]["prompt_message"]
        assert result["periodic_stop_counter"] == 1
    
    def test_dual_polling_system(self, v13_flow):
        """Test agent voting with user override capability."""
        v13_flow.create_session()
        
        # Mock agent votes
        for i, agent in enumerate(v13_flow.discussing_agents):
            if i < 2:
                # First two vote yes
                agent.return_value = {
                    "messages": [Mock(content="Yes, I think we should conclude")]
                }
            else:
                # Last one votes no
                agent.return_value = {
                    "messages": [Mock(content="No, more discussion needed")]
                }
        
        # Test end topic poll
        state = {
            "active_topic": "Test Topic",
            "current_round": 3
        }
        
        result = v13_flow.nodes.end_topic_poll_node(state)
        
        assert "conclusion_vote" in result
        assert result["conclusion_vote"]["yes_votes"] == 2
        assert result["conclusion_vote"]["total_votes"] == 3
        assert result["conclusion_vote"]["passed"] is True  # Majority
        assert len(result["conclusion_vote"]["minority_voters"]) == 1
    
    def test_edge_conditions(self, v13_flow):
        """Test v1.3 conditional edges."""
        conditions = v13_flow.conditions
        
        # Test periodic stop condition
        state = {"current_round": 5}
        assert conditions.check_periodic_stop(state) == "periodic_stop"
        
        state = {"current_round": 3}
        assert conditions.check_periodic_stop(state) == "check_votes"
        
        # Test user forced conclusion
        state = {"user_forced_conclusion": True, "conclusion_vote": {"passed": False}}
        assert conditions.evaluate_conclusion_vote(state) == "conclude_topic"
        
        # Test agent session vote
        state = {"agents_vote_end_session": True}
        assert conditions.check_agent_session_vote(state) == "end_session"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])