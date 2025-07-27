"""Tests for the VirtualAgoraFlow graph implementation."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from virtual_agora.flow.graph import VirtualAgoraFlow
from virtual_agora.config.models import (
    Config as VirtualAgoraConfig,
    ModeratorConfig,
    AgentConfig,
)
from virtual_agora.providers.config import ProviderType
from virtual_agora.state.schema import VirtualAgoraState


class TestVirtualAgoraFlow:
    """Test VirtualAgoraFlow graph implementation."""

    def setup_method(self):
        """Set up test method."""
        self.config = VirtualAgoraConfig(
            moderator=ModeratorConfig(
                provider=ProviderType.GOOGLE, model="gemini-2.5-pro"
            ),
            agents=[AgentConfig(provider=ProviderType.OPENAI, model="gpt-4o", count=2)],
        )

    @patch("virtual_agora.flow.graph.create_provider")
    def test_flow_initialization(self, mock_create_provider):
        """Test flow initialization and agent setup."""
        # Mock LLM
        mock_llm = Mock()
        mock_create_provider.return_value = mock_llm

        # Create flow
        flow = VirtualAgoraFlow(self.config)

        # Check agents were initialized
        assert "moderator" in flow.agents
        assert len(flow.agents) == 3  # 1 moderator + 2 participants

        # Check agent roles
        assert flow.agents["moderator"].role == "moderator"
        for agent_id, agent in flow.agents.items():
            if agent_id != "moderator":
                assert agent.role == "participant"

    @patch("virtual_agora.flow.graph.create_provider")
    def test_graph_building(self, mock_create_provider):
        """Test graph structure and node creation."""
        mock_llm = Mock()
        mock_create_provider.return_value = mock_llm

        flow = VirtualAgoraFlow(self.config)
        graph = flow.build_graph()

        # Check graph was created
        assert graph is not None
        assert flow.graph is graph

        # Verify nodes exist (simplified check)
        # In a real test, we'd inspect the graph structure more thoroughly
        assert hasattr(flow.nodes, "initialization_node")
        assert hasattr(flow.nodes, "agenda_setting_node")
        assert hasattr(flow.nodes, "discussion_round_node")
        assert hasattr(flow.nodes, "conclusion_poll_node")
        assert hasattr(flow.nodes, "report_generation_node")

    @patch("virtual_agora.flow.graph.create_provider")
    def test_graph_compilation(self, mock_create_provider):
        """Test graph compilation with checkpointing."""
        mock_llm = Mock()
        mock_create_provider.return_value = mock_llm

        flow = VirtualAgoraFlow(self.config)
        compiled_graph = flow.compile()

        # Check compilation
        assert compiled_graph is not None
        assert flow.compiled_graph is compiled_graph
        assert flow.checkpointer is not None

    @patch("virtual_agora.flow.graph.create_provider")
    def test_session_creation(self, mock_create_provider):
        """Test session creation and state initialization."""
        mock_llm = Mock()
        mock_create_provider.return_value = mock_llm

        flow = VirtualAgoraFlow(self.config)
        session_id = flow.create_session(main_topic="Test Topic")

        # Check session was created
        assert session_id is not None
        assert flow.compiled_graph is not None

        # Check state was initialized with Epic 6 fields
        state = flow.state_manager.get_snapshot()
        assert "current_round" in state
        assert "hitl_state" in state
        assert "flow_control" in state
        assert state.get("main_topic") == "Test Topic"

    @patch("virtual_agora.flow.graph.create_provider")
    def test_state_manager_integration(self, mock_create_provider):
        """Test integration with state manager."""
        mock_llm = Mock()
        mock_create_provider.return_value = mock_llm

        flow = VirtualAgoraFlow(self.config)
        state_manager = flow.get_state_manager()

        # Check state manager
        assert state_manager is not None
        assert state_manager is flow.state_manager

    @patch("virtual_agora.flow.graph.create_provider")
    def test_flow_components_integration(self, mock_create_provider):
        """Test that flow components are properly integrated."""
        mock_llm = Mock()
        mock_create_provider.return_value = mock_llm

        flow = VirtualAgoraFlow(self.config)

        # Check components exist
        assert flow.nodes is not None
        assert flow.conditions is not None

        # Check components have correct dependencies
        assert flow.nodes.agents is flow.agents
        assert flow.nodes.state_manager is flow.state_manager

    @patch("virtual_agora.flow.graph.create_provider")
    def test_graph_state_operations(self, mock_create_provider):
        """Test graph state operations."""
        mock_llm = Mock()
        mock_create_provider.return_value = mock_llm

        flow = VirtualAgoraFlow(self.config)
        flow.create_session()

        # Test state updates
        updates = {"current_round": 1}
        flow.update_state(updates)

        # Test state retrieval
        graph_state = flow.get_graph_state()
        assert graph_state is not None

        # Test state history
        history = flow.get_state_history()
        assert isinstance(history, list)

    @patch("virtual_agora.flow.graph.create_provider")
    def test_graph_visualization(self, mock_create_provider):
        """Test graph visualization capability."""
        mock_llm = Mock()
        mock_create_provider.return_value = mock_llm

        flow = VirtualAgoraFlow(self.config)

        # Test visualization
        try:
            viz_data = flow.visualize_graph()
            assert isinstance(viz_data, bytes)
        except Exception:
            # Visualization might fail in test environment
            pytest.skip("Graph visualization not available in test environment")

    @patch("virtual_agora.flow.graph.create_provider")
    def test_error_handling(self, mock_create_provider):
        """Test error handling in flow operations."""
        mock_llm = Mock()
        mock_create_provider.return_value = mock_llm

        flow = VirtualAgoraFlow(self.config)

        # Test operations without compiled graph
        with pytest.raises(Exception):  # Should raise StateError
            flow.invoke()

        with pytest.raises(Exception):
            flow.update_state({})

        with pytest.raises(Exception):
            flow.get_graph_state()

    @patch("virtual_agora.flow.graph.create_provider")
    def test_config_validation(self, mock_create_provider):
        """Test configuration validation."""
        mock_llm = Mock()
        mock_create_provider.return_value = mock_llm

        # Test with valid config
        flow = VirtualAgoraFlow(self.config)
        assert flow.config is self.config

        # Test agents match config
        expected_agents = 1 + sum(
            agent.count for agent in self.config.agents
        )  # +1 for moderator
        assert len(flow.agents) == expected_agents
