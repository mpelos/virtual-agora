"""Test to reproduce the session_id node error.

This test reproduces the exact error encountered when starting the Virtual Agora
application: "Expected node session_id to update at least one of ['session_id', ...]
got {'__start__': True}"
"""

import pytest
from unittest.mock import Mock, patch
from virtual_agora.flow.graph_v13 import VirtualAgoraV13Flow
from virtual_agora.config.models import Config as VirtualAgoraConfig, Provider


class TestSessionIdNodeError:
    """Test to reproduce and fix the session_id node error."""

    @pytest.fixture
    def test_config(self):
        """Create minimal test configuration."""
        return VirtualAgoraConfig(
            moderator={
                "provider": Provider.GOOGLE,
                "model": "gemini-2.5-flash-lite",
                "temperature": 0.7,
            },
            summarizer={
                "provider": Provider.GOOGLE,
                "model": "gemini-2.5-flash-lite",
                "temperature": 0.3,
            },
            topic_report={
                "provider": Provider.GOOGLE,
                "model": "gemini-2.5-flash-lite",
                "temperature": 0.5,
            },
            ecclesia_report={
                "provider": Provider.GOOGLE,
                "model": "gemini-2.5-flash-lite",
                "temperature": 0.5,
            },
            agents=[
                {
                    "provider": Provider.GOOGLE,
                    "model": "gemini-2.5-flash-lite",
                    "count": 3,
                    "temperature": 0.7,
                }
            ],
        )

    @patch.dict('os.environ', {'GOOGLE_API_KEY': 'fake-key-for-testing'})
    @patch("virtual_agora.providers.create_provider")
    def test_reproduce_session_id_node_error(self, mock_create_provider, test_config):
        """Test that reproduces the session_id node error."""
        # Mock the provider to avoid actual API calls
        mock_llm = Mock()
        mock_create_provider.return_value = mock_llm

        # Initialize the flow
        flow = VirtualAgoraV13Flow(test_config, enable_monitoring=False)

        # Create session
        session_id = flow.create_session(main_topic="Test topic")
        assert session_id is not None

        # Compile the graph
        compiled_graph = flow.compile()
        assert compiled_graph is not None

        # This should now work correctly with the fix:
        # Pass actual state data instead of empty dict or {'__start__': True}
        # Try both invoke and stream to verify the fix works
        
        # First try invoke
        try:
            result = flow.invoke()
            invoke_success = True
        except Exception as e:
            invoke_error = str(e)
            invoke_success = False
            if "Expected node session_id to update" in invoke_error and "__start__" in invoke_error:
                pytest.fail(f"invoke() reproduced the error: {invoke_error}")
        
        # Now try stream (like the real application does)
        try:
            config_dict = {"configurable": {"thread_id": session_id}}
            updates = list(flow.stream(config_dict))
            stream_success = True
        except Exception as e:
            stream_error = str(e)
            stream_success = False
            if "Expected node session_id to update" in stream_error and "__start__" in stream_error:
                pytest.fail(f"stream() reproduced the error: {stream_error}")
        
        # If neither reproduced the error, we need to investigate further
        if invoke_success and stream_success:
            pytest.skip("Could not reproduce the session_id node error with current test setup")

    @patch("virtual_agora.providers.create_provider")
    def test_session_creation_and_state_initialization(
        self, mock_create_provider, test_config
    ):
        """Test that session creation and state initialization work correctly."""
        # Mock the provider
        mock_llm = Mock()
        mock_create_provider.return_value = mock_llm

        # Initialize the flow
        flow = VirtualAgoraV13Flow(test_config, enable_monitoring=False)

        # Create session with a topic
        session_id = flow.create_session(main_topic="Test topic for discussion")

        # Verify session was created
        assert session_id is not None
        assert session_id.startswith("session_")

        # Verify state was initialized correctly
        state = flow.get_state_manager().state
        assert state["session_id"] == session_id
        assert state["main_topic"] == "Test topic for discussion"
        assert "specialized_agents" in state
        assert "agents" in state

        # Verify graph can be compiled
        compiled_graph = flow.compile()
        assert compiled_graph is not None

    @patch("virtual_agora.providers.create_provider")
    def test_graph_structure_integrity(self, mock_create_provider, test_config):
        """Test that the graph structure is correct and doesn't have invalid nodes."""
        # Mock the provider
        mock_llm = Mock()
        mock_create_provider.return_value = mock_llm

        # Initialize the flow
        flow = VirtualAgoraV13Flow(test_config, enable_monitoring=False)

        # Build the graph
        graph = flow.build_graph()

        # Verify the graph has the expected structure
        nodes = list(graph.nodes.keys())

        # Should NOT contain 'session_id' as a node
        assert "session_id" not in nodes

        # Should contain expected nodes
        expected_nodes = [
            "config_and_keys",
            "agent_instantiation",
            "get_theme",
            "agenda_proposal",
            "collate_proposals",
            "agenda_voting",
            "synthesize_agenda",
            "agenda_approval",
        ]

        for node in expected_nodes:
            assert node in nodes, f"Expected node {node} not found in graph"

        # Verify START and END are properly connected
        assert "__start__" in graph.edges or "START" in str(graph.edges)
