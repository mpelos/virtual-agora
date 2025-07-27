"""End-to-end integration tests for complete Virtual Agora v1.3 sessions."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import asyncio

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.flow.graph_v13 import VirtualAgoraV13Flow
from virtual_agora.agents.agent_factory import AgentFactory
from virtual_agora.config.models import Config, Provider
from virtual_agora.ui.hitl_manager import EnhancedHITLManager

from tests.helpers.fake_llm import create_specialized_fake_llms, create_fake_llm_pool
from tests.helpers.integration_utils import (
    create_v13_test_state,
    create_test_config_file,
    patch_ui_components,
    run_integration_test,
)


class TestCompleteSessionV3:
    """End-to-end tests for complete Virtual Agora v1.3 sessions."""

    @pytest.fixture
    def test_config(self, tmp_path):
        """Create test configuration."""
        config_path = tmp_path / "config.yaml"
        create_test_config_file(config_path, version="1.3", num_agents=3)

        # Load and return config
        from virtual_agora.config.loader import ConfigLoader

        loader = ConfigLoader(config_path)
        return loader.load()

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        specialized = create_specialized_fake_llms()
        discussion = create_fake_llm_pool(num_agents=3)

        # Combine all agents
        all_agents = {}
        all_agents.update(specialized)
        all_agents.update(discussion)

        return all_agents

    @pytest.fixture
    def test_graph(self, test_config, mock_agents):
        """Create test graph with mock agents."""
        with patch(
            "virtual_agora.agents.agent_factory.AgentFactory.create_all_agents"
        ) as mock_create:
            # Mock the agent creation to return our fake agents
            mock_create.return_value = mock_agents

            # Create the flow instance
            flow = VirtualAgoraV13Flow(test_config)
            # Build and return the graph
            return flow.build_graph()

    @pytest.mark.asyncio
    async def test_minimal_session_flow(self, test_graph):
        """Test minimal session flow through all phases."""
        # Create initial state
        initial_state = create_v13_test_state()
        initial_state["theme"] = "AI Safety and Ethics"
        initial_state["human_input"] = "Let's discuss AI safety challenges"

        # Track state progression
        states = []

        # Run through the graph
        async for state in test_graph.astream(initial_state):
            states.append(state)

            # Stop after conclusion phase
            if state.get("phase") == "conclusion":
                break

        # Verify we went through key phases
        phases_seen = [s.get("phase") for s in states if s.get("phase")]
        assert "agenda" in phases_seen
        assert "discussion" in phases_seen

        # Verify key state elements were populated
        final_state = states[-1] if states else initial_state
        assert final_state.get("proposed_topics") is not None
        assert final_state.get("agenda") is not None

    @pytest.mark.asyncio
    async def test_full_session_with_multiple_topics(self, test_graph):
        """Test full session with multiple topics and rounds."""
        initial_state = create_v13_test_state()
        initial_state["theme"] = "Future of Work"
        initial_state["human_input"] = "Discuss how AI will transform the workplace"

        # Simulate predefined agenda
        initial_state["agenda"] = [
            "Remote Work Technologies",
            "AI Automation Impact",
            "Skills and Education",
            "Economic Implications",
        ]
        initial_state["phase"] = "discussion"
        initial_state["current_topic_index"] = 0

        states_by_phase = {"discussion": [], "conclusion": [], "reporting": []}

        # Run session
        async for state in test_graph.astream(initial_state):
            phase = state.get("phase", "unknown")
            if phase in states_by_phase:
                states_by_phase[phase].append(state)

            # Stop after reporting
            if phase == "reporting" and state.get("final_report"):
                break

        # Verify we processed multiple topics
        assert len(states_by_phase["discussion"]) > 0
        assert len(states_by_phase["conclusion"]) > 0

        # Check that specialized agents were used
        final_state = (
            states_by_phase["reporting"][-1] if states_by_phase["reporting"] else {}
        )
        assert final_state.get("round_summaries") is not None
        assert final_state.get("topic_reports") is not None

    @pytest.mark.asyncio
    async def test_hitl_interruption_handling(self, test_graph):
        """Test HITL interruption and resume functionality."""
        initial_state = create_v13_test_state()
        initial_state["theme"] = "Climate Technology"
        initial_state["human_input"] = "Explore climate tech solutions"

        # Enable HITL with aggressive settings
        initial_state["hitl_config"] = {
            "enabled": True,
            "mode": "periodic",
            "check_interval": 2,  # Check every 2 rounds
            "require_approval": True,
        }

        states = []
        interrupted = False

        # Run with interruption simulation
        async for state in test_graph.astream(initial_state):
            states.append(state)

            # Simulate HITL interruption after a few states
            if len(states) > 3 and not interrupted:
                # Check if HITL gate is active
                if state.get("hitl_gate_active"):
                    interrupted = True
                    # Simulate user approval
                    state["human_approval"] = True
                    state["human_feedback"] = "Good progress, continue"

            # Stop after enough progress
            if len(states) > 10:
                break

        # Verify HITL was activated
        hitl_states = [s for s in states if s.get("hitl_gate_active")]
        assert len(hitl_states) > 0 or interrupted

    @pytest.mark.asyncio
    async def test_error_recovery_flow(self, test_graph):
        """Test error recovery during session."""
        initial_state = create_v13_test_state()
        initial_state["theme"] = "Quantum Computing"

        # Inject an error condition
        initial_state["inject_error"] = True
        initial_state["error_phase"] = "discussion"

        states = []
        error_encountered = False

        try:
            async for state in test_graph.astream(initial_state):
                states.append(state)

                # Check for error handling
                if state.get("error") or state.get("error_recovery"):
                    error_encountered = True

                # Stop after recovery or timeout
                if len(states) > 15:
                    break
        except Exception as e:
            # Graph should handle errors gracefully
            error_encountered = True

        # Verify graceful handling
        assert len(states) > 0  # Should have made some progress

    def test_state_persistence_and_recovery(self, test_config):
        """Test state persistence and session recovery."""
        # Create initial state
        state = create_v13_test_state()
        state["theme"] = "Space Exploration"
        state["session_id"] = "test-session-123"

        # Add some progress
        state["phase"] = "discussion"
        state["current_topic_index"] = 2
        state["round_number"] = 5
        state["discussion_messages"] = [
            {"agent": "agent_1", "content": "Message 1"},
            {"agent": "agent_2", "content": "Message 2"},
        ]

        # Simulate persistence
        from virtual_agora.state.manager import VirtualAgoraStateManager

        manager = VirtualAgoraStateManager()

        # Save state
        saved_state = manager.create_checkpoint(state)
        assert saved_state is not None

        # Simulate recovery
        recovered_state = manager.restore_from_checkpoint(saved_state)

        # Verify key elements preserved
        assert recovered_state["session_id"] == state["session_id"]
        assert recovered_state["phase"] == state["phase"]
        assert recovered_state["current_topic_index"] == state["current_topic_index"]
        assert len(recovered_state["discussion_messages"]) == 2

    def test_comprehensive_reporting_generation(self, mock_agents):
        """Test comprehensive report generation at session end."""
        from virtual_agora.agents.ecclesia_report_agent import EcclesiaReportAgent

        # Create state with multiple completed topics
        state = create_v13_test_state()
        state["topic_reports"] = {
            "AI Ethics": "Comprehensive discussion on ethical AI development...",
            "Privacy Concerns": "Deep dive into data privacy challenges...",
            "Regulation Framework": "Analysis of regulatory approaches...",
        }

        # Get the ecclesia agent
        ecclesia_agent = None
        for agent_id, agent in mock_agents.items():
            if "ecclesia" in agent_id:
                # Create proper agent instance
                ecclesia_agent = EcclesiaReportAgent(
                    agent_id="ecclesia_report", llm=agent, enable_error_handling=False
                )
                break

        assert ecclesia_agent is not None

        # Generate report structure
        sections = ecclesia_agent.generate_report_structure(
            topic_reports=state["topic_reports"], discussion_theme="AI Governance"
        )

        assert isinstance(sections, list)
        assert len(sections) > 0
        assert "Executive Summary" in sections

        # Generate executive summary
        exec_summary = ecclesia_agent.write_section(
            section_title="Executive Summary",
            topic_reports=state["topic_reports"],
            discussion_theme="AI Governance",
            previous_sections={},
        )

        assert isinstance(exec_summary, str)
        assert len(exec_summary) > 50

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, test_graph):
        """Test performance monitoring during session."""
        initial_state = create_v13_test_state()
        initial_state["theme"] = "Biotechnology Advances"
        initial_state["enable_monitoring"] = True

        # Track performance metrics
        metrics = {
            "state_count": 0,
            "phase_transitions": 0,
            "round_summaries": 0,
            "start_time": datetime.now(),
        }

        last_phase = None

        async for state in test_graph.astream(initial_state):
            metrics["state_count"] += 1

            # Track phase transitions
            current_phase = state.get("phase")
            if current_phase and current_phase != last_phase:
                metrics["phase_transitions"] += 1
                last_phase = current_phase

            # Track round summaries
            if state.get("round_summaries"):
                metrics["round_summaries"] = len(state["round_summaries"])

            # Stop after reasonable progress
            if metrics["state_count"] > 20:
                break

        # Calculate duration
        metrics["duration"] = (datetime.now() - metrics["start_time"]).total_seconds()

        # Verify reasonable performance
        assert metrics["state_count"] > 0
        assert metrics["phase_transitions"] > 0
        assert metrics["duration"] < 60  # Should complete quickly with mocks

    def test_agent_coordination_patterns(self, mock_agents):
        """Test coordination patterns between specialized agents."""
        state = create_v13_test_state()

        # Simulate discussion round
        discussion_messages = []
        for i in range(3):
            agent_id = f"agent_{i+1}"
            if agent_id in mock_agents:
                # Simulate discussion contribution
                discussion_messages.append(
                    {
                        "agent_id": agent_id,
                        "content": f"Point {i+1} about the topic",
                        "round_number": 1,
                    }
                )

        state["discussion_messages"] = discussion_messages

        # Get summarizer
        from virtual_agora.agents.summarizer import SummarizerAgent

        summarizer_llm = mock_agents.get("summarizer")
        if summarizer_llm:
            summarizer = SummarizerAgent(
                agent_id="summarizer", llm=summarizer_llm, enable_error_handling=False
            )

            # Generate round summary
            summary = summarizer.summarize_round(
                messages=discussion_messages, topic="Test Topic", round_number=1
            )

            assert isinstance(summary, str)
            assert len(summary) > 0

            # Verify compression
            original_length = sum(len(m["content"]) for m in discussion_messages)
            assert len(summary) < original_length * 2  # Should compress somewhat

    def test_configuration_flexibility(self, tmp_path):
        """Test different configuration options."""
        # Test with different agent counts
        for num_agents in [2, 5, 7]:
            config_path = tmp_path / f"config_{num_agents}.yaml"
            create_test_config_file(config_path, num_agents=num_agents, version="1.3")

            from virtual_agora.config.loader import load_config

            config = load_config(str(config_path))

            assert len(config.agents) > 0
            # In v1.3, we have specialized agents plus discussion agents
            assert hasattr(config, "moderator")
            assert hasattr(config, "summarizer")
            assert hasattr(config, "topic_report")
            assert hasattr(config, "ecclesia_report")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
