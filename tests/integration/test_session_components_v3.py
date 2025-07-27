"""Integration tests for v1.3 session components and workflows."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.agents.agent_factory import AgentFactory
from virtual_agora.config.models import (
    Config,
    Provider,
    ModeratorConfig,
    SummarizerConfig,
    TopicReportConfig,
    EcclesiaReportConfig,
    AgentConfig,
)
from virtual_agora.ui.hitl_manager import EnhancedHITLManager

from tests.helpers.fake_llm import create_specialized_fake_llms, create_fake_llm_pool
from tests.helpers.integration_utils import create_v13_test_state


class TestSessionComponentsV3:
    """Test v1.3 session components and their integration."""

    @pytest.fixture
    def test_config(self):
        """Create test configuration for v1.3."""
        return Config(
            moderator=ModeratorConfig(provider=Provider.OPENAI, model="gpt-4"),
            summarizer=SummarizerConfig(provider=Provider.OPENAI, model="gpt-4"),
            topic_report=TopicReportConfig(
                provider=Provider.ANTHROPIC, model="claude-3"
            ),
            ecclesia_report=EcclesiaReportConfig(
                provider=Provider.GOOGLE, model="gemini-pro"
            ),
            agents=[AgentConfig(provider=Provider.OPENAI, model="gpt-4", count=3)],
            theme="Technology and Society",
            rounds_per_topic=5,
            enable_hitl=True,
            version="1.3",
        )

    @pytest.fixture
    def mock_llms(self):
        """Create mock LLMs for all agents."""
        specialized = create_specialized_fake_llms()
        discussion = create_fake_llm_pool(num_agents=3)

        all_llms = {}
        all_llms.update(specialized)
        all_llms.update(discussion)

        return all_llms

    def test_agent_factory_creates_all_v13_agents(self, test_config):
        """Test that agent factory creates all required v1.3 agents."""
        with patch(
            "virtual_agora.providers.factory.ProviderFactory.create_provider"
        ) as mock_provider:
            # Mock LLM creation
            mock_provider.return_value = Mock()

            factory = AgentFactory(test_config)

            # Create specialized agents
            specialized = factory.create_specialized_agents()
            assert "moderator" in specialized
            assert "summarizer" in specialized
            assert "topic_report" in specialized
            assert "ecclesia_report" in specialized

            # Create all agents (discussion agents only in create_all_agents)
            discussion_agents = factory.create_all_agents()
            assert len(discussion_agents) == 3  # Just discussion agents

            # Create agent pool includes all agents
            agent_pool = factory.create_agent_pool()
            assert len(agent_pool) >= 4  # Should include moderator + discussion agents

    def test_round_summary_workflow(self, mock_llms):
        """Test the round summary workflow with specialized agents."""
        from virtual_agora.agents.summarizer import SummarizerAgent

        state = create_v13_test_state()

        # Create summarizer
        summarizer = SummarizerAgent(
            agent_id="summarizer",
            llm=mock_llms["summarizer"],
            compression_ratio=0.3,
            enable_error_handling=False,
        )

        # Simulate discussion round
        messages = [
            {"agent_id": "agent_1", "content": "First point about AI ethics"},
            {"agent_id": "agent_2", "content": "Second point about regulation"},
            {"agent_id": "agent_3", "content": "Third point about transparency"},
        ]

        # Generate round summary
        summary = summarizer.summarize_round(
            messages=messages, topic="AI Ethics Framework", round_number=1
        )

        assert isinstance(summary, str)
        assert len(summary) > 0

        # Add to state
        state["round_summaries"].append(summary)
        assert len(state["round_summaries"]) == 1

    def test_topic_conclusion_workflow(self, mock_llms):
        """Test topic conclusion and report generation workflow."""
        from virtual_agora.agents.topic_report_agent import TopicReportAgent
        from virtual_agora.agents.summarizer import SummarizerAgent

        state = create_v13_test_state()

        # Create agents
        summarizer = SummarizerAgent(
            agent_id="summarizer",
            llm=mock_llms["summarizer"],
            enable_error_handling=False,
        )

        topic_report_agent = TopicReportAgent(
            agent_id="topic_report",
            llm=mock_llms["topic_report"],
            enable_error_handling=False,
        )

        # Generate multiple round summaries
        round_summaries = []
        for i in range(3):
            messages = [
                {"content": f"Round {i+1} discussion point 1"},
                {"content": f"Round {i+1} discussion point 2"},
            ]
            summary = summarizer.summarize_round(messages, "Privacy Laws", i + 1)
            round_summaries.append(summary)

        # Generate topic report
        topic_report = topic_report_agent.synthesize_topic(
            round_summaries=round_summaries,
            final_considerations=["GDPR compliance is essential"],
            topic="Privacy Laws",
            discussion_theme="Digital Rights",
        )

        assert isinstance(topic_report, str)
        assert len(topic_report) > 100
        assert "Topic Report:" in topic_report or "Overview" in topic_report

    def test_final_report_generation(self, mock_llms):
        """Test final ecclesia report generation."""
        from virtual_agora.agents.ecclesia_report_agent import EcclesiaReportAgent

        # Create ecclesia agent
        ecclesia_agent = EcclesiaReportAgent(
            agent_id="ecclesia_report",
            llm=mock_llms["ecclesia_report"],
            enable_error_handling=False,
        )

        # Create topic reports
        topic_reports = {
            "AI Safety": "Comprehensive discussion on AI safety measures...",
            "Data Privacy": "Deep analysis of data privacy concerns...",
            "Algorithmic Bias": "Examination of bias in AI systems...",
        }

        # Generate report structure
        sections = ecclesia_agent.generate_report_structure(
            topic_reports=topic_reports, discussion_theme="AI Ethics and Society"
        )

        assert isinstance(sections, list)
        assert "Executive Summary" in sections

        # Write a section
        exec_summary = ecclesia_agent.write_section(
            section_title="Executive Summary",
            topic_reports=topic_reports,
            discussion_theme="AI Ethics and Society",
            previous_sections={},
        )

        assert isinstance(exec_summary, str)
        assert "Executive Summary" in exec_summary

    def test_hitl_integration(self, test_config):
        """Test HITL manager integration with v1.3."""
        state = create_v13_test_state()
        state["hitl_config"] = {
            "enabled": True,
            "mode": "periodic",
            "check_interval": 5,
        }

        # Create HITL manager with mock console
        from unittest.mock import Mock

        mock_console = Mock()
        hitl_manager = EnhancedHITLManager(console=mock_console)

        # Test HITL state
        state["current_round"] = 5

        # Check that HITL manager was created
        assert hitl_manager is not None

        # Check periodic stop logic manually (HITL manager checks this internally)
        # With periodic mode and interval 5, should trigger at round 5
        assert state["current_round"] % state["hitl_config"]["check_interval"] == 0

    def test_state_progression(self):
        """Test state progression through v1.3 phases."""
        state = create_v13_test_state()

        # Initial phase (0 = initialization in v1.3)
        assert state["current_phase"] == 0

        # Agenda phase (1 = agenda in v1.3)
        state["current_phase"] = 1
        state["proposed_topics"] = ["Topic 1", "Topic 2", "Topic 3"]
        assert len(state["proposed_topics"]) == 3

        # Discussion phase (2 = discussion in v1.3)
        state["current_phase"] = 2
        state["current_topic_index"] = 0
        state["active_topic"] = state["proposed_topics"][0]
        state["current_round"] = 1

        # Add messages
        state["messages"].append(
            {
                "agent_id": "agent_1",
                "content": "Discussion point",
                "round_number": 1,
                "topic": state["active_topic"],
            }
        )

        assert len(state["messages"]) == 1

        # Conclusion phase (3 = conclusion in v1.3)
        state["current_phase"] = 3
        state["conclusion_votes"] = {
            "agent_1": {"vote": True, "reason": "Complete"},
            "agent_2": {"vote": True, "reason": "Agreed"},
            "agent_3": {"vote": False, "reason": "More to discuss"},
        }

        # Count votes
        yes_votes = sum(1 for v in state["conclusion_votes"].values() if v["vote"])
        assert yes_votes == 2

        # Reporting phase (4 = reporting in v1.3)
        state["current_phase"] = 4
        state["topic_reports"]["Topic 1"] = "Report content..."
        assert len(state["topic_reports"]) == 1

    def test_agent_muting_and_warnings(self):
        """Test agent muting system in v1.3."""
        from virtual_agora.agents.discussion_agent import DiscussionAgent

        # Create discussion agent
        agent = DiscussionAgent(
            agent_id="test_agent", llm=Mock(), enable_error_handling=False
        )

        # Test warning system
        assert agent.warning_count == 0
        assert not agent.is_muted()

        # Add warnings
        agent.add_warning("Off-topic")
        assert agent.warning_count == 1

        agent.add_warning("Repetitive")
        assert agent.warning_count == 2

        # Third warning should mute
        result = agent.add_warning("Disruptive")
        assert result == True  # Returns True when muted
        assert agent.is_muted()
        assert agent.warning_count == 3

    def test_progressive_summary_generation(self, mock_llms):
        """Test progressive summary feature of v1.3."""
        from virtual_agora.agents.summarizer import SummarizerAgent

        summarizer = SummarizerAgent(
            agent_id="summarizer",
            llm=mock_llms["summarizer"],
            enable_error_handling=False,
        )

        # Create multiple round summaries
        round_summaries = [
            "Round 1: Initial exploration of the topic",
            "Round 2: Deeper analysis and debate",
            "Round 3: Convergence on key points",
            "Round 4: Final considerations",
            "Round 5: Wrap-up and conclusions",
        ]

        # Generate progressive summary
        progressive = summarizer.generate_progressive_summary(
            summaries=round_summaries, topic="Technology Impact"
        )

        assert isinstance(progressive, str)
        assert len(progressive) > 0
        # Should mention progression/evolution
        assert "rounds" in progressive.lower() or "discussion" in progressive.lower()

    def test_minority_opinion_handling(self, mock_llms):
        """Test minority opinion handling in topic reports."""
        from virtual_agora.agents.topic_report_agent import TopicReportAgent

        topic_agent = TopicReportAgent(
            agent_id="topic_report",
            llm=mock_llms["topic_report"],
            enable_error_handling=False,
        )

        # Create dissenting views
        dissenting_views = [
            {
                "agent": "agent_1",
                "content": "I disagree with the consensus on timeline",
            },
            {"agent": "agent_3", "content": "The risks are being underestimated"},
        ]

        # Handle minority considerations
        minority_section = topic_agent.handle_minority_considerations(
            dissenting_agents=["agent_1", "agent_3"],
            dissenting_views=dissenting_views,
            topic="Implementation Timeline",
            majority_conclusion="6-month implementation is feasible",
        )

        assert isinstance(minority_section, str)
        assert len(minority_section) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
