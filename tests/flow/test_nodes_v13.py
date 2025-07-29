"""Unit tests for v1.3 node implementations."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.state.manager import StateManager
from virtual_agora.flow.nodes_v13 import V13FlowNodes
from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.agents.moderator import ModeratorAgent
from virtual_agora.agents.summarizer import SummarizerAgent
from virtual_agora.agents.topic_report_agent import TopicReportAgent
from virtual_agora.agents.ecclesia_report_agent import EcclesiaReportAgent


@pytest.fixture
def mock_specialized_agents():
    """Create mock specialized agents."""
    agents = {}

    # Create moderator mock
    moderator = Mock(spec=ModeratorAgent)
    moderator.agent_id = "moderator"
    agents["moderator"] = moderator

    # Create summarizer mock
    summarizer = Mock(spec=SummarizerAgent)
    summarizer.agent_id = "summarizer"
    summarizer.summarize_topic_conclusion = Mock(
        return_value="Mock topic conclusion summary"
    )
    agents["summarizer"] = summarizer

    # Create topic report mock
    topic_report = Mock(spec=TopicReportAgent)
    topic_report.agent_id = "topic_report"
    agents["topic_report"] = topic_report

    # Create ecclesia report mock
    ecclesia_report = Mock(spec=EcclesiaReportAgent)
    ecclesia_report.agent_id = "ecclesia_report"
    agents["ecclesia_report"] = ecclesia_report

    return agents


@pytest.fixture
def mock_discussing_agents():
    """Create mock discussing agents."""
    agents = []
    for i in range(3):
        agent = Mock(spec=LLMAgent)
        agent.agent_id = f"test_agent_{i}"
        agent.role = "participant"
        agents.append(agent)
    return agents


@pytest.fixture
def mock_state_manager():
    """Create mock state manager."""
    manager = Mock(spec=StateManager)
    manager.state = {"session_id": "test_session"}
    return manager


@pytest.fixture
def flow_nodes(mock_specialized_agents, mock_discussing_agents, mock_state_manager):
    """Create V13FlowNodes instance with mocks."""
    return V13FlowNodes(
        specialized_agents=mock_specialized_agents,
        discussing_agents=mock_discussing_agents,
        state_manager=mock_state_manager,
    )


class TestPhase0Nodes:
    """Test Phase 0: Initialization nodes."""

    def test_config_and_keys_node(self, flow_nodes):
        """Test configuration and keys validation."""
        state = {}
        result = flow_nodes.config_and_keys_node(state)

        assert result["config_loaded"] is True
        assert "initialization_timestamp" in result
        assert result["system_status"] == "ready"

    def test_agent_instantiation_node(self, flow_nodes):
        """Test agent instantiation recording."""
        state = {}
        result = flow_nodes.agent_instantiation_node(state)

        assert "specialized_agents" in result
        assert "agents" in result
        assert "speaking_order" in result
        assert len(result["speaking_order"]) == 3
        assert result["agent_count"] == 3
        assert result["current_phase"] == 0

    @patch("virtual_agora.flow.nodes_v13.get_initial_topic")
    def test_get_theme_node(self, mock_get_topic, flow_nodes):
        """Test theme collection from user."""
        mock_get_topic.return_value = "Test Theme"

        state = {}
        result = flow_nodes.get_theme_node(state)

        assert result["main_topic"] == "Test Theme"
        assert "hitl_state" in result
        assert result["hitl_state"]["approval_history"][0]["type"] == "theme_input"

    def test_get_theme_node_already_set(self, flow_nodes):
        """Test theme node when theme already exists."""
        state = {"main_topic": "Existing Theme"}
        result = flow_nodes.get_theme_node(state)

        assert result == {}  # No updates needed


class TestPhase1Nodes:
    """Test Phase 1: Agenda Setting nodes."""

    def test_agenda_proposal_node(self, flow_nodes, mock_discussing_agents):
        """Test topic proposal collection."""
        state = {"main_topic": "Test Theme"}

        # Mock agent responses
        for agent in mock_discussing_agents:
            agent.return_value = {
                "messages": [Mock(content="Topic 1\nTopic 2\nTopic 3")]
            }

        result = flow_nodes.agenda_proposal_node(state)

        assert "proposed_topics" in result
        assert len(result["proposed_topics"]) == 3
        assert result["current_phase"] == 1

        # Verify each agent was called
        for agent in mock_discussing_agents:
            agent.assert_called_once()

    def test_collate_proposals_node(self, flow_nodes, mock_specialized_agents):
        """Test proposal collation by moderator."""
        state = {
            "proposed_topics": [
                {"agent_id": "agent1", "proposals": "Topic A\nTopic B"},
                {"agent_id": "agent2", "proposals": "Topic B\nTopic C"},
            ]
        }

        mock_moderator = mock_specialized_agents["moderator"]
        mock_moderator.collect_proposals.return_value = {
            "unique_topics": ["Topic A", "Topic B", "Topic C"]
        }

        result = flow_nodes.collate_proposals_node(state)

        assert result["collated_topics"] == ["Topic A", "Topic B", "Topic C"]
        assert result["total_unique_topics"] == 3
        mock_moderator.collect_proposals.assert_called_once()

    def test_synthesize_agenda_node(self, flow_nodes, mock_specialized_agents):
        """Test agenda synthesis from votes."""
        state = {
            "agenda_votes": [
                {"agent_id": "agent1", "vote": "I prefer Topic A first"},
                {"agent_id": "agent2", "vote": "Topic B should be first"},
            ],
            "collated_topics": ["Topic A", "Topic B", "Topic C"],
        }

        mock_moderator = mock_specialized_agents["moderator"]
        mock_moderator.synthesize_agenda.return_value = {
            "proposed_agenda": ["Topic B", "Topic A", "Topic C"]
        }

        result = flow_nodes.synthesize_agenda_node(state)

        assert result["proposed_agenda"] == ["Topic B", "Topic A", "Topic C"]
        assert result["agenda_synthesis_complete"] is True
        mock_moderator.synthesize_agenda.assert_called_once()


class TestPhase2Nodes:
    """Test Phase 2: Discussion Loop nodes."""

    def test_announce_item_node(self, flow_nodes, mock_specialized_agents):
        """Test topic announcement."""
        state = {"topic_queue": ["Topic A", "Topic B"], "main_topic": "Test Theme"}

        result = flow_nodes.announce_item_node(state)

        assert result["active_topic"] == "Topic A"
        assert result["current_round"] == 0
        assert "topic_start_time" in result
        assert "last_announcement" in result

    def test_discussion_round_node(
        self, flow_nodes, mock_discussing_agents, mock_specialized_agents
    ):
        """Test discussion round execution."""
        state = {
            "active_topic": "Topic A",
            "main_topic": "Test Theme",
            "speaking_order": ["test_agent_0", "test_agent_1", "test_agent_2"],
            "current_round": 0,
            "round_summaries": [],
            "messages": [],
        }

        # Mock agent responses
        for agent in mock_discussing_agents:
            agent.return_value = {
                "messages": [Mock(content="This is my thoughtful response")]
            }

        # Mock moderator relevance check
        mock_moderator = mock_specialized_agents["moderator"]
        mock_moderator.evaluate_message_relevance.return_value = {
            "is_relevant": True,
            "relevance_score": 0.9,
        }

        result = flow_nodes.discussion_round_node(state)

        assert result["current_round"] == 1
        assert len(result["messages"]) == 3
        assert result["rounds_per_topic"]["Topic A"] == 1

        # Verify speaking order (no rotation on round 1)
        assert result["speaking_order"][0] == "test_agent_0"  # No rotation yet

    def test_end_topic_poll_node(self, flow_nodes, mock_discussing_agents):
        """Test topic conclusion polling."""
        state = {"active_topic": "Topic A", "current_round": 3}

        # Mock different votes
        mock_discussing_agents[0].return_value = {
            "messages": [Mock(content="Yes, we should conclude")]
        }
        mock_discussing_agents[1].return_value = {
            "messages": [Mock(content="Yes, I agree")]
        }
        mock_discussing_agents[2].return_value = {
            "messages": [Mock(content="No, more discussion needed")]
        }

        result = flow_nodes.end_topic_poll_node(state)

        assert "topic_conclusion_votes" in result
        assert "conclusion_vote" in result
        assert result["conclusion_vote"]["yes_votes"] == 2
        assert result["conclusion_vote"]["total_votes"] == 3
        assert result["conclusion_vote"]["passed"] is True
        assert result["conclusion_vote"]["minority_voters"] == ["test_agent_2"]


class TestPhase3Nodes:
    """Test Phase 3: Topic Conclusion nodes."""

    def test_final_considerations_node_vote_based(
        self, flow_nodes, mock_discussing_agents
    ):
        """Test final considerations for minority voters."""
        state = {
            "active_topic": "Topic A",
            "user_forced_conclusion": False,
            "conclusion_vote": {"minority_voters": ["test_agent_2"]},
        }

        # Only minority agent should be called
        mock_discussing_agents[2].return_value = {
            "messages": [Mock(content="My final thoughts...")]
        }

        result = flow_nodes.final_considerations_node(state)

        assert len(result["final_considerations"]) == 1
        assert result["final_considerations"][0]["agent_id"] == "test_agent_2"

        # Verify only minority agent was called
        mock_discussing_agents[0].assert_not_called()
        mock_discussing_agents[1].assert_not_called()
        mock_discussing_agents[2].assert_called_once()

    def test_topic_report_generation_node(self, flow_nodes, mock_specialized_agents):
        """Test topic report generation."""
        state = {
            "active_topic": "Topic A",
            "main_topic": "Test Theme",
            "round_summaries": [
                {"topic": "Topic A", "summary": "Round 1 summary"},
                {"topic": "Topic A", "summary": "Round 2 summary"},
            ],
            "final_considerations": [
                {"consideration": "Final thought 1"},
                {"consideration": "Final thought 2"},
            ],
        }

        mock_topic_agent = mock_specialized_agents["topic_report"]
        mock_topic_agent.synthesize_topic.return_value = "Comprehensive topic report"

        result = flow_nodes.topic_report_generation_node(state)

        assert result["topic_summaries"]["Topic A"] == "Comprehensive topic report"
        assert result["last_topic_report"] == "Comprehensive topic report"
        assert result["current_phase"] == 3

        mock_topic_agent.synthesize_topic.assert_called_once_with(
            round_summaries=["Round 1 summary", "Round 2 summary"],
            final_considerations=["Final thought 1", "Final thought 2"],
            topic="Topic A",
            discussion_theme="Test Theme",
        )

    def test_topic_summary_generation_node(self, flow_nodes, mock_specialized_agents):
        """Test topic summary generation for conclusion."""
        state = {
            "active_topic": "Topic A",
            "round_summaries": [
                {"topic": "Topic A", "summary": "Round 1 summary"},
                {"topic": "Topic A", "summary": "Round 2 summary"},
            ],
            "consensus_proposals": {
                "Topic A_final_considerations": [
                    "Final thought 1",
                    "Final thought 2",
                ]
            },
            "topic_summaries": {},
        }

        mock_summarizer = mock_specialized_agents["summarizer"]
        mock_summarizer.summarize_topic_conclusion.return_value = (
            "Topic A conclusion summary"
        )

        result = flow_nodes.topic_summary_generation_node(state)

        assert (
            result["topic_summaries"]["Topic A_conclusion"]
            == "Topic A conclusion summary"
        )

        mock_summarizer.summarize_topic_conclusion.assert_called_once_with(
            round_summaries=["Round 1 summary", "Round 2 summary"],
            final_considerations=["Final thought 1", "Final thought 2"],
            topic="Topic A",
        )

    def test_topic_summary_generation_node_empty_data(
        self, flow_nodes, mock_specialized_agents
    ):
        """Test topic summary generation with empty data."""
        state = {
            "active_topic": "Topic B",
            "round_summaries": [],
            "consensus_proposals": {},
            "topic_summaries": {},
        }

        mock_summarizer = mock_specialized_agents["summarizer"]
        mock_summarizer.summarize_topic_conclusion.return_value = "Empty topic summary"

        result = flow_nodes.topic_summary_generation_node(state)

        assert result["topic_summaries"]["Topic B_conclusion"] == "Empty topic summary"

        mock_summarizer.summarize_topic_conclusion.assert_called_once_with(
            round_summaries=[], final_considerations=[], topic="Topic B"
        )

    def test_topic_summary_generation_node_error_handling(
        self, flow_nodes, mock_specialized_agents
    ):
        """Test topic summary generation error handling."""
        state = {
            "active_topic": "Topic C",
            "round_summaries": [],
            "consensus_proposals": {},
            "topic_summaries": {},
        }

        mock_summarizer = mock_specialized_agents["summarizer"]
        mock_summarizer.summarize_topic_conclusion.side_effect = Exception(
            "Summarizer error"
        )

        result = flow_nodes.topic_summary_generation_node(state)

        assert "Topic C_conclusion" in result["topic_summaries"]
        assert (
            "Failed to generate summary"
            in result["topic_summaries"]["Topic C_conclusion"]
        )
        assert result["summary_error"] == "Summarizer error"


class TestPhase5Nodes:
    """Test Phase 5: Final Report nodes."""

    def test_final_report_node(self, flow_nodes, mock_specialized_agents):
        """Test final report generation."""
        state = {
            "main_topic": "Test Theme",
            "topic_summaries": {
                "Topic A": "Summary A",
                "Topic B": "Summary B",
            },
        }

        mock_ecclesia = mock_specialized_agents["ecclesia_report"]
        mock_ecclesia.generate_report_structure.return_value = [
            "Executive Summary",
            "Key Findings",
            "Conclusion",
        ]
        mock_ecclesia.write_section.return_value = "Section content"

        result = flow_nodes.final_report_node(state)

        assert result["current_phase"] == 5
        assert result["report_generation_status"] == "completed"
        assert len(result["report_sections"]) == 3
        assert "Executive Summary" in result["report_sections"]

        # Verify all sections were written
        assert mock_ecclesia.write_section.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
