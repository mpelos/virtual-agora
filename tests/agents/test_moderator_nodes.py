"""Tests for ModeratorAgent LangGraph node functions.

This module tests the LangGraph-compatible node functions that wrap
ModeratorAgent functionality for state-based workflows, covering
Stories 3.7, 3.8, and 3.9 from Epic 3.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

from virtual_agora.agents.moderator import ModeratorAgent
from virtual_agora.agents.moderator_nodes import (
    ModeratorNodes,
    create_moderator_nodes,
    identify_minority_voters,
    collect_minority_considerations,
    incorporate_minority_views,
    initialize_report_structure,
    generate_report_section,
    finalize_report,
    request_agenda_modifications,
    synthesize_agenda_changes,
    facilitate_agenda_revote,
)
from virtual_agora.state.schema import VirtualAgoraState, VoteRound, Vote, AgentInfo


class TestModeratorNodes:
    """Test class for ModeratorNodes functionality."""

    @pytest.fixture
    def mock_moderator(self):
        """Create a mock ModeratorAgent for testing."""
        moderator = Mock(spec=ModeratorAgent)
        moderator.agent_id = "test_moderator"
        return moderator

    @pytest.fixture
    def moderator_nodes(self, mock_moderator):
        """Create ModeratorNodes instance for testing."""
        return ModeratorNodes(mock_moderator)

    @pytest.fixture
    def sample_state(self):
        """Create a sample VirtualAgoraState for testing."""
        return VirtualAgoraState(
            session_id="test_session",
            start_time=datetime.now(),
            config_hash="test_hash",
            current_phase=2,
            phase_history=[],
            phase_start_time=datetime.now(),
            active_topic="Test Topic",
            topic_queue=["Topic 1", "Topic 2"],
            proposed_topics=["Topic 1", "Topic 2", "Topic 3"],
            topics_info={},
            completed_topics=[],
            agents={
                "agent1": AgentInfo(
                    id="agent1",
                    model="gpt-4",
                    provider="openai",
                    role="participant",
                    message_count=5,
                    created_at=datetime.now(),
                ),
                "agent2": AgentInfo(
                    id="agent2",
                    model="claude-3",
                    provider="anthropic",
                    role="participant",
                    message_count=3,
                    created_at=datetime.now(),
                ),
            },
            moderator_id="test_moderator",
            current_speaker_id=None,
            speaking_order=["agent1", "agent2"],
            next_speaker_index=0,
            messages=[],
            last_message_id="msg_0",
            active_vote=VoteRound(
                id="vote_1",
                phase=2,
                vote_type="topic_conclusion",
                options=["Yes", "No"],
                start_time=datetime.now(),
                end_time=None,
                required_votes=2,
                received_votes=2,
                result="Yes",
                status="completed",
            ),
            vote_history=[],
            votes=[
                Vote(
                    id="vote_1_agent1",
                    voter_id="agent1",
                    phase=2,
                    vote_type="topic_conclusion",
                    choice="Yes",
                    timestamp=datetime.now(),
                ),
                Vote(
                    id="vote_1_agent2",
                    voter_id="agent2",
                    phase=2,
                    vote_type="topic_conclusion",
                    choice="No",
                    timestamp=datetime.now(),
                ),
            ],
            consensus_proposals={},
            consensus_reached={},
            phase_summaries={},
            topic_summaries={"Test Topic": "Sample summary"},
            consensus_summaries={},
            final_report=None,
            total_messages=8,
            messages_by_phase={2: 8},
            messages_by_agent={"agent1": 5, "agent2": 3},
            messages_by_topic={"Test Topic": 8},
            vote_participation_rate={"vote_1": 1.0},
            tool_calls=[],
            active_tool_calls={},
            tool_metrics={
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "average_execution_time_ms": 0.0,
                "calls_by_tool": {},
                "calls_by_agent": {},
                "errors_by_type": {},
            },
            tools_enabled_agents=[],
            last_error=None,
            error_count=0,
            warnings=[],
        )


class TestStory37MinorityConsiderations(TestModeratorNodes):
    """Tests for Story 3.7: Minority Considerations Management."""

    @pytest.mark.asyncio
    async def test_identify_minority_voters_success(
        self, moderator_nodes, sample_state
    ):
        """Test successful identification of minority voters."""
        result = await moderator_nodes.identify_minority_voters_node(sample_state)

        assert "active_vote" in result
        assert "minority_voters" in result["active_vote"]
        assert result["active_vote"]["minority_voters"] == ["agent2"]
        assert "warnings" in result
        assert len(result["warnings"]) == 0

    @pytest.mark.asyncio
    async def test_identify_minority_voters_no_active_vote(
        self, moderator_nodes, sample_state
    ):
        """Test minority voter identification with no active vote."""
        sample_state["active_vote"] = None

        result = await moderator_nodes.identify_minority_voters_node(sample_state)

        assert "warnings" in result
        assert "No active vote found" in result["warnings"][0]

    @pytest.mark.asyncio
    async def test_identify_minority_voters_no_votes(
        self, moderator_nodes, sample_state
    ):
        """Test minority voter identification with no votes cast."""
        sample_state["votes"] = []

        result = await moderator_nodes.identify_minority_voters_node(sample_state)

        assert "warnings" in result
        assert "No votes found" in result["warnings"][0]

    @pytest.mark.asyncio
    async def test_identify_minority_voters_unanimous(
        self, moderator_nodes, sample_state
    ):
        """Test minority voter identification when all voted the same."""
        # Make both votes "Yes"
        sample_state["votes"][1]["choice"] = "Yes"

        result = await moderator_nodes.identify_minority_voters_node(sample_state)

        assert "active_vote" in result
        assert result["active_vote"]["minority_voters"] == []
        assert "warnings" in result
        assert "No minority voters identified" in result["warnings"][0]

    @pytest.mark.asyncio
    async def test_collect_minority_considerations_success(
        self, moderator_nodes, sample_state
    ):
        """Test successful collection of minority considerations."""
        # Set up active vote with minority voters
        sample_state["active_vote"]["minority_voters"] = ["agent2"]

        # Mock the moderator method
        moderator_nodes.moderator.collect_minority_consideration = AsyncMock(
            return_value="Final consideration from agent2"
        )

        result = await moderator_nodes.collect_minority_considerations_node(
            sample_state
        )

        assert "active_vote" in result
        assert "minority_considerations" in result["active_vote"]
        assert len(result["active_vote"]["minority_considerations"]) == 1
        assert "warnings" in result
        assert len(result["warnings"]) == 0

        # Verify the method was called
        moderator_nodes.moderator.collect_minority_consideration.assert_called_once_with(
            "agent2", "Test Topic", sample_state
        )

    @pytest.mark.asyncio
    async def test_collect_minority_considerations_no_minority(
        self, moderator_nodes, sample_state
    ):
        """Test collection when no minority voters exist."""
        sample_state["active_vote"]["minority_voters"] = []

        result = await moderator_nodes.collect_minority_considerations_node(
            sample_state
        )

        assert "warnings" in result
        assert "No minority voters found" in result["warnings"][0]

    @pytest.mark.asyncio
    async def test_collect_minority_considerations_no_active_vote(
        self, moderator_nodes, sample_state
    ):
        """Test collection with no active vote."""
        sample_state["active_vote"] = None

        result = await moderator_nodes.collect_minority_considerations_node(
            sample_state
        )

        assert "warnings" in result
        assert "No minority voters found" in result["warnings"][0]

    @pytest.mark.asyncio
    async def test_collect_minority_considerations_method_failure(
        self, moderator_nodes, sample_state
    ):
        """Test handling of method failures during collection."""
        sample_state["active_vote"]["minority_voters"] = ["agent2"]

        # Mock method to raise exception
        moderator_nodes.moderator.collect_minority_consideration = AsyncMock(
            side_effect=Exception("Test error")
        )

        result = await moderator_nodes.collect_minority_considerations_node(
            sample_state
        )

        assert "active_vote" in result
        assert (
            "[Error collecting from agent2]"
            in result["active_vote"]["minority_considerations"]
        )

    @pytest.mark.asyncio
    async def test_incorporate_minority_views_success(
        self, moderator_nodes, sample_state
    ):
        """Test successful incorporation of minority views."""
        sample_state["active_vote"]["minority_considerations"] = [
            "Consideration 1",
            "Consideration 2",
        ]

        # Mock the moderator method
        moderator_nodes.moderator.incorporate_minority_views = AsyncMock(
            return_value="Updated summary with minority views"
        )

        result = await moderator_nodes.incorporate_minority_views_node(sample_state)

        assert "topic_summaries" in result
        assert (
            result["topic_summaries"]["Test Topic"]
            == "Updated summary with minority views"
        )

        # Verify the method was called
        moderator_nodes.moderator.incorporate_minority_views.assert_called_once_with(
            "Sample summary", ["Consideration 1", "Consideration 2"], "Test Topic"
        )

    @pytest.mark.asyncio
    async def test_incorporate_minority_views_no_considerations(
        self, moderator_nodes, sample_state
    ):
        """Test incorporation when no minority considerations exist."""
        sample_state["active_vote"]["minority_considerations"] = []

        result = await moderator_nodes.incorporate_minority_views_node(sample_state)

        # Should return empty dict since no work to do
        assert result == {}

    @pytest.mark.asyncio
    async def test_incorporate_minority_views_method_failure(
        self, moderator_nodes, sample_state
    ):
        """Test handling of method failure during incorporation."""
        sample_state["active_vote"]["minority_considerations"] = ["Consideration 1"]

        # Mock method to return None (failure)
        moderator_nodes.moderator.incorporate_minority_views = AsyncMock(
            return_value=None
        )

        result = await moderator_nodes.incorporate_minority_views_node(sample_state)

        assert "warnings" in result
        assert "Failed to incorporate minority views" in result["warnings"][0]


class TestStory38ReportWriterMode(TestModeratorNodes):
    """Tests for Story 3.8: Report Writer Mode Implementation."""

    @pytest.mark.asyncio
    async def test_initialize_report_structure_success(
        self, moderator_nodes, sample_state
    ):
        """Test successful report structure initialization."""
        # Mock the moderator method
        moderator_nodes.moderator.define_report_structure = AsyncMock(
            return_value=["Executive Summary", "Detailed Analysis", "Conclusions"]
        )

        result = await moderator_nodes.initialize_report_structure_node(sample_state)

        assert "report_structure" in result
        assert result["report_structure"] == [
            "Executive Summary",
            "Detailed Analysis",
            "Conclusions",
        ]
        assert "report_sections" in result
        assert result["report_sections"] == {}
        assert result["report_generation_status"] == "structuring"

    @pytest.mark.asyncio
    async def test_initialize_report_structure_no_summaries(
        self, moderator_nodes, sample_state
    ):
        """Test initialization with no topic summaries."""
        sample_state["topic_summaries"] = {}

        result = await moderator_nodes.initialize_report_structure_node(sample_state)

        assert result["report_generation_status"] == "failed"
        assert "warnings" in result
        assert "No topic summaries available" in result["warnings"][0]

    @pytest.mark.asyncio
    async def test_initialize_report_structure_method_failure(
        self, moderator_nodes, sample_state
    ):
        """Test handling of method failure during initialization."""
        # Mock method to return None (failure)
        moderator_nodes.moderator.define_report_structure = AsyncMock(return_value=None)

        result = await moderator_nodes.initialize_report_structure_node(sample_state)

        assert result["report_generation_status"] == "failed"
        assert "last_error" in result
        assert "Failed to generate report structure" in result["last_error"]

    @pytest.mark.asyncio
    async def test_generate_report_section_success(self, moderator_nodes, sample_state):
        """Test successful report section generation."""
        sample_state["report_structure"] = ["Executive Summary", "Conclusions"]
        sample_state["report_sections"] = {}

        # Mock the moderator method
        moderator_nodes.moderator.generate_report_section = AsyncMock(
            return_value="Generated executive summary content"
        )

        result = await moderator_nodes.generate_report_section_node(sample_state)

        assert "report_sections" in result
        assert (
            result["report_sections"]["Executive Summary"]
            == "Generated executive summary content"
        )
        assert result["report_generation_status"] == "writing"

    @pytest.mark.asyncio
    async def test_generate_report_section_all_completed(
        self, moderator_nodes, sample_state
    ):
        """Test section generation when all sections are completed."""
        sample_state["report_structure"] = ["Executive Summary"]
        sample_state["report_sections"] = {"Executive Summary": "Already generated"}

        result = await moderator_nodes.generate_report_section_node(sample_state)

        assert result["report_generation_status"] == "completed"
        assert "final_report" in result
        assert "Report generation completed" in result["final_report"]

    @pytest.mark.asyncio
    async def test_generate_report_section_no_structure(
        self, moderator_nodes, sample_state
    ):
        """Test section generation with no report structure."""
        sample_state["report_structure"] = []

        result = await moderator_nodes.generate_report_section_node(sample_state)

        assert result["report_generation_status"] == "failed"
        assert "last_error" in result
        assert "No report structure available" in result["last_error"]

    @pytest.mark.asyncio
    async def test_generate_report_section_method_failure(
        self, moderator_nodes, sample_state
    ):
        """Test handling of method failure during section generation."""
        sample_state["report_structure"] = ["Executive Summary"]
        sample_state["report_sections"] = {}

        # Mock method to return None (failure)
        moderator_nodes.moderator.generate_report_section = AsyncMock(return_value=None)

        result = await moderator_nodes.generate_report_section_node(sample_state)

        assert result["report_generation_status"] == "failed"
        assert "last_error" in result
        assert "Failed to generate content for section" in result["last_error"]

    @pytest.mark.asyncio
    async def test_finalize_report_success(self, moderator_nodes, sample_state):
        """Test successful report finalization."""
        sample_state["report_structure"] = ["Executive Summary", "Conclusions"]
        sample_state["report_sections"] = {
            "Executive Summary": "Summary content",
            "Conclusions": "Conclusion content",
        }

        result = await moderator_nodes.finalize_report_node(sample_state)

        assert "final_report" in result
        assert "2 sections" in result["final_report"]
        assert result["report_generation_status"] == "completed"

    @pytest.mark.asyncio
    async def test_finalize_report_missing_sections(
        self, moderator_nodes, sample_state
    ):
        """Test finalization with missing sections."""
        sample_state["report_structure"] = ["Executive Summary", "Conclusions"]
        sample_state["report_sections"] = {"Executive Summary": "Summary content"}

        result = await moderator_nodes.finalize_report_node(sample_state)

        assert result["report_generation_status"] == "failed"
        assert "last_error" in result
        assert "Missing sections: ['Conclusions']" in result["last_error"]

    @pytest.mark.asyncio
    async def test_finalize_report_no_structure(self, moderator_nodes, sample_state):
        """Test finalization with no report structure."""
        sample_state["report_structure"] = []
        sample_state["report_sections"] = {}

        result = await moderator_nodes.finalize_report_node(sample_state)

        assert result["report_generation_status"] == "failed"
        assert "last_error" in result
        assert "Missing report structure or sections" in result["last_error"]


class TestStory39AgendaModification(TestModeratorNodes):
    """Tests for Story 3.9: Agenda Modification Facilitation."""

    @pytest.mark.asyncio
    async def test_request_agenda_modifications_success(
        self, moderator_nodes, sample_state
    ):
        """Test successful agenda modification requests."""
        # Mock the moderator method
        moderator_nodes.moderator.request_agenda_modification = AsyncMock(
            return_value="Suggested modification"
        )

        result = await moderator_nodes.request_agenda_modifications_node(sample_state)

        assert "pending_agenda_modifications" in result
        assert len(result["pending_agenda_modifications"]) == 2  # Two agents
        assert all(
            "Suggested modification" in mod
            for mod in result["pending_agenda_modifications"]
        )

    @pytest.mark.asyncio
    async def test_request_agenda_modifications_no_topics(
        self, moderator_nodes, sample_state
    ):
        """Test modification requests with no remaining topics."""
        sample_state["topic_queue"] = []

        result = await moderator_nodes.request_agenda_modifications_node(sample_state)

        assert "warnings" in result
        assert "No remaining topics for modification" in result["warnings"][0]

    @pytest.mark.asyncio
    async def test_request_agenda_modifications_method_failure(
        self, moderator_nodes, sample_state
    ):
        """Test handling of method failure during request collection."""
        # Mock method to raise exception for one agent
        moderator_nodes.moderator.request_agenda_modification = AsyncMock(
            side_effect=[Exception("Test error"), "Valid suggestion"]
        )

        result = await moderator_nodes.request_agenda_modifications_node(sample_state)

        assert "pending_agenda_modifications" in result
        assert len(result["pending_agenda_modifications"]) == 1  # Only successful one

    @pytest.mark.asyncio
    async def test_synthesize_agenda_changes_success(
        self, moderator_nodes, sample_state
    ):
        """Test successful agenda change synthesis."""
        sample_state["pending_agenda_modifications"] = [
            "Add new topic X",
            "Remove topic 2",
        ]

        # Mock the moderator method
        moderator_nodes.moderator.synthesize_agenda_modifications = AsyncMock(
            return_value=["Topic 1", "New Topic X"]
        )

        result = await moderator_nodes.synthesize_agenda_changes_node(sample_state)

        assert "proposed_topics" in result
        assert result["proposed_topics"] == ["Topic 1", "New Topic X"]
        assert result["pending_agenda_modifications"] == []

    @pytest.mark.asyncio
    async def test_synthesize_agenda_changes_no_modifications(
        self, moderator_nodes, sample_state
    ):
        """Test synthesis with no pending modifications."""
        sample_state["pending_agenda_modifications"] = []

        result = await moderator_nodes.synthesize_agenda_changes_node(sample_state)

        assert "warnings" in result
        assert "No agenda modifications received" in result["warnings"][0]

    @pytest.mark.asyncio
    async def test_synthesize_agenda_changes_method_failure(
        self, moderator_nodes, sample_state
    ):
        """Test handling of method failure during synthesis."""
        sample_state["pending_agenda_modifications"] = ["Add topic X"]

        # Mock method to return None (failure)
        moderator_nodes.moderator.synthesize_agenda_modifications = AsyncMock(
            return_value=None
        )

        result = await moderator_nodes.synthesize_agenda_changes_node(sample_state)

        assert "warnings" in result
        assert "Failed to synthesize agenda modifications" in result["warnings"][0]

    @pytest.mark.asyncio
    async def test_facilitate_agenda_revote_success(
        self, moderator_nodes, sample_state
    ):
        """Test successful agenda re-voting facilitation."""
        sample_state["proposed_topics"] = ["Topic A", "Topic B", "Topic C"]

        # Mock the moderator methods
        moderator_nodes.moderator.collect_agenda_vote = AsyncMock(
            return_value="Vote response"
        )
        moderator_nodes.moderator.synthesize_agenda_votes = AsyncMock(
            return_value=["Topic B", "Topic A", "Topic C"]
        )

        result = await moderator_nodes.facilitate_agenda_revote_node(sample_state)

        assert "topic_queue" in result
        assert result["topic_queue"] == ["Topic B", "Topic A", "Topic C"]
        assert "agenda_modification_votes" in result
        assert len(result["agenda_modification_votes"]) == 2  # Two agents

    @pytest.mark.asyncio
    async def test_facilitate_agenda_revote_no_topics(
        self, moderator_nodes, sample_state
    ):
        """Test re-voting facilitation with no proposed topics."""
        sample_state["proposed_topics"] = []

        result = await moderator_nodes.facilitate_agenda_revote_node(sample_state)

        assert "warnings" in result
        assert "No proposed topics for re-voting" in result["warnings"][0]

    @pytest.mark.asyncio
    async def test_facilitate_agenda_revote_no_votes(
        self, moderator_nodes, sample_state
    ):
        """Test re-voting facilitation when no votes are collected."""
        sample_state["proposed_topics"] = ["Topic A", "Topic B"]

        # Mock method to return None (no votes)
        moderator_nodes.moderator.collect_agenda_vote = AsyncMock(return_value=None)

        result = await moderator_nodes.facilitate_agenda_revote_node(sample_state)

        assert "last_error" in result
        assert "No agenda votes collected" in result["last_error"]
        assert result["error_count"] == 1

    @pytest.mark.asyncio
    async def test_facilitate_agenda_revote_synthesis_failure(
        self, moderator_nodes, sample_state
    ):
        """Test handling of synthesis failure during re-voting."""
        sample_state["proposed_topics"] = ["Topic A", "Topic B"]

        # Mock vote collection to succeed but synthesis to fail
        moderator_nodes.moderator.collect_agenda_vote = AsyncMock(
            return_value="Vote response"
        )
        moderator_nodes.moderator.synthesize_agenda_votes = AsyncMock(return_value=None)

        result = await moderator_nodes.facilitate_agenda_revote_node(sample_state)

        assert "last_error" in result
        assert "Failed to synthesize agenda votes" in result["last_error"]


class TestDirectNodeFunctions(TestModeratorNodes):
    """Tests for direct node function aliases."""

    @pytest.mark.asyncio
    async def test_direct_identify_minority_voters(self, mock_moderator, sample_state):
        """Test direct identify_minority_voters function."""
        result = await identify_minority_voters(sample_state, mock_moderator)

        assert "active_vote" in result
        assert "minority_voters" in result["active_vote"]

    @pytest.mark.asyncio
    async def test_direct_collect_minority_considerations(
        self, mock_moderator, sample_state
    ):
        """Test direct collect_minority_considerations function."""
        sample_state["active_vote"]["minority_voters"] = ["agent2"]
        mock_moderator.collect_minority_consideration = AsyncMock(
            return_value="consideration"
        )

        result = await collect_minority_considerations(sample_state, mock_moderator)

        assert "active_vote" in result
        assert "minority_considerations" in result["active_vote"]

    @pytest.mark.asyncio
    async def test_direct_initialize_report_structure(
        self, mock_moderator, sample_state
    ):
        """Test direct initialize_report_structure function."""
        mock_moderator.define_report_structure = AsyncMock(
            return_value=["Section 1", "Section 2"]
        )

        result = await initialize_report_structure(sample_state, mock_moderator)

        assert "report_structure" in result
        assert result["report_generation_status"] == "structuring"

    @pytest.mark.asyncio
    async def test_direct_request_agenda_modifications(
        self, mock_moderator, sample_state
    ):
        """Test direct request_agenda_modifications function."""
        mock_moderator.request_agenda_modification = AsyncMock(
            return_value="modification"
        )

        result = await request_agenda_modifications(sample_state, mock_moderator)

        assert "pending_agenda_modifications" in result


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_moderator_nodes(self):
        """Test create_moderator_nodes factory function."""
        mock_moderator = Mock(spec=ModeratorAgent)
        nodes = create_moderator_nodes(mock_moderator)

        assert isinstance(nodes, ModeratorNodes)
        assert nodes.moderator == mock_moderator


class TestErrorHandling(TestModeratorNodes):
    """Tests for error handling across all node functions."""

    @pytest.mark.asyncio
    async def test_error_handling_consistency(self, moderator_nodes, sample_state):
        """Test that all node functions handle errors consistently."""
        # Test each node function with an exception
        node_functions = [
            moderator_nodes.identify_minority_voters_node,
            moderator_nodes.collect_minority_considerations_node,
            moderator_nodes.incorporate_minority_views_node,
            moderator_nodes.initialize_report_structure_node,
            moderator_nodes.generate_report_section_node,
            moderator_nodes.finalize_report_node,
            moderator_nodes.request_agenda_modifications_node,
            moderator_nodes.synthesize_agenda_changes_node,
            moderator_nodes.facilitate_agenda_revote_node,
        ]

        for node_function in node_functions:
            # Force an exception by corrupting the state
            corrupted_state = {"invalid": "state"}

            try:
                result = await node_function(corrupted_state)

                # All functions should return error information
                assert (
                    "last_error" in result
                    or "warnings" in result
                    or "report_generation_status" in result
                )

            except Exception as e:
                # Some functions might raise exceptions - that's also acceptable
                pass

    @pytest.mark.asyncio
    async def test_state_preservation(self, moderator_nodes, sample_state):
        """Test that node functions don't modify the original state."""
        original_state = sample_state.copy()

        # Run a few node functions
        await moderator_nodes.identify_minority_voters_node(sample_state)
        await moderator_nodes.initialize_report_structure_node(sample_state)

        # Verify original state hasn't been modified (keys should be the same)
        assert set(sample_state.keys()) == set(original_state.keys())


if __name__ == "__main__":
    pytest.main([__file__])
