"""Integration tests for ModeratorAgent workflows in Virtual Agora graph context.

These tests verify that the ModeratorAgent properly integrates with the
Virtual Agora state management and LangGraph execution environment.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models.base import BaseLanguageModel

from virtual_agora.agents.moderator import ModeratorAgent, ModeratorMode
from virtual_agora.state.schema import (
    VirtualAgoraState,
    Message,
    Vote,
    TopicInfo,
    AgentInfo,
)
from virtual_agora.state.manager import StateManager
from virtual_agora.config.models import Config as VirtualAgoraConfig
from virtual_agora.utils.exceptions import StateError


class TestModeratorAgentIntegration:
    """Integration tests for ModeratorAgent with Virtual Agora state."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = Mock(spec=BaseLanguageModel)
        llm.invoke = Mock(return_value=AIMessage(content="Test response"))
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test async response"))
        return llm

    @pytest.fixture
    def sample_config(self) -> VirtualAgoraConfig:
        """Create a sample configuration for testing."""
        config_dict = {
            "session": {"max_duration_minutes": 60, "max_messages": 100},
            "moderator": {"model": "gpt-4", "provider": "openai", "temperature": 0.7},
            "agents": [
                {
                    "model": "gpt-3.5-turbo",
                    "provider": "openai",
                    "count": 2,
                    "temperature": 0.8,
                }
            ],
        }
        return VirtualAgoraConfig(**config_dict)

    @pytest.fixture
    def state_manager(self, sample_config):
        """Create a state manager for testing."""
        return StateManager(sample_config)

    @pytest.fixture
    def moderator_agent(self, mock_llm):
        """Create a ModeratorAgent for testing."""
        return ModeratorAgent(
            agent_id="moderator",
            llm=mock_llm,
            mode="facilitation",
            turn_timeout_seconds=300,
            relevance_threshold=0.7,
            warning_threshold=2,
            mute_duration_minutes=5,
        )

    def test_moderator_agent_initialization(self, moderator_agent):
        """Test that ModeratorAgent initializes correctly with all parameters."""
        assert moderator_agent.agent_id == "moderator"
        assert moderator_agent.mode == "facilitation"
        assert moderator_agent.turn_timeout_seconds == 300
        assert moderator_agent.relevance_threshold == 0.7
        assert moderator_agent.warning_threshold == 2
        assert moderator_agent.mute_duration_minutes == 5

        # Check that collections are initialized
        assert hasattr(moderator_agent, "relevance_violations")
        assert hasattr(moderator_agent, "muted_agents")
        assert hasattr(moderator_agent, "active_polls")

    def test_round_management_workflow(self, moderator_agent):
        """Test complete round management workflow."""
        topic = "AI Ethics"
        round_number = 1

        # Test topic announcement
        announcement = moderator_agent.announce_topic(topic, round_number)
        assert f"Round {round_number}" in announcement
        assert topic in announcement

        # Test round completion signal
        completion_signal = moderator_agent.signal_round_completion(round_number, topic)
        assert f"Round {round_number}" in completion_signal
        assert topic in completion_signal

    def test_relevance_enforcement_integration(self, moderator_agent):
        """Test complete relevance enforcement workflow."""
        agent_id = "participant1"
        topic = "AI Ethics"
        moderator_agent.current_topic_context = topic

        # Test relevance scoring
        with patch.object(
            moderator_agent, "_calculate_relevance_score", return_value=0.3
        ):
            relevance_assessment = {
                "reason": "Message is off-topic",
                "suggestions": "Please focus on AI ethics",
            }

            # Issue first warning
            warning_msg = moderator_agent.issue_relevance_warning(
                agent_id, relevance_assessment
            )
            assert "Relevance Warning" in warning_msg
            assert agent_id in warning_msg

            # Check violation tracking
            assert agent_id in moderator_agent.relevance_violations
            assert (
                moderator_agent.relevance_violations[agent_id]["warnings_issued"] == 1
            )

            # Issue second warning (should trigger muting)
            second_warning = moderator_agent.issue_relevance_warning(
                agent_id, relevance_assessment
            )
            assert "Final Warning" in second_warning

            # Next violation should mute
            mute_msg = moderator_agent.mute_agent(
                agent_id, "Multiple relevance violations"
            )
            assert "has been temporarily muted" in mute_msg
            assert agent_id in moderator_agent.muted_agents

    def test_summarization_integration(self, moderator_agent):
        """Test summarization workflow with mock LLM responses."""
        with patch.object(moderator_agent.llm, "invoke") as mock_invoke:
            mock_invoke.return_value = AIMessage(content="Test summary generated")

            # Test round summary generation
            messages = [
                Message(
                    id="msg1",
                    speaker_id="participant1",
                    speaker_role="participant",
                    content="Discussion point 1",
                    timestamp=datetime.now(),
                    phase=2,
                    topic="AI Ethics",
                )
            ]

            round_summary = moderator_agent.generate_round_summary(
                1, "AI Ethics", messages
            )
            assert "Test summary generated" in round_summary

            # Test topic summary generation
            topic_summary = moderator_agent.generate_topic_summary(
                "AI Ethics", messages
            )
            assert "Test summary generated" in topic_summary

    def test_polling_workflow_integration(self, moderator_agent):
        """Test complete polling workflow."""
        topic = "AI Ethics"
        eligible_voters = ["participant1", "participant2", "participant3"]

        # Initiate poll
        poll_result = moderator_agent.initiate_conclusion_poll(
            topic, eligible_voters, 10
        )
        assert poll_result["status"] == "initiated"
        poll_id = poll_result["poll_id"]

        # Cast votes
        vote1_result = moderator_agent.cast_vote(
            poll_id, "participant1", "conclude", "We've covered enough"
        )
        assert vote1_result["status"] == "recorded"

        vote2_result = moderator_agent.cast_vote(
            poll_id, "participant2", "conclude", "Time to move on"
        )
        assert vote2_result["status"] == "recorded"

        vote3_result = moderator_agent.cast_vote(
            poll_id, "participant3", "continue", "Need more discussion"
        )
        assert vote3_result["status"] == "recorded"

        # Tally results
        tally_result = moderator_agent.tally_poll_results(poll_id)
        assert tally_result["success"] is True
        assert tally_result["decision"] == "conclude"  # 2 vs 1
        assert tally_result["vote_counts"]["conclude"] == 2
        assert tally_result["vote_counts"]["continue"] == 1

        # Handle minority considerations
        minority_msg = moderator_agent.handle_minority_considerations(
            tally_result, topic
        )
        assert "Minority Final Considerations" in minority_msg
        assert "participant3" in minority_msg  # The minority voter

    def test_mode_switching_integration(self, moderator_agent):
        """Test mode switching during operations."""
        original_mode = moderator_agent.mode
        assert original_mode == "facilitation"

        # Switch to synthesis mode
        moderator_agent.set_mode("synthesis")
        assert moderator_agent.mode == "synthesis"

        # Perform synthesis operation
        with patch.object(moderator_agent.llm, "invoke") as mock_invoke:
            mock_invoke.return_value = AIMessage(content="Synthesis complete")

            messages = [
                Message(
                    id="msg1",
                    speaker_id="participant1",
                    speaker_role="participant",
                    content="Test message",
                    timestamp=datetime.now(),
                    phase=2,
                    topic="AI Ethics",
                )
            ]

            summary = moderator_agent.generate_topic_summary("AI Ethics", messages)
            assert "Synthesis complete" in summary

        # Switch back to facilitation
        moderator_agent.set_mode("facilitation")
        assert moderator_agent.mode == "facilitation"

    def test_timeout_handling_integration(self, moderator_agent):
        """Test timeout detection and handling."""
        agent_id = "participant1"

        # Mock a timeout situation
        past_time = datetime.now() - timedelta(seconds=400)  # 400 seconds ago

        # Check timeout detection
        is_timeout = moderator_agent._is_turn_timeout(past_time)
        assert is_timeout

        # Test timeout announcement
        timeout_msg = moderator_agent.announce_turn_timeout(agent_id)
        assert "timeout" in timeout_msg.lower()
        assert agent_id in timeout_msg

    def test_error_handling_integration(self, moderator_agent):
        """Test error handling in moderator operations."""
        # Test invalid poll ID
        with pytest.raises(ValueError, match="Poll .* not found"):
            moderator_agent.cast_vote(
                "invalid_poll_id", "participant1", "yes", "reasoning"
            )

        # Test invalid choice in valid poll
        poll_result = moderator_agent.initiate_conclusion_poll(
            "Test Topic", ["participant1"], 5
        )
        poll_id = poll_result["poll_id"]

        with pytest.raises(ValueError, match="Invalid choice"):
            moderator_agent.cast_vote(
                poll_id, "participant1", "invalid_choice", "reasoning"
            )

    def test_comprehensive_discussion_cycle(self, moderator_agent):
        """Test a complete discussion cycle with multiple operations."""
        topic = "AI Ethics"
        participants = ["participant1", "participant2", "participant3"]

        # 1. Start discussion round
        announcement = moderator_agent.announce_topic(topic, 1)
        assert topic in announcement

        # 2. Simulate some relevance issues
        moderator_agent.current_topic_context = topic
        with patch.object(
            moderator_agent, "_calculate_relevance_score", return_value=0.3
        ):
            relevance_assessment = {
                "reason": "Off-topic",
                "suggestions": "Stay focused",
            }
            warning_msg = moderator_agent.issue_relevance_warning(
                "participant1", relevance_assessment
            )
            assert "Warning" in warning_msg

        # 3. Conduct conclusion poll
        poll_result = moderator_agent.initiate_conclusion_poll(topic, participants, 10)
        poll_id = poll_result["poll_id"]

        # Cast votes
        moderator_agent.cast_vote(
            poll_id, "participant1", "conclude", "Enough discussion"
        )
        moderator_agent.cast_vote(
            poll_id, "participant2", "conclude", "Time to move on"
        )
        moderator_agent.cast_vote(poll_id, "participant3", "continue", "Need more time")

        # Tally results
        tally_result = moderator_agent.tally_poll_results(poll_id)
        assert tally_result["decision"] == "conclude"

        # 4. Generate summaries
        with patch.object(moderator_agent.llm, "invoke") as mock_invoke:
            mock_invoke.return_value = AIMessage(content="Summary of discussion")

            messages = [
                Message(
                    id="msg1",
                    speaker_id="participant1",
                    speaker_role="participant",
                    content="Point about AI safety",
                    timestamp=datetime.now(),
                    phase=2,
                    topic=topic,
                )
            ]

            round_summary = moderator_agent.generate_round_summary(1, topic, messages)
            topic_summary = moderator_agent.generate_topic_summary(topic, messages)

            assert "Summary of discussion" in round_summary
            assert "Summary of discussion" in topic_summary

        # 5. Signal completion
        completion_msg = moderator_agent.signal_round_completion(1, topic)
        assert topic in completion_msg

    def test_state_consistency_tracking(self, moderator_agent):
        """Test that moderator maintains consistent internal state."""
        # Track initial state
        initial_violations = len(moderator_agent.relevance_violations)
        initial_muted = len(moderator_agent.muted_agents)
        initial_polls = len(moderator_agent.active_polls)

        # Perform operations that modify state
        moderator_agent.issue_relevance_warning(
            "agent1", {"reason": "test", "suggestions": "test"}
        )
        moderator_agent.mute_agent("agent2", "test mute")
        poll_result = moderator_agent.initiate_conclusion_poll("Test", ["agent1"], 5)

        # Verify state changes
        assert len(moderator_agent.relevance_violations) == initial_violations + 1
        assert len(moderator_agent.muted_agents) == initial_muted + 1
        assert len(moderator_agent.active_polls) == initial_polls + 1

        # Complete poll to clean up state
        moderator_agent.cast_vote(poll_result["poll_id"], "agent1", "conclude", "done")
        moderator_agent.tally_poll_results(poll_result["poll_id"])

        # Verify cleanup
        assert (
            len(moderator_agent.active_polls) == initial_polls
        )  # Poll completed, removed from active


class TestModeratorNodeIntegration:
    """Integration tests for new ModeratorAgent node functions (Stories 3.7, 3.8, 3.9)."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = Mock(spec=BaseLanguageModel)
        llm.invoke = Mock(return_value=AIMessage(content="Test response"))
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test async response"))
        return llm

    @pytest.fixture
    def moderator_agent(self, mock_llm):
        """Create a ModeratorAgent instance for testing."""
        return ModeratorAgent(
            agent_id="test_moderator", llm=mock_llm, mode="facilitation"
        )

    @pytest.fixture
    def sample_state_with_vote(self):
        """Create a sample state with voting data for testing."""
        return VirtualAgoraState(
            session_id="test_session",
            start_time=datetime.now(),
            config_hash="test_hash",
            current_phase=2,
            phase_history=[],
            phase_start_time=datetime.now(),
            active_topic="Climate Change Policy",
            topic_queue=["Economic Impact", "Technology Solutions"],
            proposed_topics=[
                "Climate Change Policy",
                "Economic Impact",
                "Technology Solutions",
            ],
            topics_info={
                "Climate Change Policy": TopicInfo(
                    topic="Climate Change Policy",
                    proposed_by="agent1",
                    start_time=datetime.now(),
                    end_time=None,
                    message_count=15,
                    status="active",
                )
            },
            completed_topics=[],
            agents={
                "agent1": AgentInfo(
                    id="agent1",
                    model="gpt-4",
                    provider="openai",
                    role="participant",
                    message_count=8,
                    created_at=datetime.now(),
                ),
                "agent2": AgentInfo(
                    id="agent2",
                    model="claude-3",
                    provider="anthropic",
                    role="participant",
                    message_count=7,
                    created_at=datetime.now(),
                ),
                "agent3": AgentInfo(
                    id="agent3",
                    model="gemini-pro",
                    provider="google",
                    role="participant",
                    message_count=5,
                    created_at=datetime.now(),
                ),
            },
            moderator_id="test_moderator",
            current_speaker_id=None,
            speaking_order=["agent1", "agent2", "agent3"],
            next_speaker_index=0,
            messages=[],
            last_message_id="msg_15",
            active_vote={
                "id": "conclude_vote_1",
                "phase": 2,
                "vote_type": "topic_conclusion",
                "options": ["Yes", "No"],
                "start_time": datetime.now(),
                "end_time": None,
                "required_votes": 3,
                "received_votes": 3,
                "result": "Yes",
                "status": "completed",
            },
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
                Vote(
                    id="vote_1_agent3",
                    voter_id="agent3",
                    phase=2,
                    vote_type="topic_conclusion",
                    choice="Yes",
                    timestamp=datetime.now(),
                ),
            ],
            consensus_proposals={},
            consensus_reached={},
            phase_summaries={},
            topic_summaries={
                "Climate Change Policy": "Comprehensive discussion on climate change policy approaches, covering mitigation strategies, adaptation measures, and international cooperation frameworks."
            },
            consensus_summaries={},
            final_report=None,
            total_messages=20,
            messages_by_phase={2: 20},
            messages_by_agent={"agent1": 8, "agent2": 7, "agent3": 5},
            messages_by_topic={"Climate Change Policy": 20},
            vote_participation_rate={"conclude_vote_1": 1.0},
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

    @pytest.mark.asyncio
    async def test_story_37_minority_considerations_workflow(
        self, moderator_agent, sample_state_with_vote
    ):
        """Test complete Story 3.7 workflow: Minority Considerations Management."""
        from virtual_agora.agents.moderator_nodes import ModeratorNodes

        nodes = ModeratorNodes(moderator_agent)

        # Step 1: Identify minority voters
        result1 = await nodes.identify_minority_voters_node(sample_state_with_vote)

        assert "active_vote" in result1
        assert "minority_voters" in result1["active_vote"]
        assert "agent2" in result1["active_vote"]["minority_voters"]

        # Update state with minority voters
        sample_state_with_vote["active_vote"]["minority_voters"] = result1[
            "active_vote"
        ]["minority_voters"]

        # Step 2: Collect minority considerations
        with patch.object(
            moderator_agent, "collect_minority_consideration", new_callable=AsyncMock
        ) as mock_collect:
            mock_collect.return_value = "I believe we need stronger enforcement mechanisms for climate policies."

            result2 = await nodes.collect_minority_considerations_node(
                sample_state_with_vote
            )

            assert "active_vote" in result2
            assert "minority_considerations" in result2["active_vote"]
            assert len(result2["active_vote"]["minority_considerations"]) == 1
            mock_collect.assert_called_once_with(
                "agent2", "Climate Change Policy", sample_state_with_vote
            )

        # Update state with considerations
        sample_state_with_vote["active_vote"]["minority_considerations"] = result2[
            "active_vote"
        ]["minority_considerations"]

        # Step 3: Incorporate minority views into summary
        with patch.object(
            moderator_agent, "incorporate_minority_views", new_callable=AsyncMock
        ) as mock_incorporate:
            mock_incorporate.return_value = (
                "Comprehensive discussion on climate change policy approaches, covering mitigation strategies, "
                "adaptation measures, and international cooperation frameworks. Minority perspective emphasized "
                "the need for stronger enforcement mechanisms for climate policies."
            )

            result3 = await nodes.incorporate_minority_views_node(
                sample_state_with_vote
            )

            assert "topic_summaries" in result3
            assert (
                "stronger enforcement mechanisms"
                in result3["topic_summaries"]["Climate Change Policy"]
            )
            mock_incorporate.assert_called_once()

    @pytest.mark.asyncio
    async def test_story_38_report_writer_workflow(
        self, moderator_agent, sample_state_with_vote
    ):
        """Test complete Story 3.8 workflow: Report Writer Mode Implementation."""
        from virtual_agora.agents.moderator_nodes import ModeratorNodes

        # Add multiple topic summaries for report generation
        sample_state_with_vote["topic_summaries"].update(
            {
                "Economic Impact": "Analysis of economic implications of climate policies on various sectors.",
                "Technology Solutions": "Review of technological approaches to climate change mitigation.",
            }
        )

        nodes = ModeratorNodes(moderator_agent)

        # Step 1: Initialize report structure
        with patch.object(
            moderator_agent, "define_report_structure", new_callable=AsyncMock
        ) as mock_structure:
            mock_structure.return_value = [
                "Executive Summary",
                "Policy Analysis",
                "Economic Considerations",
                "Technology Assessment",
                "Recommendations",
            ]

            result1 = await nodes.initialize_report_structure_node(
                sample_state_with_vote
            )

            assert result1["report_generation_status"] == "structuring"
            assert len(result1["report_structure"]) == 5
            assert result1["report_sections"] == {}
            mock_structure.assert_called_once()

        # Update state with structure
        sample_state_with_vote.update(result1)

        # Step 2: Generate report sections (simulate generating first section)
        with patch.object(
            moderator_agent, "generate_report_section", new_callable=AsyncMock
        ) as mock_section:
            mock_section.return_value = (
                "This report synthesizes insights from a comprehensive discussion on climate change policy, "
                "economic impacts, and technology solutions."
            )

            result2 = await nodes.generate_report_section_node(sample_state_with_vote)

            assert result2["report_generation_status"] == "writing"
            assert "Executive Summary" in result2["report_sections"]
            assert (
                "synthesizes insights"
                in result2["report_sections"]["Executive Summary"]
            )
            mock_section.assert_called_once()

        # Update state and generate remaining sections
        sample_state_with_vote["report_sections"] = result2["report_sections"]

        # Simulate completing all sections
        sample_state_with_vote["report_sections"] = {
            "Executive Summary": "Executive summary content",
            "Policy Analysis": "Policy analysis content",
            "Economic Considerations": "Economic considerations content",
            "Technology Assessment": "Technology assessment content",
            "Recommendations": "Recommendations content",
        }

        # Step 3: Finalize report
        result3 = await nodes.finalize_report_node(sample_state_with_vote)

        assert result3["report_generation_status"] == "completed"
        assert "5 sections" in result3["final_report"]

    @pytest.mark.asyncio
    async def test_story_39_agenda_modification_workflow(
        self, moderator_agent, sample_state_with_vote
    ):
        """Test complete Story 3.9 workflow: Agenda Modification Facilitation."""
        from virtual_agora.agents.moderator_nodes import ModeratorNodes

        # Mark first topic as completed
        sample_state_with_vote["completed_topics"] = ["Climate Change Policy"]
        sample_state_with_vote["active_topic"] = None

        nodes = ModeratorNodes(moderator_agent)

        # Step 1: Request agenda modifications
        with patch.object(
            moderator_agent, "request_agenda_modification", new_callable=AsyncMock
        ) as mock_request:
            mock_request.side_effect = [
                "Add topic: Carbon Pricing Mechanisms",
                "Remove: Technology Solutions - covered in climate policy",
                "Add topic: International Climate Agreements",
            ]

            result1 = await nodes.request_agenda_modifications_node(
                sample_state_with_vote
            )

            assert "pending_agenda_modifications" in result1
            assert len(result1["pending_agenda_modifications"]) == 3
            assert (
                "Carbon Pricing Mechanisms"
                in result1["pending_agenda_modifications"][0]
            )
            assert mock_request.call_count == 3  # Called for each agent

        # Update state with modifications
        sample_state_with_vote["pending_agenda_modifications"] = result1[
            "pending_agenda_modifications"
        ]

        # Step 2: Synthesize agenda changes
        with patch.object(
            moderator_agent, "synthesize_agenda_modifications", new_callable=AsyncMock
        ) as mock_synthesize:
            mock_synthesize.return_value = [
                "Economic Impact",
                "Carbon Pricing Mechanisms",
                "International Climate Agreements",
            ]

            result2 = await nodes.synthesize_agenda_changes_node(sample_state_with_vote)

            assert "proposed_topics" in result2
            assert len(result2["proposed_topics"]) == 3
            assert "Carbon Pricing Mechanisms" in result2["proposed_topics"]
            assert result2["pending_agenda_modifications"] == []
            mock_synthesize.assert_called_once()

        # Update state with proposed topics
        sample_state_with_vote["proposed_topics"] = result2["proposed_topics"]

        # Step 3: Facilitate agenda re-vote
        with patch.object(
            moderator_agent, "collect_agenda_vote", new_callable=AsyncMock
        ) as mock_vote:
            mock_vote.side_effect = [
                "1. Carbon Pricing Mechanisms, 2. Economic Impact, 3. International Climate Agreements",
                "1. Economic Impact, 2. International Climate Agreements, 3. Carbon Pricing Mechanisms",
                "1. International Climate Agreements, 2. Carbon Pricing Mechanisms, 3. Economic Impact",
            ]

            with patch.object(
                moderator_agent, "synthesize_agenda_votes", new_callable=AsyncMock
            ) as mock_synthesize_votes:
                mock_synthesize_votes.return_value = [
                    "Carbon Pricing Mechanisms",
                    "Economic Impact",
                    "International Climate Agreements",
                ]

                result3 = await nodes.facilitate_agenda_revote_node(
                    sample_state_with_vote
                )

                assert "topic_queue" in result3
                assert result3["topic_queue"][0] == "Carbon Pricing Mechanisms"
                assert "agenda_modification_votes" in result3
                assert len(result3["agenda_modification_votes"]) == 3
                mock_vote.assert_called()
                mock_synthesize_votes.assert_called_once()

    @pytest.mark.asyncio
    async def test_end_to_end_topic_conclusion_workflow(
        self, moderator_agent, sample_state_with_vote
    ):
        """Test end-to-end workflow from topic conclusion through minority considerations."""
        from virtual_agora.agents.moderator_nodes import ModeratorNodes

        nodes = ModeratorNodes(moderator_agent)

        # Mock all the required methods
        with (
            patch.object(
                moderator_agent,
                "collect_minority_consideration",
                new_callable=AsyncMock,
            ) as mock_collect,
            patch.object(
                moderator_agent, "incorporate_minority_views", new_callable=AsyncMock
            ) as mock_incorporate,
        ):

            mock_collect.return_value = (
                "Strong enforcement needed for effective climate policy"
            )
            mock_incorporate.return_value = "Updated summary with minority view: Strong enforcement needed for effective climate policy"

            # Complete minority considerations workflow
            result1 = await nodes.identify_minority_voters_node(sample_state_with_vote)
            sample_state_with_vote["active_vote"]["minority_voters"] = result1[
                "active_vote"
            ]["minority_voters"]

            result2 = await nodes.collect_minority_considerations_node(
                sample_state_with_vote
            )
            sample_state_with_vote["active_vote"]["minority_considerations"] = result2[
                "active_vote"
            ]["minority_considerations"]

            result3 = await nodes.incorporate_minority_views_node(
                sample_state_with_vote
            )

            # Verify the complete workflow
            assert "agent2" in sample_state_with_vote["active_vote"]["minority_voters"]
            assert (
                len(sample_state_with_vote["active_vote"]["minority_considerations"])
                == 1
            )
            assert (
                "Strong enforcement needed"
                in result3["topic_summaries"]["Climate Change Policy"]
            )

            # Verify method calls
            mock_collect.assert_called_once_with(
                "agent2", "Climate Change Policy", sample_state_with_vote
            )
            mock_incorporate.assert_called_once()

    def test_node_error_recovery(self, moderator_agent, sample_state_with_vote):
        """Test that node functions properly handle and recover from errors."""
        from virtual_agora.agents.moderator_nodes import ModeratorNodes

        nodes = ModeratorNodes(moderator_agent)

        # Test with corrupted state
        corrupted_state = {"invalid": "state_structure"}

        # All node functions should handle errors gracefully
        import asyncio

        async def test_error_handling():
            result = await nodes.identify_minority_voters_node(corrupted_state)
            # Should return error information, not crash
            assert isinstance(result, dict)
            # Should contain error information
            assert "last_error" in result or "warnings" in result

        # Run the async test
        asyncio.run(test_error_handling())
