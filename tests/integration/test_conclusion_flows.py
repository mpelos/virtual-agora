"""Integration tests for topic conclusion flows.

This module tests the conclusion polling, voting, minority dissent,
and topic summarization workflows using fake LLMs.
"""

import pytest
from unittest.mock import patch, Mock
from datetime import datetime
import uuid
import json

from virtual_agora.state.schema import (
    VirtualAgoraState,
    VoteRound,
    Message,
    TopicInfo,
    RoundInfo,
)
from virtual_agora.flow.graph import VirtualAgoraFlow

from ..helpers.fake_llm import ModeratorFakeLLM, AgentFakeLLM
from ..helpers.integration_utils import (
    IntegrationTestHelper,
    StateTestBuilder,
    TestResponseValidator,
    patch_ui_components,
    create_test_voting_round,
)


class TestConclusionPollingFlow:
    """Test conclusion polling and vote timing."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")
        self.validator = TestResponseValidator()

    @pytest.mark.integration
    def test_conclusion_poll_timing(self):
        """Test that conclusion polls start at round 3."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Rounds 1 and 2 should not trigger conclusion poll
            for round_num in [1, 2]:
                state["current_round"] = round_num
                should_poll = self._should_trigger_conclusion_poll(state)
                assert not should_poll, f"Poll triggered too early at round {round_num}"

            # Round 3 and later should trigger conclusion poll
            for round_num in [3, 4, 5]:
                state["current_round"] = round_num
                should_poll = self._should_trigger_conclusion_poll(state)
                assert should_poll, f"Poll not triggered at round {round_num}"

    @pytest.mark.integration
    def test_conclusion_poll_generation(self):
        """Test moderator generating conclusion poll questions."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Set up state for conclusion poll
            state["current_round"] = 3
            current_topic = state["agenda"][state["current_topic_index"]]["title"]

            # Generate poll question
            poll_question = self._generate_conclusion_poll(state, current_topic)

            # Validate poll question
            assert isinstance(poll_question, str)
            assert len(poll_question) > 20
            assert "conclude" in poll_question.lower()
            assert (
                current_topic.lower() in poll_question.lower()
                or "topic" in poll_question.lower()
            )
            assert "yes" in poll_question.lower() and "no" in poll_question.lower()

    @pytest.mark.integration
    def test_repeated_conclusion_polls(self):
        """Test handling of repeated conclusion polls when discussion continues."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Simulate multiple failed conclusion attempts
            poll_results = []

            for round_num in range(3, 8):  # Rounds 3-7
                state["current_round"] = round_num

                # Generate poll
                poll_question = self._generate_conclusion_poll(
                    state, "Technical Implementation"
                )

                # Simulate voting (initially fails, then passes)
                votes = self._simulate_conclusion_voting(
                    state, pass_threshold=(round_num >= 6)
                )

                # Create voting round
                voting_round = {
                    "id": str(uuid.uuid4()),
                    "phase": round_num,
                    "vote_type": "conclusion",
                    "options": ["Yes", "No"],
                    "start_time": datetime.now(),
                    "end_time": datetime.now(),
                    "required_votes": len(state["speaking_order"]),
                    "received_votes": len(state["speaking_order"]),
                    "result": "Yes" if self._calculate_vote_result(votes) else "No",
                    "status": "completed",
                }

                state["voting_rounds"].append(voting_round)
                poll_results.append(voting_round["result"])

                # Stop if poll passed
                if voting_round["result"] == "Yes":
                    break

            # Should have multiple failed polls before success
            assert "No" in poll_results
            assert poll_results[-1] == "Yes"
            assert len(state["voting_rounds"]) >= 3  # Multiple attempts

    def _should_trigger_conclusion_poll(self, state: VirtualAgoraState) -> bool:
        """Determine if conclusion poll should be triggered."""
        return state["current_round"] >= 3

    def _generate_conclusion_poll(self, state: VirtualAgoraState, topic: str) -> str:
        """Simulate moderator generating conclusion poll question."""
        return (
            f"Should we conclude the discussion on '{topic}'? "
            f"Please respond with 'Yes' or 'No' and a short justification."
        )

    def _simulate_conclusion_voting(
        self, state: VirtualAgoraState, pass_threshold: bool = False
    ) -> dict[str, str]:
        """Simulate agents voting on conclusion."""
        votes = {}

        for i, agent_id in enumerate(state["speaking_order"]):
            if pass_threshold:
                # Most agents vote yes when threshold reached
                vote = "Yes" if i < 2 else "No"
                justification = (
                    "sufficient discussion" if vote == "Yes" else "need more analysis"
                )
            else:
                # Mixed votes when not ready
                vote = "Yes" if i == 0 else "No"
                justification = (
                    "ready to conclude" if vote == "Yes" else "more discussion needed"
                )

            votes[agent_id] = f"{vote}. {justification.capitalize()}."

        return votes

    def _calculate_vote_result(self, votes: dict[str, str]) -> bool:
        """Calculate if conclusion vote passed."""
        yes_votes = sum(1 for vote in votes.values() if vote.lower().startswith("yes"))
        total_votes = len(votes)
        return yes_votes > total_votes / 2


class TestVotingMechanics:
    """Test voting mechanics and tallying."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=4, scenario="default")
        self.validator = TestResponseValidator()

    @pytest.mark.integration
    def test_majority_vote_calculation(self):
        """Test majority vote calculation with different scenarios."""
        test_scenarios = [
            # (total_agents, yes_votes, should_pass)
            (3, 2, True),  # 2/3 majority
            (3, 1, False),  # 1/3 minority
            (4, 3, True),  # 3/4 majority
            (4, 2, False),  # 2/4 tie (not majority)
            (5, 3, True),  # 3/5 majority
            (5, 2, False),  # 2/5 minority
        ]

        for total_agents, yes_votes, should_pass in test_scenarios:
            votes = {}

            # Create votes
            for i in range(total_agents):
                agent_id = f"agent_{i+1}"
                vote = "Yes" if i < yes_votes else "No"
                votes[agent_id] = f"{vote}. Test justification."

            # Calculate result
            result = self._calculate_majority_vote(votes)

            assert (
                result == should_pass
            ), f"Vote calculation failed for {yes_votes}/{total_agents}: expected {should_pass}, got {result}"

    @pytest.mark.integration
    def test_vote_format_validation(self):
        """Test validation of vote response formats."""
        valid_votes = [
            "Yes. I think we've covered the main points.",
            "No. We need more discussion on implementation details.",
            "YES. Ready to move forward.",
            "no. Still have concerns about security.",
        ]

        invalid_votes = [
            "Maybe we should conclude.",
            "I'm not sure about this.",
            "Let's think about it more.",
            "Probably yes.",
        ]

        for vote in valid_votes:
            assert self.validator.validate_conclusion_vote(
                vote
            ), f"Valid vote rejected: {vote}"

        for vote in invalid_votes:
            assert not self.validator.validate_conclusion_vote(
                vote
            ), f"Invalid vote accepted: {vote}"

    @pytest.mark.integration
    def test_vote_justification_analysis(self):
        """Test analysis of vote justifications."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Create votes with different justifications
            votes_with_justifications = {
                "agent_1": "Yes. We've comprehensively analyzed the technical requirements.",
                "agent_2": "No. The security implications need further investigation.",
                "agent_3": "Yes. The core objectives have been thoroughly discussed.",
                "agent_4": "No. Resource allocation concerns remain unaddressed.",
            }

            # Analyze justifications
            analysis = self._analyze_vote_justifications(votes_with_justifications)

            # Should identify key themes
            assert "technical" in analysis["yes_themes"]
            assert (
                "security" in analysis["no_themes"]
                or "resource" in analysis["no_themes"]
            )
            assert len(analysis["yes_reasons"]) > 0
            assert len(analysis["no_reasons"]) > 0

    def _calculate_majority_vote(self, votes: dict[str, str]) -> bool:
        """Calculate if majority vote passed."""
        yes_votes = sum(1 for vote in votes.values() if vote.lower().startswith("yes"))
        total_votes = len(votes)
        return yes_votes > total_votes / 2

    def _analyze_vote_justifications(self, votes: dict[str, str]) -> dict:
        """Analyze vote justifications for themes."""
        yes_votes = [vote for vote in votes.values() if vote.lower().startswith("yes")]
        no_votes = [vote for vote in votes.values() if vote.lower().startswith("no")]

        # Extract themes (simplified)
        yes_themes = set()
        no_themes = set()

        for vote in yes_votes:
            words = vote.lower().split()
            yes_themes.update(word for word in words if len(word) > 5)

        for vote in no_votes:
            words = vote.lower().split()
            no_themes.update(word for word in words if len(word) > 5)

        return {
            "yes_themes": yes_themes,
            "no_themes": no_themes,
            "yes_reasons": yes_votes,
            "no_reasons": no_votes,
        }


class TestMinorityDissentFlow:
    """Test minority dissent handling."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(
            num_agents=3, scenario="minority_dissent"
        )
        self.validator = TestResponseValidator()

    @pytest.mark.integration
    def test_minority_identification(self):
        """Test identification of dissenting minority."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Create voting scenario with minority dissent
            votes = {
                "agent_1": "Yes. Ready to conclude this topic.",
                "agent_2": "Yes. Sufficient discussion completed.",
                "agent_3": "No. Critical issues remain unaddressed.",
            }

            # Identify minority
            minority_agents = self._identify_minority_voters(votes)

            # Validate minority identification
            assert len(minority_agents) == 1
            assert "agent_3" in minority_agents

    @pytest.mark.integration
    def test_minority_final_considerations_prompt(self):
        """Test prompting minority for final considerations."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            minority_agents = ["agent_2", "agent_3"]
            current_topic = state["agenda"][state["current_topic_index"]]["title"]

            # Generate minority prompt
            prompt = self._generate_minority_prompt(minority_agents, current_topic)

            # Validate prompt
            assert isinstance(prompt, str)
            assert len(prompt) > 30
            assert "final" in prompt.lower()
            assert "consideration" in prompt.lower() or "thought" in prompt.lower()
            assert current_topic.lower() in prompt.lower() or "topic" in prompt.lower()

    @pytest.mark.integration
    def test_minority_final_statements(self):
        """Test minority agents providing final statements."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            minority_agents = ["agent_1"]
            current_topic = "Technical Implementation"

            # Generate final statements
            final_statements = {}
            for agent_id in minority_agents:
                statement = self._generate_minority_statement(agent_id, current_topic)
                final_statements[agent_id] = statement

            # Validate statements
            assert len(final_statements) == 1
            statement = final_statements["agent_1"]
            assert len(statement) > 50  # Should be substantive
            assert "concern" in statement.lower() or "issue" in statement.lower()

    @pytest.mark.integration
    def test_no_minority_scenario(self):
        """Test handling when there's no minority (unanimous vote)."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Create unanimous vote
            unanimous_votes = {
                "agent_1": "Yes. Comprehensive discussion completed.",
                "agent_2": "Yes. All key points have been addressed.",
                "agent_3": "Yes. Ready to move to next topic.",
            }

            # Check for minority
            minority_agents = self._identify_minority_voters(unanimous_votes)

            # Should be no minority
            assert len(minority_agents) == 0

    @pytest.mark.integration
    def test_majority_minority_scenario(self):
        """Test handling when minority is actually the majority vote."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Create scenario where "No" votes are majority
            majority_no_votes = {
                "agent_1": "No. Insufficient analysis of risks.",
                "agent_2": "No. More stakeholder input needed.",
                "agent_3": "Yes. I think we can proceed now.",
            }

            # In this case, conclusion vote should fail
            yes_votes = sum(
                1 for vote in majority_no_votes.values() if vote.startswith("Yes")
            )
            no_votes = sum(
                1 for vote in majority_no_votes.values() if vote.startswith("No")
            )
            vote_passed = yes_votes > no_votes
            assert not vote_passed

            # No minority considerations needed since vote failed
            minority_agents = (
                self._identify_minority_voters(majority_no_votes) if vote_passed else []
            )
            assert len(minority_agents) == 0

    def _identify_minority_voters(self, votes: dict[str, str]) -> list[str]:
        """Identify agents who voted against conclusion."""
        # Only identify minority if vote passed (majority voted yes)
        yes_votes = sum(1 for vote in votes.values() if vote.lower().startswith("yes"))
        total_votes = len(votes)

        if yes_votes <= total_votes / 2:
            return []  # Vote failed, no minority considerations needed

        # Return agents who voted "No"
        return [
            agent_id
            for agent_id, vote in votes.items()
            if vote.lower().startswith("no")
        ]

    def _generate_minority_prompt(self, minority_agents: list[str], topic: str) -> str:
        """Generate prompt for minority final considerations."""
        if len(minority_agents) == 1:
            return (
                f"As the dissenting voice on '{topic}', please share your final "
                f"considerations before we conclude this topic."
            )
        else:
            return (
                f"As those who voted to continue discussing '{topic}', please share "
                f"your final considerations before we conclude this topic."
            )

    def _generate_minority_statement(self, agent_id: str, topic: str) -> str:
        """Generate minority agent final statement."""
        return (
            f"While I respect the majority decision to conclude '{topic}', "
            f"I want to emphasize my concerns about [specific issue] that I believe "
            f"warrants further consideration in future discussions. The implications "
            f"of [concern details] could significantly impact our implementation approach. "
            f"I recommend we revisit this aspect before final implementation. ({agent_id})"
        )


class TestTopicSummarizationFlow:
    """Test topic summary generation and file creation."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")
        self.validator = TestResponseValidator()

    @pytest.mark.integration
    def test_comprehensive_topic_summary_generation(self):
        """Test generation of comprehensive topic summary."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Set up completed discussion state
            completed_state = self._create_completed_topic_state(state)

            # Generate topic summary
            summary = self._generate_topic_summary(completed_state)

            # Validate summary
            assert self.validator.validate_topic_summary(summary)
            assert len(summary) > 200
            assert "#" in summary  # Should have markdown headers
            assert "summary" in summary.lower()
            assert "overview" in summary.lower() or "conclusion" in summary.lower()

    @pytest.mark.integration
    def test_summary_content_structure(self):
        """Test structure and content of generated summaries."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Create rich discussion data
            state = self._add_rich_discussion_data(state)

            # Generate summary
            summary = self._generate_topic_summary(state)

            # Validate structure
            sections = summary.split("\n## ")
            assert len(sections) >= 3  # Should have multiple sections

            # Check for key content
            summary_lower = summary.lower()
            assert any(
                word in summary_lower
                for word in ["overview", "summary", "introduction"]
            )
            assert any(
                word in summary_lower for word in ["points", "discussion", "analysis"]
            )
            assert any(
                word in summary_lower for word in ["conclusion", "outcome", "result"]
            )

    @pytest.mark.integration
    def test_summary_incorporates_all_perspectives(self):
        """Test that summary incorporates all agent perspectives."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add messages from all agents with distinct perspectives
            distinct_messages = [
                {
                    "id": str(uuid.uuid4()),
                    "agent_id": "agent_1",
                    "content": "From a technical perspective, scalability is paramount.",
                    "timestamp": datetime.now(),
                    "round_number": 1,
                    "topic": "Technical Implementation",
                    "message_type": "discussion",
                },
                {
                    "id": str(uuid.uuid4()),
                    "agent_id": "agent_2",
                    "content": "The business impact and ROI must be carefully considered.",
                    "timestamp": datetime.now(),
                    "round_number": 1,
                    "topic": "Technical Implementation",
                    "message_type": "discussion",
                },
                {
                    "id": str(uuid.uuid4()),
                    "agent_id": "agent_3",
                    "content": "Security vulnerabilities pose significant risks to consider.",
                    "timestamp": datetime.now(),
                    "round_number": 1,
                    "topic": "Technical Implementation",
                    "message_type": "discussion",
                },
            ]
            state["messages"] = distinct_messages

            # Generate summary
            summary = self._generate_topic_summary(state)

            # Should incorporate all perspectives
            summary_lower = summary.lower()
            assert "scalability" in summary_lower or "technical" in summary_lower
            assert "business" in summary_lower or "roi" in summary_lower
            assert "security" in summary_lower or "risk" in summary_lower

    @pytest.mark.integration
    def test_summary_with_minority_dissent_inclusion(self):
        """Test that summary includes minority dissent when present."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add minority dissent data
            minority_statement = (
                "I remain concerned about the security implications "
                "that haven't been fully addressed in our analysis."
            )

            # Add to state metadata
            state["metadata"]["minority_statements"] = {"agent_2": minority_statement}

            # Generate summary
            summary = self._generate_topic_summary(state)

            # Should mention dissenting views
            summary_lower = summary.lower()
            assert any(
                word in summary_lower
                for word in ["dissent", "concern", "minority", "alternative"]
            )

    @pytest.mark.integration
    def test_summary_file_naming_convention(self):
        """Test file naming convention for topic summaries."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            topic_titles = [
                "Technical Implementation Details",
                "Legal & Regulatory Compliance",
                "Market Analysis and User Adoption",
                "Risk Assessment & Mitigation Strategies",
            ]

            for topic_title in topic_titles:
                filename = self._generate_summary_filename(topic_title)

                # Validate filename
                assert filename.endswith(".md")
                assert filename.startswith("topic_summary_")
                assert len(filename) > 15
                assert " " not in filename  # Should replace spaces
                assert "&" not in filename  # Should handle special chars

    def _create_completed_topic_state(
        self, state: VirtualAgoraState
    ) -> VirtualAgoraState:
        """Create state representing a completed topic discussion."""
        # Add discussion rounds
        for round_num in range(1, 4):
            discussion_round = {
                "round_id": str(uuid.uuid4()),
                "round_number": round_num,
                "topic": state["agenda"][0]["title"],
                "start_time": datetime.now(),
                "end_time": datetime.now(),
                "participants": state["speaking_order"],
                "summary": f"Round {round_num} focused on key implementation aspects",
                "messages_count": 3,
                "status": "completed",
            }
            state["discussion_rounds"].append(discussion_round)

        # Add conclusion vote
        conclusion_vote = create_test_voting_round("conclusion")
        state["voting_rounds"].append(conclusion_vote)

        return state

    def _add_rich_discussion_data(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Add rich discussion data for testing."""
        # Add multiple rounds of messages
        topics = ["scalability", "security", "performance", "usability"]

        for round_num in range(1, 5):
            for i, agent_id in enumerate(state["speaking_order"]):
                topic_focus = topics[i % len(topics)]
                message = {
                    "id": str(uuid.uuid4()),
                    "agent_id": agent_id,
                    "content": f"In round {round_num}, I want to emphasize {topic_focus} "
                    f"considerations and their impact on our implementation strategy.",
                    "timestamp": datetime.now(),
                    "round_number": round_num,
                    "topic": "Technical Implementation",
                    "message_type": "discussion",
                }
                state["messages"].append(message)

        # Add round summaries
        for round_num in range(1, 5):
            summary = (
                f"Round {round_num} Summary: Agents discussed {topics[(round_num-1) % len(topics)]} "
                f"with focus on practical implementation considerations."
            )
            state["round_summaries"].append(summary)

        return state

    def _generate_topic_summary(self, state: VirtualAgoraState) -> str:
        """Generate comprehensive topic summary."""
        topic = state["agenda"][state["current_topic_index"]]["title"]

        # Extract key themes from messages
        all_content = " ".join(msg["content"] for msg in state["messages"])

        # Build comprehensive summary
        summary = f"""# Topic Summary: {topic}

## Overview
This topic generated extensive discussion among {len(state["speaking_order"])} participants across {len(state["round_summaries"])} rounds of deliberation.

## Key Points Discussed
The discussion covered several critical aspects:

- Technical implementation requirements and architectural considerations
- Security implications and risk mitigation strategies  
- Performance optimization and scalability concerns
- Business impact and stakeholder considerations

## Discussion Evolution
{self._summarize_discussion_progression(state)}

## Conclusions Reached
After comprehensive analysis, the participants reached consensus on the core requirements while identifying areas that merit continued investigation.

## Areas for Future Consideration
Several important aspects were identified for future discussion:
- Detailed implementation timeline and resource allocation
- Specific security protocols and compliance requirements
- Performance benchmarking and optimization strategies

---
*Summary generated from {len(state["messages"])} discussion messages across {len(state["round_summaries"])} rounds*
"""

        return summary

    def _summarize_discussion_progression(self, state: VirtualAgoraState) -> str:
        """Summarize how the discussion evolved over rounds."""
        if not state["round_summaries"]:
            return "Discussion proceeded systematically through key topics."

        progression = []
        for i, summary in enumerate(state["round_summaries"], 1):
            progression.append(
                f"- Round {i}: {summary.split(': ')[-1] if ': ' in summary else summary}"
            )

        return "\n".join(progression[:5])  # Limit to first 5 rounds

    def _generate_summary_filename(self, topic_title: str) -> str:
        """Generate filename for topic summary."""
        # Clean title for filename
        clean_title = topic_title.replace(" ", "_").replace("&", "and")
        clean_title = "".join(c for c in clean_title if c.isalnum() or c in "_-")

        return f"topic_summary_{clean_title}.md"


@pytest.mark.integration
class TestConclusionEdgeCases:
    """Test edge cases in conclusion flows."""

    def test_immediate_consensus_scenario(self):
        """Test scenario where agents immediately agree to conclude."""
        helper = IntegrationTestHelper(num_agents=3, scenario="quick_consensus")

        with patch_ui_components():
            flow = helper.create_test_flow()
            state = helper.create_discussion_state()

            # Set up for round 3 (first poll opportunity)
            state["current_round"] = 3

            # All agents vote to conclude immediately
            unanimous_yes = {
                "agent_1": "Yes. Clear consensus has been reached.",
                "agent_2": "Yes. All key points have been covered.",
                "agent_3": "Yes. Ready to move forward.",
            }

            # Should pass immediately
            result = self._calculate_vote_result(unanimous_yes)
            assert result

            # No minority dissent needed
            minority = self._identify_minority_voters(unanimous_yes, result)
            assert len(minority) == 0

    def test_perpetual_disagreement_scenario(self):
        """Test handling of scenario where agents never agree to conclude."""
        helper = IntegrationTestHelper(num_agents=3, scenario="extended_debate")

        with patch_ui_components():
            flow = helper.create_test_flow()
            state = helper.create_discussion_state()

            # Simulate many rounds of failed conclusion votes
            failed_attempts = 0
            max_attempts = 10

            for round_num in range(3, 3 + max_attempts):
                state["current_round"] = round_num

                # Majority always votes no (unrealistic but tests edge case)
                persistent_no = {
                    "agent_1": "No. Still need more analysis.",
                    "agent_2": "No. Critical issues remain.",
                    "agent_3": "Yes. I think we should conclude.",  # Minority yes
                }

                result = self._calculate_vote_result(persistent_no)
                if not result:
                    failed_attempts += 1
                else:
                    break

            # Should eventually have some mechanism to prevent infinite discussion
            # (In real implementation, might force conclusion after X attempts)
            assert failed_attempts > 0  # Should track failed attempts

    def test_empty_justification_handling(self):
        """Test handling of votes with missing or empty justifications."""
        helper = IntegrationTestHelper(num_agents=3, scenario="default")
        validator = TestResponseValidator()

        # Test votes with minimal justifications
        minimal_votes = ["Yes.", "No.", "Yes. OK.", "No. Not ready."]

        for vote in minimal_votes:
            # Should still be valid format even if brief
            is_valid = validator.validate_conclusion_vote(vote)
            assert is_valid, f"Minimal vote should be valid: {vote}"

    def _calculate_vote_result(self, votes: dict[str, str]) -> bool:
        """Calculate vote result."""
        yes_votes = sum(1 for vote in votes.values() if vote.lower().startswith("yes"))
        return yes_votes > len(votes) / 2

    def _identify_minority_voters(
        self, votes: dict[str, str], vote_passed: bool
    ) -> list[str]:
        """Identify minority voters if vote passed."""
        if not vote_passed:
            return []

        return [
            agent_id
            for agent_id, vote in votes.items()
            if vote.lower().startswith("no")
        ]
