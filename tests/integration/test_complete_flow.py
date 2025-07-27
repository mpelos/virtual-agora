"""Complete flow integration tests for Virtual Agora.

This module tests the entire discussion workflow from initialization
to final report generation using fake LLMs.
"""

import pytest
import json
from unittest.mock import patch, Mock
from pathlib import Path

from virtual_agora.config.loader import ConfigLoader
from virtual_agora.flow.graph import VirtualAgoraFlow
from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.config.models import Config as VirtualAgoraConfig

from ..helpers.fake_llm import create_fake_llm_pool, ModeratorFakeLLM, AgentFakeLLM
from ..helpers.integration_utils import (
    IntegrationTestHelper,
    StateTestBuilder,
    TestResponseValidator,
    patch_ui_components,
)


class TestCompleteFlow:
    """Test complete discussion flow scenarios."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")
        self.validator = TestResponseValidator()

    @pytest.mark.integration
    def test_minimal_discussion_flow(self):
        """Test minimal complete flow with 2 agents and 1 topic."""
        # Create minimal configuration
        helper = IntegrationTestHelper(num_agents=2, scenario="quick_consensus")

        # Configure user responses
        user_responses = {
            "topic": "Future of Remote Work",
            "approve": "y",
            "continue": "n",  # End after first topic
        }
        helper.simulate_user_interactions(user_responses)

        with patch_ui_components():
            flow = helper.create_test_flow()
            initial_state = helper.create_basic_state()

            # Run the flow (simulated)
            final_state = self._simulate_complete_flow(
                flow, initial_state, max_topics=1, max_rounds_per_topic=3
            )

            # Validate completion
            helper.assert_flow_completion(final_state)
            assert len(final_state.get("topic_summaries", [])) >= 1
            assert (
                final_state["current_phase"] == 5
                or final_state["current_phase"] == "completed"
            )

    @pytest.mark.integration
    def test_standard_discussion_flow(self):
        """Test standard flow with 3 agents and multiple topics."""
        user_responses = {
            "topic": "Artificial Intelligence Ethics",
            "approve": "y",
            "continue": "y",
            "final_continue": "n",
        }
        self.test_helper.simulate_user_interactions(user_responses)

        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            initial_state = self.test_helper.create_basic_state()

            # Run multi-topic discussion
            final_state = self._simulate_complete_flow(
                flow, initial_state, max_topics=3, max_rounds_per_topic=4
            )

            # Validate comprehensive completion
            self.test_helper.assert_flow_completion(final_state)
            assert len(final_state["agenda"]) >= 2
            assert len(final_state["topic_summaries"]) >= 2
            assert len(final_state["voting_rounds"]) > 0
            assert final_state["metadata"].get("report_generated", False)

    @pytest.mark.integration
    def test_extended_debate_flow(self):
        """Test flow with extended debate before consensus."""
        helper = IntegrationTestHelper(num_agents=4, scenario="extended_debate")

        user_responses = {
            "topic": "Universal Basic Income Implementation",
            "approve": "y",
            "continue": "y",
        }
        helper.simulate_user_interactions(user_responses)

        with patch_ui_components():
            flow = helper.create_test_flow()
            initial_state = helper.create_basic_state()

            # Run extended discussion
            final_state = self._simulate_complete_flow(
                flow, initial_state, max_topics=2, max_rounds_per_topic=8
            )

            # Validate extended discussion occurred
            assert final_state["current_round"] >= 6  # Should have many rounds
            assert len(final_state["round_summaries"]) > 6
            assert len(final_state["messages"]) > 20  # Lots of discussion

            # Should still reach completion
            helper.assert_flow_completion(final_state)

    @pytest.mark.integration
    def test_minority_dissent_flow(self):
        """Test flow with minority dissent scenarios."""
        helper = IntegrationTestHelper(num_agents=3, scenario="minority_dissent")

        user_responses = {
            "topic": "Climate Change Policy",
            "approve": "y",
            "continue": "n",
        }
        helper.simulate_user_interactions(user_responses)

        with patch_ui_components():
            flow = helper.create_test_flow()
            initial_state = helper.create_basic_state()

            final_state = self._simulate_complete_flow(
                flow, initial_state, max_topics=1, max_rounds_per_topic=5
            )

            # Validate minority dissent handling
            conclusion_votes = [
                vr
                for vr in final_state.get("voting_rounds", [])
                if vr.get("vote_type") == "conclusion"
            ]
            assert len(conclusion_votes) > 0

            # Should have some "No" votes before final consensus
            early_votes = conclusion_votes[:-1] if len(conclusion_votes) > 1 else []
            has_dissent = any(
                "no" in str(vr.get("votes", {}).values()).lower() for vr in early_votes
            )
            assert has_dissent

            helper.assert_flow_completion(final_state)

    @pytest.mark.integration
    def test_error_recovery_flow(self):
        """Test flow with simulated errors and recovery."""
        helper = IntegrationTestHelper(num_agents=2, scenario="error_prone")

        user_responses = {
            "topic": "Blockchain Technology",
            "approve": "y",
            "continue": "n",
        }
        helper.simulate_user_interactions(user_responses)

        with patch_ui_components():
            flow = helper.create_test_flow()
            initial_state = helper.create_basic_state()

            # Should handle errors gracefully and still complete
            final_state = self._simulate_complete_flow(
                flow,
                initial_state,
                max_topics=1,
                max_rounds_per_topic=4,
                expect_errors=True,
            )

            # Even with errors, should reach some form of completion
            assert final_state["current_phase"] in [
                5,
                "completed",
                "error_recovery",
            ]  # 5 is final_report phase
            assert len(final_state["messages"]) > 0  # Some progress made

    @pytest.mark.integration
    def test_agenda_modification_flow(self):
        """Test flow with agenda modifications between topics."""
        user_responses = {
            "topic": "Digital Transformation Strategy",
            "approve": "y",
            "continue": "y",  # Continue after first topic
            "modify": "y",  # Modify agenda
            "final_continue": "n",
        }
        self.test_helper.simulate_user_interactions(user_responses)

        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            initial_state = self.test_helper.create_basic_state()

            final_state = self._simulate_complete_flow(
                flow,
                initial_state,
                max_topics=2,
                max_rounds_per_topic=3,
                enable_modifications=True,
            )

            # Should have gone through agenda modification
            assert (
                len(final_state.get("voting_rounds", [])) > 3
            )  # Multiple voting rounds
            self.test_helper.assert_flow_completion(final_state)

    @pytest.mark.integration
    def test_context_compression_flow(self):
        """Test flow with context window compression."""
        helper = IntegrationTestHelper(num_agents=4, scenario="default")

        user_responses = {
            "topic": "Large Scale System Architecture",
            "approve": "y",
            "continue": "y",
        }
        helper.simulate_user_interactions(user_responses)

        with patch_ui_components():
            flow = helper.create_test_flow()
            initial_state = helper.create_basic_state()

            # Run with many rounds to trigger compression
            final_state = self._simulate_complete_flow(
                flow, initial_state, max_topics=1, max_rounds_per_topic=15
            )

            # Should have triggered context compression
            assert len(final_state.get("round_summaries", [])) > 10
            # Original messages might be compressed, but summaries preserved
            helper.assert_flow_completion(final_state)

    def _simulate_complete_flow(
        self,
        flow: VirtualAgoraFlow,
        initial_state: VirtualAgoraState,
        max_topics: int = 2,
        max_rounds_per_topic: int = 5,
        expect_errors: bool = False,
        enable_modifications: bool = False,
    ) -> VirtualAgoraState:
        """Simulate a complete discussion flow.

        This is a simplified simulation of what would happen when running
        the actual LangGraph flow with fake LLMs.
        """
        state = initial_state

        # Phase 0: Initialization
        state = self._simulate_initialization(state)

        topics_completed = 0
        while topics_completed < max_topics and state.get("current_phase") != 5:

            # Phase 1: Agenda Setting (if needed)
            if not state.get("agenda") or state["current_phase"] == 1:
                state = self._simulate_agenda_setting(state)
                state = self._simulate_agenda_approval(state)

            # Phase 2: Topic Discussion
            if state.get("agenda") and state.get("current_topic_index", 0) < len(
                state["agenda"]
            ):
                state = self._simulate_topic_discussion(state, max_rounds_per_topic)

                # Phase 3: Topic Conclusion
                state = self._simulate_topic_conclusion(state)
                state = self._simulate_topic_summary(state)

                topics_completed += 1

                # Phase 4: Continuation Decision
                if topics_completed < max_topics:
                    state = self._simulate_continuation_approval(
                        state, enable_modifications
                    )
                    if enable_modifications and topics_completed == 1:
                        state = self._simulate_agenda_modification(state)

        # Phase 5: Final Report Generation
        if topics_completed > 0:
            state = self._simulate_report_generation(state)
            state["current_phase"] = 5  # final_report phase
            if "metadata" not in state:
                state["metadata"] = {}
            state["metadata"]["report_generated"] = True

        return state

    def _simulate_initialization(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Simulate initialization phase."""
        # Add agents
        agents = {}
        for i in range(self.test_helper.num_agents):
            agent_id = f"agent_{i+1}"
            agents[agent_id] = self._create_test_agent_info(agent_id)

        state["agents"] = agents
        state["speaking_order"] = list(agents.keys())
        state["current_phase"] = 1  # agenda_setting

        return state

    def _simulate_agenda_setting(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Simulate agenda setting phase."""
        # Simulate moderator requesting proposals and synthesizing agenda
        topics = [
            {
                "title": "Technical Implementation",
                "description": "Technical aspects and requirements",
                "proposed_by": "agent_1",
                "votes_for": 2,
                "votes_against": 1,
                "status": "pending",
            },
            {
                "title": "Social Impact",
                "description": "Social implications and adoption",
                "proposed_by": "agent_2",
                "votes_for": 3,
                "votes_against": 0,
                "status": "pending",
            },
            {
                "title": "Economic Considerations",
                "description": "Economic impact and sustainability",
                "proposed_by": "agent_3",
                "votes_for": 2,
                "votes_against": 1,
                "status": "pending",
            },
        ]

        state["agenda"] = topics
        state["current_phase"] = 1  # Still agenda_setting until approved

        return state

    def _simulate_agenda_approval(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Simulate HITL agenda approval."""
        state["hitl_state"]["approved"] = True
        state["current_phase"] = 2  # discussion
        state["current_topic_index"] = 0
        state["current_round"] = 1

        return state

    def _simulate_topic_discussion(
        self, state: VirtualAgoraState, max_rounds: int
    ) -> VirtualAgoraState:
        """Simulate topic discussion rounds."""
        from datetime import datetime
        import uuid

        # Store max_rounds in state for voting logic
        state["max_rounds_override"] = max_rounds

        for round_num in range(1, max_rounds + 1):
            state["current_round"] = round_num

            # Each agent speaks
            for i, agent_id in enumerate(state["speaking_order"]):
                state["current_speaker_index"] = i

                # Create message
                message = {
                    "id": str(uuid.uuid4()),
                    "agent_id": agent_id,
                    "content": f"Round {round_num} response from {agent_id} on topic {state.get('current_topic_index', 0)}",
                    "timestamp": datetime.now(),
                    "round_number": round_num,
                    "topic": (
                        state["agenda"][state.get("current_topic_index", 0)]["title"]
                        if state.get("agenda")
                        else "Test Topic"
                    ),
                    "message_type": "discussion",
                }
                if "messages" not in state:
                    state["messages"] = []
                state["messages"].append(message)

            # Add round summary
            summary = (
                f"Round {round_num} summary: Agents discussed key aspects of the topic."
            )
            if "round_summaries" not in state:
                state["round_summaries"] = []
            state["round_summaries"].append(summary)

            # Create discussion round record
            discussion_round = {
                "round_id": str(uuid.uuid4()),
                "round_number": round_num,
                "topic": (
                    state["agenda"][state.get("current_topic_index", 0)]["title"]
                    if state.get("agenda")
                    else "Test Topic"
                ),
                "start_time": datetime.now(),
                "end_time": datetime.now(),
                "participants": state["speaking_order"].copy(),
                "summary": summary,
                "messages_count": len(state["speaking_order"]),
                "status": "completed",
            }
            if "discussion_rounds" not in state:
                state["discussion_rounds"] = []
            state["discussion_rounds"].append(discussion_round)

            # Starting from round 3, conduct conclusion polls
            if round_num >= 3:
                # Simulate a conclusion vote
                state = self._simulate_topic_conclusion(state)

                # Check if vote passed
                last_vote = state["voting_rounds"][-1]
                if last_vote["result"] == "Yes":
                    # Topic concluded
                    break

        return state

    def _simulate_topic_conclusion(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Simulate topic conclusion voting."""
        from datetime import datetime
        import uuid

        # Simulate multiple voting rounds to show minority dissent
        if "voting_rounds" not in state:
            state["voting_rounds"] = []

        # Create voting rounds based on current round number
        # Early rounds should have dissent, later rounds reach consensus
        current_round = state.get("current_round", 3)

        # Check scenario type to determine voting behavior
        scenario = getattr(self.test_helper, "scenario", "default")

        # Check if we're in a long discussion mode (for context compression testing)
        max_rounds = state.get("max_rounds_override", None)

        # Simulate votes for this round
        votes = {}
        for agent_id in state["speaking_order"]:
            # Long discussion mode: vote No until we're close to max rounds
            if max_rounds and current_round < max_rounds - 2:
                vote = "No. We need more discussion to fully explore this topic."
            # Extended debate scenario: agents vote No until round 7+
            elif scenario == "extended_debate":
                if current_round < 7:
                    vote = "No. There are still important aspects to discuss in detail."
                else:
                    vote = "Yes. After thorough discussion, I'm ready to conclude."
            # Default scenario: In early rounds (3-4), some agents dissent
            elif current_round <= 4 and agent_id in ["agent_1", "agent_2"]:
                vote = "No. I believe we need more discussion on this topic."
            else:
                vote = "Yes. I think we've covered the key points sufficiently."
            votes[agent_id] = vote

        # Count votes to determine result
        yes_count = sum(1 for v in votes.values() if v.startswith("Yes"))
        no_count = sum(1 for v in votes.values() if v.startswith("No"))
        result = "Yes" if yes_count > no_count else "No"

        voting_round = {
            "id": str(uuid.uuid4()),
            "phase": current_round,
            "vote_type": "conclusion",
            "options": ["Yes", "No"],
            "start_time": datetime.now(),
            "end_time": datetime.now(),
            "required_votes": len(state["speaking_order"]),
            "received_votes": len(state["speaking_order"]),
            "votes": votes,  # Store individual votes
            "result": result,
            "status": "completed",
        }
        state["voting_rounds"].append(voting_round)

        # Track minority voters for dissent handling
        if result == "Yes":
            minority_voters = [
                agent_id for agent_id, vote in votes.items() if vote.startswith("No")
            ]
            if minority_voters:
                if "minority_dissent" not in state:
                    state["minority_dissent"] = []
                state["minority_dissent"].extend(minority_voters)

        return state

    def _simulate_topic_summary(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Simulate topic summary generation."""
        topic = state["agenda"][state["current_topic_index"]]

        summary = f"""# Topic Summary: {topic["title"]}

## Overview
Comprehensive discussion on {topic["title"]} covering multiple perspectives and considerations.

## Key Points Discussed
- Technical implementation requirements
- Stakeholder impact analysis  
- Resource allocation considerations
- Risk assessment and mitigation

## Conclusions
The agents reached consensus on core requirements while identifying areas for further investigation.
"""

        if "topic_summaries" not in state:
            state["topic_summaries"] = []
        state["topic_summaries"].append(summary)
        state["current_topic_index"] += 1

        return state

    def _simulate_continuation_approval(
        self, state: VirtualAgoraState, enable_modifications: bool = False
    ) -> VirtualAgoraState:
        """Simulate continuation approval."""
        if state.get("current_topic_index", 0) < len(state.get("agenda", [])):
            state["hitl_state"]["approved"] = True
            state["current_phase"] = (
                2 if not enable_modifications else 4
            )  # 2=discussion, 4=agenda_reevaluation
            state["current_round"] = 1
        else:
            state["current_phase"] = 5  # final_report

        return state

    def _simulate_agenda_modification(
        self, state: VirtualAgoraState
    ) -> VirtualAgoraState:
        """Simulate agenda modification."""
        # Add one more topic to test modification
        new_topic = {
            "title": "Implementation Timeline",
            "description": "Timeline and milestones for implementation",
            "proposed_by": "agent_2",
            "votes_for": 3,
            "votes_against": 0,
            "status": "pending",
        }
        if "agenda" not in state:
            state["agenda"] = []
        state["agenda"].append(new_topic)
        state["current_phase"] = 2  # discussion

        return state

    def _simulate_report_generation(
        self, state: VirtualAgoraState
    ) -> VirtualAgoraState:
        """Simulate final report generation."""
        # Simulate report structure and content generation
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["report_sections"] = [
            "Executive Summary",
            "Technical Analysis",
            "Social Impact Assessment",
            "Recommendations",
        ]

        return state

    def _create_test_agent_info(self, agent_id: str):
        """Create test agent info."""
        return {
            "agent_id": agent_id,
            "name": f"Test Agent {agent_id.split('_')[1]}",
            "role": "participant",
            "model": f"fake-model-{agent_id}",
            "provider": "fake_provider",
        }


@pytest.mark.integration
class TestFlowValidation:
    """Test flow validation and error handling."""

    def test_state_validation_throughout_flow(self):
        """Test that state remains valid throughout flow."""
        helper = IntegrationTestHelper(num_agents=2, scenario="default")
        validator = TestResponseValidator()

        with patch_ui_components():
            flow = helper.create_test_flow()
            initial_state = helper.create_basic_state()

            # Track state through key transitions
            states = []
            test_flow = TestCompleteFlow()
            test_flow.test_helper = helper

            current_state = initial_state
            states.append(("initial", current_state))

            # Initialization
            current_state = test_flow._simulate_initialization(current_state)
            states.append(("initialized", current_state))

            # Agenda setting
            current_state = test_flow._simulate_agenda_setting(current_state)
            states.append(("agenda_set", current_state))

            # Validate each state transition
            for i in range(1, len(states)):
                prev_state = states[i - 1][1]
                curr_state = states[i][1]

                is_valid = validator.validate_state_transition(prev_state, curr_state)
                assert (
                    is_valid
                ), f"Invalid transition from {states[i-1][0]} to {states[i][0]}"

                # Check flow validation
                issues = validator.validate_discussion_flow(curr_state)
                assert len(issues) == 0, f"Flow issues in {states[i][0]}: {issues}"

    def test_response_format_validation(self):
        """Test validation of LLM response formats."""
        validator = TestResponseValidator()

        # Test agenda synthesis validation
        valid_agenda = '{"proposed_agenda": ["Topic A", "Topic B", "Topic C"]}'
        assert validator.validate_agenda_synthesis(valid_agenda)

        invalid_agenda = '{"wrong_field": ["Topic A"]}'
        assert not validator.validate_agenda_synthesis(invalid_agenda)

        # Test conclusion vote validation
        assert validator.validate_conclusion_vote("Yes. I agree we should conclude.")
        assert validator.validate_conclusion_vote("No. More discussion needed.")
        assert not validator.validate_conclusion_vote("Maybe we should think about it.")

        # Test topic summary validation
        valid_summary = """# Topic Summary
        
## Overview
This is a comprehensive summary with sufficient detail.

## Key Points
- Point 1
- Point 2

## Conclusions
Final thoughts and conclusions.
"""
        assert validator.validate_topic_summary(valid_summary)

        invalid_summary = "Too short"
        assert not validator.validate_topic_summary(invalid_summary)
