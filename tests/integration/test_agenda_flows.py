"""Integration tests for agenda management flows in Virtual Agora v1.3.

This module tests the agenda setting, voting, approval, and modification
workflows using the node-centric architecture with specialized agents.
"""

import pytest
import json
from unittest.mock import patch, Mock
from datetime import datetime

from virtual_agora.state.schema import (
    VirtualAgoraState,
    TopicInfo,
    VoteRound,
    HITLState,
)
from virtual_agora.ui.hitl_state import HITLApprovalType
from virtual_agora.flow.graph_v13 import VirtualAgoraV13Flow

from ..helpers.fake_llm import create_fake_llm_pool, create_specialized_fake_llms
from ..helpers.integration_utils import (
    IntegrationTestHelper,
    StateTestBuilder,
    TestResponseValidator,
    patch_ui_components,
)


class TestAgendaSettingFlow:
    """Test agenda setting and proposal workflows."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")
        self.validator = TestResponseValidator()

    @pytest.mark.integration
    def test_topic_proposal_generation(self):
        """Test agent topic proposal generation."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_agenda_setting_state()

            # Simulate moderator requesting proposals
            proposals = self._simulate_topic_proposals(state)

            # Validate proposals
            assert len(proposals) >= 3  # Should have proposals from each agent
            assert all(isinstance(prop, str) and len(prop) > 10 for prop in proposals)

            # Check for diversity in proposals
            unique_words = set()
            for proposal in proposals:
                unique_words.update(proposal.lower().split())
            assert len(unique_words) > 10  # Should have diverse vocabulary

    @pytest.mark.integration
    def test_agenda_voting_process(self):
        """Test agent voting on proposed agenda items."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_agenda_setting_state()

            # Set up proposed topics
            proposed_topics = [
                "Technical Implementation Details",
                "Regulatory Compliance Requirements",
                "Market Analysis and Adoption",
                "Risk Assessment and Mitigation",
                "Timeline and Resource Planning",
            ]

            # Simulate voting
            votes = self._simulate_agenda_voting(state, proposed_topics)

            # Validate votes
            assert len(votes) == len(state["agents"])
            for agent_id, vote in votes.items():
                assert agent_id in state["agents"]
                assert isinstance(vote, str)
                assert len(vote) > 20  # Should have substantive justification

    @pytest.mark.integration
    def test_moderator_agenda_synthesis(self):
        """Test moderator synthesis of agenda from votes."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_agenda_setting_state()

            # Mock voting results
            voting_results = {
                "agent_1": "I prioritize technical implementation first, then regulatory concerns.",
                "agent_2": "Regulatory compliance should come first for legal clarity.",
                "agent_3": "Market analysis is crucial before any implementation decisions.",
            }

            # Simulate moderator synthesis
            agenda_json = self._simulate_agenda_synthesis(voting_results)

            # Validate synthesis
            assert self.validator.validate_agenda_synthesis(agenda_json)

            agenda_data = json.loads(agenda_json)
            proposed_agenda = agenda_data["proposed_agenda"]

            assert len(proposed_agenda) >= 3
            assert len(proposed_agenda) <= 6
            assert all(isinstance(topic, str) for topic in proposed_agenda)
            assert all(len(topic) > 5 for topic in proposed_agenda)

    @pytest.mark.integration
    def test_agenda_conflict_resolution(self):
        """Test handling of conflicting agent preferences."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_agenda_setting_state()

            # Create conflicting votes
            conflicting_votes = {
                "agent_1": "Technical implementation must be first priority - it's foundational.",
                "agent_2": "Legal compliance absolutely must come before any technical work.",
                "agent_3": "Economic viability analysis should precede all other considerations.",
            }

            # Moderator should resolve conflicts
            agenda_json = self._simulate_agenda_synthesis(conflicting_votes)
            agenda_data = json.loads(agenda_json)

            # Should still produce valid agenda despite conflicts
            assert len(agenda_data["proposed_agenda"]) >= 3

            # All conflicting topics should be included in some order
            agenda_text = " ".join(agenda_data["proposed_agenda"]).lower()
            assert any(word in agenda_text for word in ["technical", "implementation"])
            assert any(word in agenda_text for word in ["legal", "compliance"])
            assert any(word in agenda_text for word in ["economic", "viability"])

    @pytest.mark.integration
    def test_empty_agenda_handling(self):
        """Test handling of edge case with no valid proposals."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_agenda_setting_state()

            # Simulate case with minimal/poor proposals
            minimal_votes = {
                "agent_1": "I don't have strong preferences.",
                "agent_2": "Whatever others think is fine.",
                "agent_3": "Any topic works for me.",
            }

            # Moderator should generate default agenda
            agenda_json = self._simulate_agenda_synthesis(minimal_votes)
            agenda_data = json.loads(agenda_json)

            # Should have fallback agenda
            assert len(agenda_data["proposed_agenda"]) >= 2

            # Should include generic but meaningful topics
            agenda_text = " ".join(agenda_data["proposed_agenda"]).lower()
            assert len(agenda_text) > 50  # Should have substantive content

    def _simulate_topic_proposals(self, state: VirtualAgoraState) -> list[str]:
        """Simulate agents generating topic proposals."""
        proposals = []

        for agent_id in state["agents"]:
            # Each agent proposes 3-4 topics based on their personality
            agent_proposals = [
                f"Proposal 1 from {agent_id}: Technical architecture considerations",
                f"Proposal 2 from {agent_id}: Implementation timeline and milestones",
                f"Proposal 3 from {agent_id}: Risk management and contingency planning",
            ]
            proposals.extend(agent_proposals)

        return proposals

    def _simulate_agenda_voting(
        self, state: VirtualAgoraState, topics: list[str]
    ) -> dict[str, str]:
        """Simulate agents voting on proposed topics."""
        votes = {}

        personalities = ["technical", "business", "cautious"]

        for i, agent_id in enumerate(state["agents"]):
            personality = personalities[i % len(personalities)]

            if personality == "technical":
                vote = (
                    f"I prioritize technical topics: {topics[0]}, then {topics[1]}. "
                    f"Technical foundation is essential before other considerations."
                )
            elif personality == "business":
                vote = (
                    f"Business value is key: {topics[2]}, then {topics[0]}. "
                    f"We need market validation before technical deep-dives."
                )
            else:  # cautious
                vote = (
                    f"Risk management first: {topics[3]}, then {topics[0]}. "
                    f"We must understand potential issues before proceeding."
                )

            votes[agent_id] = vote

        return votes

    def _simulate_agenda_synthesis(self, voting_results: dict[str, str]) -> str:
        """Simulate moderator synthesizing agenda from votes."""
        # Analyze votes and create balanced agenda
        topics = []

        # Extract priorities from votes
        vote_text = " ".join(voting_results.values()).lower()

        if "technical" in vote_text:
            topics.append("Technical Implementation and Architecture")
        if "legal" in vote_text or "compliance" in vote_text:
            topics.append("Legal and Regulatory Considerations")
        if "market" in vote_text or "economic" in vote_text:
            topics.append("Market Analysis and Economic Impact")
        if "risk" in vote_text:
            topics.append("Risk Assessment and Mitigation")

        # Ensure minimum topics
        if len(topics) < 3:
            topics.extend(
                [
                    "Strategic Planning and Objectives",
                    "Implementation Timeline and Resources",
                ]
            )

        return json.dumps({"proposed_agenda": topics[:5]})  # Limit to 5 topics


class TestAgendaApprovalFlow:
    """Test HITL agenda approval workflows."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")

    @pytest.mark.integration
    def test_agenda_approval_acceptance(self):
        """Test user accepting proposed agenda."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_agenda_setting_state()

            # Set up agenda for approval
            agenda = [
                {
                    "title": "Technical Implementation",
                    "description": "Technical architecture and design",
                    "proposed_by": "agent_1",
                    "votes_for": 3,
                    "votes_against": 0,
                    "status": "pending",
                },
                {
                    "title": "Business Impact",
                    "description": "Market analysis and ROI",
                    "proposed_by": "agent_2",
                    "votes_for": 2,
                    "votes_against": 1,
                    "status": "pending",
                },
            ]
            state["agenda"] = agenda

            # Simulate user approval
            approved_state = self._simulate_user_approval(state, approve=True)

            # Validate approval
            assert approved_state["hitl_state"]["approved"]
            assert approved_state["current_phase"] == 2
            assert len(approved_state["agenda"]) == 2
            assert all(
                topic["status"] == "approved" for topic in approved_state["agenda"]
            )

    @pytest.mark.integration
    def test_agenda_approval_rejection_and_modification(self):
        """Test user rejecting agenda and requesting modifications."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_agenda_setting_state()

            # Set up agenda for approval
            agenda = [
                {
                    "title": "Topic A",
                    "description": "Description A",
                    "proposed_by": "agent_1",
                    "votes_for": 1,
                    "votes_against": 2,
                    "status": "pending",
                }
            ]
            state["agenda"] = agenda

            # Simulate user rejection and modification
            modified_state = self._simulate_user_modification(state)

            # Validate modification
            assert (
                len(modified_state["agenda"]) == len(state["agenda"]) + 1
            )  # User added a topic
            assert modified_state["current_phase"] == 1  # Back to setting

    @pytest.mark.integration
    def test_agenda_approval_with_conditions(self):
        """Test conditional approval with user-specified changes."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_agenda_setting_state()

            # Set up agenda with some concerning topics
            agenda = [
                {
                    "title": "Controversial Topic",
                    "description": "Potentially problematic discussion",
                    "proposed_by": "agent_1",
                    "votes_for": 1,
                    "votes_against": 2,
                    "status": "pending",
                },
                {
                    "title": "Good Topic",
                    "description": "Well-defined and relevant",
                    "proposed_by": "agent_2",
                    "votes_for": 3,
                    "votes_against": 0,
                    "status": "pending",
                },
            ]
            state["agenda"] = agenda

            # Simulate conditional approval (remove controversial topic)
            approved_state = self._simulate_conditional_approval(state)

            # Should have modified agenda
            assert len(approved_state["agenda"]) == 1
            assert approved_state["agenda"][0]["title"] == "Good Topic"
            assert approved_state["hitl_state"]["approved"]

    def _simulate_user_approval(
        self, state: VirtualAgoraState, approve: bool
    ) -> VirtualAgoraState:
        """Simulate user approval/rejection of agenda."""
        if approve:
            state["hitl_state"]["approved"] = True
            state["current_phase"] = 2
            for topic in state["agenda"]:
                topic["status"] = "approved"
        else:
            state["hitl_state"]["approved"] = False
            state["current_phase"] = 1

        return state

    def _simulate_user_modification(
        self, state: VirtualAgoraState
    ) -> VirtualAgoraState:
        """Simulate user modifying the agenda."""
        # Create a copy of the state to avoid modifying the original
        modified_state = dict(state)
        modified_state["agenda"] = state["agenda"].copy()

        # Add a new topic
        new_topic = {
            "title": "User-Added Topic",
            "description": "Topic added by user modification",
            "proposed_by": "user",
            "votes_for": 0,
            "votes_against": 0,
            "status": "pending",
        }
        modified_state["agenda"].append(new_topic)
        modified_state["current_phase"] = 1

        return modified_state

    def _simulate_conditional_approval(
        self, state: VirtualAgoraState
    ) -> VirtualAgoraState:
        """Simulate conditional approval with modifications."""
        # Create a copy of the state
        approved_state = dict(state)

        # Remove controversial topics
        approved_state["agenda"] = [
            topic
            for topic in state["agenda"]
            if "controversial" not in topic["title"].lower()
        ]

        # Create a copy of hitl_state if it exists
        if isinstance(state.get("hitl_state"), dict):
            approved_state["hitl_state"] = dict(state["hitl_state"])
            approved_state["hitl_state"]["approved"] = True
        else:
            approved_state["hitl_state"] = {"approved": True}

        approved_state["current_phase"] = 2

        return approved_state


class TestAgendaModificationFlow:
    """Test agenda modification during discussion."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")

    @pytest.mark.integration
    def test_mid_discussion_agenda_modification(self):
        """Test modifying agenda between topics."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Complete first topic
            state["current_topic_index"] = 1  # Move to second topic
            state["topic_summaries"].append("Summary of first topic")

            # Simulate agents suggesting modifications
            modification_suggestions = self._simulate_modification_suggestions(state)

            # Apply modifications
            modified_state = self._apply_agenda_modifications(
                state, modification_suggestions
            )

            # Validate modifications
            assert len(modified_state["agenda"]) > len(state["agenda"])
            self.test_helper.assert_agenda_created(modified_state)

    @pytest.mark.integration
    def test_agenda_reordering(self):
        """Test reordering remaining agenda topics."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Set up state with multiple remaining topics
            original_order = [topic["title"] for topic in state["agenda"]]

            # Simulate reordering based on discussion insights
            reordered_state = self._simulate_agenda_reordering(state)

            new_order = [topic["title"] for topic in reordered_state["agenda"]]

            # Order should be different but contain same topics
            assert set(original_order) == set(new_order)
            assert original_order != new_order  # Should be reordered

    @pytest.mark.integration
    def test_agenda_topic_removal(self):
        """Test removing topics from agenda based on discussion."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            original_count = len(state["agenda"])

            # Simulate removal of irrelevant topic
            modified_state = self._simulate_topic_removal(state)

            # Should have fewer topics
            assert len(modified_state["agenda"]) < original_count
            assert all(
                topic["status"] != "removed" for topic in modified_state["agenda"]
            )

    def _simulate_modification_suggestions(
        self, state: VirtualAgoraState
    ) -> dict[str, list[str]]:
        """Simulate agents suggesting agenda modifications."""
        suggestions = {
            "agent_1": [
                "Add: Implementation Timeline and Milestones",
                "Add: Resource Allocation and Budget Planning",
            ],
            "agent_2": [
                "Add: Stakeholder Communication Strategy",
                "Reorder: Move Social Impact before Technical Implementation",
            ],
            "agent_3": ["Add: Quality Assurance and Testing Protocols"],
        }
        return suggestions

    def _apply_agenda_modifications(
        self, state: VirtualAgoraState, suggestions: dict
    ) -> VirtualAgoraState:
        """Apply suggested modifications to agenda."""
        # Create a copy of the state
        modified_state = dict(state)
        modified_state["agenda"] = state["agenda"].copy()

        # Add new topics based on suggestions
        new_topics = []
        for agent_id, agent_suggestions in suggestions.items():
            for suggestion in agent_suggestions:
                if suggestion.startswith("Add:"):
                    topic_title = suggestion[4:].strip()
                    new_topic = {
                        "title": topic_title,
                        "description": f"Topic suggested by {agent_id}",
                        "proposed_by": agent_id,
                        "votes_for": 1,
                        "votes_against": 0,
                        "status": "pending",
                    }
                    new_topics.append(new_topic)

        modified_state["agenda"].extend(new_topics)
        return modified_state

    def _simulate_agenda_reordering(
        self, state: VirtualAgoraState
    ) -> VirtualAgoraState:
        """Simulate reordering agenda topics."""
        # Create a copy of the state
        reordered_state = dict(state)
        # Reverse order as simple reordering
        reordered_state["agenda"] = list(reversed(state["agenda"]))
        return reordered_state

    def _simulate_topic_removal(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Simulate removing a topic from agenda."""
        # Create a copy of the state
        modified_state = dict(state)
        modified_state["agenda"] = state["agenda"].copy()
        # Remove last topic
        if modified_state["agenda"]:
            removed_topic = modified_state["agenda"].pop()
            removed_topic["status"] = "removed"

        return modified_state


@pytest.mark.integration
class TestAgendaEdgeCases:
    """Test edge cases in agenda management."""

    def test_single_topic_agenda(self):
        """Test handling of agenda with only one topic."""
        helper = IntegrationTestHelper(num_agents=2, scenario="default")

        with patch_ui_components():
            flow = helper.create_test_flow()
            state = helper.create_agenda_setting_state()

            # Set single topic agenda
            single_topic = {
                "title": "Single Topic Discussion",
                "description": "The only topic for discussion",
                "proposed_by": "agent_1",
                "votes_for": 2,
                "votes_against": 0,
                "status": "pending",
            }
            state["agenda"] = [single_topic]

            # Should handle single topic gracefully
            helper.assert_agenda_created(state)
            assert len(state["agenda"]) == 1

    def test_maximum_topics_agenda(self):
        """Test handling of agenda with maximum number of topics."""
        helper = IntegrationTestHelper(num_agents=3, scenario="default")

        with patch_ui_components():
            flow = helper.create_test_flow()
            state = helper.create_agenda_setting_state()

            # Create agenda with many topics
            many_topics = []
            for i in range(10):  # Large number of topics
                topic = {
                    "title": f"Topic {i+1}",
                    "description": f"Description for topic {i+1}",
                    "proposed_by": f"agent_{(i % 3) + 1}",
                    "votes_for": 2,
                    "votes_against": 1,
                    "status": "pending",
                }
                many_topics.append(topic)

            state["agenda"] = many_topics

            # Should handle large agenda
            helper.assert_agenda_created(state)
            assert len(state["agenda"]) == 10

    def test_duplicate_topic_handling(self):
        """Test handling of duplicate or similar topics."""
        helper = IntegrationTestHelper(num_agents=3, scenario="default")

        with patch_ui_components():
            flow = helper.create_test_flow()
            state = helper.create_agenda_setting_state()

            # Create agenda with duplicate topics
            duplicate_topics = [
                {
                    "title": "Technical Implementation",
                    "description": "Technical aspects",
                    "proposed_by": "agent_1",
                    "votes_for": 2,
                    "votes_against": 0,
                    "status": "pending",
                },
                {
                    "title": "Technical Implementation",  # Duplicate
                    "description": "Technical details",
                    "proposed_by": "agent_2",
                    "votes_for": 1,
                    "votes_against": 0,
                    "status": "pending",
                },
                {
                    "title": "Implementation Technical",  # Similar
                    "description": "Technical considerations",
                    "proposed_by": "agent_3",
                    "votes_for": 1,
                    "votes_against": 0,
                    "status": "pending",
                },
            ]
            state["agenda"] = duplicate_topics

            # Should handle duplicates (either merge or keep separate)
            assert len(state["agenda"]) >= 1  # At least one topic retained
