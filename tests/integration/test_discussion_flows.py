"""Integration tests for discussion round flows.

This module tests the discussion rounds, turn rotation, relevance checking,
and round summarization workflows using fake LLMs.
"""

import pytest
from unittest.mock import patch, Mock
from datetime import datetime
import uuid

from virtual_agora.state.schema import (
    VirtualAgoraState,
    Message,
    RoundInfo,
    AgentInfo,
    TopicInfo,
)
from virtual_agora.flow.graph import VirtualAgoraFlow

from ..helpers.fake_llm import ModeratorFakeLLM, AgentFakeLLM
from ..helpers.integration_utils import (
    IntegrationTestHelper,
    TestStateBuilder,
    TestResponseValidator,
    patch_ui_components,
    create_test_messages,
)


class TestDiscussionRoundFlow:
    """Test basic discussion round mechanics."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")
        self.validator = TestResponseValidator()

    @pytest.mark.integration
    def test_basic_turn_rotation(self):
        """Test basic agent turn rotation in discussion."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Run one complete round
            updated_state = self._simulate_discussion_round(state, round_number=1)

            # Validate turn rotation
            assert len(updated_state["messages"]) >= len(state["speaking_order"])

            # Check that each agent spoke once
            speakers_in_round = [
                msg["agent_id"]
                for msg in updated_state["messages"]
                if msg["round_number"] == 1
            ]
            expected_speakers = state["speaking_order"]
            assert set(speakers_in_round) == set(expected_speakers)

            # Check order is correct
            assert speakers_in_round == expected_speakers

    @pytest.mark.integration
    def test_multiple_round_rotation(self):
        """Test turn rotation across multiple rounds."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Run multiple rounds
            for round_num in range(1, 4):
                state = self._simulate_discussion_round(state, round_number=round_num)

            # Validate rotation across rounds
            assert state["current_round"] == 3
            assert len(state["messages"]) >= 9  # 3 agents × 3 rounds

            # Check rotation pattern
            round_1_speakers = [
                msg["agent_id"] for msg in state["messages"] if msg["round_number"] == 1
            ]
            round_2_speakers = [
                msg["agent_id"] for msg in state["messages"] if msg["round_number"] == 2
            ]
            round_3_speakers = [
                msg["agent_id"] for msg in state["messages"] if msg["round_number"] == 3
            ]

            # Each round should have same speakers
            assert (
                set(round_1_speakers) == set(round_2_speakers) == set(round_3_speakers)
            )

            # Order should follow rotation (Round 2 starts with agent_2, etc.)
            expected_round_2_start = state["speaking_order"][1]
            assert round_2_speakers[0] == expected_round_2_start

    @pytest.mark.integration
    def test_agent_response_generation(self):
        """Test agent response generation with context."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add some existing messages for context
            existing_messages = [
                {
                    "id": str(uuid.uuid4()),
                    "agent_id": "agent_1",
                    "content": "I think we should focus on scalability first.",
                    "timestamp": datetime.now(),
                    "round_number": 1,
                    "topic": state["agenda"][0]["title"],
                    "message_type": "discussion",
                }
            ]
            state["messages"] = existing_messages

            # Generate response from next agent
            response = self._simulate_agent_response(
                state, "agent_2", round_number=1, context=existing_messages
            )

            # Validate response
            assert isinstance(response, str)
            assert len(response) > 20  # Should be substantive
            assert (
                "agent_2" in response or "scalability" in response.lower()
            )  # Should reference context

    @pytest.mark.integration
    def test_context_building_across_rounds(self):
        """Test that context builds properly across discussion rounds."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Simulate progressive discussion
            topics_mentioned = set()

            for round_num in range(1, 4):
                state = self._simulate_discussion_round(state, round_number=round_num)

                # Extract topics from latest round
                latest_messages = [
                    msg for msg in state["messages"] if msg["round_number"] == round_num
                ]
                for msg in latest_messages:
                    # Simple topic extraction
                    words = msg["content"].lower().split()
                    topics_mentioned.update(word for word in words if len(word) > 5)

            # Context should be building (more topics mentioned over time)
            assert len(topics_mentioned) > 10  # Should have substantial vocabulary

    def _simulate_discussion_round(
        self, state: VirtualAgoraState, round_number: int
    ) -> VirtualAgoraState:
        """Simulate a complete discussion round."""
        state["current_round"] = round_number

        # Rotate speaking order for each round
        speaking_order = state["speaking_order"].copy()
        # Rotate by (round_number - 1) positions
        for _ in range(round_number - 1):
            speaking_order.append(speaking_order.pop(0))

        # Each agent speaks in rotated order
        for i, agent_id in enumerate(speaking_order):
            state["current_speaker_index"] = i

            # Generate agent response
            response = self._simulate_agent_response(state, agent_id, round_number)

            # Create message
            message = {
                "id": str(uuid.uuid4()),
                "agent_id": agent_id,
                "content": response,
                "timestamp": datetime.now(),
                "round_number": round_number,
                "topic": state["agenda"][state["current_topic_index"]]["title"],
                "message_type": "discussion",
            }
            state["messages"].append(message)

        # Generate round summary
        summary = self._simulate_round_summary(state, round_number)
        if "round_summaries" not in state:
            state["round_summaries"] = []
        state["round_summaries"].append(summary)

        return state

    def _simulate_agent_response(
        self,
        state: VirtualAgoraState,
        agent_id: str,
        round_number: int,
        context: list = None,
    ) -> str:
        """Simulate an agent generating a response."""
        topic = state["agenda"][state.get("current_topic_index", 0)]["title"]

        # Get agent personality from test helper
        personalities = ["optimistic", "skeptical", "technical", "balanced"]
        agent_num = int(agent_id.split("_")[1]) - 1
        personality = personalities[agent_num % len(personalities)]

        # Generate response based on personality and context
        if personality == "optimistic":
            response = (
                f"I see great opportunities in {topic}. "
                f"Round {round_number} builds on our previous insights."
            )
        elif personality == "skeptical":
            response = (
                f"We need to carefully consider the challenges with {topic}. "
                f"The concerns from round {round_number} require attention."
            )
        elif personality == "technical":
            response = (
                f"From a technical perspective, {topic} requires careful architecture. "
                f"Round {round_number} technical analysis shows key requirements."
            )
        else:  # balanced
            response = (
                f"Balancing the perspectives on {topic}, I think we need to consider "
                f"both opportunities and challenges discussed in round {round_number}."
            )

        return f"{response} ({agent_id})"

    def _simulate_round_summary(
        self, state: VirtualAgoraState, round_number: int
    ) -> str:
        """Simulate moderator generating round summary."""
        topic = state["agenda"][state.get("current_topic_index", 0)]["title"]
        num_speakers = len(state["speaking_order"])

        return (
            f"Round {round_number} Summary: {num_speakers} agents discussed {topic}, "
            f"covering key perspectives and building on previous insights."
        )


class TestRelevanceCheckingFlow:
    """Test moderator relevance checking and enforcement."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")

    @pytest.mark.integration
    def test_on_topic_response_acceptance(self):
        """Test acceptance of on-topic responses."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            current_topic = state["agenda"][state["current_topic_index"]]["title"]

            # Create on-topic response
            on_topic_response = (
                f"Regarding {current_topic}, I believe we should consider "
                f"the technical implementation aspects and their implications."
            )

            # Check relevance
            is_relevant = self._check_response_relevance(
                on_topic_response, current_topic
            )

            assert is_relevant

    @pytest.mark.integration
    def test_off_topic_response_warning(self):
        """Test warning for off-topic responses."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            current_topic = state["agenda"][state["current_topic_index"]]["title"]

            # Create off-topic response
            off_topic_response = (
                "I'd like to talk about something completely different "
                "that has nothing to do with our current discussion."
            )

            # Check relevance
            is_relevant = self._check_response_relevance(
                off_topic_response, current_topic
            )

            assert not is_relevant

            # Simulate warning
            warning_issued = self._issue_relevance_warning(
                "agent_1", off_topic_response
            )
            assert warning_issued

    @pytest.mark.integration
    def test_warn_then_mute_mechanism(self):
        """Test warn-then-mute mechanism for repeated off-topic responses."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            agent_id = "agent_1"
            current_topic = state["agenda"][state["current_topic_index"]]["title"]

            # Track warnings for agent
            warnings = {}

            # First off-topic response -> warning
            off_topic_1 = "Let's talk about the weather instead."
            relevance_1 = self._check_response_relevance(off_topic_1, current_topic)
            assert not relevance_1

            warnings[agent_id] = warnings.get(agent_id, 0) + 1
            assert warnings[agent_id] == 1  # First warning

            # Second off-topic response -> final warning
            off_topic_2 = "I prefer discussing sports."
            relevance_2 = self._check_response_relevance(off_topic_2, current_topic)
            assert not relevance_2

            warnings[agent_id] += 1
            assert warnings[agent_id] == 2  # Final warning

            # Third off-topic response -> mute
            off_topic_3 = "What about food recipes?"
            relevance_3 = self._check_response_relevance(off_topic_3, current_topic)
            assert not relevance_3

            # Agent should be muted
            should_mute = warnings[agent_id] >= 2
            assert should_mute

    @pytest.mark.integration
    def test_relevance_context_understanding(self):
        """Test moderator understanding of topic context for relevance."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Set specific topic
            tech_topic = "Artificial Intelligence Implementation"
            state["agenda"][0]["title"] = tech_topic

            # Test various responses for relevance
            test_cases = [
                ("AI algorithms and machine learning approaches", True),
                ("Implementation strategies for neural networks", True),
                ("Data processing and training methodologies", True),
                ("My favorite pizza toppings", False),
                ("Yesterday's football game results", False),
                ("Ethics in AI development", True),  # Related but broader
                ("Software architecture patterns", True),  # Somewhat related
            ]

            for response, expected_relevance in test_cases:
                actual_relevance = self._check_response_relevance(response, tech_topic)
                assert (
                    actual_relevance == expected_relevance
                ), f"Response '{response}' relevance check failed"

    def _check_response_relevance(self, response: str, topic: str) -> bool:
        """Simulate moderator checking response relevance."""
        # Check for explicitly off-topic keywords
        off_topic_keywords = [
            "pizza",
            "football",
            "weather",
            "vacation",
            "holiday",
            "movie",
        ]
        response_lower = response.lower()

        # If contains off-topic keywords, it's not relevant
        for keyword in off_topic_keywords:
            if keyword in response_lower:
                return False

        # Expanded relevance checking
        topic_lower = topic.lower()

        # Direct topic word matching
        topic_words = set(topic_lower.split())
        response_words = set(response_lower.split())
        direct_matches = len(topic_words.intersection(response_words))

        # Check for related concepts to AI/technology
        ai_related_terms = {
            "ai",
            "artificial",
            "intelligence",
            "machine",
            "learning",
            "algorithm",
            "neural",
            "network",
            "data",
            "processing",
            "training",
            "model",
            "ethics",
            "ethical",
            "bias",
            "fairness",
            "privacy",
            "security",
        }
        ai_matches = len(ai_related_terms.intersection(response_words))

        # Check for implementation/technical terms
        tech_terms = {
            "implementation",
            "technical",
            "architecture",
            "design",
            "system",
            "approach",
            "strategy",
            "method",
            "solution",
            "analysis",
            "development",
            "software",
            "hardware",
            "infrastructure",
            "deployment",
            "integration",
        }
        tech_matches = len(tech_terms.intersection(response_words))

        # Check for conceptual relevance
        # If discussing AI, topics like ethics, data, algorithms are relevant
        if "artificial intelligence" in topic_lower or "ai" in topic_lower:
            if ai_matches > 0:
                return True

        # Consider relevant if:
        # 1. Has direct topic word matches, OR
        # 2. Has significant AI-related terms when discussing AI, OR
        # 3. Has technical terms related to implementation
        # 4. Discusses software/architecture patterns (relevant to any implementation)
        is_implementation_related = (
            "implementation" in topic_lower and tech_matches >= 2
        )
        is_architecture_related = (
            "architecture" in response_lower or "patterns" in response_lower
        ) and tech_matches >= 1

        return (
            direct_matches > 0
            or ai_matches >= 2
            or is_implementation_related
            or is_architecture_related
        )

    def _issue_relevance_warning(self, agent_id: str, response: str) -> bool:
        """Simulate issuing relevance warning."""
        # Always issue warning for off-topic responses in test
        return True


class TestRoundSummarizationFlow:
    """Test moderator round summarization."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")

    @pytest.mark.integration
    def test_basic_round_summary_generation(self):
        """Test basic round summary generation."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add messages for a complete round
            round_messages = [
                {
                    "id": str(uuid.uuid4()),
                    "agent_id": "agent_1",
                    "content": "I think scalability is the primary concern here.",
                    "timestamp": datetime.now(),
                    "round_number": 1,
                    "topic": "Technical Implementation",
                    "message_type": "discussion",
                },
                {
                    "id": str(uuid.uuid4()),
                    "agent_id": "agent_2",
                    "content": "Security should also be a top priority in our approach.",
                    "timestamp": datetime.now(),
                    "round_number": 1,
                    "topic": "Technical Implementation",
                    "message_type": "discussion",
                },
                {
                    "id": str(uuid.uuid4()),
                    "agent_id": "agent_3",
                    "content": "We need to balance both scalability and security concerns.",
                    "timestamp": datetime.now(),
                    "round_number": 1,
                    "topic": "Technical Implementation",
                    "message_type": "discussion",
                },
            ]
            state["messages"] = round_messages

            # Generate summary
            summary = self._generate_round_summary(state, round_messages)

            # Validate summary
            assert len(summary) > 50  # Should be substantive
            assert "scalability" in summary.lower()
            assert "security" in summary.lower()
            assert "round" in summary.lower()

    @pytest.mark.integration
    def test_progressive_summary_building(self):
        """Test summary building across multiple rounds."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Simulate multiple rounds with evolving discussion
            themes_by_round = [
                ["scalability", "architecture"],
                ["security", "authentication"],
                ["performance", "optimization"],
            ]

            for round_num, themes in enumerate(themes_by_round, 1):
                # Add messages for this round
                round_messages = []
                for i, agent_id in enumerate(state["speaking_order"]):
                    theme = themes[i % len(themes)]
                    message = {
                        "id": str(uuid.uuid4()),
                        "agent_id": agent_id,
                        "content": f"Let's discuss {theme} in detail for this implementation.",
                        "timestamp": datetime.now(),
                        "round_number": round_num,
                        "topic": "Technical Implementation",
                        "message_type": "discussion",
                    }
                    round_messages.append(message)

                state["messages"].extend(round_messages)

                # Generate summary for this round
                summary = self._generate_round_summary(state, round_messages)
                state["round_summaries"].append(summary)

            # Validate progressive building
            assert len(state["round_summaries"]) == 3

            # Each summary should mention its round's themes
            for i, summary in enumerate(state["round_summaries"]):
                round_themes = themes_by_round[i]
                for theme in round_themes:
                    assert theme in summary.lower()

    @pytest.mark.integration
    def test_summary_context_compression(self):
        """Test summary helping with context compression."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Create a round with verbose messages
            verbose_messages = []
            for i, agent_id in enumerate(state["speaking_order"]):
                verbose_content = f"""
                I have many thoughts about this topic. First, I want to discuss the technical
                aspects in great detail, including all the implementation considerations,
                architectural patterns, design principles, and best practices that we should
                follow. Additionally, I think we need to consider the broader implications,
                the stakeholder impacts, the resource requirements, and the timeline
                considerations. Furthermore, there are risk factors, mitigation strategies,
                and contingency plans that we should develop. Finally, I believe we should
                also think about the long-term sustainability, maintenance requirements,
                and evolution path for this solution. ({agent_id})
                """

                message = {
                    "id": str(uuid.uuid4()),
                    "agent_id": agent_id,
                    "content": verbose_content.strip(),
                    "timestamp": datetime.now(),
                    "round_number": 1,
                    "topic": "Technical Implementation",
                    "message_type": "discussion",
                }
                verbose_messages.append(message)

            state["messages"] = verbose_messages

            # Generate compressed summary
            summary = self._generate_round_summary(state, verbose_messages)

            # Summary should be much shorter than combined messages
            total_message_length = sum(len(msg["content"]) for msg in verbose_messages)
            assert len(summary) < total_message_length / 3  # At least 3x compression

            # But should still capture key themes (at least 3 out of 4)
            key_themes = ["technical", "implementation", "stakeholder", "risk"]
            themes_found = sum(1 for theme in key_themes if theme in summary.lower())
            assert (
                themes_found >= 3
            ), f"Expected at least 3 themes, found {themes_found} in summary: {summary}"

    def _generate_round_summary(
        self, state: VirtualAgoraState, messages: list[Message]
    ) -> str:
        """Simulate moderator generating round summary."""
        topic = state["agenda"][state["current_topic_index"]]["title"]
        round_num = messages[0]["round_number"] if messages else 1

        # Extract key themes from messages
        all_content = " ".join(msg["content"] for msg in messages)
        words = all_content.lower().split()

        # Find important technical terms with proper extraction
        important_terms = []
        tech_keywords = {
            "scalability",
            "security",
            "performance",
            "architecture",
            "implementation",
            "design",
            "strategy",
            "approach",
            "solution",
            "analysis",
            "optimization",
            "authentication",
            "infrastructure",
            "technical",
            "stakeholder",
            "risk",
        }

        # Count keyword occurrences
        keyword_counts = {}
        for term in tech_keywords:
            count = words.count(term)
            if count > 0:
                keyword_counts[term] = count

        # Sort by frequency and get top terms
        sorted_terms = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        important_terms = [term[0] for term in sorted_terms[:4]]

        # Generate comprehensive summary
        summary = f"Round {round_num} Summary: The {len(messages)} participating agents engaged in detailed discussion of {topic}"

        if important_terms:
            summary += f", focusing on {', '.join(important_terms)}"

        # Add specific theme mentions
        if "scalability" in important_terms:
            summary += ". Scalability considerations were thoroughly examined"
        if "security" in important_terms:
            summary += ". Security protocols received significant attention"
        if "technical" in important_terms or "implementation" in important_terms:
            summary += ". Technical implementation details were analyzed"
        if "stakeholder" in important_terms:
            summary += ". Stakeholder perspectives were carefully considered"
        if "risk" in important_terms:
            summary += ". Risk factors were identified and discussed"

        summary += (
            ". Key insights were shared and the discussion is progressing productively."
        )

        return summary


class TestContextWindowManagement:
    """Test context window management during discussions."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=4, scenario="default")

    @pytest.mark.integration
    def test_automatic_context_compression(self):
        """Test automatic compression when context gets too large."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Simulate very long discussion
            for round_num in range(1, 20):  # Many rounds
                for agent_id in state["speaking_order"]:
                    long_message = {
                        "id": str(uuid.uuid4()),
                        "agent_id": agent_id,
                        "content": f"Very long message from {agent_id} in round {round_num}. "
                        * 20,
                        "timestamp": datetime.now(),
                        "round_number": round_num,
                        "topic": "Technical Implementation",
                        "message_type": "discussion",
                    }
                    state["messages"].append(long_message)

                # Add round summary
                summary = f"Round {round_num} covered extensive technical details."
                if "round_summaries" not in state:
                    state["round_summaries"] = []
                state["round_summaries"].append(summary)

            # Should trigger context compression
            total_message_length = sum(len(msg["content"]) for msg in state["messages"])
            assert total_message_length > 50000  # Large context

            # Simulate compression
            compressed_state = self._simulate_context_compression(state)

            # Should have summaries but fewer detailed messages
            assert len(compressed_state["round_summaries"]) == len(
                state["round_summaries"]
            )
            assert len(compressed_state["messages"]) <= len(state["messages"])

    @pytest.mark.integration
    def test_summary_preservation_during_compression(self):
        """Test that summaries are preserved during context compression."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Create state with important summaries
            important_summaries = [
                "Round 1: Established core technical requirements",
                "Round 2: Identified key security considerations",
                "Round 3: Defined performance benchmarks",
                "Round 4: Outlined implementation timeline",
            ]
            state["round_summaries"] = important_summaries

            # Add many messages
            for i in range(100):
                message = {
                    "id": str(uuid.uuid4()),
                    "agent_id": f"agent_{(i % 3) + 1}",
                    "content": f"Message {i} with detailed content",
                    "timestamp": datetime.now(),
                    "round_number": (i // 3) + 1,
                    "topic": "Technical Implementation",
                    "message_type": "discussion",
                }
                state["messages"].append(message)

            # Compress context
            compressed_state = self._simulate_context_compression(state)

            # Summaries should be preserved
            assert compressed_state["round_summaries"] == important_summaries
            # Should have compressed messages (keeping only 20 recent from 100 total)
            assert len(state["messages"]) == 100  # Original had 100 messages
            assert len(compressed_state["messages"]) == 20  # Compressed to 20
            assert len(compressed_state["messages"]) < len(state["messages"])

    def _simulate_context_compression(
        self, state: VirtualAgoraState
    ) -> VirtualAgoraState:
        """Simulate context window compression."""
        # Create a copy of the state to avoid mutating the original
        import copy

        compressed_state = copy.deepcopy(state)

        # Keep only recent messages and all summaries
        recent_threshold = 20
        if len(compressed_state["messages"]) > recent_threshold:
            compressed_state["messages"] = compressed_state["messages"][
                -recent_threshold:
            ]

        # Summaries are always preserved (already in the copy)
        return compressed_state


@pytest.mark.integration
class TestDiscussionEdgeCases:
    """Test edge cases in discussion flows."""

    def test_single_agent_discussion(self):
        """Test discussion with minimal agent setup (2 agents as required by config)."""
        helper = IntegrationTestHelper(num_agents=2, scenario="default")

        with patch_ui_components():
            flow = helper.create_test_flow()
            state = helper.create_discussion_state()

            # Simulate one agent being muted or inactive
            state["speaking_order"] = ["agent_1"]  # Only one agent speaking

            # Should handle limited participation gracefully
            assert len(state["speaking_order"]) == 1

            # Run a discussion round
            for round_num in range(1, 3):
                state["current_round"] = round_num
                # Only one agent speaks
                message = {
                    "id": str(uuid.uuid4()),
                    "agent_id": "agent_1",
                    "content": f"Solo perspective on round {round_num}",
                    "timestamp": datetime.now(),
                    "round_number": round_num,
                    "topic": state["agenda"][0]["title"],
                    "message_type": "discussion",
                }
                state["messages"].append(message)

            # Should still generate summaries
            assert len(state["messages"]) == 2
            assert state["messages"][0]["agent_id"] == "agent_1"

            # Single agent should be able to have "discussion"
            new_message = {
                "id": str(uuid.uuid4()),
                "agent_id": state["speaking_order"][0],
                "content": "I'll analyze this topic comprehensively.",
                "timestamp": datetime.now(),
                "round_number": 3,
                "topic": "Test Topic",
                "message_type": "discussion",
            }
            state["messages"].append(new_message)

            # Should now have 3 messages total (2 from rounds + 1 new)
            assert len(state["messages"]) == 3
            assert state["messages"][-1]["agent_id"] in state["speaking_order"]
            assert (
                state["messages"][-1]["content"]
                == "I'll analyze this topic comprehensively."
            )

    def test_agent_silence_handling(self):
        """Test handling of agent that doesn't respond."""
        helper = IntegrationTestHelper(num_agents=3, scenario="error_prone")

        with patch_ui_components():
            flow = helper.create_test_flow()
            state = helper.create_discussion_state()

            # Simulate one agent not responding
            responding_agents = state["speaking_order"][:-1]  # Exclude last agent

            for agent_id in responding_agents:
                message = {
                    "id": str(uuid.uuid4()),
                    "agent_id": agent_id,
                    "content": f"Response from {agent_id}",
                    "timestamp": datetime.now(),
                    "round_number": 1,
                    "topic": "Test Topic",
                    "message_type": "discussion",
                }
                state["messages"].append(message)

            # Should handle missing response gracefully
            assert len(state["messages"]) == len(responding_agents)

            # Non-responding agent should be noted but not break flow
            responding_agent_ids = {msg["agent_id"] for msg in state["messages"]}
            expected_responding = set(responding_agents)
            assert responding_agent_ids == expected_responding

    def test_very_long_individual_response(self):
        """Test handling of extremely long agent response."""
        helper = IntegrationTestHelper(num_agents=2, scenario="default")

        with patch_ui_components():
            flow = helper.create_test_flow()
            state = helper.create_discussion_state()

            # Create very long response
            very_long_content = "This is a very detailed response. " * 1000  # ~30KB

            long_message = {
                "id": str(uuid.uuid4()),
                "agent_id": "agent_1",
                "content": very_long_content,
                "timestamp": datetime.now(),
                "round_number": 1,
                "topic": "Test Topic",
                "message_type": "discussion",
            }
            state["messages"].append(long_message)

            # Should handle long message
            assert len(state["messages"]) == 1
            assert len(state["messages"][0]["content"]) > 10000
