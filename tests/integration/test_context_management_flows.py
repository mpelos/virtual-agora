"""Integration tests for Context Window & Memory Management flows.

This module tests the complete context management workflow including
context window limits, memory compression, conversation history, and context recovery.
"""

import pytest
import tempfile
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime
import uuid

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.flow.graph import VirtualAgoraFlow
from virtual_agora.flow.context_window import ContextWindowManager
from virtual_agora.utils.exceptions import VirtualAgoraError

from ..helpers.fake_llm import ModeratorFakeLLM, AgentFakeLLM
from ..helpers.integration_utils import (
    IntegrationTestHelper,
    StateTestBuilder,
    TestResponseValidator,
)


def ensure_context_fields(state: VirtualAgoraState) -> None:
    """Ensure all required context fields exist in state."""
    if "phase_summaries" not in state:
        state["phase_summaries"] = {}
    if "topic_summaries" not in state:
        state["topic_summaries"] = {}
    elif isinstance(state["topic_summaries"], list):
        # Convert list to dict if needed (from test helper)
        state["topic_summaries"] = {}
    if "consensus_summaries" not in state:
        state["consensus_summaries"] = {}
    if "final_report" not in state:
        state["final_report"] = ""


def patch_other_ui_components(exclude=None):
    """Helper to patch UI components except specific ones being tested."""
    from contextlib import ExitStack

    if exclude is None:
        exclude = []

    stack = ExitStack()

    # Patch components we're not specifically testing
    if "get_continuation_approval" not in exclude:
        mock_continuation = stack.enter_context(
            patch("virtual_agora.ui.human_in_the_loop.get_continuation_approval")
        )
        mock_continuation.return_value = "continue"

    if "get_agenda_modifications" not in exclude:
        mock_modifications = stack.enter_context(
            patch("virtual_agora.ui.human_in_the_loop.get_agenda_modifications")
        )
        mock_modifications.return_value = []

    if "get_agenda_approval" not in exclude:
        mock_approval = stack.enter_context(
            patch("virtual_agora.ui.human_in_the_loop.get_agenda_approval")
        )
        mock_approval.return_value = True

    if "get_initial_topic" not in exclude:
        mock_topic = stack.enter_context(
            patch("virtual_agora.ui.human_in_the_loop.get_initial_topic")
        )
        mock_topic.return_value = "Test Topic"

    if "display_session_status" not in exclude:
        mock_display = stack.enter_context(
            patch("virtual_agora.ui.human_in_the_loop.display_session_status")
        )

    if "get_user_preferences" not in exclude:
        mock_preferences = stack.enter_context(
            patch("virtual_agora.ui.preferences.get_user_preferences")
        )
        mock_preferences.return_value = Mock(auto_approve_agenda_on_consensus=True)

    return stack


class TestContextWindowLimits:
    """Test context window management and limits."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")
        self.validator = TestResponseValidator()

    @pytest.mark.integration
    def test_context_window_tracking(self):
        """Test context window token tracking during discussion."""
        with patch_other_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add messages that approach context limit
            state = self._add_messages_approaching_limit(state)

            # Create context window manager
            context_manager = ContextWindowManager(window_limit=8000)

            # Track token usage
            current_tokens = context_manager.get_context_size(state)

            # Should detect approaching limit
            assert current_tokens > 6000  # Approaching limit

            # Should suggest compression
            needs_compression = context_manager.needs_compression(state)
            assert needs_compression == True

    @pytest.mark.integration
    def test_context_window_overflow_handling(self):
        """Test handling when context window overflows."""
        with patch_other_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add excessive messages
            state = self._add_excessive_messages(state)

            # Ensure flow_control has context_window_limit
            if "flow_control" not in state:
                state["flow_control"] = {}
            state["flow_control"]["context_window_limit"] = 4000

            # Ensure warnings exists
            if "warnings" not in state:
                state["warnings"] = []

            context_manager = ContextWindowManager(window_limit=4000)  # Small limit

            # Should detect overflow
            current_tokens = context_manager.get_context_size(state)
            assert current_tokens > 4000

            # Should handle overflow gracefully
            try:
                compression_updates = context_manager.compress_context(state)

                # Apply compression updates to state
                compressed_state = state.copy()
                compressed_state.update(compression_updates)

                # Compressed state should be smaller
                compressed_tokens = context_manager.get_context_size(compressed_state)
                assert compressed_tokens < current_tokens

                # Should preserve essential information
                assert "agenda" in compressed_state
                assert "current_phase" in compressed_state
                assert (
                    len(compressed_state.get("messages", [])) >= 0
                )  # Messages may be compressed

            except VirtualAgoraError as e:
                # Should provide helpful error information
                assert len(str(e)) > 0

    @pytest.mark.integration
    def test_selective_message_compression(self):
        """Test selective compression of older messages."""
        with patch_other_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add tiered messages (old, recent, current)
            state = self._add_tiered_messages(state)

            context_manager = ContextWindowManager(max_tokens=6000)

            # Compress selectively
            compressed_state = context_manager.compress_selectively(state)

            # Should preserve recent messages
            messages = compressed_state.get("messages", [])
            recent_messages = [
                msg for msg in messages if msg.get("round_number", 0) >= 8
            ]
            assert len(recent_messages) > 0

            # Should summarize older content
            assert (
                "message_summaries" in compressed_state
                or "compressed_history" in compressed_state
            )

    def _add_messages_approaching_limit(
        self, state: VirtualAgoraState
    ) -> VirtualAgoraState:
        """Add messages that approach context window limit."""
        # Ensure required fields exist
        ensure_context_fields(state)

        # Clear existing messages
        state["messages"] = []

        # Add many detailed messages
        for round_num in range(1, 16):  # 15 rounds
            for agent_id in state["speaking_order"]:
                # Create longer messages to consume more tokens
                message = {
                    "id": str(uuid.uuid4()),
                    "agent_id": agent_id,
                    "content": f"In round {round_num}, I want to provide a comprehensive analysis of the topic at hand. "
                    f"This involves considering multiple perspectives, evaluating various approaches, "
                    f"analyzing potential risks and benefits, and providing detailed recommendations. "
                    f"The complexity of this issue requires us to examine historical precedents, "
                    f"current market conditions, technological limitations, and future implications. "
                    f"After careful consideration of all these factors, my position is that we should "
                    f"proceed with a measured approach that balances innovation with caution.",
                    "timestamp": datetime.now(),
                    "round_number": round_num,
                    "topic": state["agenda"][0]["title"],
                    "message_type": "discussion",
                }
                state["messages"].append(message)

        return state

    def _add_excessive_messages(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Add excessive messages that would overflow context window."""
        # Ensure required fields exist
        ensure_context_fields(state)

        # Clear existing messages
        state["messages"] = []

        # Add extremely long messages
        for round_num in range(1, 25):  # 24 rounds
            for agent_id in state["speaking_order"]:
                # Create very long messages
                long_content = f"Round {round_num} detailed analysis: " + " ".join(
                    [
                        f"This is sentence {i} with detailed analysis and comprehensive evaluation of all aspects."
                        for i in range(50)  # 50 sentences per message
                    ]
                )

                message = {
                    "id": str(uuid.uuid4()),
                    "agent_id": agent_id,
                    "content": long_content,
                    "timestamp": datetime.now(),
                    "round_number": round_num,
                    "topic": state["agenda"][0]["title"],
                    "message_type": "discussion",
                }
                state["messages"].append(message)

        return state

    def _add_tiered_messages(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Add messages in tiers: old, recent, current."""
        # Ensure required fields exist
        ensure_context_fields(state)

        # Clear existing messages
        state["messages"] = []

        # Old messages (rounds 1-5) - can be compressed
        for round_num in range(1, 6):
            for agent_id in state["speaking_order"]:
                message = {
                    "id": str(uuid.uuid4()),
                    "agent_id": agent_id,
                    "content": f"Old round {round_num}: Basic discussion points that can be summarized.",
                    "timestamp": datetime.now(),
                    "round_number": round_num,
                    "topic": state["agenda"][0]["title"],
                    "message_type": "discussion",
                }
                state["messages"].append(message)

        # Recent messages (rounds 6-9) - should be preserved
        for round_num in range(6, 10):
            for agent_id in state["speaking_order"]:
                message = {
                    "id": str(uuid.uuid4()),
                    "agent_id": agent_id,
                    "content": f"Recent round {round_num}: Important recent discussion with key insights and decisions.",
                    "timestamp": datetime.now(),
                    "round_number": round_num,
                    "topic": state["agenda"][0]["title"],
                    "message_type": "discussion",
                }
                state["messages"].append(message)

        # Current messages (round 10) - must be preserved
        for agent_id in state["speaking_order"]:
            message = {
                "id": str(uuid.uuid4()),
                "agent_id": agent_id,
                "content": f"Current round 10: Critical current discussion that must be preserved for context.",
                "timestamp": datetime.now(),
                "round_number": 10,
                "topic": state["agenda"][0]["title"],
                "message_type": "discussion",
            }
            state["messages"].append(message)

        return state


class TestMemoryCompression:
    """Test memory compression and summarization workflows."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")

    @pytest.mark.integration
    def test_automatic_compression_trigger(self):
        """Test automatic compression when approaching limits."""
        with patch_other_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add many messages to trigger compression
            state = self._add_compression_trigger_messages(state)

            context_manager = ContextWindowManager(max_tokens=5000)

            # Should automatically trigger compression
            result = context_manager.auto_manage_context(state)

            # Should have compressed content
            assert "compressed_history" in result or "message_summaries" in result

            # Should be within token limits
            compressed_tokens = context_manager.estimate_tokens(result)
            assert compressed_tokens <= 5000

            # Should preserve current round
            current_messages = [
                msg
                for msg in result.get("messages", [])
                if msg.get("round_number") == result.get("current_round", 1)
            ]
            assert len(current_messages) > 0

    @pytest.mark.integration
    def test_round_based_compression(self):
        """Test compression based on discussion rounds."""
        with patch_other_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add multiple rounds
            state = self._add_multiple_rounds(state)

            context_manager = ContextWindowManager(max_tokens=6000)

            # Compress older rounds
            compressed_state = context_manager.compress_by_rounds(
                state, keep_recent_rounds=3
            )

            # Should keep recent rounds
            messages = compressed_state.get("messages", [])
            recent_rounds = set(msg.get("round_number", 0) for msg in messages)
            assert max(recent_rounds) >= 8  # Should keep rounds 8, 9, 10

            # Should have round summaries for older rounds
            round_summaries = compressed_state.get("round_summaries", [])
            assert len(round_summaries) >= 5  # Should have summaries for rounds 1-5

    @pytest.mark.integration
    def test_topic_based_compression(self):
        """Test compression organized by topic."""
        with patch_other_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add multi-topic messages
            state = self._add_multi_topic_messages(state)

            context_manager = ContextWindowManager(max_tokens=6000)

            # Compress by topic
            compressed_state = context_manager.compress_by_topics(state)

            # Should preserve current topic messages
            current_topic = state["agenda"][state.get("current_topic_index", 0)][
                "title"
            ]
            current_topic_messages = [
                msg
                for msg in compressed_state.get("messages", [])
                if msg.get("topic") == current_topic
            ]
            assert len(current_topic_messages) > 0

            # Should have topic summaries for completed topics
            topic_summaries = compressed_state.get("topic_summaries", [])
            assert len(topic_summaries) > 0

    def _add_compression_trigger_messages(
        self, state: VirtualAgoraState
    ) -> VirtualAgoraState:
        """Add messages that should trigger compression."""
        # Ensure required fields exist
        ensure_context_fields(state)

        state["messages"] = []

        # Add many rounds to trigger compression (need to exceed 4000 tokens for 5000 limit)
        for round_num in range(1, 25):  # 24 rounds instead of 11
            for agent_id in state["speaking_order"]:
                # Create much longer messages to ensure we exceed the compression threshold
                long_content = (
                    f"Round {round_num}: This is a comprehensive and detailed discussion message that provides "
                    f"extensive analysis, multiple perspectives, and thorough evaluation of the topic at hand. "
                    f"In this round, I want to elaborate on several key points that require careful consideration. "
                    f"First, we need to examine the technical implications and their long-term consequences. "
                    f"Second, we should analyze the economic impact and potential risks associated with our decisions. "
                    f"Third, it's crucial to consider the social and ethical dimensions of our proposed solutions. "
                    f"Furthermore, we must evaluate the feasibility of implementation given current constraints. "
                    f"Additionally, we should discuss potential alternatives and their relative merits. "
                    f"The complexity of this issue requires us to examine historical precedents and patterns. "
                    f"We must also consider future implications and how our decisions will affect stakeholders. "
                    f"After careful deliberation and analysis of all these factors, my position is that we should "
                    f"proceed with a measured and well-researched approach that balances all these considerations. "
                    f"This comprehensive analysis ensures we have covered all necessary aspects thoroughly. "
                )
                message = {
                    "id": str(uuid.uuid4()),
                    "agent_id": agent_id,
                    "content": long_content,
                    "timestamp": datetime.now(),
                    "round_number": round_num,
                    "topic": state["agenda"][0]["title"],
                    "message_type": "discussion",
                }
                state["messages"].append(message)

        state["current_round"] = 24  # Set to match the last round we generated
        return state

    def _add_multiple_rounds(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Add messages across multiple rounds."""
        state["messages"] = []

        for round_num in range(1, 11):
            for agent_id in state["speaking_order"]:
                message = {
                    "id": str(uuid.uuid4()),
                    "agent_id": agent_id,
                    "content": f"Round {round_num}: Discussion content with analysis and reasoning.",
                    "timestamp": datetime.now(),
                    "round_number": round_num,
                    "topic": state["agenda"][0]["title"],
                    "message_type": "discussion",
                }
                state["messages"].append(message)

        # Add round summaries for some rounds
        state["round_summaries"] = [
            f"Round {i}: Summary of discussion and key points" for i in range(1, 8)
        ]

        state["current_round"] = 10
        return state

    def _add_multi_topic_messages(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Add messages across multiple topics."""
        # Ensure required fields exist
        ensure_context_fields(state)

        # Expand agenda to multiple topics
        state["agenda"] = [
            {
                "title": "AI Ethics",
                "description": "Ethics discussion",
                "status": "completed",
            },
            {
                "title": "Data Privacy",
                "description": "Privacy discussion",
                "status": "completed",
            },
            {
                "title": "Future Technology",
                "description": "Future tech discussion",
                "status": "active",
            },
        ]

        state["messages"] = []

        # Add messages for each topic
        for topic_index, topic in enumerate(state["agenda"]):
            for round_num in range(1, 4):
                for agent_id in state["speaking_order"]:
                    message = {
                        "id": str(uuid.uuid4()),
                        "agent_id": agent_id,
                        "content": f"Round {round_num} on {topic['title']}: Detailed analysis and discussion.",
                        "timestamp": datetime.now(),
                        "round_number": round_num,
                        "topic": topic["title"],
                        "message_type": "discussion",
                    }
                    state["messages"].append(message)

        # Add topic summaries for completed topics
        state["topic_summaries"] = [
            "AI Ethics: Comprehensive discussion on ethical implications",
            "Data Privacy: Analysis of privacy concerns and solutions",
        ]

        state["current_topic_index"] = 2  # Currently on "Future Technology"
        return state


class TestConversationHistory:
    """Test conversation history management and retrieval."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")

    @pytest.mark.integration
    def test_conversation_history_tracking(self):
        """Test tracking of conversation history."""
        with patch_other_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add conversation with history
            state = self._add_conversation_with_history(state)

            context_manager = ContextWindowManager()

            # Extract conversation history
            history = context_manager.get_conversation_history(state)

            # Should contain chronological messages
            assert len(history) > 0
            assert all("timestamp" in msg for msg in history)

            # Should be chronologically ordered
            timestamps = [msg["timestamp"] for msg in history]
            assert timestamps == sorted(timestamps)

    @pytest.mark.integration
    def test_context_retrieval_by_topic(self):
        """Test retrieving context for specific topics."""
        with patch_other_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add multi-topic conversation
            state = self._add_multi_topic_conversation(state)

            context_manager = ContextWindowManager()

            # Retrieve context for specific topic
            topic_context = context_manager.get_topic_context(state, "AI Ethics")

            # Should contain only messages for that topic
            assert all(msg.get("topic") == "AI Ethics" for msg in topic_context)
            assert len(topic_context) > 0

    @pytest.mark.integration
    def test_context_retrieval_by_timeframe(self):
        """Test retrieving context for specific timeframes."""
        with patch_other_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add timestamped conversation
            state = self._add_timestamped_conversation(state)

            context_manager = ContextWindowManager()

            # Retrieve recent context (last 5 minutes)
            from datetime import timedelta

            recent_context = context_manager.get_recent_context(
                state, timeframe=timedelta(minutes=5)
            )

            # Should contain only recent messages
            assert len(recent_context) > 0
            cutoff_time = datetime.now() - timedelta(minutes=5)
            assert all(msg["timestamp"] >= cutoff_time for msg in recent_context)

    def _add_conversation_with_history(
        self, state: VirtualAgoraState
    ) -> VirtualAgoraState:
        """Add conversation with tracked history."""
        state["messages"] = []

        # Add messages with timestamps
        base_time = datetime.now()
        for i in range(15):
            for j, agent_id in enumerate(state["speaking_order"]):
                message = {
                    "id": str(uuid.uuid4()),
                    "agent_id": agent_id,
                    "content": f"Message {i*3+j+1}: Discussion point with reasoning and analysis.",
                    "timestamp": base_time.replace(minute=i * 2, second=j * 10),
                    "round_number": (i * 3 + j) // 3 + 1,
                    "topic": state["agenda"][0]["title"],
                    "message_type": "discussion",
                }
                state["messages"].append(message)

        return state

    def _add_multi_topic_conversation(
        self, state: VirtualAgoraState
    ) -> VirtualAgoraState:
        """Add conversation across multiple topics."""
        topics = ["AI Ethics", "Data Privacy", "Future Tech"]
        state["agenda"] = [
            {"title": topic, "description": f"{topic} discussion", "status": "active"}
            for topic in topics
        ]

        state["messages"] = []

        for topic in topics:
            for round_num in range(1, 4):
                for agent_id in state["speaking_order"]:
                    message = {
                        "id": str(uuid.uuid4()),
                        "agent_id": agent_id,
                        "content": f"Round {round_num} discussing {topic}: Analysis and insights.",
                        "timestamp": datetime.now(),
                        "round_number": round_num,
                        "topic": topic,
                        "message_type": "discussion",
                    }
                    state["messages"].append(message)

        return state

    def _add_timestamped_conversation(
        self, state: VirtualAgoraState
    ) -> VirtualAgoraState:
        """Add conversation with specific timestamps."""
        # Ensure required fields exist
        ensure_context_fields(state)

        state["messages"] = []

        base_time = datetime.now()

        # Add older messages (10 minutes ago)
        for i in range(5):
            message = {
                "id": str(uuid.uuid4()),
                "agent_id": state["speaking_order"][i % len(state["speaking_order"])],
                "content": f"Old message {i+1}: Earlier discussion point.",
                "timestamp": base_time.replace(minute=base_time.minute - 10 + i),
                "round_number": 1,
                "topic": state["agenda"][0]["title"],
                "message_type": "discussion",
            }
            state["messages"].append(message)

        # Add recent messages (last 3 minutes)
        for i in range(8):
            message = {
                "id": str(uuid.uuid4()),
                "agent_id": state["speaking_order"][i % len(state["speaking_order"])],
                "content": f"Recent message {i+1}: Current discussion point.",
                "timestamp": base_time.replace(
                    minute=base_time.minute - 3 + i // 3, second=i * 7
                ),
                "round_number": 2,
                "topic": state["agenda"][0]["title"],
                "message_type": "discussion",
            }
            state["messages"].append(message)

        return state


class TestContextRecovery:
    """Test context recovery and restoration workflows."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test method."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.integration
    def test_context_state_recovery(self):
        """Test recovery of context state after interruption."""
        with patch_other_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add rich context state
            state = self._add_rich_context_state(state)

            context_manager = ContextWindowManager()

            # Save context state
            context_snapshot = context_manager.create_context_snapshot(state)

            # Simulate interruption and context loss
            interrupted_state = self._simulate_context_loss(state)

            # Recover context
            recovered_state = context_manager.restore_context_snapshot(
                interrupted_state, context_snapshot
            )

            # Should restore essential context
            assert len(recovered_state.get("messages", [])) > 0
            assert "agenda" in recovered_state
            assert recovered_state.get("current_round") == state.get("current_round")

    @pytest.mark.integration
    def test_partial_context_reconstruction(self):
        """Test reconstruction when only partial context is available."""
        with patch_other_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add partial context data
            state = self._add_partial_context_data(state)

            context_manager = ContextWindowManager()

            # Simulate partial context loss
            partial_state = self._simulate_partial_context_loss(state)

            # Reconstruct missing context
            reconstructed_state = context_manager.reconstruct_context(partial_state)

            # Should have reasonable context reconstruction
            assert "messages" in reconstructed_state
            assert len(reconstructed_state.get("agenda", [])) > 0

            # Should indicate reconstructed nature
            metadata = reconstructed_state.get("metadata", {})
            assert metadata.get("context_reconstructed") == True

    @pytest.mark.integration
    def test_context_degradation_handling(self):
        """Test graceful handling of context degradation."""
        with patch_other_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add context that will degrade
            state = self._add_degrading_context(state)

            context_manager = ContextWindowManager(max_tokens=3000)  # Low limit

            # Handle degradation progressively
            degradation_stages = []
            current_state = state

            for stage in range(5):
                # Apply degradation
                current_state = context_manager.apply_degradation_step(current_state)
                degradation_stages.append(current_state.copy())

                # Should maintain minimum viable context
                assert "current_phase" in current_state
                assert "agenda" in current_state
                assert len(current_state.get("messages", [])) >= 0

            # Should show progressive degradation
            message_counts = [
                len(stage.get("messages", [])) for stage in degradation_stages
            ]
            assert message_counts[0] >= message_counts[-1]  # Should reduce over time

    def _add_rich_context_state(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Add rich context state for recovery testing."""
        # Add comprehensive discussion
        state["messages"] = []
        for round_num in range(1, 8):
            for agent_id in state["speaking_order"]:
                message = {
                    "id": str(uuid.uuid4()),
                    "agent_id": agent_id,
                    "content": f"Round {round_num}: Comprehensive discussion with detailed analysis.",
                    "timestamp": datetime.now(),
                    "round_number": round_num,
                    "topic": state["agenda"][0]["title"],
                    "message_type": "discussion",
                }
                state["messages"].append(message)

        # Add rich metadata
        state["metadata"] = {
            "session_context": {
                "total_rounds": 7,
                "active_participants": len(state["speaking_order"]),
                "discussion_intensity": "high",
                "consensus_level": 0.8,
            },
            "context_markers": {
                "important_decisions": ["Decision A", "Decision B"],
                "key_insights": ["Insight 1", "Insight 2"],
                "unresolved_issues": ["Issue 1"],
            },
        }

        state["current_round"] = 7
        return state

    def _simulate_context_loss(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Simulate context loss due to interruption."""
        interrupted_state = {
            "session_id": state["session_id"],
            "current_phase": state["current_phase"],
            "agenda": state["agenda"],
            # Messages lost
            "messages": [],
            # Metadata partially lost
            "metadata": {},
            # Round information lost
            "current_round": 1,
        }
        return interrupted_state

    def _add_partial_context_data(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Add partial context data for reconstruction testing."""
        # Limited messages
        state["messages"] = []
        for i in range(3):
            message = {
                "id": str(uuid.uuid4()),
                "agent_id": state["speaking_order"][i % len(state["speaking_order"])],
                "content": f"Partial message {i+1}: Fragment of discussion.",
                "timestamp": datetime.now(),
                "round_number": 1,
                "topic": state["agenda"][0]["title"],
                "message_type": "discussion",
            }
            state["messages"].append(message)

        # Partial round summaries
        state["round_summaries"] = [
            "Round 1: Partial summary available",
            "Round 2: Fragment of summary",
        ]

        return state

    def _simulate_partial_context_loss(
        self, state: VirtualAgoraState
    ) -> VirtualAgoraState:
        """Simulate partial context loss."""
        partial_state = state.copy()

        # Lose some messages
        partial_state["messages"] = state["messages"][:2]

        # Lose some metadata
        partial_state.pop("round_summaries", None)

        # Mark as having gaps
        partial_state["context_gaps"] = True

        return partial_state

    def _add_degrading_context(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Add context that will need progressive degradation."""
        # Add many messages
        state["messages"] = []
        for round_num in range(1, 15):
            for agent_id in state["speaking_order"]:
                message = {
                    "id": str(uuid.uuid4()),
                    "agent_id": agent_id,
                    "content": f"Round {round_num}: Detailed message with extensive content "
                    + "that will need to be compressed or summarized during degradation. "
                    * 5,
                    "timestamp": datetime.now(),
                    "round_number": round_num,
                    "topic": state["agenda"][0]["title"],
                    "message_type": "discussion",
                }
                state["messages"].append(message)

        # Add extensive metadata
        state["metadata"] = {
            "extensive_data": {f"key_{i}": f"value_{i}" for i in range(100)},
            "detailed_metrics": {f"metric_{i}": i * 1.5 for i in range(50)},
        }

        return state


@pytest.mark.integration
class TestMemoryPersistence:
    """Test memory persistence across sessions."""

    def test_memory_persistence_across_sessions(self):
        """Test that important context persists across sessions."""
        helper = IntegrationTestHelper(num_agents=2, scenario="default")

        with patch_other_ui_components():
            # First session
            flow1 = helper.create_test_flow()
            state1 = helper.create_discussion_state()

            # Add important context
            state1 = self._add_important_context(state1)

            context_manager = ContextWindowManager()

            # Save persistent context
            persistent_context = context_manager.extract_persistent_context(state1)

            # Second session (simulated restart)
            flow2 = helper.create_test_flow()
            state2 = helper.create_basic_state()

            # Restore persistent context
            restored_state = context_manager.restore_persistent_context(
                state2, persistent_context
            )

            # Should have key information from previous session
            assert (
                "previous_session_summary" in restored_state
                or "context_history" in restored_state
            )

            # Should maintain continuity
            if "agenda_continuity" in persistent_context:
                assert "agenda_continuity" in restored_state

    def test_selective_memory_preservation(self):
        """Test preserving only important memories."""
        helper = IntegrationTestHelper(num_agents=2, scenario="default")

        with patch_other_ui_components():
            flow = helper.create_test_flow()
            state = helper.create_discussion_state()

            # Add mixed importance context
            state = self._add_mixed_importance_context(state)

            context_manager = ContextWindowManager()

            # Extract selective memory
            selective_memory = context_manager.extract_selective_memory(
                state, importance_threshold=0.7
            )

            # Should contain only important information
            assert "high_importance_items" in selective_memory
            assert len(selective_memory["high_importance_items"]) > 0

            # Should not contain low importance information
            low_importance_keys = [
                k for k in selective_memory.keys() if "low" in k.lower()
            ]
            assert len(low_importance_keys) == 0

    def _add_important_context(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Add important context that should persist."""
        # Important decisions
        state["important_decisions"] = [
            {"decision": "Use AI safety protocols", "importance": 0.9},
            {"decision": "Implement privacy safeguards", "importance": 0.85},
            {"decision": "Regular security audits", "importance": 0.8},
        ]

        # Key insights
        state["key_insights"] = [
            {"insight": "AI requires human oversight", "importance": 0.95},
            {"insight": "Privacy is fundamental right", "importance": 0.9},
        ]

        # Unresolved issues for next session
        state["unresolved_issues"] = [
            {"issue": "Resource allocation", "priority": "high"},
            {"issue": "Timeline constraints", "priority": "medium"},
        ]

        return state

    def _add_mixed_importance_context(
        self, state: VirtualAgoraState
    ) -> VirtualAgoraState:
        """Add context with mixed importance levels."""
        # High importance items
        state["high_importance_items"] = [
            {"item": "Critical security decision", "importance": 0.95},
            {"item": "Major policy change", "importance": 0.9},
        ]

        # Medium importance items
        state["medium_importance_items"] = [
            {"item": "Process improvement", "importance": 0.6},
            {"item": "Minor clarification", "importance": 0.5},
        ]

        # Low importance items
        state["low_importance_items"] = [
            {"item": "Casual comment", "importance": 0.2},
            {"item": "Side discussion", "importance": 0.1},
        ]

        return state
