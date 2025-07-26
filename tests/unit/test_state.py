"""Unit tests for Virtual Agora state management."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from virtual_agora.config.models import (
    Config as VirtualAgoraConfig,
    ModeratorConfig,
    AgentConfig,
    Provider,
)
from virtual_agora.state.schema import (
    VirtualAgoraState,
    AgentInfo,
    Message,
    Vote,
    PhaseTransition,
    VoteRound,
    TopicInfo,
    Agenda,
)
from virtual_agora.state.manager import StateManager
from virtual_agora.state.validators import StateValidator
from virtual_agora.state.utils import (
    format_state_summary,
    calculate_statistics,
    calculate_phase_durations,
    get_phase_messages,
    get_topic_messages,
    get_agent_messages,
    get_vote_results,
    export_for_analysis,
)
from virtual_agora.state.graph_integration import VirtualAgoraGraph
from virtual_agora.utils.exceptions import ValidationError, StateError


@pytest.fixture
def sample_config():
    """Create a sample configuration."""
    return VirtualAgoraConfig(
        moderator=ModeratorConfig(provider=Provider.OPENAI, model="gpt-4o"),
        agents=[
            AgentConfig(provider=Provider.OPENAI, model="gpt-4o-mini", count=2),
            AgentConfig(
                provider=Provider.ANTHROPIC, model="claude-3-5-haiku-latest", count=1
            ),
        ],
    )


@pytest.fixture
def state_manager(sample_config):
    """Create a state manager with sample config."""
    return StateManager(sample_config)


@pytest.fixture
def initialized_state(state_manager):
    """Create an initialized state."""
    return state_manager.initialize_state("test_session_001")


class TestStateSchema:
    """Test state schema definitions."""

    def test_agent_info_structure(self):
        """Test AgentInfo TypedDict structure."""
        now = datetime.now()
        agent = AgentInfo(
            id="agent_1",
            model="gpt-4o",
            provider="openai",
            role="participant",
            message_count=0,
            created_at=now,
        )

        assert agent["id"] == "agent_1"
        assert agent["model"] == "gpt-4o"
        assert agent["provider"] == "openai"
        assert agent["role"] == "participant"
        assert agent["message_count"] == 0
        assert agent["created_at"] == now

    def test_message_structure(self):
        """Test Message TypedDict structure."""
        now = datetime.now()
        message = Message(
            id="msg_001",
            speaker_id="agent_1",
            speaker_role="participant",
            content="This is a test message",
            timestamp=now,
            phase=1,
            topic="Test Topic",
        )

        assert message["id"] == "msg_001"
        assert message["speaker_id"] == "agent_1"
        assert message["speaker_role"] == "participant"
        assert message["content"] == "This is a test message"
        assert message["timestamp"] == now
        assert message["phase"] == 1
        assert message["topic"] == "Test Topic"

    def test_vote_structure(self):
        """Test Vote TypedDict structure."""
        now = datetime.now()
        vote = Vote(
            id="vote_001",
            voter_id="agent_1",
            phase=1,
            vote_type="topic_selection",
            choice="Topic A",
            timestamp=now,
        )

        assert vote["id"] == "vote_001"
        assert vote["voter_id"] == "agent_1"
        assert vote["phase"] == 1
        assert vote["vote_type"] == "topic_selection"
        assert vote["choice"] == "Topic A"
        assert vote["timestamp"] == now


class TestStateManager:
    """Test StateManager functionality."""

    def test_initialize_state(self, state_manager):
        """Test state initialization."""
        state = state_manager.initialize_state("test_001")

        assert state["session_id"] == "test_001"
        assert state["current_phase"] == 0
        assert state["moderator_id"] == "moderator"
        assert len(state["agents"]) == 4  # 1 moderator + 3 participants
        assert len(state["speaking_order"]) == 3  # Only participants
        assert state["current_speaker_id"] == "moderator"
        assert state["total_messages"] == 0
        assert state["active_topic"] is None
        assert state["topic_queue"] == []

    def test_auto_session_id(self, state_manager):
        """Test automatic session ID generation."""
        state = state_manager.initialize_state()

        assert state["session_id"] is not None
        assert len(state["session_id"]) > 10

    def test_agent_creation(self, state_manager):
        """Test proper agent creation from config."""
        state = state_manager.initialize_state()

        # Check moderator
        assert "moderator" in state["agents"]
        assert state["agents"]["moderator"]["role"] == "moderator"
        assert state["agents"]["moderator"]["model"] == "gpt-4o"

        # Check participants
        participants = [
            a for a in state["agents"].values() if a["role"] == "participant"
        ]
        assert len(participants) == 3

        # Check naming convention
        agent_names = [a["id"] for a in participants]
        assert "gpt-4o-mini_1" in agent_names
        assert "gpt-4o-mini_2" in agent_names
        assert "claude-3-5-haiku-latest_3" in agent_names

    def test_phase_transition_valid(self, state_manager, initialized_state):
        """Test valid phase transitions."""
        # 0 -> 1
        state_manager.transition_phase(1)
        assert state_manager.state["current_phase"] == 1
        assert len(state_manager.state["phase_history"]) == 1

        # 1 -> 2 (requires topics)
        state_manager.state["topic_queue"] = ["Topic 1"]
        state_manager.transition_phase(2)
        assert state_manager.state["current_phase"] == 2

    def test_phase_transition_invalid(self, state_manager, initialized_state):
        """Test invalid phase transitions."""
        # Cannot go from 0 to 2
        with pytest.raises(ValidationError, match="Invalid phase transition"):
            state_manager.transition_phase(2)

        # Cannot go to phase 2 without topics
        state_manager.transition_phase(1)
        with pytest.raises(ValidationError, match="Cannot start Discussion"):
            state_manager.transition_phase(2)

    def test_add_message(self, state_manager, initialized_state):
        """Test adding messages."""
        # Moderator can speak in phase 0
        msg = state_manager.add_message("moderator", "Welcome to Virtual Agora!")

        assert msg["id"] == "msg_000001"
        assert msg["speaker_id"] == "moderator"
        assert msg["content"] == "Welcome to Virtual Agora!"
        assert msg["phase"] == 0
        assert msg["topic"] is None

        # Check state updates
        assert state_manager.state["total_messages"] == 1
        assert state_manager.state["messages_by_phase"][0] == 1
        assert state_manager.state["messages_by_agent"]["moderator"] == 1
        assert state_manager.state["agents"]["moderator"]["message_count"] == 1

    def test_speaker_validation(self, state_manager, initialized_state):
        """Test speaker validation."""
        # Non-moderator cannot speak in phase 0
        with pytest.raises(ValidationError, match="Only the moderator"):
            state_manager.add_message("gpt-4o-mini_1", "Hello!")

        # Move to phase 1
        state_manager.transition_phase(1)

        # Wrong speaker
        with pytest.raises(ValidationError, match="not .* turn"):
            state_manager.add_message("gpt-4o-mini_2", "Hello!")

    def test_voting_system(self, state_manager, initialized_state):
        """Test voting functionality."""
        # Move to phase 1
        state_manager.transition_phase(1)

        # Start a vote
        vote_round = state_manager.start_vote(
            "topic_selection", ["Topic A", "Topic B", "Topic C"]
        )

        assert vote_round["id"] == "vote_0001"
        assert vote_round["vote_type"] == "topic_selection"
        assert vote_round["options"] == ["Topic A", "Topic B", "Topic C"]
        assert vote_round["required_votes"] == 3  # 3 participants
        assert vote_round["status"] == "active"

        # Cast votes
        state_manager.cast_vote("gpt-4o-mini_1", "Topic A")
        state_manager.cast_vote("gpt-4o-mini_2", "Topic B")
        state_manager.cast_vote("claude-3-5-haiku-latest_3", "Topic A")

        # Vote should be completed
        assert state_manager.state["active_vote"] is None
        assert len(state_manager.state["vote_history"]) == 1
        assert state_manager.state["vote_history"][0]["result"] == "Topic A"

    def test_duplicate_vote_prevention(self, state_manager, initialized_state):
        """Test that agents cannot vote twice."""
        state_manager.transition_phase(1)
        state_manager.start_vote("test", ["A", "B"])

        state_manager.cast_vote("gpt-4o-mini_1", "A")

        with pytest.raises(ValidationError, match="already voted"):
            state_manager.cast_vote("gpt-4o-mini_1", "B")

    def test_topic_management(self, state_manager, initialized_state):
        """Test topic proposal and activation."""
        state_manager.transition_phase(1)

        # Propose topics
        state_manager.propose_topic("AI Ethics", "gpt-4o-mini_1")
        state_manager.propose_topic("Future of Work", "gpt-4o-mini_2")

        assert len(state_manager.state["proposed_topics"]) == 2
        assert "AI Ethics" in state_manager.state["topics_info"]

        # Set topic queue
        state_manager.set_topic_queue(["Future of Work", "AI Ethics"])
        assert state_manager.state["topic_queue"] == ["Future of Work", "AI Ethics"]

        # Move to discussion (automatically activates first topic)
        state_manager.state["topic_queue"] = ["Future of Work", "AI Ethics"]
        state_manager.transition_phase(2)

        # Check that first topic was automatically activated
        assert state_manager.state["active_topic"] == "Future of Work"
        assert state_manager.state["topic_queue"] == ["AI Ethics"]
        assert (
            state_manager.state["topics_info"]["Future of Work"]["status"] == "active"
        )

    def test_complete_topic(self, state_manager, initialized_state):
        """Test topic completion."""
        # Setup
        state_manager.transition_phase(1)
        state_manager.propose_topic("Test Topic", "moderator")
        state_manager.set_topic_queue(["Test Topic"])
        state_manager.transition_phase(2)  # Automatically activates "Test Topic"

        # Complete topic
        state_manager.complete_topic()

        assert state_manager.state["active_topic"] is None
        assert "Test Topic" in state_manager.state["completed_topics"]
        assert state_manager.state["topics_info"]["Test Topic"]["status"] == "completed"

    def test_export_session(self, state_manager, initialized_state):
        """Test session export."""
        # Add some data
        state_manager.transition_phase(1)
        state_manager.add_message(
            state_manager.state["current_speaker_id"], "Test message"
        )

        export = state_manager.export_session()

        # Check datetime conversion
        assert isinstance(export["start_time"], str)
        assert isinstance(export["messages"][0]["timestamp"], str)

        # Check structure
        assert "session_id" in export
        assert "agents" in export
        assert "messages" in export


class TestStateValidator:
    """Test StateValidator functionality."""

    def test_phase_transition_validation(self):
        """Test phase transition rules."""
        validator = StateValidator()

        # Create minimal state for testing
        state = {
            "current_phase": 0,
            "topic_queue": [],
            "active_topic": None,
        }

        # Valid: 0 -> 1
        validator.validate_phase_transition(state, 1)

        # Invalid: 0 -> 2
        with pytest.raises(ValidationError):
            validator.validate_phase_transition(state, 2)

        # Invalid: 1 -> 2 without topics
        state["current_phase"] = 1
        with pytest.raises(ValidationError, match="without topics"):
            validator.validate_phase_transition(state, 2)

        # Valid: 1 -> 2 with topics
        state["topic_queue"] = ["Topic 1"]
        validator.validate_phase_transition(state, 2)

    def test_speaker_validation(self):
        """Test speaker validation rules."""
        validator = StateValidator()

        state = {
            "current_phase": 0,
            "agents": {
                "moderator": {"role": "moderator"},
                "agent_1": {"role": "participant"},
            },
            "current_speaker_id": "moderator",
        }

        # Valid: moderator in phase 0
        validator.validate_speaker(state, "moderator")

        # Invalid: participant in phase 0
        with pytest.raises(ValidationError, match="Only the moderator"):
            validator.validate_speaker(state, "agent_1")

        # Invalid: not their turn
        state["current_phase"] = 1
        state["current_speaker_id"] = "agent_1"
        with pytest.raises(ValidationError, match="not .* turn"):
            validator.validate_speaker(state, "moderator")

    def test_vote_validation(self):
        """Test vote validation."""
        validator = StateValidator()

        state = {
            "agents": {"agent_1": {}},
            "active_vote": None,
            "votes": [],
        }

        # No active vote
        with pytest.raises(ValidationError, match="No active vote"):
            validator.validate_vote(state, "agent_1", "Choice")

        # Valid vote
        state["active_vote"] = {
            "id": "vote_001",
            "status": "active",
            "options": ["A", "B"],
        }
        validator.validate_vote(state, "agent_1", "A")

        # Invalid choice
        with pytest.raises(ValidationError, match="Invalid vote choice"):
            validator.validate_vote(state, "agent_1", "C")

    def test_message_format_validation(self):
        """Test message format validation."""
        validator = StateValidator()

        # Valid message
        message = Message(
            id="msg_001",
            speaker_id="agent_1",
            speaker_role="participant",
            content="Hello",
            timestamp=datetime.now(),
            phase=1,
            topic=None,
        )
        validator.validate_message_format(message)

        # Empty content
        message["content"] = "  "
        with pytest.raises(ValidationError, match="cannot be empty"):
            validator.validate_message_format(message)

        # Invalid role
        message["content"] = "Hello"
        message["speaker_role"] = "observer"
        with pytest.raises(ValidationError, match="Invalid speaker role"):
            validator.validate_message_format(message)

    def test_state_consistency_check(self, state_manager, initialized_state):
        """Test state consistency validation."""
        validator = StateValidator()

        # Manually corrupt state
        state_manager.state["total_messages"] = 10  # Wrong count

        warnings = validator.validate_state_consistency(state_manager.state)

        assert len(warnings) > 0
        assert any("message count mismatch" in w.lower() for w in warnings)


class TestStateUtils:
    """Test state utility functions."""

    def test_format_state_summary(self, state_manager, initialized_state):
        """Test state summary formatting."""
        summary = format_state_summary(state_manager.state)

        assert "Virtual Agora State Summary" in summary
        assert "Session ID: test_session_001" in summary
        assert "Current Phase: Initialization" in summary
        assert "Agents: 4 total" in summary

    def test_calculate_statistics(self, state_manager, initialized_state):
        """Test statistics calculation."""
        # Add some data
        state_manager.add_message("moderator", "Welcome!")
        state_manager.transition_phase(1)

        stats = calculate_statistics(state_manager.state)

        assert stats["session"]["id"] == "test_session_001"
        assert stats["agents"]["total"] == 4
        assert stats["agents"]["participants"] == 3
        assert stats["messages"]["total"] == 1
        assert stats["messages"]["by_role"]["moderator"] == 1
        assert stats["messages"]["by_role"]["participant"] == 0

    def test_calculate_phase_durations(self, state_manager, initialized_state):
        """Test phase duration calculation."""
        # Add phase transitions with time gaps
        base_time = datetime.now() - timedelta(minutes=10)
        state_manager.state["start_time"] = base_time

        state_manager.state["phase_history"] = [
            PhaseTransition(
                from_phase=0,
                to_phase=1,
                timestamp=base_time + timedelta(minutes=2),
                reason="test",
                triggered_by="system",
            ),
            PhaseTransition(
                from_phase=1,
                to_phase=2,
                timestamp=base_time + timedelta(minutes=5),
                reason="test",
                triggered_by="system",
            ),
        ]
        state_manager.state["current_phase"] = 2

        durations = calculate_phase_durations(state_manager.state)

        assert durations[0] == pytest.approx(120, abs=1)  # 2 minutes
        assert durations[1] == pytest.approx(180, abs=1)  # 3 minutes
        assert durations[2] > 0  # Current phase

    def test_get_phase_messages(self, state_manager, initialized_state):
        """Test filtering messages by phase."""
        # Add messages in different phases
        state_manager.add_message("moderator", "Phase 0 message")
        state_manager.transition_phase(1)
        state_manager.state["current_speaker_id"] = "gpt-4o-mini_1"
        state_manager.add_message("gpt-4o-mini_1", "Phase 1 message")

        phase_0_msgs = get_phase_messages(state_manager.state, 0)
        phase_1_msgs = get_phase_messages(state_manager.state, 1)

        assert len(phase_0_msgs) == 1
        assert len(phase_1_msgs) == 1
        assert phase_0_msgs[0]["content"] == "Phase 0 message"
        assert phase_1_msgs[0]["content"] == "Phase 1 message"

    def test_get_vote_results(self, state_manager, initialized_state):
        """Test vote result calculation."""
        # Setup vote
        state_manager.transition_phase(1)
        state_manager.start_vote("test", ["A", "B"])
        vote_id = state_manager.state["active_vote"]["id"]

        # Cast votes
        state_manager.cast_vote("gpt-4o-mini_1", "A")
        state_manager.cast_vote("gpt-4o-mini_2", "A")
        state_manager.cast_vote("claude-3-5-haiku-latest_3", "B")

        results = get_vote_results(state_manager.state, vote_id)

        assert results["A"] == 2
        assert results["B"] == 1

    def test_export_for_analysis(self, state_manager, initialized_state):
        """Test analysis export format."""
        # Add some data
        state_manager.add_message("moderator", "Test message")

        export = export_for_analysis(state_manager.state)

        assert "session_info" in export
        assert "agents" in export
        assert "messages" in export
        assert "statistics" in export

        # Check message format
        assert len(export["messages"]) == 1
        msg = export["messages"][0]
        assert "word_count" in msg
        assert "char_count" in msg
        assert msg["word_count"] == 2  # "Test message"


class TestGraphIntegration:
    """Test LangGraph integration."""

    @pytest.mark.parametrize(
        "test_name, test_func",
        [
            ("test_graph_creation", lambda config: VirtualAgoraGraph(config)),
            (
                "test_create_session",
                lambda config: VirtualAgoraGraph(config).create_session("test-session"),
            ),
            (
                "test_graph_nodes",
                lambda config: VirtualAgoraGraph(config).build_graph().nodes,
            ),
            (
                "test_conditional_edges",
                lambda config: VirtualAgoraGraph(config).build_graph().edges,
            ),
            (
                "test_topic_continuation_logic",
                lambda config: VirtualAgoraGraph(config)._should_continue_discussion(
                    VirtualAgoraState(
                        agenda=Agenda(topics=["a", "b"]), active_topic="a", messages=[]
                    )
                ),
            ),
            (
                "test_state_update_through_graph",
                lambda config: (
                    graph := VirtualAgoraGraph(config),
                    graph.create_session("test-session"),
                    graph.update_state(updates={"current_phase": 1}),
                ),
            ),
        ],
    )
    @patch("virtual_agora.state.graph_integration.create_provider")
    def test_graph_integration_functions(
        self, mock_create_provider, test_name, test_func, sample_config
    ):
        """Test various graph integration functions with mocked provider creation."""
        mock_create_provider.return_value = Mock()
        test_func(sample_config)
