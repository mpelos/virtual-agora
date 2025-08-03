"""State validation logic for Virtual Agora.

This module provides validation functions to ensure state transitions
and updates follow the rules of the Virtual Agora discussion format.
"""

from typing import List, Dict, Optional, Set
from datetime import datetime

from virtual_agora.state.schema import (
    VirtualAgoraState,
    AgentInfo,
    Message,
    Vote,
    PhaseTransition,
    VoteRound,
)
from virtual_agora.utils.exceptions import ValidationError


class StateValidator:
    """Validates state transitions and updates for Virtual Agora."""

    # Valid phase transitions
    VALID_PHASE_TRANSITIONS = {
        0: [1],  # Initialization -> Agenda Setting
        1: [2],  # Agenda Setting -> Discussion
        2: [2, 3],  # Discussion -> Discussion (continue) or Consensus
        3: [2, 4],  # Consensus -> Discussion (new topic) or Summary
        4: [],  # Summary -> End (no transitions)
    }

    # Phase names for error messages
    PHASE_NAMES = {
        0: "Initialization",
        1: "Agenda Setting",
        2: "Discussion",
        3: "Consensus Building",
        4: "Summary Generation",
    }

    def __init__(self):
        """Initialize the state validator."""
        pass

    def validate_phase_transition(
        self, state: VirtualAgoraState, new_phase: int
    ) -> None:
        """Validate a phase transition.

        Args:
            state: Current state
            new_phase: Target phase

        Raises:
            ValidationError: If the transition is invalid
        """
        current_phase = state["current_phase"]

        if new_phase not in self.VALID_PHASE_TRANSITIONS.get(current_phase, []):
            raise ValidationError(
                f"Invalid phase transition: {self.PHASE_NAMES[current_phase]} "
                f"({current_phase}) -> {self.PHASE_NAMES.get(new_phase, 'Unknown')} "
                f"({new_phase})"
            )

        # Additional validation based on phase requirements
        if new_phase == 2 and current_phase == 1:
            # Moving from Agenda Setting to Discussion requires topics
            if not state["topic_queue"]:
                raise ValidationError(
                    "Cannot start Discussion phase without topics in the queue"
                )

        if new_phase == 3:
            # Moving to Consensus requires an active topic
            if not state["active_topic"]:
                raise ValidationError(
                    "Cannot enter Consensus phase without an active topic"
                )

        if new_phase == 4:
            # Moving to Summary requires all topics to be completed
            if state["topic_queue"] or state["active_topic"]:
                raise ValidationError("Cannot enter Summary phase with pending topics")

    def validate_speaker(self, state: VirtualAgoraState, speaker_id: str) -> None:
        """Validate that a speaker is allowed to speak.

        Args:
            state: Current state
            speaker_id: ID of the agent trying to speak

        Raises:
            ValidationError: If the speaker is not allowed
        """
        # Check if agent exists
        if speaker_id not in state["agents"]:
            raise ValidationError(f"Unknown agent: {speaker_id}")

        # Phase-specific validation first
        phase = state["current_phase"]
        agent = state["agents"][speaker_id]

        if phase == 0:
            # Only moderator speaks during initialization
            if agent["role"] != "moderator":
                raise ValidationError(
                    "Only the moderator can speak during Initialization"
                )

        if phase == 4:
            # Only moderator speaks during summary
            if agent["role"] != "moderator":
                raise ValidationError(
                    "Only the moderator can speak during Summary Generation"
                )

        # Check if it's the agent's turn
        if state["current_speaker_id"] != speaker_id:
            current = state["current_speaker_id"] or "None"
            raise ValidationError(
                f"It's not {speaker_id}'s turn to speak. " f"Current speaker: {current}"
            )

    def validate_vote(
        self, state: VirtualAgoraState, voter_id: str, choice: str
    ) -> None:
        """Validate a vote.

        Args:
            state: Current state
            voter_id: ID of the voting agent
            choice: The vote choice

        Raises:
            ValidationError: If the vote is invalid
        """
        # Check if agent exists
        if voter_id not in state["agents"]:
            raise ValidationError(f"Unknown agent: {voter_id}")

        # Check if there's an active vote
        if not state["active_vote"]:
            raise ValidationError("No active vote to participate in")

        active_vote = state["active_vote"]

        # Check if vote is still open
        if active_vote["status"] != "active":
            raise ValidationError(
                f"Vote is not active (status: {active_vote['status']})"
            )

        # Check if choice is valid
        if choice not in active_vote["options"]:
            raise ValidationError(
                f"Invalid vote choice: {choice}. "
                f"Valid options: {', '.join(active_vote['options'])}"
            )

        # Check if agent already voted
        vote_id = active_vote["id"]
        existing_votes = [
            v
            for v in state["votes"]
            if v["voter_id"] == voter_id
            and v.get("metadata", {}).get("vote_round_id") == vote_id
        ]
        if existing_votes:
            raise ValidationError(f"Agent {voter_id} has already voted in this round")

    def validate_topic_transition(
        self, state: VirtualAgoraState, new_topic: Optional[str]
    ) -> None:
        """Validate a topic transition.

        Args:
            state: Current state
            new_topic: New topic to activate (None to deactivate)

        Raises:
            ValidationError: If the transition is invalid
        """
        phase = state["current_phase"]

        # Topics can only be active during Discussion and Consensus phases
        if new_topic and phase not in [2, 3]:
            raise ValidationError(
                f"Cannot activate topic during {self.PHASE_NAMES[phase]} phase"
            )

        # New topic must be from the queue
        if new_topic and new_topic not in state["topic_queue"]:
            if new_topic not in state["completed_topics"]:
                raise ValidationError(f"Topic '{new_topic}' is not in the queue")
            else:
                raise ValidationError(f"Topic '{new_topic}' has already been discussed")

    def validate_message_format(self, message: Message) -> None:
        """Validate message format.

        Args:
            message: Message to validate

        Raises:
            ValidationError: If the message format is invalid
        """
        required_fields = [
            "id",
            "speaker_id",
            "speaker_role",
            "content",
            "timestamp",
            "phase",
        ]

        for field in required_fields:
            if field not in message:
                raise ValidationError(f"Message missing required field: {field}")

        if not message["content"].strip():
            raise ValidationError("Message content cannot be empty")

        if message["speaker_role"] not in ["moderator", "participant"]:
            raise ValidationError(f"Invalid speaker role: {message['speaker_role']}")

    def validate_agent_info(self, agent: AgentInfo) -> None:
        """Validate agent information.

        Args:
            agent: Agent info to validate

        Raises:
            ValidationError: If the agent info is invalid
        """
        required_fields = [
            "id",
            "model",
            "provider",
            "role",
            "message_count",
            "created_at",
        ]

        for field in required_fields:
            if field not in agent:
                raise ValidationError(f"Agent missing required field: {field}")

        if agent["role"] not in ["moderator", "participant"]:
            raise ValidationError(f"Invalid agent role: {agent['role']}")

        if agent["message_count"] < 0:
            raise ValidationError("Message count cannot be negative")

    def validate_speaking_order(
        self, state: VirtualAgoraState, speaking_order: List[str]
    ) -> None:
        """Validate a speaking order.

        Args:
            state: Current state
            speaking_order: Proposed speaking order

        Raises:
            ValidationError: If the speaking order is invalid
        """
        # All agents in speaking order must exist
        agent_ids = set(state["agents"].keys())
        order_ids = set(speaking_order)

        unknown = order_ids - agent_ids
        if unknown:
            raise ValidationError(
                f"Unknown agents in speaking order: {', '.join(unknown)}"
            )

        # In discussion phases, all participants should be in the order
        if state["current_phase"] in [1, 2, 3]:
            participants = {
                aid
                for aid, info in state["agents"].items()
                if info["role"] == "participant"
            }
            missing = participants - order_ids
            if missing:
                raise ValidationError(
                    f"Missing participants in speaking order: {', '.join(missing)}"
                )

    def validate_state_consistency(self, state: VirtualAgoraState) -> List[str]:
        """Perform comprehensive state consistency check.

        Args:
            state: State to validate

        Returns:
            List of warning messages (empty if state is fully consistent)
        """
        warnings = []

        # Check message counts
        actual_total = len(state["messages"])
        if state["total_messages"] != actual_total:
            warnings.append(
                f"Total message count mismatch: "
                f"recorded={state['total_messages']}, actual={actual_total}"
            )

        # Check agent message counts
        agent_msg_counts: Dict[str, int] = {}
        for msg in state["messages"]:
            # Safely extract speaker_id from both AIMessage objects and dict formats
            if hasattr(
                msg, "content"
            ):  # LangChain BaseMessage (AIMessage, HumanMessage, etc.)
                speaker_id = getattr(msg, "additional_kwargs", {}).get(
                    "speaker_id", "unknown"
                )
            else:  # Virtual Agora dict format
                speaker_id = msg.get("speaker_id", "unknown")

            agent_msg_counts[speaker_id] = agent_msg_counts.get(speaker_id, 0) + 1

        for agent_id, count in agent_msg_counts.items():
            if agent_id in state["agents"]:
                recorded = state["agents"][agent_id]["message_count"]
                if recorded != count:
                    warnings.append(
                        f"Message count mismatch for {agent_id}: "
                        f"recorded={recorded}, actual={count}"
                    )

        # Check vote participation
        if state["vote_history"]:
            for vote_round in state["vote_history"]:
                if vote_round["status"] == "completed":
                    expected = vote_round["required_votes"]
                    actual = vote_round["received_votes"]
                    if actual < expected:
                        warnings.append(
                            f"Incomplete vote round {vote_round['id']}: "
                            f"{actual}/{expected} votes"
                        )

        # Check topic status
        for topic, info in state["topics_info"].items():
            if info["status"] == "active" and topic != state["active_topic"]:
                warnings.append(f"Topic '{topic}' marked as active but not current")

        return warnings
