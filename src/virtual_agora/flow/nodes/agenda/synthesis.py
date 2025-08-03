"""Agenda synthesis node for Virtual Agora flow.

This node uses the moderator agent to synthesize votes into a final agenda
by analyzing all votes, breaking ties, and producing a structured agenda.
"""

from typing import Dict, Any, List

from virtual_agora.flow.nodes.base import FlowNode
from virtual_agora.state.schema import VirtualAgoraState, Agenda
from virtual_agora.agents.moderator import ModeratorAgent
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class SynthesizeAgendaNode(FlowNode):
    """Handles synthesis of votes into final agenda.

    This node invokes the ModeratorAgent to:
    1. Analyze all votes
    2. Break ties
    3. Produce JSON agenda
    """

    def __init__(self, moderator_agent: ModeratorAgent):
        """Initialize the agenda synthesis node.

        Args:
            moderator_agent: Moderator agent to handle agenda synthesis
        """
        super().__init__()
        self.moderator_agent = moderator_agent

    def execute(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Execute agenda synthesis process.

        Args:
            state: Current Virtual Agora state

        Returns:
            State updates with synthesized agenda
        """
        logger.info("Node: synthesize_agenda - Moderator creating final agenda")

        votes = state["votes"]  # Use the correct schema field
        topics = state["topic_queue"]

        try:
            # Extract vote content from Vote objects
            # Handle case where votes might be nested lists due to reducer behavior
            flat_votes = []
            for v in votes:
                if isinstance(v, list):
                    # If vote is wrapped in a list, flatten it
                    flat_votes.extend(v)
                elif isinstance(v, dict):
                    # Normal vote object
                    flat_votes.append(v)
                else:
                    logger.warning(f"Unexpected vote type: {type(v)}, value: {v}")

            topic_votes = [
                v for v in flat_votes if v.get("vote_type") == "topic_selection"
            ]
            vote_responses = [v["choice"] for v in topic_votes]
            voter_ids = [v["voter_id"] for v in topic_votes]

            # Invoke moderator for synthesis
            # Convert votes to the expected format: List[Dict[str, str]]
            agent_votes = [
                {"agent_id": voter_id, "vote": vote_response}
                for voter_id, vote_response in zip(voter_ids, vote_responses)
            ]
            agenda_json = self.moderator_agent.synthesize_agenda(agent_votes)

            # Extract the proposed agenda
            if isinstance(agenda_json, dict) and "proposed_agenda" in agenda_json:
                final_agenda = agenda_json["proposed_agenda"]
            else:
                # Fallback if JSON parsing fails
                final_agenda = topics[:5]  # Take first 5 topics

        except Exception as e:
            logger.error(f"Failed to synthesize agenda: {e}")
            # Fallback: use topics in original order
            final_agenda = topics[:5]

        agenda_obj: Agenda = {
            "topics": final_agenda,
            "current_topic_index": 0,
            "completed_topics": [],
        }

        updates = {
            "agenda": agenda_obj,
            "proposed_agenda": final_agenda,  # For HITL approval
            "current_phase": 3,  # Move to next phase after synthesis
        }

        logger.info(f"Synthesized agenda with {len(final_agenda)} topics")

        return updates

    def validate_preconditions(self, state: VirtualAgoraState) -> bool:
        """Validate that node can execute.

        Args:
            state: Current Virtual Agora state

        Returns:
            True if preconditions are met
        """
        # Check that we have votes to synthesize
        votes = state.get("votes", [])
        if not votes:
            logger.warning("Cannot synthesize agenda: no votes in state")
            return False

        # Check that we have topics to work with
        topic_queue = state.get("topic_queue", [])
        if not topic_queue:
            logger.warning("Cannot synthesize agenda: no topic_queue in state")
            return False

        # Check that moderator agent is available
        if not self.moderator_agent:
            logger.warning("Cannot synthesize agenda: no moderator agent available")
            return False

        return True

    def get_node_name(self) -> str:
        """Get human-readable node name."""
        return "SynthesizeAgenda"
