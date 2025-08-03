"""Agenda voting node for Virtual Agora flow.

This node handles voting on agenda topic ordering by collecting
votes from discussing agents on their preferred discussion order.
"""

import uuid
from typing import Dict, Any, List
from datetime import datetime

from virtual_agora.flow.nodes.base import FlowNode
from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class AgendaVotingNode(FlowNode):
    """Handles voting on agenda topic ordering.

    This node collects votes from discussing agents on their preferred
    order for discussing the topics in the queue.
    """

    def __init__(self, discussing_agents: List[LLMAgent]):
        """Initialize the agenda voting node.

        Args:
            discussing_agents: List of agents that will vote on topic order
        """
        super().__init__()
        self.discussing_agents = discussing_agents

    def execute(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Execute agenda voting process.

        Args:
            state: Current Virtual Agora state

        Returns:
            State updates with collected votes
        """
        logger.info("Node: agenda_voting - Collecting votes on topics")

        topics = state["topic_queue"]
        votes = []

        # Format topics for presentation
        topics_formatted = "\n".join(
            f"{i+1}. {topic}" for i, topic in enumerate(topics)
        )

        for agent in self.discussing_agents:
            prompt = f"""Vote on your preferred discussion order for these topics:
            {topics_formatted}

            Express your preferences in natural language. You may rank all topics
            or just indicate your top priorities."""

            try:
                # Call agent with proper state and prompt
                response_dict = agent(state, prompt=prompt)

                # Extract vote content
                messages = response_dict.get("messages", [])
                if messages:
                    vote_content = (
                        messages[-1].content
                        if hasattr(messages[-1], "content")
                        else str(messages[-1])
                    )
                    vote_obj = {
                        "id": f"vote_{uuid.uuid4().hex[:8]}",
                        "voter_id": agent.agent_id,
                        "phase": 1,  # Agenda voting phase
                        "vote_type": "topic_selection",
                        "choice": vote_content,
                        "timestamp": datetime.now(),
                    }
                    votes.append(vote_obj)
                    logger.info(f"Collected vote from {agent.agent_id}")
            except Exception as e:
                logger.error(f"Failed to get vote from {agent.agent_id}: {e}")
                vote_obj = {
                    "id": f"vote_{uuid.uuid4().hex[:8]}",
                    "voter_id": agent.agent_id,
                    "phase": 1,
                    "vote_type": "topic_selection",
                    "choice": "No preference",
                    "timestamp": datetime.now(),
                    "metadata": {"error": str(e)},
                }
                votes.append(vote_obj)

        # Pass vote batch to reducer
        updates = {
            "votes": votes,  # Vote batch - handled properly by enhanced reducer
            "current_phase": 2,  # Progress to next phase after voting
        }

        logger.info(f"Collected {len(votes)} votes for agenda ordering")
        return updates

    def validate_preconditions(self, state: VirtualAgoraState) -> bool:
        """Validate that node can execute.

        Args:
            state: Current Virtual Agora state

        Returns:
            True if preconditions are met
        """
        # Check that we have topics to vote on
        topic_queue = state.get("topic_queue", [])
        if not topic_queue:
            logger.warning("Cannot vote on agenda: no topic_queue in state")
            return False

        # Check that we have agents to vote
        if not self.discussing_agents:
            logger.warning("Cannot vote on agenda: no discussing agents available")
            return False

        return True

    def get_node_name(self) -> str:
        """Get human-readable node name."""
        return "AgendaVoting"
