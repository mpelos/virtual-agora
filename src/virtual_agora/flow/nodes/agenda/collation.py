"""Proposal collation node for Virtual Agora flow.

This node handles the consolidation and deduplication of topic proposals
using the moderator agent to create a unified topic list.
"""

from typing import Dict, Any, List
from datetime import datetime

from virtual_agora.flow.nodes.base import FlowNode
from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.agents.moderator import ModeratorAgent
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class CollateProposalsNode(FlowNode):
    """Handles collation and deduplication of topic proposals.

    This node invokes the ModeratorAgent as a tool to:
    1. Read all proposals
    2. Remove duplicates
    3. Create unified list
    """

    def __init__(self, moderator_agent: ModeratorAgent):
        """Initialize the collation node.

        Args:
            moderator_agent: Moderator agent to handle proposal collation
        """
        super().__init__()
        self.moderator_agent = moderator_agent

    def execute(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Execute proposal collation and deduplication.

        Args:
            state: Current Virtual Agora state

        Returns:
            State updates with unified topic queue
        """
        logger.info("Node: collate_proposals - Moderator processing proposals")

        proposals = state["proposed_topics"]

        try:
            # Invoke moderator as a tool to collect proposals
            # Pass the proposals list directly as it already has the right format
            unified_list = self.moderator_agent.collect_proposals(proposals)

            logger.info(f"Moderator compiled {len(unified_list)} unique topics")

        except Exception as e:
            logger.error(f"Failed to collate proposals: {e}")
            # Fallback: extract topics manually
            unified_list = self._fallback_collation(proposals)

        updates = {
            "topic_queue": unified_list,
            "current_phase": 1,  # Move to next phase after collating
        }

        return updates

    def _fallback_collation(self, proposals: List[Dict[str, Any]]) -> List[str]:
        """Fallback method for manual proposal collation.

        Args:
            proposals: List of proposal dictionaries

        Returns:
            List of deduplicated topic strings
        """
        unified_list = []
        for p in proposals:
            # Simple extraction - look for numbered items
            text = p["proposals"]
            lines = text.split("\n")
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    # Remove numbering/bullets
                    topic = line.lstrip("0123456789.-) ").strip()
                    if topic and topic not in unified_list:
                        unified_list.append(topic)

        unified_list = unified_list[:10]  # Limit to 10 topics
        return unified_list

    def validate_preconditions(self, state: VirtualAgoraState) -> bool:
        """Validate that node can execute.

        Args:
            state: Current Virtual Agora state

        Returns:
            True if preconditions are met
        """
        # Check that we have proposals to collate
        proposed_topics = state.get("proposed_topics", [])
        if not proposed_topics:
            logger.warning("Cannot collate proposals: no proposed_topics in state")
            return False

        # Check that moderator agent is available
        if not self.moderator_agent:
            logger.warning("Cannot collate proposals: no moderator agent available")
            return False

        return True

    def get_node_name(self) -> str:
        """Get human-readable node name."""
        return "CollateProposals"
