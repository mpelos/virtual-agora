"""Factory for creating agenda nodes with proper dependency injection.

This module provides centralized creation and configuration of agenda nodes
for Step 3.2 of the architecture refactoring. It ensures all agenda nodes
receive proper dependencies and handles fallback strategies for nodes
not yet extracted.
"""

from typing import Dict, Any, List, Optional

from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.agents.moderator import ModeratorAgent
from virtual_agora.flow.nodes.base import FlowNode
from virtual_agora.utils.logging import get_logger

# Import extracted agenda nodes
from .proposal import AgendaProposalNode
from .refinement import TopicRefinementNode
from .collation import CollateProposalsNode
from .voting import AgendaVotingNode
from .synthesis import SynthesizeAgendaNode

logger = get_logger(__name__)


class AgendaNodeFactory:
    """Factory for creating agenda nodes with proper dependency injection.

    This factory centralizes the creation of all agenda-related nodes,
    ensuring consistent dependency injection and configuration. It provides
    a clean interface for the graph to obtain properly configured agenda nodes
    without needing to know implementation details.

    Key responsibilities:
    - Validate required dependencies (moderator agent, discussing agents)
    - Create properly configured instances of extracted agenda nodes
    - Handle fallback strategies for nodes not yet extracted
    - Provide metadata about node capabilities and requirements
    """

    def __init__(
        self, moderator_agent: ModeratorAgent, discussing_agents: List[LLMAgent]
    ):
        """Initialize agenda node factory.

        Args:
            moderator_agent: Moderator agent for proposal collation and synthesis
            discussing_agents: List of discussing agents for proposals and voting

        Raises:
            ValueError: If required dependencies are missing or invalid
        """
        if not moderator_agent:
            raise ValueError("Moderator agent is required for agenda node factory")

        if not discussing_agents:
            raise ValueError(
                "At least one discussing agent is required for agenda node factory"
            )

        if not isinstance(moderator_agent, ModeratorAgent):
            raise ValueError(f"Expected ModeratorAgent, got {type(moderator_agent)}")

        # Validate discussing agents
        for i, agent in enumerate(discussing_agents):
            if not isinstance(agent, LLMAgent):
                raise ValueError(f"Expected LLMAgent at index {i}, got {type(agent)}")

        self.moderator_agent = moderator_agent
        self.discussing_agents = discussing_agents

        logger.info(
            f"Initialized AgendaNodeFactory with moderator '{moderator_agent.agent_id}' "
            f"and {len(discussing_agents)} discussing agents"
        )

    def create_agenda_proposal_node(self) -> AgendaProposalNode:
        """Create agenda proposal node for topic collection.

        Returns:
            Configured AgendaProposalNode instance
        """
        logger.debug("Creating AgendaProposalNode")
        return AgendaProposalNode(discussing_agents=self.discussing_agents)

    def create_topic_refinement_node(self) -> TopicRefinementNode:
        """Create topic refinement node for proposal enhancement.

        Returns:
            Configured TopicRefinementNode instance
        """
        logger.debug("Creating TopicRefinementNode")
        return TopicRefinementNode(discussing_agents=self.discussing_agents)

    def create_collate_proposals_node(self) -> CollateProposalsNode:
        """Create proposal collation node for unified topic list.

        Returns:
            Configured CollateProposalsNode instance
        """
        logger.debug("Creating CollateProposalsNode")
        return CollateProposalsNode(moderator_agent=self.moderator_agent)

    def create_agenda_voting_node(self) -> AgendaVotingNode:
        """Create agenda voting node for topic prioritization.

        Returns:
            Configured AgendaVotingNode instance
        """
        logger.debug("Creating AgendaVotingNode")
        return AgendaVotingNode(discussing_agents=self.discussing_agents)

    def create_synthesize_agenda_node(self) -> SynthesizeAgendaNode:
        """Create agenda synthesis node for final agenda creation.

        Returns:
            Configured SynthesizeAgendaNode instance
        """
        logger.debug("Creating SynthesizeAgendaNode")
        return SynthesizeAgendaNode(moderator_agent=self.moderator_agent)

    def create_agenda_approval_node(self) -> Optional[FlowNode]:
        """Create agenda approval node for user validation.

        Note: AgendaApprovalNode is not yet extracted from Step 3.1.
        This method returns None to indicate fallback to V13 wrapper is needed.

        Returns:
            None (indicating V13 wrapper should be used)
        """
        logger.debug(
            "AgendaApprovalNode not yet extracted - returning None for V13 fallback"
        )
        return None

    def create_all_agenda_nodes(self) -> Dict[str, Optional[FlowNode]]:
        """Create all available agenda nodes.

        Returns:
            Dictionary mapping node names to FlowNode instances.
            None values indicate V13 wrapper fallback is needed.
        """
        logger.info("Creating all agenda nodes")

        nodes = {
            "agenda_proposal": self.create_agenda_proposal_node(),
            "topic_refinement": self.create_topic_refinement_node(),
            "collate_proposals": self.create_collate_proposals_node(),
            "agenda_voting": self.create_agenda_voting_node(),
            "synthesize_agenda": self.create_synthesize_agenda_node(),
            "agenda_approval": self.create_agenda_approval_node(),
        }

        # Count successfully created nodes
        created_count = sum(1 for node in nodes.values() if node is not None)
        fallback_count = len(nodes) - created_count

        logger.info(
            f"Created {created_count} extracted agenda nodes, "
            f"{fallback_count} nodes will use V13 fallback"
        )

        return nodes

    def get_node_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata about all agenda nodes.

        Returns:
            Dictionary mapping node names to their metadata
        """
        return {
            "agenda_proposal": {
                "extracted": True,
                "dependencies": ["discussing_agents"],
                "phase": 1,
                "description": "Collect topic proposals from discussing agents",
            },
            "topic_refinement": {
                "extracted": True,
                "dependencies": ["discussing_agents"],
                "phase": 1,
                "description": "Refine and enhance proposed topics",
            },
            "collate_proposals": {
                "extracted": True,
                "dependencies": ["moderator_agent"],
                "phase": 1,
                "description": "Collate proposals into unified topic list",
            },
            "agenda_voting": {
                "extracted": True,
                "dependencies": ["discussing_agents"],
                "phase": 1,
                "description": "Collect votes on topic prioritization",
            },
            "synthesize_agenda": {
                "extracted": True,
                "dependencies": ["moderator_agent"],
                "phase": 1,
                "description": "Synthesize votes into final agenda",
            },
            "agenda_approval": {
                "extracted": False,
                "dependencies": ["v13_wrapper"],
                "phase": 1,
                "description": "Human-in-the-loop agenda approval (V13 wrapper)",
            },
        }

    def validate_dependencies(self) -> Dict[str, bool]:
        """Validate that all required dependencies are available.

        Returns:
            Dictionary mapping dependency names to their availability status
        """
        return {
            "moderator_agent_available": self.moderator_agent is not None,
            "moderator_agent_valid": isinstance(self.moderator_agent, ModeratorAgent),
            "discussing_agents_available": bool(self.discussing_agents),
            "discussing_agents_count": len(self.discussing_agents),
            "all_agents_valid": all(
                isinstance(agent, LLMAgent) for agent in self.discussing_agents
            ),
        }

    def get_factory_summary(self) -> Dict[str, Any]:
        """Get summary information about the factory and its capabilities.

        Returns:
            Dictionary with factory summary
        """
        dependencies = self.validate_dependencies()
        metadata = self.get_node_metadata()

        extracted_nodes = [name for name, meta in metadata.items() if meta["extracted"]]
        fallback_nodes = [
            name for name, meta in metadata.items() if not meta["extracted"]
        ]

        return {
            "total_agenda_nodes": len(metadata),
            "extracted_nodes": extracted_nodes,
            "extracted_count": len(extracted_nodes),
            "fallback_nodes": fallback_nodes,
            "fallback_count": len(fallback_nodes),
            "dependencies_valid": all(dependencies.values()),
            "dependency_status": dependencies,
            "moderator_agent_id": (
                self.moderator_agent.agent_id if self.moderator_agent else None
            ),
            "discussing_agent_ids": [
                agent.agent_id for agent in self.discussing_agents
            ],
        }
