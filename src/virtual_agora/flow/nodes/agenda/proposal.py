"""Agenda proposal node for Virtual Agora flow.

This node handles the collection of topic proposals from discussing agents
during the agenda-setting phase of the discussion.
"""

from typing import Dict, Any, List
from datetime import datetime

from virtual_agora.flow.nodes.base import FlowNode
from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.providers.config import ProviderType
from virtual_agora.ui.discussion_display import display_agent_response
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


def get_provider_type_from_agent_id(agent_id: str) -> ProviderType:
    """Extract provider type from agent ID.

    Agent IDs typically follow patterns like:
    - gpt-4o-1, gpt-4o-2 -> OpenAI
    - claude-3-opus-1 -> Anthropic
    - gemini-2.5-pro-1 -> Google
    - grok-beta-1 -> Grok
    """
    agent_id_lower = agent_id.lower()

    if "gpt" in agent_id_lower or "openai" in agent_id_lower:
        return ProviderType.OPENAI
    elif "claude" in agent_id_lower or "anthropic" in agent_id_lower:
        return ProviderType.ANTHROPIC
    elif "gemini" in agent_id_lower or "google" in agent_id_lower:
        return ProviderType.GOOGLE
    elif "grok" in agent_id_lower:
        return ProviderType.GROK
    else:
        # Default fallback
        return ProviderType.OPENAI


class AgendaProposalNode(FlowNode):
    """Handles topic proposal collection from discussing agents.

    This node:
    1. Prompts each discussing agent for 3-5 topics
    2. Collects all proposals
    3. Updates state with raw proposals
    """

    def __init__(self, discussing_agents: List[LLMAgent]):
        """Initialize the agenda proposal node.

        Args:
            discussing_agents: List of agents to collect proposals from
        """
        super().__init__()
        self.discussing_agents = discussing_agents

    def execute(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Execute topic proposal collection.

        Args:
            state: Current Virtual Agora state

        Returns:
            State updates with collected proposals
        """
        logger.info("Node: agenda_proposal - Collecting topic proposals")

        theme = state["main_topic"]
        proposals = []

        # Request proposals from each discussing agent
        logger.info(
            f"Collecting topic proposals from {len(self.discussing_agents)} agents..."
        )
        for i, agent in enumerate(self.discussing_agents):
            logger.info(
                f"Getting proposals from {agent.agent_id} ({i+1}/{len(self.discussing_agents)})"
            )

            prompt = f"""Based on the theme '{theme}', propose 3-5 strategic topics that will serve as a compass to guide our discussion toward the best possible conclusion.

Think strategically: What key areas need to be explored and in what logical order to build comprehensive understanding? Consider topics as building blocks that lead from foundational concepts to deeper insights.

Your topics should:
- Address essential aspects that must be discussed to reach meaningful conclusions
- Build upon each other in a logical progression
- Cover different dimensions of the theme to ensure comprehensive coverage
- Be designed to facilitate knowledge building throughout the discussion

Frame each topic as a stepping stone toward collective understanding. Consider: What needs to be discussed and in what order to arrive at the most insightful and complete conclusion?"""

            try:
                # Call agent with proper state and prompt
                response_dict = agent(state, prompt=prompt)

                # Extract response content
                messages = response_dict.get("messages", [])
                if messages:
                    response_content = (
                        messages[-1].content
                        if hasattr(messages[-1], "content")
                        else str(messages[-1])
                    )
                    proposals.append(
                        {"agent_id": agent.agent_id, "proposals": response_content}
                    )
                    logger.info(f"Collected proposals from {agent.agent_id}")

                    # Display the agent's topic proposal in the UI
                    provider_type = get_provider_type_from_agent_id(agent.agent_id)
                    display_agent_response(
                        agent_id=agent.agent_id,
                        provider=provider_type,
                        content=response_content,
                        round_number=0,  # Phase 1 - agenda setting
                        topic="Strategic Topic Proposals",
                        timestamp=datetime.now(),
                    )
            except Exception as e:
                logger.error(f"Failed to get proposals from {agent.agent_id}: {e}")
                proposals.append(
                    {
                        "agent_id": agent.agent_id,
                        "proposals": "Failed to provide proposals",
                        "error": str(e),
                    }
                )

                # Display the failure for transparency
                provider_type = get_provider_type_from_agent_id(agent.agent_id)
                display_agent_response(
                    agent_id=agent.agent_id,
                    provider=provider_type,
                    content=f"❌ Failed to provide topic proposals: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}",
                    round_number=0,  # Phase 1 - agenda setting
                    topic="Strategic Topic Proposals",
                    timestamp=datetime.now(),
                )

        updates = {
            "proposed_topics": proposals,
            "current_phase": 1,
            "phase_start_time": datetime.now(),
        }

        logger.info(f"Collected proposals from {len(proposals)} agents")

        return updates

    def validate_preconditions(self, state: VirtualAgoraState) -> bool:
        """Validate that node can execute.

        Args:
            state: Current Virtual Agora state

        Returns:
            True if preconditions are met
        """
        # Check that main topic is set
        if not state.get("main_topic"):
            logger.warning("Cannot collect proposals: main_topic not set")
            return False

        # Check that we have agents to query
        if not self.discussing_agents:
            logger.warning("Cannot collect proposals: no discussing agents available")
            return False

        return True

    def get_node_name(self) -> str:
        """Get human-readable node name."""
        return "AgendaProposal"
