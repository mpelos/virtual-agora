"""Topic refinement node for Virtual Agora flow.

This node handles collaborative refinement of topic proposals where agents
can review all initial proposals and refine their own suggestions.
"""

from typing import Dict, Any, List
from datetime import datetime

from virtual_agora.flow.nodes.base import FlowNode
from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.providers.config import ProviderType
from virtual_agora.ui.discussion_display import display_agent_response
from virtual_agora.utils.logging import get_logger

# Import utility function from proposal module
from .proposal import get_provider_type_from_agent_id

logger = get_logger(__name__)


class TopicRefinementNode(FlowNode):
    """Handles collaborative refinement of topic proposals.

    This node:
    1. Shows all initial proposals to each agent
    2. Allows agents to refine, merge, or replace their topics
    3. Enables collaborative consensus building
    4. Maintains strategic focus on conclusion-oriented progression
    """

    def __init__(self, discussing_agents: List[LLMAgent]):
        """Initialize the topic refinement node.

        Args:
            discussing_agents: List of agents to collect refinements from
        """
        super().__init__()
        self.discussing_agents = discussing_agents

    def execute(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Execute collaborative topic refinement.

        Args:
            state: Current Virtual Agora state

        Returns:
            State updates with refined proposals
        """
        logger.info("Node: topic_refinement - Collaborative topic refinement")

        initial_proposals = state["proposed_topics"]
        theme = state["main_topic"]
        refined_proposals = []

        # Create comprehensive view of all initial proposals for context
        all_proposals_text = "\n\n".join(
            [
                f"**{proposal['agent_id']}** proposed:\n{proposal['proposals']}"
                for proposal in initial_proposals
            ]
        )

        # Request refinements from each discussing agent
        logger.info(f"Refining topics with {len(self.discussing_agents)} agents...")
        for i, agent in enumerate(self.discussing_agents):
            logger.info(
                f"Refining topics with {agent.agent_id} ({i+1}/{len(self.discussing_agents)})"
            )

            # Find this agent's initial proposal
            agent_initial_proposal = next(
                (p for p in initial_proposals if p["agent_id"] == agent.agent_id),
                {"proposals": "No initial proposal found"},
            )

            prompt = f"""Now that all agents have proposed initial topics for the theme '{theme}', you can see everyone's suggestions below. This is your opportunity to refine your topics based on collective wisdom.

ALL INITIAL PROPOSALS:
{all_proposals_text}

YOUR INITIAL PROPOSAL:
{agent_initial_proposal['proposals']}

COLLABORATIVE REFINEMENT TASK:
Review all proposals with a strategic lens. Your goal is to help design the optimal discussion flow that will lead to the best possible conclusions. Consider:

1. **Strategic Synthesis**: How can topics be combined or refined to create better logical progression?
2. **Gap Analysis**: What essential aspects are missing that need to be addressed?
3. **Flow Optimization**: What order would build knowledge most effectively?
4. **Collaboration Opportunities**: How can your topics complement others' suggestions?

Refine your 3-5 topics considering:
- Merge similar topics from different agents into stronger, more comprehensive ones
- Identify and fill any critical gaps in coverage
- Ensure logical progression from foundational to advanced concepts
- Build upon others' insights while maintaining your unique perspective
- Keep the strategic focus: "What pathway will lead to the most comprehensive conclusion?"

Provide your refined topic proposals, incorporating insights from the collaborative review."""

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
                    refined_proposals.append(
                        {"agent_id": agent.agent_id, "proposals": response_content}
                    )
                    logger.info(f"Collected refined proposals from {agent.agent_id}")

                    # Display the agent's refined topic proposals in the UI
                    provider_type = get_provider_type_from_agent_id(agent.agent_id)
                    display_agent_response(
                        agent_id=agent.agent_id,
                        provider=provider_type,
                        content=response_content,
                        round_number=0,  # Phase 1 - agenda setting
                        topic="Collaborative Topic Refinement",
                        timestamp=datetime.now(),
                    )
            except Exception as e:
                logger.error(
                    f"Failed to get refined proposals from {agent.agent_id}: {e}"
                )
                # Fallback to original proposal
                refined_proposals.append(agent_initial_proposal)

                # Display the failure for transparency
                provider_type = get_provider_type_from_agent_id(agent.agent_id)
                display_agent_response(
                    agent_id=agent.agent_id,
                    provider=provider_type,
                    content=f"❌ Failed to refine topics, using original proposal: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}",
                    round_number=0,  # Phase 1 - agenda setting
                    topic="Collaborative Topic Refinement",
                    timestamp=datetime.now(),
                )

        updates = {
            "proposed_topics": refined_proposals,  # Replace initial with refined proposals
            "initial_proposals": initial_proposals,  # Keep original for reference
            "refinement_completed": True,
        }

        logger.info(f"Completed topic refinement with {len(refined_proposals)} agents")

        return updates

    def validate_preconditions(self, state: VirtualAgoraState) -> bool:
        """Validate that node can execute.

        Args:
            state: Current Virtual Agora state

        Returns:
            True if preconditions are met
        """
        # Check that we have initial proposals
        proposed_topics = state.get("proposed_topics", [])
        if not proposed_topics:
            logger.warning("Cannot refine topics: no proposed_topics in state")
            return False

        # Check that main topic is set
        if not state.get("main_topic"):
            logger.warning("Cannot refine topics: main_topic not set")
            return False

        # Check that we have agents to query
        if not self.discussing_agents:
            logger.warning("Cannot refine topics: no discussing agents available")
            return False

        return True

    def get_node_name(self) -> str:
        """Get human-readable node name."""
        return "TopicRefinement"
