"""Main Agenda Management System for Virtual Agora.

This module provides the core AgendaManager class that orchestrates
the entire agenda lifecycle from proposal collection through synthesis,
voting, modifications, and analytics.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from virtual_agora.agenda.models import (
    Proposal,
    Vote,
    VoteType,
    AgendaItem,
    AgendaState,
    ProposalCollection,
    VoteCollection,
    AgendaModification,
    TopicTransition,
    AgendaAnalytics,
    AgendaSynthesisResult,
    EdgeCaseEvent,
    ProposalStatus,
    VoteStatus,
    AgendaStatus,
)
from virtual_agora.agenda.voting import VotingOrchestrator
from virtual_agora.agenda.synthesis import AgendaSynthesizer
from virtual_agora.agenda.analytics import AgendaAnalyticsCollector
from virtual_agora.agents.discussion_agent import DiscussionAgent
from virtual_agora.agents.moderator import ModeratorAgent
from virtual_agora.utils.logging import get_logger
from virtual_agora.utils.exceptions import StateError, AgentError
from virtual_agora.state.schema import VirtualAgoraState

logger = get_logger(__name__)


class ProposalCollector:
    """Handles collection of topic proposals from agents."""

    def __init__(self, moderator: ModeratorAgent, timeout_seconds: int = 300):
        self.moderator = moderator
        self.timeout_seconds = timeout_seconds

    async def collect_proposals(
        self,
        agents: Dict[str, DiscussionAgent],
        main_topic: str,
        session_id: str,
        proposals_per_agent: int = 5,
    ) -> ProposalCollection:
        """Collect proposals from all agents in parallel.

        Args:
            agents: Dictionary of agent_id -> DiscussionAgent
            main_topic: The main discussion topic
            session_id: Current session ID
            proposals_per_agent: Number of proposals to request per agent

        Returns:
            ProposalCollection with all collected proposals
        """
        logger.info(f"Starting proposal collection for {len(agents)} agents")

        collection = ProposalCollection(
            session_id=session_id,
            requested_agents=list(agents.keys()),
            timeout_seconds=self.timeout_seconds,
            status=ProposalStatus.PENDING,
        )

        try:
            # Use ThreadPoolExecutor for parallel proposal collection
            with ThreadPoolExecutor(max_workers=len(agents)) as executor:
                # Submit all proposal requests
                future_to_agent = {
                    executor.submit(
                        self._collect_single_agent_proposals,
                        agent_id,
                        agent,
                        main_topic,
                        proposals_per_agent,
                    ): agent_id
                    for agent_id, agent in agents.items()
                }

                # Wait for completion with timeout
                completed_futures = []
                try:
                    for future in as_completed(
                        future_to_agent, timeout=self.timeout_seconds
                    ):
                        completed_futures.append(future)
                        agent_id = future_to_agent[future]

                        try:
                            proposals = future.result()
                            collection.proposals.extend(proposals)
                            collection.responding_agents.append(agent_id)
                            logger.debug(
                                f"Collected {len(proposals)} proposals from {agent_id}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Error collecting proposals from {agent_id}: {e}"
                            )
                            # Agent will be marked as timeout if not in responding_agents

                except asyncio.TimeoutError:
                    logger.warning(
                        f"Proposal collection timeout after {self.timeout_seconds}s"
                    )

                # Cancel remaining futures and identify timeout agents
                for future in future_to_agent:
                    if future not in completed_futures:
                        future.cancel()
                        agent_id = future_to_agent[future]
                        collection.timeout_agents.append(agent_id)

            # Remove duplicates while preserving attribution
            collection.proposals = self._deduplicate_proposals(collection.proposals)
            collection.end_time = datetime.now()
            collection.status = ProposalStatus.COLLECTED

            logger.info(
                f"Proposal collection completed: {len(collection.proposals)} unique proposals, "
                f"{len(collection.responding_agents)}/{len(collection.requested_agents)} agents responded"
            )

            return collection

        except Exception as e:
            collection.status = ProposalStatus.ERROR
            collection.error_details = str(e)
            collection.end_time = datetime.now()
            logger.error(f"Error in proposal collection: {e}")
            raise

    def _collect_single_agent_proposals(
        self, agent_id: str, agent: DiscussionAgent, main_topic: str, count: int
    ) -> List[Proposal]:
        """Collect proposals from a single agent.

        Args:
            agent_id: ID of the agent
            agent: DiscussionAgent instance
            main_topic: Main discussion topic
            count: Number of proposals to request

        Returns:
            List of Proposal objects
        """
        try:
            # Use the existing propose_topics method
            topic_strings = agent.propose_topics(main_topic)

            # Convert to Proposal objects
            proposals = []
            for topic in topic_strings[:count]:  # Limit to requested count
                proposal = Proposal(
                    agent_id=agent_id, topic=topic, status=ProposalStatus.COLLECTED
                )
                proposals.append(proposal)

            return proposals

        except Exception as e:
            logger.error(f"Error getting proposals from agent {agent_id}: {e}")
            return []

    def _deduplicate_proposals(self, proposals: List[Proposal]) -> List[Proposal]:
        """Remove duplicate proposals while preserving attribution.

        Args:
            proposals: List of proposals to deduplicate

        Returns:
            List of unique proposals with merged attribution
        """
        # Group proposals by normalized topic
        topic_groups: Dict[str, List[Proposal]] = {}

        for proposal in proposals:
            normalized_topic = self._normalize_topic(proposal.topic)
            if normalized_topic not in topic_groups:
                topic_groups[normalized_topic] = []
            topic_groups[normalized_topic].append(proposal)

        # Keep one proposal per group, merging attribution
        unique_proposals = []
        for topic, group in topic_groups.items():
            # Use the first proposal as the base
            base_proposal = group[0]

            # If there are duplicates, add attribution info
            if len(group) > 1:
                base_proposal.metadata["duplicate_count"] = len(group)
                base_proposal.metadata["proposed_by_agents"] = [
                    p.agent_id for p in group
                ]
                logger.debug(
                    f"Merged {len(group)} duplicate proposals for topic: {topic}"
                )

            unique_proposals.append(base_proposal)

        return unique_proposals

    def _normalize_topic(self, topic: str) -> str:
        """Normalize topic string for comparison.

        Args:
            topic: Topic string to normalize

        Returns:
            Normalized topic string
        """
        return topic.lower().strip().replace("  ", " ")


class AgendaManager:
    """Main class for managing the entire agenda lifecycle."""

    def __init__(self, moderator: ModeratorAgent, session_id: str):
        """Initialize the agenda manager.

        Args:
            moderator: ModeratorAgent instance for synthesis
            session_id: Current session ID
        """
        self.moderator = moderator
        self.session_id = session_id

        # Initialize components
        self.proposal_collector = ProposalCollector(moderator)
        self.voting_orchestrator = VotingOrchestrator(moderator)
        self.synthesizer = AgendaSynthesizer(moderator)
        self.analytics = AgendaAnalyticsCollector(session_id)

        # Current state
        self.state = AgendaState(session_id=session_id)
        self.edge_cases: List[EdgeCaseEvent] = []

        logger.info(f"AgendaManager initialized for session {session_id}")

    async def initialize_agenda(
        self,
        agents: Dict[str, DiscussionAgent],
        main_topic: str,
        proposals_per_agent: int = 5,
    ) -> AgendaState:
        """Initialize the agenda through the full democratic process.

        Args:
            agents: Dictionary of agent_id -> DiscussionAgent
            main_topic: Main discussion topic
            proposals_per_agent: Number of proposals per agent

        Returns:
            Updated AgendaState
        """
        logger.info("Starting agenda initialization process")
        self.state.status = AgendaStatus.COLLECTING_PROPOSALS

        try:
            # Step 1: Collect proposals
            proposal_collection = await self.proposal_collector.collect_proposals(
                agents, main_topic, self.session_id, proposals_per_agent
            )
            self.state.proposal_collections.append(proposal_collection)

            # Handle edge case: no proposals
            if not proposal_collection.proposals:
                return await self._handle_empty_proposals(agents, main_topic)

            # Step 2: Collect votes
            self.state.status = AgendaStatus.COLLECTING_VOTES
            vote_collection = await self.voting_orchestrator.collect_votes(
                agents,
                [p.topic for p in proposal_collection.proposals],
                VoteType.INITIAL_AGENDA,
            )
            self.state.vote_collections.append(vote_collection)

            # Handle edge case: no votes
            if not vote_collection.votes:
                return await self._handle_no_votes(proposal_collection.proposals)

            # Step 3: Synthesize agenda
            self.state.status = AgendaStatus.SYNTHESIZING
            synthesis_result = self.synthesizer.synthesize_agenda(
                proposal_collection.proposals, vote_collection.votes
            )

            # Check if synthesis used fallback (indicating failures)
            if synthesis_result.synthesis_attempts >= self.synthesizer.max_retries:
                edge_case = EdgeCaseEvent(
                    session_id=self.session_id,
                    event_type="initialization_error",
                    description=f"Synthesis failed after {synthesis_result.synthesis_attempts} attempts, used fallback",
                    resolution_strategy="fallback_synthesis",
                    system_response="Used fallback synthesis with proposal ranking",
                    recovered_successfully=True,
                )
                self.edge_cases.append(edge_case)

            # Step 4: Create agenda items
            agenda_items = []
            for rank, topic in enumerate(synthesis_result.proposed_agenda, 1):
                # Find original proposal(s) for this topic
                original_proposals = [
                    p
                    for p in proposal_collection.proposals
                    if self.proposal_collector._normalize_topic(p.topic)
                    == self.proposal_collector._normalize_topic(topic)
                ]

                proposed_by = [p.agent_id for p in original_proposals]
                description = (
                    original_proposals[0].description if original_proposals else None
                )

                agenda_item = AgendaItem(
                    topic=topic,
                    description=description,
                    rank=rank,
                    proposed_by=proposed_by,
                    vote_score=self._calculate_vote_score(topic, vote_collection.votes),
                )
                agenda_items.append(agenda_item)

            self.state.current_agenda = agenda_items
            self.state.status = AgendaStatus.COMPLETED
            self.state.updated_at = datetime.now()

            # Update analytics
            await self.analytics.record_agenda_initialization(
                proposal_collection, vote_collection, self.state
            )

            logger.info(
                f"Agenda initialization completed with {len(agenda_items)} topics"
            )
            return self.state

        except Exception as e:
            self.state.status = AgendaStatus.ERROR
            logger.error(f"Error in agenda initialization: {e}")

            # Record edge case
            edge_case = EdgeCaseEvent(
                session_id=self.session_id,
                event_type="initialization_error",
                description=f"Failed to initialize agenda: {str(e)}",
                resolution_strategy="fallback_to_manual_agenda",
                system_response="Creating fallback agenda from proposals",
                recovered_successfully=False,
            )
            self.edge_cases.append(edge_case)

            raise

    def _calculate_vote_score(self, topic: str, votes: List[Vote]) -> float:
        """Calculate a voting score for a topic.

        Args:
            topic: Topic to calculate score for
            votes: List of all votes

        Returns:
            Calculated vote score
        """
        score = 0.0
        normalized_topic = self.proposal_collector._normalize_topic(topic)

        for vote in votes:
            if vote.parsed_preferences:
                # Find topic position in preferences (lower index = higher score)
                for i, preference in enumerate(vote.parsed_preferences):
                    if (
                        self.proposal_collector._normalize_topic(preference)
                        == normalized_topic
                    ):
                        # Score decreases with position (1st choice = 1.0, 2nd = 0.8, etc.)
                        position_score = max(0.1, 1.0 - (i * 0.2))
                        score += position_score
                        break

        return score

    async def _handle_empty_proposals(
        self, agents: Dict[str, DiscussionAgent], main_topic: str
    ) -> AgendaState:
        """Handle the edge case of no proposals being collected.

        Args:
            agents: Available agents
            main_topic: Main topic

        Returns:
            Updated state with fallback agenda
        """
        logger.warning("No proposals collected, creating fallback agenda")

        edge_case = EdgeCaseEvent(
            session_id=self.session_id,
            event_type="empty_proposals",
            description="No proposals were collected from any agent",
            resolution_strategy="create_fallback_agenda",
            system_response="Creating agenda based on main topic breakdown",
            affected_agents=list(agents.keys()),
            recovered_successfully=True,
        )
        self.edge_cases.append(edge_case)

        # Create fallback agenda based on main topic
        fallback_topics = [
            f"Introduction to {main_topic}",
            f"Key aspects of {main_topic}",
            f"Challenges in {main_topic}",
            f"Future of {main_topic}",
            f"Conclusion on {main_topic}",
        ]

        agenda_items = []
        for rank, topic in enumerate(fallback_topics, 1):
            agenda_item = AgendaItem(
                topic=topic,
                description=f"Generated fallback topic {rank}",
                rank=rank,
                proposed_by=["system"],
                vote_score=0.0,
            )
            agenda_items.append(agenda_item)

        self.state.current_agenda = agenda_items
        self.state.status = AgendaStatus.COMPLETED
        self.state.updated_at = datetime.now()

        return self.state

    async def _handle_no_votes(self, proposals: List[Proposal]) -> AgendaState:
        """Handle the edge case of no votes being collected.

        Args:
            proposals: Available proposals

        Returns:
            Updated state with fallback ranking
        """
        logger.warning("No votes collected, using proposal order for agenda")

        edge_case = EdgeCaseEvent(
            session_id=self.session_id,
            event_type="no_votes",
            description="No votes were collected from any agent",
            resolution_strategy="use_proposal_order",
            system_response="Ranking topics by proposal order",
            recovered_successfully=True,
        )
        self.edge_cases.append(edge_case)

        # Create agenda from proposals in order received
        agenda_items = []
        for rank, proposal in enumerate(proposals, 1):
            agenda_item = AgendaItem(
                topic=proposal.topic,
                description=proposal.description,
                rank=rank,
                proposed_by=[proposal.agent_id],
                vote_score=0.0,
            )
            agenda_items.append(agenda_item)

        self.state.current_agenda = agenda_items
        self.state.status = AgendaStatus.COMPLETED
        self.state.updated_at = datetime.now()

        return self.state

    async def modify_agenda(
        self, agents: Dict[str, DiscussionAgent], remaining_topics: List[str]
    ) -> AgendaState:
        """Handle agenda modification between topics.

        Args:
            agents: Available agents
            remaining_topics: Topics still to be discussed

        Returns:
            Updated agenda state
        """
        logger.info("Starting agenda modification process")

        try:
            # Request modification suggestions from all agents
            modifications = await self._collect_modification_suggestions(
                agents, remaining_topics
            )

            if not modifications:
                logger.info("No agenda modifications suggested")
                return self.state

            # If there are modifications, collect votes on them
            vote_collection = await self.voting_orchestrator.collect_votes(
                agents,
                remaining_topics
                + [mod.new_topic for mod in modifications if mod.new_topic],
                VoteType.AGENDA_MODIFICATION,
            )

            # Re-synthesize agenda with modifications
            synthesis_result = self.synthesizer.synthesize_modified_agenda(
                self.state.current_agenda, modifications, vote_collection.votes
            )

            # Update agenda
            self._apply_agenda_modifications(synthesis_result.proposed_agenda)
            self.state.version += 1
            self.state.last_modification = datetime.now()
            self.state.updated_at = datetime.now()

            # Record analytics
            await self.analytics.record_agenda_modification(
                modifications, vote_collection, self.state
            )

            logger.info(f"Agenda modified, new version: {self.state.version}")
            return self.state

        except Exception as e:
            logger.error(f"Error in agenda modification: {e}")

            edge_case = EdgeCaseEvent(
                session_id=self.session_id,
                event_type="modification_error",
                description=f"Failed to modify agenda: {str(e)}",
                resolution_strategy="continue_with_current_agenda",
                system_response="Continuing with existing agenda",
                recovered_successfully=True,
            )
            self.edge_cases.append(edge_case)

            return self.state

    async def _collect_modification_suggestions(
        self, agents: Dict[str, DiscussionAgent], remaining_topics: List[str]
    ) -> List[AgendaModification]:
        """Collect modification suggestions from agents.

        Args:
            agents: Available agents
            remaining_topics: Topics remaining to discuss

        Returns:
            List of AgendaModification objects
        """
        modifications = []

        # Use the moderator to request modifications
        modification_text = await self.moderator.request_agenda_modification(
            remaining_topics, list(agents.keys())
        )

        # Parse modification suggestions (simplified implementation)
        # In practice, this would use more sophisticated NLP
        for agent_id in agents.keys():
            if f"add" in modification_text.lower():
                # Extract new topic suggestions (simplified)
                # Look for "emerging technologies" or "technology" in the suggestion
                if (
                    "emerging technologies" in modification_text.lower()
                    or "technology" in modification_text.lower()
                ):
                    new_topic = "New Technology Topic"
                else:
                    new_topic = "Additional Discussion Topic"

                mod = AgendaModification(
                    agent_id=agent_id,
                    modification_type="add",
                    new_topic=new_topic,
                    justification="Suggested by agent during modification phase",
                )
                modifications.append(mod)

        return modifications

    def _apply_agenda_modifications(self, new_agenda: List[str]) -> None:
        """Apply agenda modifications to current state.

        Args:
            new_agenda: New ordered list of topics
        """
        # Create new agenda items
        new_items = []
        for rank, topic in enumerate(new_agenda, 1):
            # Find existing item if it exists
            existing_item = None
            for item in self.state.current_agenda:
                if item.topic == topic:
                    existing_item = item
                    break

            if existing_item:
                existing_item.rank = rank
                new_items.append(existing_item)
            else:
                # New topic
                new_item = AgendaItem(
                    topic=topic, rank=rank, proposed_by=["modification"]
                )
                new_items.append(new_item)

        self.state.current_agenda = new_items

    def transition_to_topic(self, topic: str, agent_ids: List[str]) -> TopicTransition:
        """Handle transition to a new topic.

        Args:
            topic: Topic to transition to
            agent_ids: List of agent IDs to reset

        Returns:
            TopicTransition record
        """
        logger.info(f"Transitioning to topic: {topic}")

        # Find the agenda item
        agenda_item = None
        for item in self.state.current_agenda:
            if item.topic == topic:
                agenda_item = item
                break

        if not agenda_item:
            raise ValueError(f"Topic not found in agenda: {topic}")

        # Record transition
        transition = TopicTransition(
            session_id=self.session_id,
            from_topic=self.state.active_topic,
            to_topic=topic,
            transition_type="start",
            agent_states_reset=agent_ids,
        )

        # Update state
        previous_topic = self.state.active_topic
        self.state.active_topic = topic
        agenda_item.status = "active"
        agenda_item.discussion_started = datetime.now()

        # Mark previous topic as completed if it exists
        if previous_topic:
            for item in self.state.current_agenda:
                if item.topic == previous_topic:
                    item.status = "completed"
                    item.discussion_ended = datetime.now()
                    if previous_topic not in self.state.completed_topics:
                        self.state.completed_topics.append(previous_topic)
                    break

        self.state.updated_at = datetime.now()

        # Record analytics
        self.analytics.record_topic_transition(transition)

        return transition

    def get_current_state(self) -> AgendaState:
        """Get the current agenda state.

        Returns:
            Current AgendaState
        """
        return self.state

    def get_analytics(self) -> AgendaAnalytics:
        """Get agenda analytics.

        Returns:
            AgendaAnalytics object
        """
        return self.analytics.generate_analytics(self.state, self.edge_cases)

    def is_agenda_complete(self) -> bool:
        """Check if all topics in the agenda have been completed.

        Returns:
            True if agenda is complete, False otherwise
        """
        total_topics = len(self.state.current_agenda)
        completed_topics = len(self.state.completed_topics)
        return completed_topics >= total_topics
