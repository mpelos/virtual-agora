"""Agenda Analytics and Reporting System.

This module provides comprehensive analytics and metrics collection
for agenda management operations.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
import statistics

from virtual_agora.agenda.models import (
    Proposal,
    Vote,
    VoteCollection,
    ProposalCollection,
    AgendaState,
    AgendaAnalytics,
    TopicTransition,
    AgendaModification,
    EdgeCaseEvent,
)
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class AgendaAnalyticsCollector:
    """Collects and analyzes agenda management metrics."""

    def __init__(self, session_id: str):
        """Initialize the analytics collector.

        Args:
            session_id: Session ID for analytics
        """
        self.session_id = session_id
        self.timeline_events = []
        self.proposal_collections = []
        self.vote_collections = []
        self.topic_transitions = []
        self.modifications = []
        self.edge_cases = []

    async def record_agenda_initialization(
        self,
        proposal_collection: ProposalCollection,
        vote_collection: VoteCollection,
        agenda_state: AgendaState,
    ) -> None:
        """Record the agenda initialization process.

        Args:
            proposal_collection: Collected proposals
            vote_collection: Collected votes
            agenda_state: Final agenda state
        """
        self.proposal_collections.append(proposal_collection)
        self.vote_collections.append(vote_collection)

        # Record timeline events
        self.timeline_events.append(
            {
                "timestamp": proposal_collection.start_time,
                "event_type": "proposal_collection_start",
                "details": {
                    "requested_agents": len(proposal_collection.requested_agents),
                    "timeout_seconds": proposal_collection.timeout_seconds,
                },
            }
        )

        self.timeline_events.append(
            {
                "timestamp": proposal_collection.end_time or datetime.now(),
                "event_type": "proposal_collection_end",
                "details": {
                    "proposals_collected": len(proposal_collection.proposals),
                    "responding_agents": len(proposal_collection.responding_agents),
                    "timeout_agents": len(proposal_collection.timeout_agents),
                    "completion_rate": proposal_collection.completion_rate,
                },
            }
        )

        self.timeline_events.append(
            {
                "timestamp": vote_collection.start_time,
                "event_type": "vote_collection_start",
                "details": {
                    "topic_options": len(vote_collection.topic_options),
                    "requested_agents": len(vote_collection.requested_agents),
                },
            }
        )

        self.timeline_events.append(
            {
                "timestamp": vote_collection.end_time or datetime.now(),
                "event_type": "vote_collection_end",
                "details": {
                    "votes_collected": len(vote_collection.votes),
                    "responding_agents": len(vote_collection.responding_agents),
                    "participation_rate": vote_collection.participation_rate,
                },
            }
        )

        self.timeline_events.append(
            {
                "timestamp": agenda_state.created_at,
                "event_type": "agenda_synthesized",
                "details": {
                    "agenda_length": len(agenda_state.current_agenda),
                    "version": agenda_state.version,
                },
            }
        )

        logger.info("Recorded agenda initialization analytics")

    def record_topic_transition(self, transition: TopicTransition) -> None:
        """Record a topic transition.

        Args:
            transition: Topic transition record
        """
        self.topic_transitions.append(transition)

        self.timeline_events.append(
            {
                "timestamp": transition.timestamp,
                "event_type": f"topic_{transition.transition_type}",
                "details": {
                    "from_topic": transition.from_topic,
                    "to_topic": transition.to_topic,
                    "duration_seconds": transition.duration_seconds,
                    "message_count": transition.message_count,
                    "participant_count": transition.participant_count,
                },
            }
        )

        logger.debug(
            f"Recorded topic transition: {transition.from_topic} -> {transition.to_topic}"
        )

    async def record_agenda_modification(
        self,
        modifications: List[AgendaModification],
        vote_collection: VoteCollection,
        agenda_state: AgendaState,
    ) -> None:
        """Record agenda modification process.

        Args:
            modifications: List of modifications
            vote_collection: Votes on modifications
            agenda_state: Updated agenda state
        """
        self.modifications.extend(modifications)
        self.vote_collections.append(vote_collection)

        self.timeline_events.append(
            {
                "timestamp": datetime.now(),
                "event_type": "agenda_modification",
                "details": {
                    "modifications_count": len(modifications),
                    "new_version": agenda_state.version,
                    "modification_types": [
                        mod.modification_type for mod in modifications
                    ],
                },
            }
        )

        logger.info(f"Recorded agenda modification with {len(modifications)} changes")

    def record_edge_case(self, edge_case: EdgeCaseEvent) -> None:
        """Record an edge case event.

        Args:
            edge_case: Edge case event
        """
        self.edge_cases.append(edge_case)

        self.timeline_events.append(
            {
                "timestamp": edge_case.timestamp,
                "event_type": "edge_case",
                "details": {
                    "case_type": edge_case.event_type,
                    "description": edge_case.description,
                    "resolved": edge_case.recovered_successfully,
                    "affected_agents": len(edge_case.affected_agents),
                },
            }
        )

        logger.warning(f"Recorded edge case: {edge_case.event_type}")

    def generate_analytics(
        self, agenda_state: AgendaState, edge_cases: List[EdgeCaseEvent]
    ) -> AgendaAnalytics:
        """Generate comprehensive agenda analytics.

        Args:
            agenda_state: Current agenda state
            edge_cases: List of edge cases encountered

        Returns:
            AgendaAnalytics object
        """
        logger.info("Generating comprehensive agenda analytics")

        analytics = AgendaAnalytics(session_id=self.session_id)

        # Basic proposal metrics
        analytics.total_proposals = sum(
            len(pc.proposals) for pc in self.proposal_collections
        )

        unique_topics = set()
        for pc in self.proposal_collections:
            for proposal in pc.proposals:
                unique_topics.add(proposal.topic.lower().strip())
        analytics.unique_topics_proposed = len(unique_topics)

        # Proposal acceptance rate
        if analytics.total_proposals > 0:
            accepted_topics = len(agenda_state.current_agenda)
            analytics.proposal_acceptance_rate = (
                accepted_topics / analytics.total_proposals
            )

        # Vote participation metrics
        if self.vote_collections:
            participation_rates = [
                vc.participation_rate for vc in self.vote_collections
            ]
            analytics.average_vote_participation = statistics.mean(participation_rates)

        # Modification metrics
        analytics.agenda_modifications_count = len(self.modifications)

        # Topic completion metrics
        analytics.topics_completed = len(agenda_state.completed_topics)
        completed_transitions = [
            t for t in self.topic_transitions if t.transition_type == "complete"
        ]

        if completed_transitions:
            durations = [
                t.duration_seconds for t in completed_transitions if t.duration_seconds
            ]
            if durations:
                analytics.average_topic_duration_minutes = (
                    statistics.mean(durations) / 60
                )

        # Agent participation analysis
        analytics.agent_participation_rates = self._calculate_agent_participation()

        # Topic proposal distribution
        analytics.topic_proposal_distribution = self._calculate_proposal_distribution()

        # Voting patterns
        analytics.voting_patterns = self._analyze_voting_patterns()

        # Modification patterns
        analytics.modification_patterns = self._analyze_modification_patterns()

        # Timeline events
        analytics.timeline_events = sorted(
            self.timeline_events, key=lambda x: x["timestamp"]
        )

        return analytics

    def _calculate_agent_participation(self) -> Dict[str, float]:
        """Calculate participation rates for each agent.

        Returns:
            Dictionary mapping agent_id to participation rate
        """
        agent_participation = defaultdict(list)

        # Analyze proposal participation
        for pc in self.proposal_collections:
            total_requested = len(pc.requested_agents)
            for agent_id in pc.requested_agents:
                participated = agent_id in pc.responding_agents
                agent_participation[agent_id].append(1.0 if participated else 0.0)

        # Analyze vote participation
        for vc in self.vote_collections:
            total_requested = len(vc.requested_agents)
            for agent_id in vc.requested_agents:
                participated = agent_id in vc.responding_agents
                agent_participation[agent_id].append(1.0 if participated else 0.0)

        # Calculate average participation rates
        participation_rates = {}
        for agent_id, participations in agent_participation.items():
            if participations:
                participation_rates[agent_id] = statistics.mean(participations)
            else:
                participation_rates[agent_id] = 0.0

        return participation_rates

    def _calculate_proposal_distribution(self) -> Dict[str, int]:
        """Calculate proposal distribution by agent.

        Returns:
            Dictionary mapping agent_id to proposal count
        """
        distribution = defaultdict(int)

        for pc in self.proposal_collections:
            for proposal in pc.proposals:
                distribution[proposal.agent_id] += 1

        return dict(distribution)

    def _analyze_voting_patterns(self) -> Dict[str, Any]:
        """Analyze voting patterns and preferences.

        Returns:
            Dictionary with voting pattern analysis
        """
        patterns = {
            "total_votes": 0,
            "valid_votes": 0,
            "average_preferences_per_vote": 0.0,
            "most_controversial_topic": None,
            "most_consensus_topic": None,
            "preference_consistency": 0.0,
            "vote_quality_distribution": {},
        }

        all_votes = []
        for vc in self.vote_collections:
            all_votes.extend(vc.votes)

        if not all_votes:
            return patterns

        patterns["total_votes"] = len(all_votes)
        valid_votes = [v for v in all_votes if v.parsed_preferences]
        patterns["valid_votes"] = len(valid_votes)

        if valid_votes:
            # Average preferences per vote
            pref_lengths = [len(v.parsed_preferences) for v in valid_votes]
            patterns["average_preferences_per_vote"] = statistics.mean(pref_lengths)

            # Topic consensus analysis
            topic_positions = defaultdict(list)
            for vote in valid_votes:
                for pos, topic in enumerate(vote.parsed_preferences, 1):
                    topic_positions[topic].append(pos)

            if topic_positions:
                # Most controversial (highest position variance)
                controversies = {}
                consensuses = {}

                for topic, positions in topic_positions.items():
                    if len(positions) > 1:
                        variance = statistics.variance(positions)
                        controversies[topic] = variance
                        consensuses[topic] = statistics.mean(positions)

                if controversies:
                    patterns["most_controversial_topic"] = min(
                        controversies, key=controversies.get
                    )
                    patterns["most_consensus_topic"] = min(
                        consensuses, key=consensuses.get
                    )

            # Vote quality distribution
            quality_ranges = {"high": 0, "medium": 0, "low": 0}
            for vote in valid_votes:
                if vote.confidence_score:
                    if vote.confidence_score >= 0.8:
                        quality_ranges["high"] += 1
                    elif vote.confidence_score >= 0.5:
                        quality_ranges["medium"] += 1
                    else:
                        quality_ranges["low"] += 1

            patterns["vote_quality_distribution"] = quality_ranges

        return patterns

    def _analyze_modification_patterns(self) -> Dict[str, int]:
        """Analyze agenda modification patterns.

        Returns:
            Dictionary with modification pattern counts
        """
        patterns = defaultdict(int)

        for mod in self.modifications:
            patterns[mod.modification_type] += 1
            patterns["total"] += 1

        return dict(patterns)

    def generate_participation_report(self) -> Dict[str, Any]:
        """Generate detailed participation report.

        Returns:
            Participation analysis report
        """
        report = {
            "overview": {},
            "agent_details": {},
            "trends": {},
            "recommendations": [],
        }

        # Overview statistics
        all_agents = set()
        total_opportunities = 0
        total_participations = 0

        for pc in self.proposal_collections:
            all_agents.update(pc.requested_agents)
            total_opportunities += len(pc.requested_agents)
            total_participations += len(pc.responding_agents)

        for vc in self.vote_collections:
            all_agents.update(vc.requested_agents)
            total_opportunities += len(vc.requested_agents)
            total_participations += len(vc.responding_agents)

        report["overview"] = {
            "total_agents": len(all_agents),
            "total_opportunities": total_opportunities,
            "total_participations": total_participations,
            "overall_participation_rate": (
                total_participations / total_opportunities
                if total_opportunities > 0
                else 0
            ),
        }

        # Agent-specific details
        agent_participation = self._calculate_agent_participation()
        for agent_id, rate in agent_participation.items():
            report["agent_details"][agent_id] = {
                "participation_rate": rate,
                "performance_category": self._categorize_participation(rate),
            }

        # Generate recommendations
        low_participation_agents = [
            agent_id for agent_id, rate in agent_participation.items() if rate < 0.7
        ]

        if low_participation_agents:
            report["recommendations"].append(
                f"Consider investigating low participation from agents: {', '.join(low_participation_agents)}"
            )

        if report["overview"]["overall_participation_rate"] < 0.8:
            report["recommendations"].append(
                "Overall participation rate is below 80%. Consider adjusting timeout settings or agent engagement strategies."
            )

        return report

    def _categorize_participation(self, rate: float) -> str:
        """Categorize participation rate.

        Args:
            rate: Participation rate (0.0 to 1.0)

        Returns:
            Category string
        """
        if rate >= 0.9:
            return "excellent"
        elif rate >= 0.7:
            return "good"
        elif rate >= 0.5:
            return "moderate"
        else:
            return "poor"

    def generate_timeline_report(self) -> Dict[str, Any]:
        """Generate timeline analysis report.

        Returns:
            Timeline analysis report
        """
        if not self.timeline_events:
            return {"events": [], "analysis": {}}

        # Sort events by timestamp
        sorted_events = sorted(self.timeline_events, key=lambda x: x["timestamp"])

        # Calculate durations between key events
        durations = {}
        event_pairs = [
            ("proposal_collection_start", "proposal_collection_end"),
            ("vote_collection_start", "vote_collection_end"),
            ("topic_start", "topic_complete"),
        ]

        for start_type, end_type in event_pairs:
            start_events = [e for e in sorted_events if e["event_type"] == start_type]
            end_events = [e for e in sorted_events if e["event_type"] == end_type]

            if start_events and end_events:
                durations_list = []
                for start_event in start_events:
                    for end_event in end_events:
                        if end_event["timestamp"] > start_event["timestamp"]:
                            duration = (
                                end_event["timestamp"] - start_event["timestamp"]
                            ).total_seconds()
                            durations_list.append(duration)
                            break

                if durations_list:
                    durations[f"{start_type}_to_{end_type}"] = {
                        "average_seconds": statistics.mean(durations_list),
                        "count": len(durations_list),
                    }

        return {
            "events": [
                {
                    "timestamp": event["timestamp"].isoformat(),
                    "event_type": event["event_type"],
                    "details": event["details"],
                }
                for event in sorted_events
            ],
            "durations": durations,
            "total_events": len(sorted_events),
        }

    def export_analytics_data(self) -> Dict[str, Any]:
        """Export all analytics data for external analysis.

        Returns:
            Complete analytics data export
        """
        return {
            "session_id": self.session_id,
            "export_timestamp": datetime.now().isoformat(),
            "proposal_collections": [
                {
                    "id": pc.id,
                    "start_time": pc.start_time.isoformat(),
                    "end_time": pc.end_time.isoformat() if pc.end_time else None,
                    "proposals_count": len(pc.proposals),
                    "requested_agents": len(pc.requested_agents),
                    "responding_agents": len(pc.responding_agents),
                    "timeout_agents": len(pc.timeout_agents),
                    "completion_rate": pc.completion_rate,
                }
                for pc in self.proposal_collections
            ],
            "vote_collections": [
                {
                    "id": vc.id,
                    "vote_type": vc.vote_type,
                    "start_time": vc.start_time.isoformat(),
                    "end_time": vc.end_time.isoformat() if vc.end_time else None,
                    "votes_count": len(vc.votes),
                    "requested_agents": len(vc.requested_agents),
                    "responding_agents": len(vc.responding_agents),
                    "participation_rate": vc.participation_rate,
                }
                for vc in self.vote_collections
            ],
            "topic_transitions": [
                {
                    "id": tt.id,
                    "from_topic": tt.from_topic,
                    "to_topic": tt.to_topic,
                    "transition_type": tt.transition_type,
                    "timestamp": tt.timestamp.isoformat(),
                    "duration_seconds": tt.duration_seconds,
                    "message_count": tt.message_count,
                }
                for tt in self.topic_transitions
            ],
            "modifications": [
                {
                    "id": mod.id,
                    "agent_id": mod.agent_id,
                    "modification_type": mod.modification_type,
                    "timestamp": mod.timestamp.isoformat(),
                    "applied": mod.applied,
                }
                for mod in self.modifications
            ],
            "edge_cases": [
                {
                    "id": ec.id,
                    "event_type": ec.event_type,
                    "timestamp": ec.timestamp.isoformat(),
                    "resolved": ec.recovered_successfully,
                    "affected_agents": len(ec.affected_agents),
                }
                for ec in self.edge_cases
            ],
            "timeline_events": self.timeline_events,
        }
