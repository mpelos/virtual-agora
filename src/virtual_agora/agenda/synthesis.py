"""Agenda Synthesis and Ranking Engine.

This module handles the synthesis of proposals and votes into a ranked
agenda using the moderator agent's reasoning capabilities.
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from virtual_agora.agenda.models import (
    Proposal,
    Vote,
    AgendaItem,
    AgendaSynthesisResult,
    AgendaModification,
)
from virtual_agora.agents.moderator import ModeratorAgent
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TopicScore:
    """Score information for a topic during synthesis."""

    topic: str
    raw_score: float
    proposal_count: int
    vote_positions: List[int]  # Positions in votes (1st, 2nd, etc.)
    first_choice_count: int
    total_mentions: int
    proposing_agents: List[str]
    normalized_score: float = 0.0


class AgendaSynthesizer:
    """Synthesizes proposals and votes into a ranked agenda."""

    def __init__(self, moderator: ModeratorAgent, max_retries: int = 3):
        """Initialize the synthesizer.

        Args:
            moderator: ModeratorAgent for synthesis
            max_retries: Maximum synthesis attempts
        """
        self.moderator = moderator
        self.max_retries = max_retries

    def synthesize_agenda(
        self, proposals: List[Proposal], votes: List[Vote]
    ) -> AgendaSynthesisResult:
        """Synthesize proposals and votes into a ranked agenda.

        Args:
            proposals: List of topic proposals
            votes: List of votes from agents

        Returns:
            AgendaSynthesisResult with ranked agenda
        """
        logger.info(
            f"Starting agenda synthesis with {len(proposals)} proposals and {len(votes)} votes"
        )

        # Step 1: Calculate topic scores
        topic_scores = self._calculate_topic_scores(proposals, votes)

        # Step 2: Use moderator for intelligent synthesis
        synthesis_attempts = 0
        last_error = None

        while synthesis_attempts < self.max_retries:
            synthesis_attempts += 1

            try:
                # Prepare synthesis context
                synthesis_context = self._prepare_synthesis_context(
                    proposals, votes, topic_scores
                )

                # Request synthesis from moderator
                # Convert our data to format expected by moderator
                topics = list(topic_scores.keys())
                vote_strings = [
                    f"{vote.agent_id}: {vote.vote_content}" for vote in votes
                ]

                synthesis_response = self.moderator.synthesize_agenda(
                    topics, vote_strings
                )

                # ModeratorAgent.synthesize_agenda returns a List[str] directly
                if isinstance(synthesis_response, list):
                    parsed_agenda = synthesis_response
                else:
                    # Fallback: try to parse as JSON if it's a string
                    parsed_agenda = self._parse_agenda_response(synthesis_response)

                if parsed_agenda:
                    # Apply tie-breaking if needed
                    final_agenda, tie_breaks = self._apply_tie_breaking(
                        parsed_agenda, topic_scores
                    )

                    result = AgendaSynthesisResult(
                        proposed_agenda=final_agenda,
                        synthesis_explanation=self._generate_explanation(
                            final_agenda, topic_scores, tie_breaks
                        ),
                        tie_breaks_applied=tie_breaks,
                        confidence_score=self._calculate_confidence_score(
                            final_agenda, topic_scores
                        ),
                        synthesis_attempts=synthesis_attempts,
                    )

                    logger.info(
                        f"Agenda synthesis completed successfully after {synthesis_attempts} attempts"
                    )
                    return result

            except Exception as e:
                last_error = e
                logger.warning(f"Synthesis attempt {synthesis_attempts} failed: {e}")
                if synthesis_attempts < self.max_retries:
                    logger.info(
                        f"Retrying synthesis (attempt {synthesis_attempts + 1}/{self.max_retries})"
                    )

        # If all attempts failed, use fallback synthesis
        logger.error(
            f"All synthesis attempts failed. Using fallback. Last error: {last_error}"
        )
        return self._fallback_synthesis(proposals, votes, topic_scores)

    def _calculate_topic_scores(
        self, proposals: List[Proposal], votes: List[Vote]
    ) -> Dict[str, TopicScore]:
        """Calculate comprehensive scores for each topic.

        Args:
            proposals: List of proposals
            votes: List of votes

        Returns:
            Dictionary mapping topic to TopicScore
        """
        topic_scores = {}

        # Initialize scores from proposals
        for proposal in proposals:
            topic = self._normalize_topic(proposal.topic)
            if topic not in topic_scores:
                topic_scores[topic] = TopicScore(
                    topic=proposal.topic,  # Keep original formatting
                    raw_score=0.0,
                    proposal_count=0,
                    vote_positions=[],
                    first_choice_count=0,
                    total_mentions=0,
                    proposing_agents=[],
                )

            score_data = topic_scores[topic]
            score_data.proposal_count += 1
            score_data.proposing_agents.append(proposal.agent_id)
            score_data.raw_score += 1.0  # Base score for being proposed

        # Add voting scores
        valid_votes = [v for v in votes if v.parsed_preferences]

        for vote in valid_votes:
            for position, topic_name in enumerate(vote.parsed_preferences, 1):
                normalized_topic = self._normalize_topic(topic_name)

                if normalized_topic in topic_scores:
                    score_data = topic_scores[normalized_topic]
                    score_data.vote_positions.append(position)
                    score_data.total_mentions += 1

                    # Position-based scoring (1st choice = 5 points, 2nd = 4, etc.)
                    position_score = max(1, 6 - position)
                    score_data.raw_score += position_score

                    # Track first choices
                    if position == 1:
                        score_data.first_choice_count += 1

        # Normalize scores
        if topic_scores:
            max_score = max(score.raw_score for score in topic_scores.values())
            for score in topic_scores.values():
                score.normalized_score = (
                    score.raw_score / max_score if max_score > 0 else 0.0
                )

        return topic_scores

    def _prepare_synthesis_context(
        self,
        proposals: List[Proposal],
        votes: List[Vote],
        topic_scores: Dict[str, TopicScore],
    ) -> Dict[str, Any]:
        """Prepare context for moderator synthesis.

        Args:
            proposals: List of proposals
            votes: List of votes
            topic_scores: Calculated topic scores

        Returns:
            Context dictionary for synthesis
        """
        # Summarize proposals
        proposals_summary = []
        for proposal in proposals:
            proposals_summary.append(
                {
                    "topic": proposal.topic,
                    "agent": proposal.agent_id,
                    "description": proposal.description,
                }
            )

        # Summarize votes
        votes_summary = []
        for vote in votes:
            if vote.parsed_preferences:
                votes_summary.append(
                    {
                        "agent": vote.agent_id,
                        "preferences": vote.parsed_preferences,
                        "confidence": vote.confidence_score,
                    }
                )

        # Prepare scoring data
        scoring_data = []
        for topic, score in sorted(
            topic_scores.items(), key=lambda x: x[1].raw_score, reverse=True
        ):
            scoring_data.append(
                {
                    "topic": score.topic,
                    "score": score.raw_score,
                    "proposal_count": score.proposal_count,
                    "first_choice_count": score.first_choice_count,
                    "total_mentions": score.total_mentions,
                    "average_position": (
                        sum(score.vote_positions) / len(score.vote_positions)
                        if score.vote_positions
                        else 0
                    ),
                }
            )

        return {
            "proposals_summary": proposals_summary,
            "votes_summary": votes_summary,
            "scoring_data": scoring_data,
        }

    def _parse_agenda_response(self, response: str) -> Optional[List[str]]:
        """Parse moderator's agenda response to extract JSON.

        Args:
            response: Moderator's response text

        Returns:
            List of topics or None if parsing failed
        """
        try:
            # Look for JSON in the response
            json_match = re.search(
                r'\{[^}]*"proposed_agenda"[^}]*\}', response, re.DOTALL
            )
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                if "proposed_agenda" in data and isinstance(
                    data["proposed_agenda"], list
                ):
                    return data["proposed_agenda"]

            # Try to parse the entire response as JSON
            data = json.loads(response.strip())
            if "proposed_agenda" in data and isinstance(data["proposed_agenda"], list):
                return data["proposed_agenda"]

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response content: {response}")

        return None

    def _apply_tie_breaking(
        self, agenda: List[str], topic_scores: Dict[str, TopicScore]
    ) -> Tuple[List[str], List[str]]:
        """Apply tie-breaking logic to agenda.

        Args:
            agenda: Proposed agenda
            topic_scores: Topic scores

        Returns:
            Tuple of (final_agenda, tie_breaks_applied)
        """
        tie_breaks = []

        # Check for topics with similar scores that might need tie-breaking
        scored_topics = []
        for topic in agenda:
            normalized = self._normalize_topic(topic)
            if normalized in topic_scores:
                scored_topics.append((topic, topic_scores[normalized]))

        # Apply tie-breaking rules
        final_agenda = []
        i = 0

        while i < len(scored_topics):
            current_topic, current_score = scored_topics[i]

            # Look for ties (topics with very similar scores)
            tied_topics = [(current_topic, current_score)]
            j = i + 1

            while j < len(scored_topics):
                next_topic, next_score = scored_topics[j]
                if abs(current_score.raw_score - next_score.raw_score) < 0.1:
                    tied_topics.append((next_topic, next_score))
                    j += 1
                else:
                    break

            if len(tied_topics) > 1:
                # Apply tie-breaking rules
                tied_topics.sort(
                    key=lambda x: (
                        -x[1].first_choice_count,  # More first choices wins
                        -x[1].proposal_count,  # More proposals wins
                        x[1].topic.lower(),  # Alphabetical as final tie-breaker
                    )
                )

                tie_break_info = f"Tie between {len(tied_topics)} topics resolved by first choice count"
                tie_breaks.append(tie_break_info)
                logger.debug(f"Applied tie-breaking: {tie_break_info}")

            # Add resolved topics to final agenda
            for topic, _ in tied_topics:
                final_agenda.append(topic)

            i = j

        return final_agenda, tie_breaks

    def _calculate_confidence_score(
        self, agenda: List[str], topic_scores: Dict[str, TopicScore]
    ) -> float:
        """Calculate confidence score for the synthesized agenda.

        Args:
            agenda: Final agenda
            topic_scores: Topic scores

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not agenda or not topic_scores:
            return 0.0

        total_confidence = 0.0

        for i, topic in enumerate(agenda):
            normalized = self._normalize_topic(topic)
            if normalized in topic_scores:
                score = topic_scores[normalized]

                # Higher confidence for:
                # - Topics with more first choices
                # - Topics with consistent ranking
                # - Topics proposed by multiple agents

                position_confidence = max(
                    0.1, 1.0 - (i * 0.1)
                )  # Decreases with position
                first_choice_confidence = min(1.0, score.first_choice_count * 0.3)
                consistency_confidence = 1.0 if score.vote_positions else 0.0
                proposal_confidence = min(1.0, score.proposal_count * 0.2)

                topic_confidence = (
                    position_confidence * 0.3
                    + first_choice_confidence * 0.3
                    + consistency_confidence * 0.2
                    + proposal_confidence * 0.2
                )

                total_confidence += topic_confidence

        return total_confidence / len(agenda)

    def _generate_explanation(
        self,
        agenda: List[str],
        topic_scores: Dict[str, TopicScore],
        tie_breaks: List[str],
    ) -> str:
        """Generate explanation for the synthesized agenda.

        Args:
            agenda: Final agenda
            topic_scores: Topic scores
            tie_breaks: Applied tie-breaks

        Returns:
            Explanation text
        """
        explanation_parts = []

        explanation_parts.append(
            "Agenda synthesized based on agent proposals and voting preferences:"
        )

        for i, topic in enumerate(agenda, 1):
            normalized = self._normalize_topic(topic)
            if normalized in topic_scores:
                score = topic_scores[normalized]
                explanation_parts.append(
                    f"{i}. {topic} - "
                    f"Score: {score.raw_score:.1f}, "
                    f"First choices: {score.first_choice_count}, "
                    f"Total mentions: {score.total_mentions}"
                )

        if tie_breaks:
            explanation_parts.append("\nTie-breaking applied:")
            for tie_break in tie_breaks:
                explanation_parts.append(f"- {tie_break}")

        return "\n".join(explanation_parts)

    def _fallback_synthesis(
        self,
        proposals: List[Proposal],
        votes: List[Vote],
        topic_scores: Dict[str, TopicScore],
    ) -> AgendaSynthesisResult:
        """Provide fallback synthesis when moderator synthesis fails.

        Args:
            proposals: List of proposals
            votes: List of votes
            topic_scores: Topic scores

        Returns:
            Fallback AgendaSynthesisResult
        """
        logger.info("Using fallback synthesis based on calculated scores")

        # Sort topics by score
        sorted_topics = sorted(
            topic_scores.items(),
            key=lambda x: (x[1].raw_score, x[1].first_choice_count),
            reverse=True,
        )

        # Create agenda from top-scoring topics
        fallback_agenda = [score.topic for _, score in sorted_topics]

        return AgendaSynthesisResult(
            proposed_agenda=fallback_agenda,
            synthesis_explanation="Fallback synthesis based on calculated topic scores",
            tie_breaks_applied=[
                "Fallback synthesis - no moderator tie-breaking applied"
            ],
            confidence_score=0.7,  # Lower confidence for fallback
            synthesis_attempts=self.max_retries,
        )

    def synthesize_modified_agenda(
        self,
        current_agenda: List[AgendaItem],
        modifications: List[AgendaModification],
        votes: List[Vote],
    ) -> AgendaSynthesisResult:
        """Synthesize a modified agenda incorporating changes.

        Args:
            current_agenda: Current agenda items
            modifications: Proposed modifications
            votes: Votes on modified topics

        Returns:
            AgendaSynthesisResult with modified agenda
        """
        logger.info(
            f"Synthesizing modified agenda with {len(modifications)} modifications"
        )

        # Create pseudo-proposals from current agenda and modifications
        pseudo_proposals = []

        # Add current topics
        for item in current_agenda:
            if item.topic not in [
                item.topic for item in current_agenda if item.status == "completed"
            ]:
                proposal = Proposal(
                    agent_id="current_agenda",
                    topic=item.topic,
                    description=item.description,
                )
                pseudo_proposals.append(proposal)

        # Add modification proposals
        for mod in modifications:
            if mod.modification_type == "add" and mod.new_topic:
                proposal = Proposal(
                    agent_id=mod.agent_id,
                    topic=mod.new_topic,
                    description=mod.new_description,
                )
                pseudo_proposals.append(proposal)

        # Use regular synthesis with the combined proposals
        return self.synthesize_agenda(pseudo_proposals, votes)

    def _normalize_topic(self, topic: str) -> str:
        """Normalize topic string for comparison.

        Args:
            topic: Topic to normalize

        Returns:
            Normalized topic string
        """
        return topic.lower().strip().replace("  ", " ")

    def get_synthesis_metrics(self, result: AgendaSynthesisResult) -> Dict[str, Any]:
        """Get metrics about the synthesis process.

        Args:
            result: Synthesis result

        Returns:
            Dictionary with synthesis metrics
        """
        return {
            "agenda_length": len(result.proposed_agenda),
            "synthesis_attempts": result.synthesis_attempts,
            "confidence_score": result.confidence_score,
            "tie_breaks_count": len(result.tie_breaks_applied),
            "synthesis_timestamp": result.timestamp.isoformat(),
            "has_explanation": bool(result.synthesis_explanation),
        }
