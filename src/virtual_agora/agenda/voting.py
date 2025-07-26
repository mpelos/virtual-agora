"""Voting Orchestration Engine for Agenda Management.

This module handles the collection and processing of votes from agents
for agenda setting and modification decisions.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

from virtual_agora.agenda.models import Vote, VoteType, VoteCollection, VoteStatus
from virtual_agora.agents.discussion_agent import DiscussionAgent
from virtual_agora.agents.moderator import ModeratorAgent
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class VoteParser:
    """Parses natural language votes into structured preferences."""

    def __init__(self):
        # Common ranking patterns
        self.ranking_patterns = [
            r"(\d+)\.?\s*(.+)",  # "1. Topic A"
            r"first[:\s]+(.+)",  # "First: Topic A"
            r"second[:\s]+(.+)",  # "Second: Topic B"
            r"third[:\s]+(.+)",  # "Third: Topic C"
            r"priority[:\s]+(.+)",  # "Priority: Topic A"
            r"prefer[:\s]+(.+)",  # "Prefer: Topic A"
        ]

        # Order words to numbers
        self.order_words = {
            "first": 1,
            "second": 2,
            "third": 3,
            "fourth": 4,
            "fifth": 5,
            "sixth": 6,
            "seventh": 7,
            "eighth": 8,
            "ninth": 9,
            "tenth": 10,
        }

    def parse_vote(self, vote_content: str, available_topics: List[str]) -> List[str]:
        """Parse natural language vote into ordered topic preferences.

        Args:
            vote_content: Natural language vote text
            available_topics: List of available topics to choose from

        Returns:
            List of topics in preference order
        """
        preferences = []
        vote_lower = vote_content.lower()

        # Try to find explicit rankings
        ranked_topics = self._extract_ranked_topics(vote_content, available_topics)
        if ranked_topics:
            return ranked_topics

        # Try to find topics mentioned in order
        mentioned_topics = self._extract_mentioned_topics(
            vote_content, available_topics
        )
        if mentioned_topics:
            return mentioned_topics

        # Fallback: score topics by mention frequency and context
        scored_topics = self._score_topics_by_context(vote_content, available_topics)
        return scored_topics

    def _extract_ranked_topics(
        self, vote_content: str, available_topics: List[str]
    ) -> List[str]:
        """Extract explicitly ranked topics from vote content.

        Args:
            vote_content: Vote text
            available_topics: Available topics

        Returns:
            List of topics in ranked order
        """
        ranked_items = []

        # Look for numbered lists or explicit rankings
        for pattern in self.ranking_patterns:
            matches = re.findall(pattern, vote_content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    rank_indicator, topic_text = match
                    # Try to convert rank to number
                    try:
                        rank = int(rank_indicator)
                    except ValueError:
                        rank = self.order_words.get(rank_indicator.lower(), 999)
                else:
                    rank = 1
                    topic_text = match

                # Check if the topic_text contains multiple topics (compound text)
                # If it contains words like "then", "and", "," it likely has multiple topics
                if self._contains_multiple_topics(topic_text, available_topics):
                    # Skip compound text - let _extract_mentioned_topics handle it
                    continue

                # Find matching topic
                matched_topic = self._find_best_topic_match(
                    topic_text, available_topics
                )
                if matched_topic:
                    ranked_items.append((rank, matched_topic))

        # Sort by rank and return topics
        ranked_items.sort(key=lambda x: x[0])
        return [topic for rank, topic in ranked_items]

    def _contains_multiple_topics(self, text: str, available_topics: List[str]) -> bool:
        """Check if text contains multiple topics.

        Args:
            text: Text to check
            available_topics: Available topics

        Returns:
            True if text likely contains multiple topics
        """
        text_lower = text.lower()

        # Check for common separators/connectors
        separators = ["then", "and", ",", "after", "followed by", "next"]
        has_separators = any(sep in text_lower for sep in separators)

        if not has_separators:
            return False

        # Count how many topics are mentioned
        topic_count = 0
        for topic in available_topics:
            if topic.lower() in text_lower:
                topic_count += 1

        return topic_count > 1

    def _extract_mentioned_topics(
        self, vote_content: str, available_topics: List[str]
    ) -> List[str]:
        """Extract topics in order of mention.

        Args:
            vote_content: Vote text
            available_topics: Available topics

        Returns:
            List of topics in order mentioned
        """
        mentioned = []
        vote_lower = vote_content.lower()

        # Find topics in order of appearance
        topic_positions = []
        for topic in available_topics:
            pos = vote_lower.find(topic.lower())
            if pos != -1:
                topic_positions.append((pos, topic))

        # Sort by position and return topics
        topic_positions.sort(key=lambda x: x[0])
        return [topic for pos, topic in topic_positions]

    def _score_topics_by_context(
        self, vote_content: str, available_topics: List[str]
    ) -> List[str]:
        """Score topics by contextual indicators in vote content.

        Args:
            vote_content: Vote text
            available_topics: Available topics

        Returns:
            List of topics sorted by score
        """
        topic_scores = {}
        vote_lower = vote_content.lower()

        # Positive indicators
        positive_words = [
            "prefer",
            "like",
            "important",
            "priority",
            "first",
            "main",
            "key",
        ]
        # Negative indicators
        negative_words = ["avoid", "skip", "later", "last", "less", "minor"]

        for topic in available_topics:
            score = 0
            topic_lower = topic.lower()

            # Base score for mention
            if topic_lower in vote_lower:
                score += 1

                # Context scoring
                topic_start = vote_lower.find(topic_lower)
                context_before = vote_lower[max(0, topic_start - 50) : topic_start]
                context_after = vote_lower[
                    topic_start + len(topic_lower) : topic_start + len(topic_lower) + 50
                ]

                # Check for positive context
                for word in positive_words:
                    if word in context_before or word in context_after:
                        score += 2

                # Check for negative context
                for word in negative_words:
                    if word in context_before or word in context_after:
                        score -= 1

            topic_scores[topic] = score

        # Sort by score (descending) and return topics with score > 0
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, score in sorted_topics if score > 0]

    def _find_best_topic_match(
        self, text: str, available_topics: List[str]
    ) -> Optional[str]:
        """Find the best matching topic for given text.

        Args:
            text: Text to match
            available_topics: Available topics

        Returns:
            Best matching topic or None
        """
        text_lower = text.lower().strip()

        # Exact match first
        for topic in available_topics:
            if topic.lower() == text_lower:
                return topic

        # Partial match - prioritize topics that appear earlier in the text
        best_match = None
        earliest_position = len(text_lower)

        for topic in available_topics:
            topic_lower = topic.lower()
            if topic_lower in text_lower:
                position = text_lower.find(topic_lower)
                if position < earliest_position:
                    earliest_position = position
                    best_match = topic
            elif text_lower in topic_lower:
                # Text is contained in topic (less likely but possible)
                if 0 < earliest_position:  # Only use if no direct match found
                    earliest_position = 0
                    best_match = topic

        if best_match:
            return best_match

        # Fuzzy match (simplified - could use more sophisticated matching)
        words = text_lower.split()
        for topic in available_topics:
            topic_words = topic.lower().split()
            common_words = set(words) & set(topic_words)
            if len(common_words) > 0:
                return topic

        return None


class VotingOrchestrator:
    """Orchestrates voting rounds for agenda decisions."""

    def __init__(self, moderator: ModeratorAgent, timeout_seconds: int = 180):
        """Initialize the voting orchestrator.

        Args:
            moderator: ModeratorAgent for vote collection
            timeout_seconds: Timeout for vote collection
        """
        self.moderator = moderator
        self.timeout_seconds = timeout_seconds
        self.vote_parser = VoteParser()

    async def collect_votes(
        self,
        agents: Dict[str, DiscussionAgent],
        topic_options: List[str],
        vote_type: VoteType,
        session_id: Optional[str] = None,
    ) -> VoteCollection:
        """Collect votes from all agents in parallel.

        Args:
            agents: Dictionary of agent_id -> DiscussionAgent
            topic_options: List of topics to vote on
            vote_type: Type of vote being collected
            session_id: Optional session ID

        Returns:
            VoteCollection with all collected votes
        """
        logger.info(
            f"Starting vote collection for {len(agents)} agents on {len(topic_options)} topics"
        )

        collection = VoteCollection(
            session_id=session_id or "unknown",
            vote_type=vote_type,
            topic_options=topic_options,
            requested_agents=list(agents.keys()),
            timeout_seconds=self.timeout_seconds,
            status=VoteStatus.PENDING,
        )

        try:
            # Use ThreadPoolExecutor for parallel vote collection
            with ThreadPoolExecutor(max_workers=len(agents)) as executor:
                # Submit all vote requests
                future_to_agent = {
                    executor.submit(
                        self._collect_single_agent_vote,
                        agent_id,
                        agent,
                        topic_options,
                        vote_type,
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
                            vote = future.result()
                            if vote:
                                collection.votes.append(vote)
                                collection.responding_agents.append(agent_id)
                                logger.debug(f"Collected vote from {agent_id}")
                            else:
                                logger.warning(f"Empty vote from {agent_id}")
                        except Exception as e:
                            logger.error(f"Error collecting vote from {agent_id}: {e}")

                except asyncio.TimeoutError:
                    logger.warning(
                        f"Vote collection timeout after {self.timeout_seconds}s"
                    )

                # Cancel remaining futures and identify timeout agents
                for future in future_to_agent:
                    if future not in completed_futures:
                        future.cancel()
                        agent_id = future_to_agent[future]
                        collection.timeout_agents.append(agent_id)

            # Validate and parse votes
            collection.votes = self._validate_and_parse_votes(
                collection.votes, topic_options
            )
            collection.end_time = datetime.now()
            collection.status = VoteStatus.SUBMITTED

            logger.info(
                f"Vote collection completed: {len(collection.votes)} valid votes, "
                f"{len(collection.responding_agents)}/{len(collection.requested_agents)} agents responded"
            )

            return collection

        except Exception as e:
            collection.status = VoteStatus.INVALID
            collection.error_details = str(e)
            collection.end_time = datetime.now()
            logger.error(f"Error in vote collection: {e}")
            raise

    def _collect_single_agent_vote(
        self,
        agent_id: str,
        agent: DiscussionAgent,
        topic_options: List[str],
        vote_type: VoteType,
    ) -> Optional[Vote]:
        """Collect a vote from a single agent.

        Args:
            agent_id: ID of the agent
            agent: DiscussionAgent instance
            topic_options: Available topics to vote on
            vote_type: Type of vote

        Returns:
            Vote object or None if failed
        """
        try:
            # Request vote from agent using moderator
            vote_prompt = self._create_vote_prompt(topic_options, vote_type)

            # Use agent's voting method if available, otherwise use general response
            if hasattr(agent, "vote_on_agenda"):
                vote_content = agent.vote_on_agenda(topic_options)
            elif hasattr(agent, "cast_vote"):
                vote_content = agent.cast_vote(topic_options, vote_type.value)
            else:
                # Fallback to generating response with voting prompt
                vote_content = agent.generate_response(vote_prompt)

            if not vote_content:
                logger.warning(f"Empty vote content from agent {agent_id}")
                return None

            vote = Vote(
                agent_id=agent_id,
                vote_type=vote_type,
                vote_content=vote_content,
                status=VoteStatus.SUBMITTED,
            )

            return vote

        except Exception as e:
            logger.error(f"Error getting vote from agent {agent_id}: {e}")
            return None

    def _create_vote_prompt(self, topic_options: List[str], vote_type: VoteType) -> str:
        """Create a voting prompt for agents.

        Args:
            topic_options: Available topics
            vote_type: Type of vote

        Returns:
            Voting prompt string
        """
        if vote_type == VoteType.INITIAL_AGENDA:
            prompt = (
                "Please rank the following topics in order of your preference for discussion. "
                "You can provide your ranking in any clear format (numbered list, explicit preferences, etc.).\n\n"
                "Available topics:\n"
            )
        elif vote_type == VoteType.AGENDA_MODIFICATION:
            prompt = (
                "Please indicate your preferences for the modified agenda topics. "
                "Rank them in order of importance for continued discussion.\n\n"
                "Topics for consideration:\n"
            )
        else:
            prompt = (
                "Please vote on the following topics. "
                "Indicate your preferences in order.\n\n"
                "Topics:\n"
            )

        for i, topic in enumerate(topic_options, 1):
            prompt += f"{i}. {topic}\n"

        prompt += "\nProvide your ranking or preferences:"

        return prompt

    def _validate_and_parse_votes(
        self, votes: List[Vote], topic_options: List[str]
    ) -> List[Vote]:
        """Validate votes and parse preferences.

        Args:
            votes: List of votes to validate
            topic_options: Available topic options

        Returns:
            List of validated votes with parsed preferences
        """
        valid_votes = []

        for vote in votes:
            try:
                # Parse vote content into preferences
                preferences = self.vote_parser.parse_vote(
                    vote.vote_content, topic_options
                )

                if not preferences:
                    logger.warning(f"Could not parse preferences from vote {vote.id}")
                    vote.status = VoteStatus.INVALID
                    continue

                # Update vote with parsed preferences
                vote.parsed_preferences = preferences
                vote.confidence_score = self._calculate_confidence_score(
                    vote.vote_content, preferences, topic_options
                )
                vote.status = VoteStatus.SUBMITTED

                valid_votes.append(vote)

            except Exception as e:
                logger.error(f"Error validating vote {vote.id}: {e}")
                vote.status = VoteStatus.INVALID

        return valid_votes

    def _calculate_confidence_score(
        self, vote_content: str, parsed_preferences: List[str], topic_options: List[str]
    ) -> float:
        """Calculate confidence score for a parsed vote.

        Args:
            vote_content: Original vote content
            parsed_preferences: Parsed preferences
            topic_options: Available topics

        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 0.0

        # Base score for having preferences
        if parsed_preferences:
            score += 0.3

        # Score for clarity (explicit numbering, clear language)
        if any(char.isdigit() for char in vote_content):
            score += 0.2
            # Extra bonus for well-formatted numbered lists
            if (
                "\n" in vote_content
                and len(
                    [
                        line
                        for line in vote_content.split("\n")
                        if line.strip().startswith(tuple("123456789"))
                    ]
                )
                >= 2
            ):
                score += 0.1

        # Score for completeness (mentioning multiple topics)
        mentioned_count = len(parsed_preferences)
        total_topics = len(topic_options)
        completeness = min(mentioned_count / max(total_topics, 1), 1.0)
        score += completeness * 0.3

        # Score for explicit language
        explicit_words = ["first", "second", "prefer", "rank", "order", "priority"]
        if any(word in vote_content.lower() for word in explicit_words):
            score += 0.2

        return min(score, 1.0)

    async def collect_modification_votes(
        self,
        agents: Dict[str, DiscussionAgent],
        current_topics: List[str],
        proposed_modifications: List[str],
        session_id: Optional[str] = None,
    ) -> VoteCollection:
        """Collect votes specifically for agenda modifications.

        Args:
            agents: Available agents
            current_topics: Current agenda topics
            proposed_modifications: Proposed new/modified topics
            session_id: Session ID

        Returns:
            VoteCollection for modifications
        """
        all_topics = current_topics + proposed_modifications

        return await self.collect_votes(
            agents, all_topics, VoteType.AGENDA_MODIFICATION, session_id
        )

    def analyze_vote_distribution(self, votes: List[Vote]) -> Dict[str, Any]:
        """Analyze the distribution and patterns in votes.

        Args:
            votes: List of votes to analyze

        Returns:
            Dictionary with vote analysis
        """
        analysis = {
            "total_votes": len(votes),
            "valid_votes": len([v for v in votes if v.status == VoteStatus.SUBMITTED]),
            "invalid_votes": len([v for v in votes if v.status == VoteStatus.INVALID]),
            "average_confidence": 0.0,
            "topic_mentions": {},
            "preference_patterns": {},
            "consensus_indicators": {},
        }

        if not votes:
            return analysis

        valid_votes = [
            v
            for v in votes
            if v.status == VoteStatus.SUBMITTED and v.parsed_preferences
        ]

        if not valid_votes:
            return analysis

        # Calculate average confidence
        confidences = [
            v.confidence_score for v in valid_votes if v.confidence_score is not None
        ]
        if confidences:
            analysis["average_confidence"] = sum(confidences) / len(confidences)

        # Analyze topic mentions
        topic_counts = {}
        for vote in valid_votes:
            for topic in vote.parsed_preferences:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

        analysis["topic_mentions"] = topic_counts

        # Analyze preference patterns (first choice, second choice, etc.)
        position_analysis = {}
        for vote in valid_votes:
            for i, topic in enumerate(vote.parsed_preferences):
                position = i + 1
                if position not in position_analysis:
                    position_analysis[position] = {}
                position_analysis[position][topic] = (
                    position_analysis[position].get(topic, 0) + 1
                )

        analysis["preference_patterns"] = position_analysis

        # Calculate consensus indicators
        if topic_counts:
            total_mentions = sum(topic_counts.values())
            max_mentions = max(topic_counts.values())
            analysis["consensus_indicators"] = {
                "most_popular_topic": max(topic_counts, key=topic_counts.get),
                "max_support_percentage": (max_mentions / len(valid_votes)) * 100,
                "topic_diversity": len(topic_counts),
                "agreement_level": max_mentions / len(valid_votes),
            }

        return analysis
