"""Discussion Agent implementation for Virtual Agora.

This module provides the DiscussionAgent class that extends LLMAgent with
discussion-specific functionality for participating in structured debates.
"""

from typing import Optional, List, Dict, Any, Union, Tuple
from datetime import datetime
import json
import re
from enum import Enum

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.state.schema import Message, Vote, VirtualAgoraState
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class AgentState(Enum):
    """Agent states during discussion."""

    ACTIVE = "active"
    WARNED = "warned"
    MUTED = "muted"


class VoteType(Enum):
    """Types of votes an agent can cast."""

    AGENDA_SELECTION = "agenda_selection"
    TOPIC_CONCLUSION = "topic_conclusion"
    AGENDA_MODIFICATION = "agenda_modification"


class DiscussionAgent(LLMAgent):
    """Discussion agent that participates in structured debates.

    This class extends LLMAgent with functionality specific to Virtual Agora
    discussions, including topic proposals, voting, and state management.
    """

    def __init__(
        self,
        agent_id: str,
        llm: BaseChatModel,
        system_prompt: Optional[str] = None,
        enable_error_handling: bool = True,
        max_retries: int = 3,
        fallback_llm: Optional[BaseChatModel] = None,
        max_context_messages: int = 10,
        warning_threshold: int = 2,
    ):
        """Initialize the discussion agent.

        Args:
            agent_id: Unique identifier for the agent
            llm: LangChain chat model instance
            system_prompt: Optional system prompt (uses default if None)
            enable_error_handling: Whether to enable enhanced error handling
            max_retries: Maximum number of retries for failed operations
            fallback_llm: Optional fallback LLM for error recovery
            max_context_messages: Maximum number of context messages to maintain
            warning_threshold: Number of warnings before agent is muted
        """
        # Use participant role and discussion-specific system prompt
        if system_prompt is None:
            system_prompt = self._get_discussion_system_prompt()

        super().__init__(
            agent_id=agent_id,
            llm=llm,
            role="participant",
            system_prompt=system_prompt,
            enable_error_handling=enable_error_handling,
            max_retries=max_retries,
            fallback_llm=fallback_llm,
        )

        # Discussion-specific attributes
        self.max_context_messages = max_context_messages
        self.warning_threshold = warning_threshold

        # Agent state tracking
        self.current_state = AgentState.ACTIVE
        self.warning_count = 0
        self.current_topic = None
        self.vote_history: List[Vote] = []

        logger.info(
            f"Initialized DiscussionAgent {agent_id} with "
            f"max_context={max_context_messages}, warning_threshold={warning_threshold}"
        )

    def _get_discussion_system_prompt(self) -> str:
        """Get the discussion-specific system prompt."""
        return (
            "You are a thoughtful participant in a structured discussion. "
            "You will be given a topic and context from previous turns. "
            "Your goal is to provide a well-reasoned, concise comment that builds upon the conversation. "
            "Stay strictly on the topic provided by the Moderator. "
            "Be prepared to propose discussion topics, vote on agendas, "
            "and vote on when to conclude a topic when asked. "
            "Always provide clear reasoning for your positions and votes."
        )

    def reset_for_new_topic(self, topic: str) -> None:
        """Reset agent state for a new topic.

        Args:
            topic: The new topic being discussed
        """
        self.current_topic = topic
        self.warning_count = 0
        self.current_state = AgentState.ACTIVE

        logger.info(f"Agent {self.agent_id} reset for new topic: {topic}")

    def add_warning(self, reason: str) -> bool:
        """Add a warning to the agent.

        Args:
            reason: Reason for the warning

        Returns:
            True if agent should be muted, False otherwise
        """
        self.warning_count += 1
        logger.warning(
            f"Agent {self.agent_id} warned (count: {self.warning_count}): {reason}"
        )

        if self.warning_count >= self.warning_threshold:
            self.current_state = AgentState.MUTED
            logger.warning(f"Agent {self.agent_id} has been muted")
            return True
        else:
            self.current_state = AgentState.WARNED
            return False

    def is_muted(self) -> bool:
        """Check if the agent is currently muted."""
        return self.current_state == AgentState.MUTED

    def get_agent_state_info(self) -> Dict[str, Any]:
        """Get current agent state information.

        Returns:
            Dictionary containing agent state information
        """
        return {
            "agent_id": self.agent_id,
            "state": self.current_state.value,
            "warning_count": self.warning_count,
            "current_topic": self.current_topic,
            "vote_count": len(self.vote_history),
            "message_count": self.message_count,
        }

    def propose_topics(
        self, main_topic: str, context_messages: Optional[List[Message]] = None
    ) -> List[str]:
        """Propose 3-5 discussion topics based on the main topic.

        Args:
            main_topic: The main topic for the discussion
            context_messages: Optional context from previous messages

        Returns:
            List of proposed sub-topics
        """
        if self.is_muted():
            logger.warning(f"Muted agent {self.agent_id} cannot propose topics")
            return []

        prompt = self._build_topic_proposal_prompt(main_topic, context_messages)

        try:
            response = self.generate_response(prompt, context_messages)
            topics = self._parse_topic_proposals(response)

            logger.info(
                f"Agent {self.agent_id} proposed {len(topics)} topics for '{main_topic}'"
            )
            return topics

        except Exception as e:
            logger.error(f"Error in topic proposal for agent {self.agent_id}: {e}")
            return []

    def _build_topic_proposal_prompt(
        self, main_topic: str, context_messages: Optional[List[Message]] = None
    ) -> str:
        """Build prompt for topic proposal.

        Args:
            main_topic: The main topic for discussion
            context_messages: Optional context messages

        Returns:
            Formatted prompt for topic proposal
        """
        prompt_parts = [
            f"Based on the main topic '{main_topic}', please propose 3-5 specific sub-topics for structured discussion.",
            "",
            "Your proposals should be:",
            "- Specific and actionable discussion points",
            "- Relevant to the main topic",
            "- Diverse in scope and perspective",
            "- Suitable for debate among multiple participants",
            "",
            "Format your response as a numbered list:",
            "1. [First sub-topic]",
            "2. [Second sub-topic]",
            "3. [Third sub-topic]",
            "4. [Fourth sub-topic] (optional)",
            "5. [Fifth sub-topic] (optional)",
        ]

        if context_messages:
            prompt_parts.insert(
                1, "Consider the following context from our discussion:"
            )
            for msg in context_messages[-3:]:  # Last 3 messages for context
                prompt_parts.insert(
                    2, f"- {msg['speaker_id']}: {msg['content'][:100]}..."
                )
            prompt_parts.insert(2 + len(context_messages[-3:]), "")

        return "\n".join(prompt_parts)

    def _parse_topic_proposals(self, response: str) -> List[str]:
        """Parse topic proposals from agent response.

        Args:
            response: Raw response from the agent

        Returns:
            List of parsed topic proposals
        """
        topics = []

        # Look for numbered list format
        numbered_pattern = r"^\d+\.\s*(.+)$"

        for line in response.strip().split("\n"):
            line = line.strip()
            match = re.match(numbered_pattern, line)
            if match:
                topic = match.group(1).strip()
                if topic and len(topic) > 5:  # Basic validation
                    topics.append(topic)

        # Fallback: try to extract topics from bullet points or other formats
        if not topics:
            bullet_pattern = r"^[-*•]\s*(.+)$"
            for line in response.strip().split("\n"):
                line = line.strip()
                match = re.match(bullet_pattern, line)
                if match:
                    topic = match.group(1).strip()
                    if topic and len(topic) > 5:
                        topics.append(topic)

        # Ensure we have 3-5 topics and remove duplicates
        unique_topics = []
        for topic in topics:
            if topic not in unique_topics:
                unique_topics.append(topic)

        return unique_topics[:5]  # Max 5 topics

    def vote_on_agenda(
        self,
        proposed_topics: List[str],
        context_messages: Optional[List[Message]] = None,
    ) -> str:
        """Vote on the proposed agenda topics.

        Args:
            proposed_topics: List of proposed topics to vote on
            context_messages: Optional context from previous messages

        Returns:
            Natural language vote response with reasoning
        """
        if self.is_muted():
            logger.warning(f"Muted agent {self.agent_id} cannot vote on agenda")
            return "I am currently muted and cannot participate in voting."

        prompt = self._build_agenda_voting_prompt(proposed_topics, context_messages)

        try:
            response = self.generate_response(prompt, context_messages)

            # Record the vote
            vote = Vote(
                id=f"{self.agent_id}_agenda_{datetime.now().isoformat()}",
                voter_id=self.agent_id,
                phase=1,  # Agenda setting phase
                vote_type=VoteType.AGENDA_SELECTION.value,
                choice=response,
                timestamp=datetime.now(),
            )
            self.vote_history.append(vote)

            logger.info(f"Agent {self.agent_id} voted on agenda")
            return response

        except Exception as e:
            logger.error(f"Error in agenda voting for agent {self.agent_id}: {e}")
            return "I apologize, but I'm having difficulty processing the agenda vote."

    def _build_agenda_voting_prompt(
        self,
        proposed_topics: List[str],
        context_messages: Optional[List[Message]] = None,
    ) -> str:
        """Build prompt for agenda voting.

        Args:
            proposed_topics: List of topics to vote on
            context_messages: Optional context messages

        Returns:
            Formatted prompt for agenda voting
        """
        prompt_parts = [
            "Please review the following proposed discussion topics and provide your preferred order for discussion.",
            "",
            "Proposed topics:",
        ]

        for i, topic in enumerate(proposed_topics, 1):
            prompt_parts.append(f"{i}. {topic}")

        prompt_parts.extend(
            [
                "",
                "Please respond with:",
                "1. Your preferred order of topics (by number or title)",
                "2. Brief reasoning for your preferences",
                "3. Any topics you think should be prioritized or deprioritized",
                "",
                "Format your response in natural language, clearly stating your preferences and reasoning.",
            ]
        )

        return "\n".join(prompt_parts)

    def vote_on_topic_conclusion(
        self, current_topic: str, context_messages: Optional[List[Message]] = None
    ) -> Tuple[bool, str]:
        """Vote on whether the current topic discussion should conclude.

        Args:
            current_topic: The topic currently being discussed
            context_messages: Optional context from recent discussion

        Returns:
            Tuple of (vote_result, reasoning) where vote_result is True for "Yes"
        """
        if self.is_muted():
            logger.warning(
                f"Muted agent {self.agent_id} cannot vote on topic conclusion"
            )
            return False, "I am currently muted and cannot participate in voting."

        prompt = self._build_conclusion_voting_prompt(current_topic, context_messages)

        try:
            response = self.generate_response(prompt, context_messages)
            vote_result, reasoning = self._parse_conclusion_vote(response)

            # Record the vote
            vote = Vote(
                id=f"{self.agent_id}_conclusion_{datetime.now().isoformat()}",
                voter_id=self.agent_id,
                phase=3,  # Consensus phase
                vote_type=VoteType.TOPIC_CONCLUSION.value,
                choice="Yes" if vote_result else "No",
                timestamp=datetime.now(),
                metadata={"reasoning": reasoning, "topic": current_topic},
            )
            self.vote_history.append(vote)

            logger.info(
                f"Agent {self.agent_id} voted {'Yes' if vote_result else 'No'} "
                f"on concluding topic '{current_topic}'"
            )
            return vote_result, reasoning

        except Exception as e:
            logger.error(f"Error in conclusion voting for agent {self.agent_id}: {e}")
            return (
                False,
                "I apologize, but I'm having difficulty processing the conclusion vote.",
            )

    def _build_conclusion_voting_prompt(
        self, current_topic: str, context_messages: Optional[List[Message]] = None
    ) -> str:
        """Build prompt for topic conclusion voting.

        Args:
            current_topic: Current topic being discussed
            context_messages: Optional context messages

        Returns:
            Formatted prompt for conclusion voting
        """
        prompt_parts = [
            f"Should we conclude the discussion on '{current_topic}'?",
            "",
            "Please respond with 'Yes' or 'No' and provide a short justification.",
        ]

        if context_messages:
            recent_messages = context_messages[-5:]  # Last 5 messages
            if recent_messages:
                prompt_parts.extend(
                    [
                        "",
                        "Consider the recent discussion:",
                    ]
                )
                for msg in recent_messages:
                    prompt_parts.append(
                        f"- {msg['speaker_id']}: {msg['content'][:150]}..."
                    )

        prompt_parts.extend(
            [
                "",
                "Consider factors such as:",
                "- Have the key aspects of the topic been adequately discussed?",
                "- Are there important points that still need to be addressed?",
                "- Has the discussion reached a natural conclusion?",
                "- Would more discussion be productive?",
                "",
                "Format your response clearly starting with 'Yes' or 'No', followed by your reasoning.",
            ]
        )

        return "\n".join(prompt_parts)

    def _parse_conclusion_vote(self, response: str) -> Tuple[bool, str]:
        """Parse conclusion vote from agent response.

        Args:
            response: Raw response from the agent

        Returns:
            Tuple of (vote_result, reasoning)
        """
        response_lower = response.lower().strip()

        # Look for clear Yes/No at the beginning
        if response_lower.startswith("yes"):
            vote_result = True
        elif response_lower.startswith("no"):
            vote_result = False
        else:
            # Try to find Yes/No in the first sentence
            first_sentence = response.split(".")[0].lower()
            if "yes" in first_sentence and "no" not in first_sentence:
                vote_result = True
            elif "no" in first_sentence and "yes" not in first_sentence:
                vote_result = False
            else:
                # Default to No if unclear
                vote_result = False
                logger.warning(
                    f"Unclear vote response from {self.agent_id}: {response[:100]}"
                )

        # Extract reasoning (everything after the vote)
        reasoning = response.strip()
        if reasoning.lower().startswith(("yes", "no")):
            # Remove the vote word and any following punctuation/whitespace
            reasoning = re.sub(
                r"^(yes|no)[.,\s]*", "", reasoning, flags=re.IGNORECASE
            ).strip()

        if not reasoning:
            reasoning = "No specific reasoning provided."

        return vote_result, reasoning

    def generate_discussion_response(
        self,
        topic: str,
        context_messages: Optional[List[Message]] = None,
        phase: int = 2,
    ) -> str:
        """Generate a response for topic discussion.

        Args:
            topic: The current topic being discussed
            context_messages: Context from recent discussion
            phase: Current discussion phase

        Returns:
            Generated discussion response
        """
        if self.is_muted():
            logger.warning(
                f"Muted agent {self.agent_id} cannot participate in discussion"
            )
            return ""

        # Update current topic
        if self.current_topic != topic:
            self.reset_for_new_topic(topic)

        # Limit context to most recent messages
        limited_context = None
        if context_messages:
            limited_context = context_messages[-self.max_context_messages :]

        prompt = self._build_discussion_prompt(topic, limited_context, phase)

        try:
            response = self.generate_response(prompt, limited_context)

            logger.info(
                f"Agent {self.agent_id} generated discussion response for topic '{topic}' "
                f"({len(response)} chars)"
            )
            return response

        except Exception as e:
            logger.error(
                f"Error generating discussion response for agent {self.agent_id}: {e}"
            )
            return "I apologize, but I'm having difficulty contributing to this discussion."

    def _build_discussion_prompt(
        self,
        topic: str,
        context_messages: Optional[List[Message]] = None,
        phase: int = 2,
    ) -> str:
        """Build prompt for discussion response.

        Args:
            topic: Current topic
            context_messages: Context messages
            phase: Discussion phase

        Returns:
            Formatted discussion prompt
        """
        prompt_parts = [
            f"We are currently discussing: {topic}",
            "",
            "Please provide a thoughtful, well-reasoned comment that contributes to this discussion.",
        ]

        if context_messages:
            prompt_parts.extend(
                [
                    "",
                    "Consider the recent discussion context:",
                ]
            )
            for msg in context_messages:
                # Show speaker and abbreviated content
                content_preview = msg["content"][:200]
                if len(msg["content"]) > 200:
                    content_preview += "..."
                prompt_parts.append(f"- {msg['speaker_id']}: {content_preview}")

        prompt_parts.extend(
            [
                "",
                "Your response should:",
                "- Stay strictly on the topic",
                "- Build upon previous comments where relevant",
                "- Provide substantive contribution to the discussion",
                "- Be concise but thoughtful (aim for 2-4 sentences)",
                "- Maintain a respectful and constructive tone",
            ]
        )

        return "\n".join(prompt_parts)

    def provide_minority_consideration(
        self, topic: str, context_messages: Optional[List[Message]] = None
    ) -> str:
        """Provide final considerations as part of the minority on a concluded topic.

        Args:
            topic: The topic that was concluded
            context_messages: Context from the full discussion

        Returns:
            Final minority consideration response
        """
        if self.is_muted():
            logger.warning(
                f"Muted agent {self.agent_id} cannot provide minority consideration"
            )
            return ""

        prompt = self._build_minority_consideration_prompt(topic, context_messages)

        try:
            response = self.generate_response(prompt, context_messages)

            logger.info(
                f"Agent {self.agent_id} provided minority consideration for topic '{topic}'"
            )
            return response

        except Exception as e:
            logger.error(
                f"Error in minority consideration for agent {self.agent_id}: {e}"
            )
            return "I apologize, but I'm having difficulty providing my final considerations."

    def _build_minority_consideration_prompt(
        self, topic: str, context_messages: Optional[List[Message]] = None
    ) -> str:
        """Build prompt for minority consideration.

        Args:
            topic: The concluded topic
            context_messages: Full discussion context

        Returns:
            Formatted minority consideration prompt
        """
        prompt_parts = [
            f"The discussion on '{topic}' has been concluded by majority vote, but you voted to continue.",
            "",
            "As part of the minority, you have an opportunity to provide your final considerations on this topic.",
            "Please share any important points, concerns, or perspectives that you feel should be on record before we move on.",
        ]

        if context_messages:
            # Show more context for final considerations
            relevant_messages = [
                msg for msg in context_messages if msg.get("topic") == topic
            ]
            if relevant_messages:
                prompt_parts.extend(
                    [
                        "",
                        "Consider the full discussion on this topic:",
                    ]
                )
                for msg in relevant_messages[-8:]:  # Last 8 relevant messages
                    content_preview = msg["content"][:150]
                    if len(msg["content"]) > 150:
                        content_preview += "..."
                    prompt_parts.append(f"- {msg['speaker_id']}: {content_preview}")

        prompt_parts.extend(
            [
                "",
                "Your final considerations should:",
                "- Highlight any overlooked aspects of the topic",
                "- Express concerns about concluding the discussion",
                "- Provide valuable insights for the final record",
                "- Be concise but substantive (2-5 sentences)",
            ]
        )

        return "\n".join(prompt_parts)
