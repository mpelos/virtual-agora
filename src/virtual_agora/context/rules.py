"""Context assembly business rules for Virtual Agora.

This module defines and enforces the business rules for context assembly
across different phases and rounds of Virtual Agora discussions.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, replace
from enum import Enum
import copy

from virtual_agora.state.schema import VirtualAgoraState, RoundSummary
from virtual_agora.context.message_processor import MessageProcessor, ProcessedMessage
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class ContextType(Enum):
    """Types of context that can be assembled for agents."""

    DISCUSSION_ROUND = "discussion_round"
    REPORT_GENERATION = "report_generation"
    SUMMARIZATION = "summarization"
    MODERATION = "moderation"


@dataclass
class ContextRuleSet:
    """Defines rules for a specific context type."""

    include_theme: bool = True
    include_topic: bool = True
    include_context_documents: bool = False
    include_round_summaries: bool = False
    include_user_messages: bool = False
    include_current_round: bool = False
    max_round_summaries: int = 10
    max_user_messages: int = 20
    max_current_round_messages: int = 50
    user_message_format: str = "[Round Moderator - Round {round}]: {content}"
    colleague_message_format: str = "[{speaker}]: {content}"


class ContextRules:
    """Enforces business rules for context assembly in Virtual Agora discussions.

    This class encapsulates the business logic for what context should be
    provided to different types of agents in different phases and rounds.
    """

    # Define rule sets for different context types
    RULE_SETS = {
        ContextType.DISCUSSION_ROUND: ContextRuleSet(
            include_theme=True,
            include_topic=True,
            include_context_documents=True,  # For complex topics
            include_round_summaries=True,  # From previous rounds
            include_user_messages=True,  # User participation
            include_current_round=True,  # Colleague messages
            max_round_summaries=10,
            max_user_messages=20,
            max_current_round_messages=50,
        ),
        ContextType.REPORT_GENERATION: ContextRuleSet(
            include_theme=False,
            include_topic=True,
            include_context_documents=False,
            include_round_summaries=True,
            include_user_messages=False,
            include_current_round=False,
            max_round_summaries=20,
        ),
        ContextType.SUMMARIZATION: ContextRuleSet(
            include_theme=False,
            include_topic=True,
            include_context_documents=False,
            include_round_summaries=False,
            include_user_messages=False,
            include_current_round=True,
            max_current_round_messages=100,
        ),
        ContextType.MODERATION: ContextRuleSet(
            include_theme=False,
            include_topic=True,
            include_context_documents=False,
            include_round_summaries=False,
            include_user_messages=False,
            include_current_round=False,
        ),
    }

    @staticmethod
    def get_context_requirements(
        context_type: ContextType, current_round: int, state: VirtualAgoraState
    ) -> Tuple[ContextRuleSet, Dict[str, Any]]:
        """Determine context requirements based on type and round.

        Args:
            context_type: Type of context being assembled
            current_round: Current round number
            state: Current state for additional context

        Returns:
            Tuple of (rule_set, additional_parameters)
        """
        base_rules = replace(ContextRules.RULE_SETS[context_type])
        additional_params = {}

        # Apply round-specific modifications for discussion rounds
        if context_type == ContextType.DISCUSSION_ROUND:
            if current_round == 0:
                # Round 0: No previous context available
                base_rules.include_round_summaries = False
                base_rules.include_user_messages = False
                additional_params["round_zero"] = True
                logger.debug("Applied round 0 rules: no previous context")

            elif current_round == 1:
                # Round 1: Potentially some summaries, limited user messages
                base_rules.max_round_summaries = 5
                base_rules.max_user_messages = 10
                additional_params["early_round"] = True
                logger.debug("Applied round 1 rules: limited previous context")

            else:
                # Round 2+: Full context with limits
                additional_params["established_round"] = True
                logger.debug(f"Applied round {current_round} rules: full context")

        # Apply topic complexity considerations
        topic_complexity = ContextRules._assess_topic_complexity(state)
        if context_type == ContextType.DISCUSSION_ROUND:
            # Always include context documents for discussion agents
            # Users expect context files to be available when placed in context directory
            base_rules.include_context_documents = True
            additional_params["topic_complexity"] = topic_complexity

        logger.info(
            f"Context requirements for {context_type.value}, round {current_round}: "
            f"summaries={base_rules.include_round_summaries}, "
            f"user_msgs={base_rules.include_user_messages}, "
            f"current_round={base_rules.include_current_round}, "
            f"context_docs={base_rules.include_context_documents}"
        )

        return base_rules, additional_params

    @staticmethod
    def _assess_topic_complexity(state: VirtualAgoraState) -> str:
        """Assess topic complexity to determine if context documents are needed.

        Args:
            state: Current state

        Returns:
            Complexity level: 'simple' or 'complex'
        """
        current_topic = state.get("active_topic", "")
        main_topic = state.get("main_topic", "")

        # Simple heuristics for complexity assessment
        complex_indicators = [
            len(current_topic.split()) > 10,  # Long topic titles
            "technical" in current_topic.lower(),
            "implement" in current_topic.lower(),
            "analysis" in current_topic.lower(),
            "strategy" in current_topic.lower(),
            len(main_topic.split()) > 15,  # Complex main themes
        ]

        complexity = "complex" if any(complex_indicators) else "simple"
        logger.debug(f"Assessed topic complexity: {complexity} for '{current_topic}'")
        return complexity

    @staticmethod
    def validate_context_assembly(
        context_type: ContextType,
        current_round: int,
        theme: Optional[str],
        current_topic: Optional[str],
        round_summaries: List[RoundSummary],
        user_messages: List[ProcessedMessage],
        current_round_messages: List[ProcessedMessage],
    ) -> Tuple[bool, List[str], float]:
        """Validate that assembled context meets business rules.

        Args:
            context_type: Type of context being validated
            current_round: Current round number
            theme: Discussion theme
            current_topic: Current topic
            round_summaries: Round summaries provided
            user_messages: User messages provided
            current_round_messages: Current round messages provided

        Returns:
            Tuple of (is_valid, issues, compliance_score)
        """
        issues = []
        rule_set = ContextRules.RULE_SETS[context_type]

        # Check required elements
        if rule_set.include_theme and not theme:
            issues.append("Missing required theme")

        if rule_set.include_topic and not current_topic:
            issues.append("Missing required current topic")

        # Check round-specific requirements
        if context_type == ContextType.DISCUSSION_ROUND:
            if current_round > 1:  # Round summaries only available from round 2+
                if rule_set.include_round_summaries and not round_summaries:
                    issues.append(f"Missing round summaries for round {current_round}")

                if rule_set.include_user_messages and current_round > 1:
                    # User messages should be available from round 1+
                    user_msg_count = len(user_messages)
                    if user_msg_count == 0:
                        logger.debug(
                            "No user messages found - this is normal if user hasn't participated"
                        )
                    else:
                        logger.debug(
                            f"Found {user_msg_count} user messages for validation"
                        )

        # Check limits
        if len(round_summaries) > rule_set.max_round_summaries:
            issues.append(
                f"Too many round summaries: {len(round_summaries)} > {rule_set.max_round_summaries}"
            )

        if len(user_messages) > rule_set.max_user_messages:
            issues.append(
                f"Too many user messages: {len(user_messages)} > {rule_set.max_user_messages}"
            )

        if len(current_round_messages) > rule_set.max_current_round_messages:
            issues.append(
                f"Too many current round messages: {len(current_round_messages)} > {rule_set.max_current_round_messages}"
            )

        # Calculate compliance score
        total_checks = 8  # Number of validation checks
        failed_checks = len(issues)
        compliance_score = max(0.0, (total_checks - failed_checks) / total_checks)

        is_valid = len(issues) == 0

        if is_valid:
            logger.debug(
                f"Context validation passed for {context_type.value}, round {current_round}"
            )
        else:
            logger.warning(
                f"Context validation failed for {context_type.value}, round {current_round}: {issues}"
            )

        return is_valid, issues, compliance_score

    @staticmethod
    def enforce_message_limits(
        messages: List[ProcessedMessage], max_count: int, strategy: str = "recent"
    ) -> List[ProcessedMessage]:
        """Enforce message count limits using specified strategy.

        Args:
            messages: List of messages to limit
            max_count: Maximum number of messages to return
            strategy: Limiting strategy ('recent', 'oldest', 'balanced')

        Returns:
            Limited list of messages
        """
        if len(messages) <= max_count:
            return messages

        if strategy == "recent":
            # Keep most recent messages
            limited = messages[-max_count:]
        elif strategy == "oldest":
            # Keep oldest messages
            limited = messages[:max_count]
        elif strategy == "balanced":
            # Keep some from beginning and some from end
            half = max_count // 2
            limited = messages[:half] + messages[-(max_count - half) :]
        else:
            # Default to recent
            limited = messages[-max_count:]

        logger.debug(
            f"Applied {strategy} limiting: {len(messages)} -> {len(limited)} messages"
        )
        return limited

    @staticmethod
    def get_round_context_requirements(
        current_round: int, agent_position: int, total_agents: int
    ) -> Dict[str, Any]:
        """Get specific context requirements for an agent in a round.

        Args:
            current_round: Current round number
            agent_position: Agent's position in speaking order (0-based)
            total_agents: Total number of agents in round

        Returns:
            Dictionary of context requirements
        """
        requirements = {
            "include_previous_speakers": agent_position > 0,
            "max_previous_speakers": agent_position,
            "include_round_announcement": True,
            "include_speaking_position": True,
            "position_context": f"{agent_position + 1}/{total_agents}",
        }

        # Round-specific adjustments
        if current_round == 0:
            requirements["context_emphasis"] = "initial_discussion"
        elif current_round == 1:
            requirements["context_emphasis"] = "building_on_previous"
        else:
            requirements["context_emphasis"] = "continued_deliberation"

        logger.debug(
            f"Round context requirements for agent {agent_position + 1}/{total_agents} in round {current_round}"
        )
        return requirements

    @staticmethod
    def format_user_message_for_context(
        processed_msg: ProcessedMessage, format_template: Optional[str] = None
    ) -> str:
        """Format user message according to business rules.

        Args:
            processed_msg: Processed user message
            format_template: Optional custom format template

        Returns:
            Formatted message content
        """
        if not format_template:
            format_template = "[Round Moderator - Round {round}]: {content}"

        return format_template.format(
            round=processed_msg.round_number,
            content=processed_msg.content,
            speaker=processed_msg.speaker_id,
            topic=processed_msg.topic,
        )

    @staticmethod
    def format_colleague_message_for_context(
        processed_msg: ProcessedMessage, format_template: Optional[str] = None
    ) -> str:
        """Format colleague message according to business rules.

        Args:
            processed_msg: Processed colleague message
            format_template: Optional custom format template

        Returns:
            Formatted message content
        """
        if not format_template:
            format_template = "[{speaker}]: {content}"

        return format_template.format(
            speaker=processed_msg.speaker_id,
            content=processed_msg.content,
            round=processed_msg.round_number,
            topic=processed_msg.topic,
        )
