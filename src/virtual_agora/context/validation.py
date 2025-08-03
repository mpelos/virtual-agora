"""Context validation and debugging tools for Virtual Agora.

This module provides comprehensive validation and debugging capabilities
for the context assembly system, helping identify and resolve context-related issues.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.context.types import ContextData
from virtual_agora.context.message_processor import MessageProcessor, ProcessedMessage
from virtual_agora.context.rules import ContextRules, ContextType
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ContextValidationResult:
    """Results of context validation."""

    is_valid: bool
    compliance_score: float
    validation_issues: List[str]
    validation_warnings: List[str]
    context_summary: Dict[str, Any]
    timestamp: datetime
    agent_id: str
    round_number: int


@dataclass
class ContextDebugInfo:
    """Detailed debugging information for context assembly."""

    agent_id: str
    round_number: int
    topic: str
    message_counts: Dict[str, int]
    rule_compliance: Dict[str, bool]
    processing_time_ms: float
    memory_usage_bytes: int
    errors: List[str]
    warnings: List[str]
    context_builder_used: str
    message_formats: List[str]
    timestamp: datetime


class ContextValidator:
    """Comprehensive context validation and debugging system."""

    @staticmethod
    def validate_context_data(
        context_data: ContextData,
        agent_id: str,
        current_round: int,
        context_type: ContextType = ContextType.DISCUSSION_ROUND,
    ) -> ContextValidationResult:
        """Perform comprehensive validation of context data.

        Args:
            context_data: Context data to validate
            agent_id: Agent ID for tracking
            current_round: Current round number
            context_type: Type of context being validated

        Returns:
            ContextValidationResult with detailed validation information
        """
        start_time = datetime.now()
        issues = []
        warnings = []

        # Basic structure validation
        if not context_data.system_prompt:
            issues.append("Missing system prompt")

        # Validate round-specific requirements
        rule_set, _ = ContextRules.get_context_requirements(
            context_type, current_round, {}
        )

        # Check required elements based on rules
        if rule_set.include_theme and not context_data.has_user_input:
            if current_round > 0:  # Theme is more critical in later rounds
                issues.append("Missing theme/user input")
            else:
                warnings.append("No theme provided (acceptable for round 0)")

        if rule_set.include_round_summaries and current_round > 0:
            if not context_data.has_round_summaries:
                warnings.append(
                    f"No round summaries available for round {current_round}"
                )
            elif len(context_data.round_summaries) > rule_set.max_round_summaries:
                warnings.append(
                    f"Too many round summaries: {len(context_data.round_summaries)} > {rule_set.max_round_summaries}"
                )

        if rule_set.include_user_messages and current_round > 1:
            if not context_data.has_user_participation_messages:
                warnings.append(
                    "No user participation messages (user may not have participated)"
                )

        if rule_set.include_current_round:
            if not context_data.has_current_round_messages and current_round > 0:
                warnings.append(
                    "No current round messages (agent may be speaking first)"
                )

        # Validate formatted context messages if available
        if hasattr(context_data, "formatted_context_messages"):
            formatted_count = len(context_data.formatted_context_messages or [])
            if formatted_count == 0 and current_round > 0:
                warnings.append("No formatted context messages available")
            elif formatted_count > 50:  # Reasonable limit
                warnings.append(
                    f"Large number of formatted messages: {formatted_count}"
                )

        # Check metadata consistency
        if context_data.metadata:
            metadata_agent_id = context_data.metadata.get("agent_id")
            if metadata_agent_id and metadata_agent_id != agent_id:
                issues.append(
                    f"Agent ID mismatch: expected {agent_id}, got {metadata_agent_id}"
                )

            metadata_round = context_data.metadata.get("current_round")
            if metadata_round is not None and metadata_round != current_round:
                issues.append(
                    f"Round mismatch: expected {current_round}, got {metadata_round}"
                )

        # Calculate compliance score
        total_checks = 10  # Number of validation checks performed
        failed_checks = len(issues)
        warning_penalty = len(warnings) * 0.1  # Minor penalty for warnings
        compliance_score = max(
            0.0, (total_checks - failed_checks - warning_penalty) / total_checks
        )

        # Create context summary
        context_summary = {
            "has_system_prompt": bool(context_data.system_prompt),
            "has_theme": context_data.has_user_input,
            "has_context_documents": context_data.has_context_documents,
            "round_summaries_count": len(context_data.round_summaries or []),
            "user_messages_count": len(context_data.user_participation_messages or []),
            "current_round_messages_count": len(
                context_data.current_round_messages or []
            ),
            "formatted_messages_count": len(
                getattr(context_data, "formatted_context_messages", []) or []
            ),
            "metadata_present": bool(context_data.metadata),
            "context_type": context_type.value,
            "rule_set_applied": {
                "include_theme": rule_set.include_theme,
                "include_round_summaries": rule_set.include_round_summaries,
                "include_user_messages": rule_set.include_user_messages,
                "include_current_round": rule_set.include_current_round,
            },
        }

        is_valid = len(issues) == 0

        result = ContextValidationResult(
            is_valid=is_valid,
            compliance_score=compliance_score,
            validation_issues=issues,
            validation_warnings=warnings,
            context_summary=context_summary,
            timestamp=start_time,
            agent_id=agent_id,
            round_number=current_round,
        )

        # Log validation results
        if is_valid:
            logger.info(
                f"Context validation passed for {agent_id} (round {current_round}): "
                f"compliance {compliance_score:.2f}, {len(warnings)} warnings"
            )
        else:
            logger.warning(
                f"Context validation failed for {agent_id} (round {current_round}): "
                f"{len(issues)} issues, {len(warnings)} warnings"
            )
            for issue in issues:
                logger.warning(f"  Issue: {issue}")

        return result

    @staticmethod
    def debug_message_processing(
        state: VirtualAgoraState, agent_id: str, current_round: int, topic: str
    ) -> ContextDebugInfo:
        """Generate detailed debugging information for message processing.

        Args:
            state: Application state
            agent_id: Agent ID for debugging
            current_round: Current round number
            topic: Current topic

        Returns:
            ContextDebugInfo with comprehensive debugging data
        """
        start_time = datetime.now()
        errors = []
        warnings = []

        try:
            # Analyze all messages in state
            all_messages = state.get("messages", [])
            message_counts = {
                "total": len(all_messages),
                "current_round": 0,
                "user_participation": 0,
                "agent_messages": 0,
                "system_messages": 0,
                "format_errors": 0,
            }

            message_formats = []

            for msg in all_messages:
                try:
                    processed = MessageProcessor.standardize_message(msg)

                    # Count by type
                    if processed.speaker_role == "user":
                        if (
                            processed.metadata.get("participation_type")
                            == "user_turn_participation"
                        ):
                            message_counts["user_participation"] += 1
                    elif processed.speaker_role in ["participant", "moderator"]:
                        message_counts["agent_messages"] += 1
                    elif processed.speaker_role == "system":
                        message_counts["system_messages"] += 1

                    # Count current round
                    if (
                        processed.round_number == current_round
                        and processed.topic == topic
                    ):
                        message_counts["current_round"] += 1

                    # Track format types
                    format_type = (
                        f"{processed.original_format}_{processed.speaker_role}"
                    )
                    if format_type not in message_formats:
                        message_formats.append(format_type)

                except Exception as e:
                    message_counts["format_errors"] += 1
                    errors.append(f"Message processing error: {str(e)}")

            # Check rule compliance
            rule_set, _ = ContextRules.get_context_requirements(
                ContextType.DISCUSSION_ROUND, current_round, state
            )

            rule_compliance = {
                "theme_available": bool(state.get("main_topic")),
                "round_summaries_available": len(state.get("round_summaries", [])) > 0,
                "user_messages_found": message_counts["user_participation"] > 0,
                "current_round_messages_found": message_counts["current_round"] > 0,
                "message_format_consistency": message_counts["format_errors"] == 0,
            }

            # Performance metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            memory_estimate = len(json.dumps(state, default=str)) if state else 0

            # Generate warnings
            if message_counts["format_errors"] > 0:
                warnings.append(
                    f"{message_counts['format_errors']} messages had format errors"
                )

            if message_counts["current_round"] == 0 and current_round > 0:
                warnings.append("No messages found for current round")

            if message_counts["user_participation"] == 0 and current_round > 1:
                warnings.append("No user participation messages found")

            if len(message_formats) > 5:
                warnings.append(
                    f"Many different message formats detected: {len(message_formats)}"
                )

        except Exception as e:
            errors.append(f"Debug analysis failed: {str(e)}")
            # Provide minimal debug info
            message_counts = {"error": True}
            rule_compliance = {"error": True}
            processing_time = 0
            memory_estimate = 0
            message_formats = ["error"]

        return ContextDebugInfo(
            agent_id=agent_id,
            round_number=current_round,
            topic=topic,
            message_counts=message_counts,
            rule_compliance=rule_compliance,
            processing_time_ms=processing_time,
            memory_usage_bytes=memory_estimate,
            errors=errors,
            warnings=warnings,
            context_builder_used="unknown",  # To be filled by caller
            message_formats=message_formats,
            timestamp=start_time,
        )

    @staticmethod
    def log_context_debug_summary(debug_info: ContextDebugInfo) -> None:
        """Log a comprehensive debug summary.

        Args:
            debug_info: Debug information to log
        """
        logger.info(f"=== Context Debug Summary for {debug_info.agent_id} ===")
        logger.info(f"Round: {debug_info.round_number}, Topic: '{debug_info.topic}'")
        logger.info(f"Context Builder: {debug_info.context_builder_used}")
        logger.info(f"Processing Time: {debug_info.processing_time_ms:.1f}ms")
        logger.info(f"Memory Usage: {debug_info.memory_usage_bytes:,} bytes")

        # Message counts
        logger.info("Message Counts:")
        for key, value in debug_info.message_counts.items():
            logger.info(f"  {key}: {value}")

        # Rule compliance
        logger.info("Rule Compliance:")
        for key, value in debug_info.rule_compliance.items():
            status = "✓" if value else "✗"
            logger.info(f"  {status} {key}")

        # Message formats
        if debug_info.message_formats:
            logger.info(f"Message Formats: {', '.join(debug_info.message_formats)}")

        # Errors and warnings
        if debug_info.errors:
            logger.error("Errors:")
            for error in debug_info.errors:
                logger.error(f"  {error}")

        if debug_info.warnings:
            logger.warning("Warnings:")
            for warning in debug_info.warnings:
                logger.warning(f"  {warning}")

        logger.info("=== End Context Debug Summary ===")

    @staticmethod
    def validate_message_consistency(
        messages: List[Any],
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate message consistency across different formats.

        Args:
            messages: List of messages to validate

        Returns:
            Tuple of (is_consistent, issues, analysis)
        """
        try:
            is_valid, issues = MessageProcessor.validate_message_consistency(messages)

            # Additional analysis
            format_counts = {"dict": 0, "langchain": 0, "unknown": 0}
            speaker_roles = set()
            topics = set()
            rounds = set()

            for msg in messages:
                try:
                    processed = MessageProcessor.standardize_message(msg)
                    format_counts[processed.original_format] += 1
                    speaker_roles.add(processed.speaker_role)
                    if processed.topic:
                        topics.add(processed.topic)
                    rounds.add(processed.round_number)
                except Exception:
                    format_counts["unknown"] += 1

            analysis = {
                "total_messages": len(messages),
                "format_distribution": format_counts,
                "unique_speakers": len(speaker_roles),
                "unique_topics": len(topics),
                "unique_rounds": len(rounds),
                "speaker_roles": list(speaker_roles),
                "topics": list(topics),
                "rounds": sorted(list(rounds)),
            }

            return is_valid, issues, analysis

        except Exception as e:
            return False, [f"Validation failed: {str(e)}"], {"error": True}


class ContextDebugger:
    """Interactive debugging tools for context assembly."""

    @staticmethod
    def trace_context_assembly(
        state: VirtualAgoraState, agent_id: str, **context_kwargs
    ) -> Dict[str, Any]:
        """Trace the complete context assembly process step by step.

        Args:
            state: Application state
            agent_id: Agent ID to trace
            **context_kwargs: Context assembly parameters

        Returns:
            Dictionary with detailed trace information
        """
        trace = {
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "steps": [],
            "total_time_ms": 0,
            "errors": [],
            "final_context": None,
        }

        start_time = datetime.now()

        try:
            # Step 1: Context builder selection
            step_start = datetime.now()
            context_type = context_kwargs.get("context_type", "discussion")
            if "current_round" in context_kwargs and "agent_position" in context_kwargs:
                context_type = "discussion_round"

            trace["steps"].append(
                {
                    "step": 1,
                    "name": "Context Builder Selection",
                    "details": {"selected_type": context_type},
                    "time_ms": (datetime.now() - step_start).total_seconds() * 1000,
                }
            )

            # Step 2: Rule determination
            step_start = datetime.now()
            current_round = context_kwargs.get("current_round", 0)
            rule_set, additional_params = ContextRules.get_context_requirements(
                ContextType.DISCUSSION_ROUND, current_round, state
            )

            trace["steps"].append(
                {
                    "step": 2,
                    "name": "Rule Determination",
                    "details": {
                        "rule_set": {
                            "include_theme": rule_set.include_theme,
                            "include_round_summaries": rule_set.include_round_summaries,
                            "include_user_messages": rule_set.include_user_messages,
                            "include_current_round": rule_set.include_current_round,
                        },
                        "additional_params": additional_params,
                    },
                    "time_ms": (datetime.now() - step_start).total_seconds() * 1000,
                }
            )

            # Step 3: Message processing
            step_start = datetime.now()
            all_messages = state.get("messages", [])
            topic = context_kwargs.get("topic", state.get("active_topic"))

            # Filter user messages
            user_messages = MessageProcessor.filter_user_participation_messages(
                all_messages, topic, exclude_round=current_round
            )

            # Filter current round messages
            current_round_messages = MessageProcessor.filter_messages_by_round(
                all_messages, current_round, topic
            )

            trace["steps"].append(
                {
                    "step": 3,
                    "name": "Message Processing",
                    "details": {
                        "total_messages": len(all_messages),
                        "user_messages_found": len(user_messages),
                        "current_round_messages_found": len(current_round_messages),
                        "topic": topic,
                    },
                    "time_ms": (datetime.now() - step_start).total_seconds() * 1000,
                }
            )

            # Step 4: Context assembly
            step_start = datetime.now()
            try:
                from virtual_agora.context.builders import get_context_builder

                context_builder = get_context_builder(context_type)
                context_data = context_builder.build_context(
                    state=state,
                    system_prompt="",  # Not needed for tracing
                    agent_id=agent_id,
                    **context_kwargs,
                )

                trace["steps"].append(
                    {
                        "step": 4,
                        "name": "Context Assembly",
                        "details": {
                            "context_builder_type": context_type,
                            "success": True,
                            "context_summary": {
                                "has_theme": context_data.has_user_input,
                                "has_summaries": context_data.has_round_summaries,
                                "has_user_messages": context_data.has_user_participation_messages,
                                "has_current_round": context_data.has_current_round_messages,
                                "formatted_messages_count": len(
                                    getattr(
                                        context_data, "formatted_context_messages", []
                                    )
                                    or []
                                ),
                            },
                        },
                        "time_ms": (datetime.now() - step_start).total_seconds() * 1000,
                    }
                )

                trace["final_context"] = (
                    context_data.metadata if context_data.metadata else {}
                )

            except Exception as e:
                trace["steps"].append(
                    {
                        "step": 4,
                        "name": "Context Assembly",
                        "details": {"success": False, "error": str(e)},
                        "time_ms": (datetime.now() - step_start).total_seconds() * 1000,
                    }
                )
                trace["errors"].append(f"Context assembly failed: {str(e)}")

        except Exception as e:
            trace["errors"].append(f"Trace failed: {str(e)}")

        trace["total_time_ms"] = (datetime.now() - start_time).total_seconds() * 1000

        return trace
