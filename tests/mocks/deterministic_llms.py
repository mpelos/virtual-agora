"""Deterministic LLM implementations for integration testing.

This module provides LLM implementations that behave deterministically but
integrate properly with LangGraph and the Virtual Agora execution flow.
These LLMs return consistent, realistic responses based on role and context,
allowing for reliable integration testing while testing real execution paths.
"""

import uuid
from typing import Any, Dict, List, Optional, Union, Iterator, Callable
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.runnables import RunnableConfig
from pydantic import Field

from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class BaseDeterministicLLM(BaseChatModel):
    """Base class for deterministic LLM implementations.

    This class provides the foundation for creating LLMs that return
    consistent, predictable responses while maintaining full compatibility
    with LangGraph and LangChain interfaces.
    """

    # LLM configuration
    model_name: str = Field(default="deterministic-llm")
    role: str = Field(default="general")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=4000)
    provider: str = Field(default="deterministic")

    # Response configuration
    response_patterns: Dict[str, str] = Field(default_factory=dict)
    context_extractors: Dict[str, Callable] = Field(default_factory=dict)
    interrupt_triggers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Tracking fields
    call_count: int = Field(default=0)
    interaction_history: List[Dict[str, Any]] = Field(default_factory=list)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.debug(
            f"Initialized {self.__class__.__name__} with role={self.role}, model={self.model_name}"
        )

    @property
    def _llm_type(self) -> str:
        return "deterministic"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a deterministic response based on role and context."""
        self.call_count += 1

        # Extract context from messages
        context = self.extract_context(messages)
        logger.debug(f"[{self.role}] Call #{self.call_count} - Context: {context}")

        # Get response template based on context
        response_content = self.get_response_content(context)

        # Check if we should trigger an interrupt
        interrupt_info = self.check_interrupt_trigger(context)

        # Create realistic metadata
        metadata = self.get_response_metadata(context, response_content)

        # Create AI message
        ai_message = AIMessage(
            content=response_content, response_metadata=metadata, id=str(uuid.uuid4())
        )

        # Record interaction
        self.interaction_history.append(
            {
                "call_count": self.call_count,
                "context": context,
                "response": response_content,
                "interrupt_triggered": interrupt_info is not None,
                "timestamp": datetime.now(),
            }
        )

        # Trigger interrupt if needed (for GraphInterrupt simulation)
        if interrupt_info:
            logger.info(f"[{self.role}] Triggering interrupt: {interrupt_info['type']}")
            self.trigger_interrupt(interrupt_info)

        return ChatResult(
            generations=[ChatGeneration(message=ai_message)], llm_output=metadata
        )

    def extract_context(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Extract context information from message history.

        Args:
            messages: List of messages in the conversation

        Returns:
            Dictionary containing extracted context information
        """
        context = {
            "message_count": len(messages),
            "role": self.role,
            "call_count": self.call_count,
            "has_system_message": any(
                isinstance(msg, SystemMessage) for msg in messages
            ),
            "last_message_type": messages[-1].__class__.__name__ if messages else None,
            "conversation_length": sum(
                len(msg.content) for msg in messages if hasattr(msg, "content")
            ),
        }

        # Extract last user message
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                context["last_user_message"] = msg.content
                break
        else:
            context["last_user_message"] = ""

        # Extract system prompt
        for msg in messages:
            if isinstance(msg, SystemMessage):
                context["system_prompt"] = msg.content
                break
        else:
            context["system_prompt"] = ""

        # Apply custom context extractors
        for extractor_name, extractor_func in self.context_extractors.items():
            try:
                context[extractor_name] = extractor_func(messages)
            except Exception as e:
                logger.warning(f"Context extractor {extractor_name} failed: {e}")
                context[extractor_name] = None

        return context

    def get_response_content(self, context: Dict[str, Any]) -> str:
        """Generate response content based on context.

        Args:
            context: Extracted context information

        Returns:
            Response content string
        """
        # Try to match context to response patterns
        for pattern_name, template in self.response_patterns.items():
            if self.context_matches_pattern(context, pattern_name):
                return self.format_response_template(template, context)

        # Fallback to default response
        return self.get_default_response(context)

    def context_matches_pattern(
        self, context: Dict[str, Any], pattern_name: str
    ) -> bool:
        """Check if context matches a specific response pattern.

        Args:
            context: Extracted context information
            pattern_name: Name of the pattern to check

        Returns:
            True if context matches the pattern
        """
        # Default implementation - subclasses should override
        return pattern_name == "default"

    def format_response_template(self, template: str, context: Dict[str, Any]) -> str:
        """Format a response template with context values.

        Args:
            template: Response template string
            context: Context values for formatting

        Returns:
            Formatted response string
        """
        try:
            return template.format(**context)
        except KeyError as e:
            logger.warning(f"Missing context key {e} for template formatting")
            return template

    def get_default_response(self, context: Dict[str, Any]) -> str:
        """Get default response when no pattern matches.

        Args:
            context: Extracted context information

        Returns:
            Default response string
        """
        return f"I am {self.role} agent responding to your message. [Call #{context['call_count']}]"

    def check_interrupt_trigger(
        self, context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Check if current context should trigger a GraphInterrupt.

        Args:
            context: Extracted context information

        Returns:
            Interrupt information if interrupt should be triggered, None otherwise
        """
        for trigger_name, trigger_config in self.interrupt_triggers.items():
            if self.should_trigger_interrupt(context, trigger_name, trigger_config):
                return {
                    "type": trigger_config["type"],
                    "name": trigger_name,
                    "data": trigger_config.get("data", {}),
                    "context": context,
                }
        return None

    def should_trigger_interrupt(
        self, context: Dict[str, Any], trigger_name: str, trigger_config: Dict[str, Any]
    ) -> bool:
        """Check if specific interrupt trigger should fire.

        Args:
            context: Extracted context information
            trigger_name: Name of the trigger
            trigger_config: Trigger configuration

        Returns:
            True if interrupt should be triggered
        """
        # Default implementation - subclasses should override
        return False

    def trigger_interrupt(self, interrupt_info: Dict[str, Any]) -> None:
        """Trigger a GraphInterrupt for testing user input scenarios.

        Args:
            interrupt_info: Information about the interrupt to trigger
        """
        # This is handled by the specific LLM implementations
        # The base class just logs the attempt
        logger.info(f"[{self.role}] Would trigger interrupt: {interrupt_info['type']}")

    def get_response_metadata(
        self, context: Dict[str, Any], response_content: str
    ) -> Dict[str, Any]:
        """Generate realistic response metadata.

        Args:
            context: Extracted context information
            response_content: Generated response content

        Returns:
            Dictionary containing response metadata
        """
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "role": self.role,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "token_usage": {
                "prompt_tokens": max(10, len(str(context)) // 4),
                "completion_tokens": max(5, len(response_content) // 4),
                "total_tokens": max(
                    15, (len(str(context)) + len(response_content)) // 4
                ),
            },
            "finish_reason": "stop",
            "call_count": self.call_count,
            "deterministic": True,
            "timestamp": datetime.now().isoformat(),
        }

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async version of _generate."""
        return self._generate(messages, stop, run_manager, **kwargs)

    def get_interaction_summary(self) -> Dict[str, Any]:
        """Get summary of all interactions with this LLM.

        Returns:
            Dictionary containing interaction statistics
        """
        return {
            "total_calls": self.call_count,
            "role": self.role,
            "model_name": self.model_name,
            "interactions": len(self.interaction_history),
            "interrupts_triggered": sum(
                1
                for interaction in self.interaction_history
                if interaction["interrupt_triggered"]
            ),
            "average_response_length": sum(
                len(interaction["response"]) for interaction in self.interaction_history
            )
            // max(1, len(self.interaction_history)),
            "first_call": (
                self.interaction_history[0]["timestamp"]
                if self.interaction_history
                else None
            ),
            "last_call": (
                self.interaction_history[-1]["timestamp"]
                if self.interaction_history
                else None
            ),
        }

    def reset_state(self) -> None:
        """Reset the LLM state for new test runs."""
        self.call_count = 0
        self.interaction_history = []
        logger.debug(f"Reset state for {self.__class__.__name__}")
