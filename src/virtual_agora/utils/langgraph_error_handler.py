"""LangGraph-specific error handling patterns for Virtual Agora.

This module provides enhanced error handling using LangGraph patterns including
RetryPolicy, ValidationNode, and fallback chain builders.
"""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    Union,
    TypeVar,
)
from functools import wraps
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.prompts import ChatPromptTemplate

try:
    from langgraph.pregel import RetryPolicy
    from langgraph.prebuilt import ValidationNode

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    RetryPolicy = None
    ValidationNode = None

from virtual_agora.utils.logging import get_logger
from virtual_agora.utils.exceptions import (
    VirtualAgoraError,
    ProviderError,
    TimeoutError,
    NetworkTransientError,
    RecoverableError,
    ValidationError,
)


logger = get_logger(__name__)
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class LangGraphErrorHandler:
    """Enhanced error handler using LangGraph patterns."""

    @staticmethod
    def create_retry_policy(
        max_attempts: int = 3,
        retry_on: Optional[Callable[[Exception], bool]] = None,
    ) -> Optional["RetryPolicy"]:
        """Create a LangGraph RetryPolicy for provider operations.

        Args:
            max_attempts: Maximum number of retry attempts
            retry_on: Function to determine if error should be retried

        Returns:
            RetryPolicy instance or None if LangGraph not available
        """
        if not LANGGRAPH_AVAILABLE or RetryPolicy is None:
            logger.warning("LangGraph not available, RetryPolicy cannot be created")
            return None

        if retry_on is None:
            # Default retry on common provider errors
            def default_retry_on(error: Exception) -> bool:
                return isinstance(
                    error,
                    (
                        ProviderError,
                        TimeoutError,
                        NetworkTransientError,
                        RecoverableError,
                    ),
                )

            retry_on = default_retry_on

        return RetryPolicy(retry_on=retry_on, max_attempts=max_attempts)

    @staticmethod
    def create_fallback_chain(
        primary_llm: BaseChatModel,
        fallback_llms: List[BaseChatModel],
        error_handler: Optional[Callable[[Exception], Dict[str, Any]]] = None,
    ) -> Runnable:
        """Create a fallback chain with error handling.

        Args:
            primary_llm: Primary LLM to use
            fallback_llms: List of fallback LLMs
            error_handler: Function to convert errors to messages

        Returns:
            Runnable chain with fallbacks
        """
        if error_handler is None:

            def default_error_handler(error: Exception) -> Dict[str, Any]:
                """Default error handler that adds error context."""
                return {
                    "messages": [
                        HumanMessage(
                            content=f"Previous attempt failed with error: {str(error)}. "
                            f"Please try again with a different approach."
                        )
                    ]
                }

            error_handler = default_error_handler

        # Start with primary LLM
        chain = primary_llm

        # Add fallbacks
        if fallback_llms:
            # Create fallback chains with error handling
            from langchain_core.runnables import RunnableLambda

            fallback_chains = []
            for fallback_llm in fallback_llms:
                # Each fallback includes error context injection
                fallback_chain = RunnableLambda(error_handler) | fallback_llm
                fallback_chains.append(fallback_chain)

            # Apply fallbacks to primary chain
            chain = chain.with_fallbacks(
                fallback_chains,
                exceptions_to_handle=(Exception,),
                exception_key="error",
            )

        return chain

    @staticmethod
    def create_self_correcting_chain(
        llm: BaseChatModel,
        max_retries: int = 3,
        include_error_context: bool = True,
    ) -> Runnable:
        """Create a self-correcting chain that retries with error context.

        Args:
            llm: Language model to use
            max_retries: Maximum retry attempts
            include_error_context: Whether to include error in retry prompt

        Returns:
            Self-correcting chain
        """

        def insert_error_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Insert error context into messages."""
            error = inputs.pop("error", None)
            if error and include_error_context:
                messages = inputs.get("messages", [])
                error_msg = HumanMessage(
                    content=f"The previous attempt failed with error: {str(error)}. "
                    f"Please correct your response and try again."
                )
                inputs["messages"] = messages + [error_msg]
            return inputs

        # Create fallback chain that includes error context
        # Use RunnableLambda to make the function compatible with pipe operator
        from langchain_core.runnables import RunnableLambda

        fallback_chain = RunnableLambda(insert_error_context) | llm

        # Apply retries with the fallback
        return llm.with_fallbacks(
            fallbacks=[fallback_chain] * max_retries, exception_key="error"
        )

    @staticmethod
    def with_retry_policy(
        func: F,
        retry_policy: Optional["RetryPolicy"] = None,
    ) -> F:
        """Decorator to apply retry policy to a function.

        Args:
            func: Function to wrap
            retry_policy: RetryPolicy to apply

        Returns:
            Wrapped function
        """
        # Create retry policy if needed
        if retry_policy is None:
            retry_policy = LangGraphErrorHandler.create_retry_policy()

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if retry policy was successfully created
            if retry_policy is None:
                # No retry policy available, call function directly
                return func(*args, **kwargs)

            # Apply retry logic
            last_error = None
            for attempt in range(retry_policy.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if (
                        not retry_policy.retry_on(e)
                        or attempt == retry_policy.max_attempts - 1
                    ):
                        raise
                    logger.warning(
                        f"Attempt {attempt + 1} failed with {type(e).__name__}: {e}. "
                        f"Retrying..."
                    )

            # Should not reach here, but just in case
            if last_error:
                raise last_error
            raise RuntimeError("Unexpected state in retry logic")

        return wrapper

    @staticmethod
    def create_validation_node(
        tools: List[Any],
        format_error: Optional[Callable[[BaseException, Any, Type], str]] = None,
    ) -> Optional["ValidationNode"]:
        """Create a ValidationNode for tool response validation.

        Args:
            tools: List of tools to validate
            format_error: Function to format validation errors

        Returns:
            ValidationNode instance or None if not available
        """
        if not LANGGRAPH_AVAILABLE or ValidationNode is None:
            logger.warning("LangGraph not available, ValidationNode cannot be created")
            return None

        if format_error is None:

            def default_format_error(
                error: BaseException, call: Any, schema: Type
            ) -> str:
                return (
                    f"Validation Error: {repr(error)}\n"
                    f"Please ensure your response matches the expected schema."
                )

            format_error = default_format_error

        # ValidationNode API may vary, try to create it
        try:
            return ValidationNode(
                tools=tools,
                format_error=format_error,
            )
        except TypeError:
            # Try alternative API
            try:
                return ValidationNode(tools)
            except:
                logger.warning("Unable to create ValidationNode with current API")
                return None

    @staticmethod
    def bind_validator_with_retries(
        llm: BaseChatModel,
        tools: List[Any],
        max_attempts: int = 3,
        validator: Optional["ValidationNode"] = None,
    ) -> Runnable:
        """Bind validators with retry logic to an LLM.

        Args:
            llm: Language model to bind
            tools: Tools to validate against
            max_attempts: Maximum validation attempts
            validator: ValidationNode to use (creates default if None)

        Returns:
            LLM with validation and retry logic
        """
        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(tools)

        # Create validator if not provided
        if validator is None:
            validator = LangGraphErrorHandler.create_validation_node(tools)

        # If no validator available, return LLM with tools
        if validator is None:
            logger.warning(
                "ValidationNode not available, returning LLM without validation"
            )
            return llm_with_tools

        # Create retry strategy for validation errors
        def validation_error_handler(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Handle validation errors by providing feedback."""
            error = inputs.pop("error", None)
            if error:
                messages = inputs.get("messages", [])
                error_msg = HumanMessage(
                    content=f"Validation failed: {str(error)}. "
                    f"Please correct your tool call and try again."
                )
                inputs["messages"] = messages + [error_msg]
            return inputs

        # Create validation chain with retries
        from langchain_core.runnables import RunnableLambda

        validation_chain = (
            RunnableLambda(validation_error_handler) | llm_with_tools | validator
        )

        # Apply fallbacks for retry
        return llm_with_tools.with_fallbacks(
            fallbacks=[validation_chain] * (max_attempts - 1),
            exception_key="error",
            exceptions_to_handle=(ValidationError,),
        )


def create_provider_error_chain(
    providers: List[BaseChatModel],
    max_retries_per_provider: int = 3,
) -> Runnable:
    """Create an error-resilient chain across multiple providers.

    This function creates a chain that automatically falls back to different
    providers when errors occur, with retry logic for each provider.

    Args:
        providers: List of provider instances
        max_retries_per_provider: Max retries for each provider

    Returns:
        Error-resilient provider chain
    """
    if not providers:
        raise ValueError("At least one provider must be specified")

    handler = LangGraphErrorHandler()

    # Create self-correcting chain for primary provider
    primary = handler.create_self_correcting_chain(
        providers[0], max_retries=max_retries_per_provider
    )

    # Create fallback providers if available
    if len(providers) > 1:
        fallbacks = []
        for provider in providers[1:]:
            fallback = handler.create_self_correcting_chain(
                provider, max_retries=max_retries_per_provider
            )
            fallbacks.append(fallback)

        # Create complete fallback chain
        return handler.create_fallback_chain(primary, fallbacks)

    return primary


def with_langgraph_error_handling(
    func: F,
    max_attempts: int = 3,
    retry_on_errors: Optional[List[Type[Exception]]] = None,
) -> F:
    """Decorator to add LangGraph error handling to a function.

    Args:
        func: Function to wrap
        max_attempts: Maximum retry attempts
        retry_on_errors: List of error types to retry on

    Returns:
        Wrapped function with error handling
    """
    handler = LangGraphErrorHandler()

    # Create retry policy
    retry_policy = handler.create_retry_policy(
        max_attempts=max_attempts,
        retry_on=lambda e: (
            any(isinstance(e, err_type) for err_type in retry_on_errors)
            if retry_on_errors
            else None
        ),
    )

    # Apply retry policy
    return handler.with_retry_policy(func, retry_policy)
