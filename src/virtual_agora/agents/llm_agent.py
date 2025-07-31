"""LLM Agent wrapper for Virtual Agora.

This module provides a wrapper around LangChain chat models to add
Virtual Agora-specific functionality for discussion agents.
"""

from typing import (
    Optional,
    List,
    Dict,
    Any,
    Union,
    AsyncIterator,
    Iterator,
    Annotated,
    Sequence,
)
from datetime import datetime
import uuid
import asyncio
from threading import Lock

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph.message import add_messages
from langgraph.types import StreamWriter
from langgraph.prebuilt import ToolNode

from virtual_agora.utils.logging import get_logger
from virtual_agora.utils.document_context import load_context_documents
from virtual_agora.state.schema import (
    AgentInfo,
    Message,
    VirtualAgoraState,
    MessagesState,
)

# Import LangGraph error handling if available
try:
    from virtual_agora.utils.langgraph_error_handler import (
        LangGraphErrorHandler,
        with_langgraph_error_handling,
    )

    LANGGRAPH_ERROR_HANDLING = True
except ImportError:
    LANGGRAPH_ERROR_HANDLING = False
    LangGraphErrorHandler = None
    with_langgraph_error_handling = None

# Import LangGraph RetryPolicy if available
try:
    from langgraph.pregel import RetryPolicy
except ImportError:
    RetryPolicy = None


logger = get_logger(__name__)


class LLMAgent:
    """Wrapper for LangChain chat models with Virtual Agora functionality.

    This class wraps a LangChain chat model and adds:
    - Agent identity management
    - Message formatting for discussions
    - Integration with Virtual Agora state
    - Logging and monitoring
    - Direct usage as LangGraph StateGraph node
    - Thread-safe operation for concurrent execution
    """

    def __init__(
        self,
        agent_id: str,
        llm: BaseChatModel,
        role: str = "participant",
        system_prompt: Optional[str] = None,
        enable_error_handling: bool = True,
        max_retries: int = 3,
        fallback_llm: Optional[BaseChatModel] = None,
        tools: Optional[Sequence[BaseTool]] = None,
    ):
        """Initialize the LLM agent.

        Args:
            agent_id: Unique identifier for the agent
            llm: LangChain chat model instance
            role: Agent role (moderator or participant)
            system_prompt: Optional system prompt for the agent
            enable_error_handling: Whether to enable enhanced error handling
            max_retries: Maximum number of retries for failed operations
            fallback_llm: Optional fallback LLM for error recovery
            tools: Optional list of tools the agent can use
        """
        self.agent_id = agent_id
        self.llm = llm
        self.role = role
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.message_count = 0
        self.created_at = datetime.now()
        self.enable_error_handling = enable_error_handling and LANGGRAPH_ERROR_HANDLING
        self.max_retries = max_retries
        self.fallback_llm = fallback_llm
        self.tools = list(tools) if tools else []
        self._tool_bound_llm = None
        self._tool_node = None

        # Extract model info from LLM
        self.model = getattr(llm, "model_name", "unknown")
        self.provider = self._extract_provider_name()

        # Thread safety
        self._count_lock = Lock()

        # Configure error handling if enabled
        if self.enable_error_handling:
            self._configure_error_handling()

        # Bind tools if provided
        if self.tools:
            self._bind_tools()

        logger.info(
            f"Initialized agent {agent_id} with role={role}, "
            f"model={self.model}, provider={self.provider}, "
            f"error_handling={self.enable_error_handling}, "
            f"tools={len(self.tools)}"
        )

    def _extract_provider_name(self) -> str:
        """Extract provider name from LLM class."""
        class_name = self.llm.__class__.__name__
        if "OpenAI" in class_name:
            return "openai"
        elif "Anthropic" in class_name:
            return "anthropic"
        elif "Google" in class_name or "Gemini" in class_name:
            return "google"
        else:
            return "unknown"

    def _configure_error_handling(self) -> None:
        """Configure enhanced error handling for the agent."""
        if not LANGGRAPH_ERROR_HANDLING or LangGraphErrorHandler is None:
            logger.warning(
                f"LangGraph error handling not available for agent {self.agent_id}"
            )
            return

        handler = LangGraphErrorHandler()

        # Create self-correcting chain for the main LLM
        self.llm = handler.create_self_correcting_chain(
            self.llm, max_retries=self.max_retries, include_error_context=True
        )

        # Add fallback if provided
        if self.fallback_llm:
            fallback_chain = handler.create_self_correcting_chain(
                self.fallback_llm,
                max_retries=self.max_retries,
                include_error_context=True,
            )
            self.llm = handler.create_fallback_chain(self.llm, [fallback_chain])

        logger.info(f"Configured error handling for agent {self.agent_id}")

    def _bind_tools(self) -> None:
        """Bind tools to the LLM using LangChain's bind_tools pattern."""
        if not self.tools:
            return

        try:
            # Bind tools to the LLM
            self._tool_bound_llm = self.llm.bind_tools(self.tools)

            # Create ToolNode for tool execution
            self._tool_node = ToolNode(self.tools)

            logger.info(f"Bound {len(self.tools)} tools to agent {self.agent_id}")
        except Exception as e:
            logger.warning(
                f"Failed to bind tools for agent {self.agent_id}: {e}. "
                "Agent will function without tools."
            )
            self._tool_bound_llm = None
            self._tool_node = None

    def bind_tools(self, tools: Sequence[BaseTool]) -> None:
        """Bind tools to the agent after initialization.

        Args:
            tools: List of tools to bind to the agent
        """
        self.tools = list(tools)
        self._bind_tools()

    def get_bound_llm(self) -> BaseChatModel:
        """Get the LLM with tools bound if available, otherwise the base LLM.

        Returns:
            The tool-bound LLM if tools are bound, otherwise the base LLM
        """
        return self._tool_bound_llm if self._tool_bound_llm else self.llm

    def get_tool_node(self) -> Optional[ToolNode]:
        """Get the ToolNode for this agent if tools are configured.

        Returns:
            ToolNode instance or None if no tools are configured
        """
        return self._tool_node

    def has_tools(self) -> bool:
        """Check if the agent has tools configured.

        Returns:
            True if tools are configured, False otherwise
        """
        return bool(self.tools and self._tool_bound_llm)

    def get_retry_policy(self) -> Optional["RetryPolicy"]:
        """Get LangGraph RetryPolicy for this agent.

        Returns:
            RetryPolicy instance or None if not available
        """
        if not self.enable_error_handling or RetryPolicy is None:
            return None

        from virtual_agora.utils.exceptions import (
            ProviderError,
            TimeoutError,
            NetworkTransientError,
        )

        return RetryPolicy(
            retry_on=lambda e: isinstance(
                e, (ProviderError, TimeoutError, NetworkTransientError)
            ),
            max_attempts=self.max_retries,
        )

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt based on role."""
        if self.role == "moderator":
            return (
                "You are the impartial Moderator of 'Virtual Agora.' "
                "Your role is NOT to have an opinion on the topic. "
                "Your sole responsibilities are process-oriented: "
                "1. Facilitate the creation of a discussion agenda by requesting proposals and tallying votes from agents. "
                "2. Announce the current topic and turn order. "
                "3. Ensure all agents' comments are relevant to the current sub-topic. "
                "4. Summarize discussion rounds. "
                "5. Conduct polls to decide when a topic is finished. "
                "You must communicate clearly. When a structured output like JSON is required, "
                "you must adhere to it strictly."
            )
        else:
            return (
                "You are a thoughtful participant in a democratic assembly engaged in structured deliberation. "
                "This is not just a discussion - it's a formal democratic process where your voice matters. "
                "ASSEMBLY CONTEXT: "
                "You are one of several assembly members deliberating on important topics through democratic discourse. "
                "Other members will share their perspectives, and you should listen carefully to their viewpoints. "
                "This is a collaborative process aimed at reaching shared understanding through respectful debate. "
                "YOUR APPROACH: "
                "- Be strong in your convictions and opinions, even when others disagree with you "
                "- Your goal is collaborative discussion that leads to deeper understanding for all participants "
                "- Build upon, challenge, or expand on points made by previous speakers "
                "- Maintain respectful but firm discourse as befits a democratic assembly "
                "- Provide well-reasoned, substantive contributions that advance the deliberation "
                "INSTRUCTIONS: "
                "You will receive the current topic and messages from fellow assembly members who spoke before you. "
                "Listen to their perspectives, then contribute your own thoughtful response that engages with their points. "
                "Stay strictly focused on the topic provided by the Moderator while fostering productive democratic dialogue. "
                "Be prepared to propose discussion topics, vote on agendas, "
                "and vote on when to conclude a topic when asked."
            )

    def get_agent_info(self) -> AgentInfo:
        """Get agent information for state tracking.

        Returns:
            AgentInfo dictionary
        """
        return AgentInfo(
            id=self.agent_id,
            model=self.model,
            provider=self.provider,
            role=self.role,
            message_count=self.message_count,
            created_at=self.created_at,
        )

    def format_messages(
        self,
        prompt: str,
        context_messages: Optional[List[Message]] = None,
        include_system: bool = True,
    ) -> List[BaseMessage]:
        """Format messages for the LLM.

        Args:
            prompt: The current prompt/question
            context_messages: Previous messages for context
            include_system: Whether to include system prompt

        Returns:
            List of formatted messages
        """
        messages: List[BaseMessage] = []

        # Add system prompt
        if include_system and self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))

        # Add context messages
        if context_messages:
            for msg in context_messages:
                if msg["speaker_role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    # Use AIMessage for all agent messages
                    messages.append(AIMessage(content=msg["content"]))

        # Load document context and inject before current prompt
        document_context = load_context_documents()
        enhanced_prompt = prompt
        if document_context:
            enhanced_prompt = f"{document_context}\n{prompt}"

        # Add current prompt (with document context if available)
        messages.append(HumanMessage(content=enhanced_prompt))

        return messages

    def generate_response(
        self,
        prompt: str,
        context_messages: Optional[List[Message]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate a response to a prompt.

        Args:
            prompt: The prompt to respond to
            context_messages: Previous messages for context
            temperature: Override temperature for this call
            max_tokens: Override max_tokens for this call
            **kwargs: Additional parameters for the LLM

        Returns:
            Generated response text
        """
        # Format messages
        messages = self.format_messages(prompt, context_messages)

        # Prepare kwargs
        llm_kwargs = {}
        if temperature is not None:
            llm_kwargs["temperature"] = temperature
        if max_tokens is not None:
            llm_kwargs["max_tokens"] = max_tokens
        llm_kwargs.update(kwargs)

        # Log the prompt
        logger.debug(f"Agent {self.agent_id} generating response to: {prompt[:100]}...")

        # Generate response with error handling
        try:
            # Use tool-bound LLM if available
            llm_to_use = self.get_bound_llm()

            if llm_kwargs:
                # Use bind to set parameters
                response = llm_to_use.bind(**llm_kwargs).invoke(messages)
            else:
                response = llm_to_use.invoke(messages)

            # Extract text content
            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)

            if not isinstance(response_text, str):
                response_text = str(response_text)

            # Check for empty or whitespace-only responses
            if not response_text or not response_text.strip():
                logger.warning(
                    f"Agent {self.agent_id} received empty response from LLM. "
                    f"Response: '{response_text}'"
                )
                # Provide a fallback response indicating the issue
                response_text = (
                    f"[Agent {self.agent_id} received an empty response from the LLM. "
                    "This may indicate an API issue, rate limiting, or content filtering.]"
                )

            # Update message count with thread safety
            with self._count_lock:
                self.message_count += 1

            logger.info(
                f"Agent {self.agent_id} generated response "
                f"({len(response_text)} chars)"
            )

            return response_text

        except Exception as e:
            # Log error with context
            logger.error(
                f"Agent {self.agent_id} failed to generate response: {e}",
                extra={
                    "agent_id": self.agent_id,
                    "provider": self.provider,
                    "model": self.model,
                    "error_type": type(e).__name__,
                    "prompt_length": len(prompt),
                    "context_messages": (
                        len(context_messages) if context_messages else 0
                    ),
                },
            )

            # If error handling is disabled or no fallback, re-raise
            if not self.enable_error_handling or not self.fallback_llm:
                raise

            # If we have enhanced error handling, the error should have been handled
            # by the self-correcting chain. If we're here, it means all retries failed.
            # Convert to a more specific error type if possible
            from virtual_agora.utils.exceptions import ProviderError

            if isinstance(e, ProviderError):
                raise
            else:
                raise ProviderError(
                    f"Failed to generate response after {self.max_retries} attempts",
                    provider=self.provider,
                    details={"original_error": str(e)},
                )

    async def generate_response_async(
        self,
        prompt: str,
        context_messages: Optional[List[Message]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate a response asynchronously.

        Args:
            prompt: The prompt to respond to
            context_messages: Previous messages for context
            temperature: Override temperature for this call
            max_tokens: Override max_tokens for this call
            **kwargs: Additional parameters for the LLM

        Returns:
            Generated response text
        """
        # Format messages
        messages = self.format_messages(prompt, context_messages)

        # Prepare kwargs
        llm_kwargs = {}
        if temperature is not None:
            llm_kwargs["temperature"] = temperature
        if max_tokens is not None:
            llm_kwargs["max_tokens"] = max_tokens
        llm_kwargs.update(kwargs)

        logger.debug(
            f"Agent {self.agent_id} generating async response to: {prompt[:100]}..."
        )

        try:
            # Use tool-bound LLM if available
            llm_to_use = self.get_bound_llm()

            if llm_kwargs:
                response = await llm_to_use.bind(**llm_kwargs).ainvoke(messages)
            else:
                response = await llm_to_use.ainvoke(messages)

            # Extract text content
            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)

            if not isinstance(response_text, str):
                response_text = str(response_text)

            # Check for empty or whitespace-only responses
            if not response_text or not response_text.strip():
                logger.warning(
                    f"Agent {self.agent_id} received empty async response from LLM. "
                    f"Response: '{response_text}'"
                )
                # Provide a fallback response indicating the issue
                response_text = (
                    f"[Agent {self.agent_id} received an empty response from the LLM. "
                    "This may indicate an API issue, rate limiting, or content filtering.]"
                )

            # Update message count with thread safety
            with self._count_lock:
                self.message_count += 1

            logger.info(
                f"Agent {self.agent_id} generated async response "
                f"({len(response_text)} chars)"
            )

            return response_text

        except Exception as e:
            # Log error with context
            logger.error(
                f"Agent {self.agent_id} failed to generate async response: {e}",
                extra={
                    "agent_id": self.agent_id,
                    "provider": self.provider,
                    "model": self.model,
                    "error_type": type(e).__name__,
                    "prompt_length": len(prompt),
                    "context_messages": (
                        len(context_messages) if context_messages else 0
                    ),
                },
            )

            # If error handling is disabled or no fallback, re-raise
            if not self.enable_error_handling or not self.fallback_llm:
                raise

            # Convert to provider error if needed
            from virtual_agora.utils.exceptions import ProviderError

            if isinstance(e, ProviderError):
                raise
            else:
                raise ProviderError(
                    f"Failed to generate async response after {self.max_retries} attempts",
                    provider=self.provider,
                    details={"original_error": str(e)},
                )

    def stream_response(
        self,
        prompt: str,
        context_messages: Optional[List[Message]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Iterator[str]:
        """Stream a response token by token.

        Args:
            prompt: The prompt to respond to
            context_messages: Previous messages for context
            temperature: Override temperature for this call
            max_tokens: Override max_tokens for this call
            **kwargs: Additional parameters for the LLM

        Yields:
            Response tokens as they are generated
        """
        # Format messages
        messages = self.format_messages(prompt, context_messages)

        # Prepare kwargs
        llm_kwargs = {}
        if temperature is not None:
            llm_kwargs["temperature"] = temperature
        if max_tokens is not None:
            llm_kwargs["max_tokens"] = max_tokens
        llm_kwargs.update(kwargs)

        logger.debug(f"Agent {self.agent_id} streaming response to: {prompt[:100]}...")

        try:
            # Use tool-bound LLM if available
            llm_to_use = self.get_bound_llm()

            if llm_kwargs:
                stream = llm_to_use.bind(**llm_kwargs).stream(messages)
            else:
                stream = llm_to_use.stream(messages)

            full_response = ""
            for chunk in stream:
                if hasattr(chunk, "content"):
                    content = chunk.content
                else:
                    content = str(chunk)

                full_response += content
                yield content

            # Update message count with thread safety
            with self._count_lock:
                self.message_count += 1

            logger.info(
                f"Agent {self.agent_id} streamed response "
                f"({len(full_response)} chars)"
            )

        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to stream response: {e}")
            raise

    async def stream_response_async(
        self,
        prompt: str,
        context_messages: Optional[List[Message]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream a response asynchronously.

        Args:
            prompt: The prompt to respond to
            context_messages: Previous messages for context
            temperature: Override temperature for this call
            max_tokens: Override max_tokens for this call
            **kwargs: Additional parameters for the LLM

        Yields:
            Response tokens as they are generated
        """
        # Format messages
        messages = self.format_messages(prompt, context_messages)

        # Prepare kwargs
        llm_kwargs = {}
        if temperature is not None:
            llm_kwargs["temperature"] = temperature
        if max_tokens is not None:
            llm_kwargs["max_tokens"] = max_tokens
        llm_kwargs.update(kwargs)

        logger.debug(
            f"Agent {self.agent_id} async streaming response to: {prompt[:100]}..."
        )

        try:
            # Use tool-bound LLM if available
            llm_to_use = self.get_bound_llm()

            if llm_kwargs:
                stream = llm_to_use.bind(**llm_kwargs).astream(messages)
            else:
                stream = llm_to_use.astream(messages)

            full_response = ""
            async for chunk in stream:
                if hasattr(chunk, "content"):
                    content = chunk.content
                else:
                    content = str(chunk)

                full_response += content
                yield content

            # Update message count with thread safety
            with self._count_lock:
                self.message_count += 1

            logger.info(
                f"Agent {self.agent_id} async streamed response "
                f"({len(full_response)} chars)"
            )

        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to async stream response: {e}")
            raise

    def create_message(self, content: str, topic: Optional[str] = None) -> Message:
        """Create a Message object for state tracking.

        Args:
            content: Message content
            topic: Current discussion topic

        Returns:
            Message dictionary
        """
        return Message(
            id=str(uuid.uuid4()),
            speaker_id=self.agent_id,
            speaker_role=self.role,
            content=content,
            timestamp=datetime.now(),
            phase=-1,  # Will be set by state manager
            topic=topic,
        )

    # ===== LangGraph StateGraph Integration Methods =====

    def __call__(
        self,
        state: Union[MessagesState, VirtualAgoraState, Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        *,
        writer: Optional[StreamWriter] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make LLMAgent callable as a LangGraph node.

        This method allows the agent to be used directly in a StateGraph:
        ```python
        graph.add_node("agent", agent_instance)
        ```

        Args:
            state: Current graph state (MessagesState or VirtualAgoraState)
            config: Optional LangGraph configuration
            writer: Optional stream writer for custom output
            **kwargs: Additional arguments (e.g., prompt override)

        Returns:
            State updates dictionary
        """
        # Handle different state types
        if isinstance(state, dict):
            # Check for VirtualAgoraState first (it also has messages)
            if "current_phase" in state and "agents" in state:
                # VirtualAgoraState
                return self._handle_virtual_agora_state(state, config, writer, **kwargs)
            elif "messages" in state:
                # MessagesState or similar
                return self._handle_messages_state(state, config, writer, **kwargs)
            else:
                # Generic dict state
                return self._handle_generic_state(state, config, writer, **kwargs)
        else:
            raise TypeError(f"Unsupported state type: {type(state)}")

    async def __acall__(
        self,
        state: Union[MessagesState, VirtualAgoraState, Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        *,
        writer: Optional[StreamWriter] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Async version of __call__ for LangGraph node usage.

        Args:
            state: Current graph state
            config: Optional LangGraph configuration
            writer: Optional stream writer
            **kwargs: Additional arguments

        Returns:
            State updates dictionary
        """
        # Handle different state types asynchronously
        if isinstance(state, dict):
            # Check for VirtualAgoraState first (it also has messages)
            if "current_phase" in state and "agents" in state:
                return await self._handle_virtual_agora_state_async(
                    state, config, writer, **kwargs
                )
            elif "messages" in state:
                return await self._handle_messages_state_async(
                    state, config, writer, **kwargs
                )
            else:
                return await self._handle_generic_state_async(
                    state, config, writer, **kwargs
                )
        else:
            raise TypeError(f"Unsupported state type: {type(state)}")

    def _handle_messages_state(
        self,
        state: Dict[str, Any],
        config: Optional[RunnableConfig],
        writer: Optional[StreamWriter],
        **kwargs,
    ) -> Dict[str, Any]:
        """Handle MessagesState for simple message-based interactions.

        Args:
            state: State with messages
            config: LangGraph config
            writer: Stream writer
            **kwargs: Additional arguments

        Returns:
            Updates with new message
        """
        messages = state.get("messages", [])

        # Check if we need to handle tool calls from the last message
        if (
            messages
            and isinstance(messages[-1], AIMessage)
            and hasattr(messages[-1], "tool_calls")
        ):
            tool_calls = getattr(messages[-1], "tool_calls", [])
            if tool_calls and self._tool_node:
                # Execute tool calls using our ToolNode
                logger.info(
                    f"Agent {self.agent_id} executing {len(tool_calls)} tool calls"
                )
                tool_results = self._tool_node.invoke(state, config)

                # Stream tool results if writer provided
                if writer:
                    writer.write(tool_results)

                return tool_results

        # Get prompt from kwargs or last message
        if "prompt" in kwargs:
            prompt = kwargs["prompt"]
        elif messages and hasattr(messages[-1], "content"):
            prompt = messages[-1].content
        else:
            raise ValueError("No prompt provided and no messages in state")

        # Format messages for context (exclude last if it's the prompt)
        context_messages = []
        if messages and not kwargs.get("prompt"):
            # Convert BaseMessage objects to our Message format
            for msg in messages[:-1]:  # Exclude last message (the prompt)
                if hasattr(msg, "content"):
                    context_messages.append(
                        {
                            "id": str(uuid.uuid4()),
                            "speaker_id": getattr(msg, "name", "unknown"),
                            "speaker_role": (
                                "user"
                                if msg.__class__.__name__ == "HumanMessage"
                                else "assistant"
                            ),
                            "content": msg.content,
                            "timestamp": datetime.now(),
                            "phase": -1,
                            "topic": None,
                        }
                    )

        # Generate response using tool-bound LLM if available
        llm_to_use = self.get_bound_llm()

        # If we have tools, the response might include tool calls
        if self.tools:
            # Use the tool-bound LLM to potentially generate tool calls
            messages_for_llm = self.format_messages(prompt, context_messages)

            # Get LLM kwargs
            llm_kwargs = {}
            if "temperature" in kwargs:
                llm_kwargs["temperature"] = kwargs["temperature"]
            if "max_tokens" in kwargs:
                llm_kwargs["max_tokens"] = kwargs["max_tokens"]

            # Invoke LLM
            if llm_kwargs:
                llm_response = llm_to_use.bind(**llm_kwargs).invoke(messages_for_llm)
            else:
                llm_response = llm_to_use.invoke(messages_for_llm)

            # Check if response contains tool calls
            if isinstance(llm_response, AIMessage) and hasattr(
                llm_response, "tool_calls"
            ):
                tool_calls = getattr(llm_response, "tool_calls", [])
                if tool_calls:
                    # Return the AI message with tool calls
                    ai_message = AIMessage(
                        content=llm_response.content or "",
                        name=self.agent_id,
                        tool_calls=tool_calls,
                    )

                    # Update message count
                    with self._count_lock:
                        self.message_count += 1

                    # Stream if writer provided
                    if writer:
                        writer.write({"messages": [ai_message]})

                    return {"messages": [ai_message]}

            # Extract response text
            response_text = (
                llm_response.content
                if hasattr(llm_response, "content")
                else str(llm_response)
            )
        else:
            # No tools, use regular response generation
            response_text = self.generate_response(
                prompt,
                context_messages,
                temperature=kwargs.get("temperature"),
                max_tokens=kwargs.get("max_tokens"),
            )

        # Create AI message with agent name
        ai_message = AIMessage(content=response_text, name=self.agent_id)

        # Stream if writer is provided
        if writer:
            writer.write({"messages": [ai_message]})

        # Return state update
        return {"messages": [ai_message]}

    async def _handle_messages_state_async(
        self,
        state: Dict[str, Any],
        config: Optional[RunnableConfig],
        writer: Optional[StreamWriter],
        **kwargs,
    ) -> Dict[str, Any]:
        """Async handler for MessagesState.

        Args:
            state: State with messages
            config: LangGraph config
            writer: Stream writer
            **kwargs: Additional arguments

        Returns:
            Updates with new message
        """
        messages = state.get("messages", [])

        # Check if we need to handle tool calls from the last message
        if (
            messages
            and isinstance(messages[-1], AIMessage)
            and hasattr(messages[-1], "tool_calls")
        ):
            tool_calls = getattr(messages[-1], "tool_calls", [])
            if tool_calls and self._tool_node:
                # Execute tool calls using our ToolNode
                logger.info(
                    f"Agent {self.agent_id} executing {len(tool_calls)} tool calls"
                )
                # ToolNode doesn't have async support yet, so use sync version
                tool_results = self._tool_node.invoke(state, config)

                # Stream tool results if writer provided
                if writer:
                    await writer.awrite(tool_results)

                return tool_results

        # Get prompt from kwargs or last message
        if "prompt" in kwargs:
            prompt = kwargs["prompt"]
        elif messages and hasattr(messages[-1], "content"):
            prompt = messages[-1].content
        else:
            raise ValueError("No prompt provided and no messages in state")

        # Format messages for context
        context_messages = []
        if messages and not kwargs.get("prompt"):
            for msg in messages[:-1]:
                if hasattr(msg, "content"):
                    context_messages.append(
                        {
                            "id": str(uuid.uuid4()),
                            "speaker_id": getattr(msg, "name", "unknown"),
                            "speaker_role": (
                                "user"
                                if msg.__class__.__name__ == "HumanMessage"
                                else "assistant"
                            ),
                            "content": msg.content,
                            "timestamp": datetime.now(),
                            "phase": -1,
                            "topic": None,
                        }
                    )

        # Generate response using tool-bound LLM if available
        llm_to_use = self.get_bound_llm()

        # If we have tools, the response might include tool calls
        if self.tools:
            # Use the tool-bound LLM to potentially generate tool calls
            messages_for_llm = self.format_messages(prompt, context_messages)

            # Get LLM kwargs
            llm_kwargs = {}
            if "temperature" in kwargs:
                llm_kwargs["temperature"] = kwargs["temperature"]
            if "max_tokens" in kwargs:
                llm_kwargs["max_tokens"] = kwargs["max_tokens"]

            # Invoke LLM
            if llm_kwargs:
                llm_response = await llm_to_use.bind(**llm_kwargs).ainvoke(
                    messages_for_llm
                )
            else:
                llm_response = await llm_to_use.ainvoke(messages_for_llm)

            # Check if response contains tool calls
            if isinstance(llm_response, AIMessage) and hasattr(
                llm_response, "tool_calls"
            ):
                tool_calls = getattr(llm_response, "tool_calls", [])
                if tool_calls:
                    # Return the AI message with tool calls
                    ai_message = AIMessage(
                        content=llm_response.content or "",
                        name=self.agent_id,
                        tool_calls=tool_calls,
                    )

                    # Update message count
                    with self._count_lock:
                        self.message_count += 1

                    # Stream if writer provided
                    if writer:
                        await writer.awrite({"messages": [ai_message]})

                    return {"messages": [ai_message]}

            # Extract response text
            response_text = (
                llm_response.content
                if hasattr(llm_response, "content")
                else str(llm_response)
            )
        else:
            # No tools, use regular response generation
            response_text = await self.generate_response_async(
                prompt,
                context_messages,
                temperature=kwargs.get("temperature"),
                max_tokens=kwargs.get("max_tokens"),
            )

        # Create AI message
        ai_message = AIMessage(content=response_text, name=self.agent_id)

        # Stream if writer is provided
        if writer:
            await writer.awrite({"messages": [ai_message]})

        return {"messages": [ai_message]}

    def _handle_virtual_agora_state(
        self,
        state: Dict[str, Any],
        config: Optional[RunnableConfig],
        writer: Optional[StreamWriter],
        **kwargs,
    ) -> Dict[str, Any]:
        """Handle VirtualAgoraState for full discussion context.

        Args:
            state: Full Virtual Agora state
            config: LangGraph config
            writer: Stream writer
            **kwargs: Additional arguments

        Returns:
            State updates
        """
        # Extract relevant context from state
        phase = state.get("current_phase", 0)
        topic = state.get("active_topic")
        messages = state.get("messages", [])

        # Get prompt based on phase and context
        prompt = kwargs.get("prompt", "")
        if not prompt:
            if phase == 1:  # Agenda setting
                prompt = "Please propose 3-5 discussion topics for our session."
            elif phase == 2:  # Discussion
                prompt = f"Please share your thoughts on: {topic}"
            elif phase == 3:  # Consensus
                prompt = f"Should we conclude our discussion on '{topic}'? Please vote Yes or No with reasoning."
            else:
                prompt = "Please provide your input."

        # Use context_messages from kwargs (colleague messages) or filter from state as fallback
        context_messages = kwargs.get("context_messages", [])

        # If no context_messages provided via kwargs, fall back to state-based filtering
        if not context_messages:
            context_messages = []
            for msg in messages:
                # Handle both dict messages and BaseMessage objects
                if isinstance(msg, dict):
                    if msg.get("phase") == phase or (
                        topic and msg.get("topic") == topic
                    ):
                        context_messages.append(msg)
                elif hasattr(msg, "additional_kwargs"):
                    # BaseMessage objects might have metadata in additional_kwargs
                    metadata = getattr(msg, "additional_kwargs", {})
                    if metadata.get("phase") == phase or (
                        topic and metadata.get("topic") == topic
                    ):
                        # Convert to dict format for consistency
                        context_messages.append(
                            {
                                "id": getattr(msg, "id", str(uuid.uuid4())),
                                "speaker_id": getattr(msg, "name", "unknown"),
                                "speaker_role": "assistant",
                                "content": msg.content,
                                "timestamp": datetime.now(),
                                "phase": metadata.get("phase", -1),
                                "topic": metadata.get("topic"),
                            }
                        )
        else:
            # Convert HumanMessage objects from discussion to format expected by agent
            formatted_context_messages = []
            for msg in context_messages:
                if isinstance(msg, HumanMessage):
                    formatted_context_messages.append(
                        {
                            "id": str(uuid.uuid4()),
                            "speaker_id": getattr(msg, "name", "unknown"),
                            "speaker_role": "user",  # Colleague messages come as HumanMessage
                            "content": msg.content,
                            "timestamp": datetime.now(),
                            "phase": phase,
                            "topic": topic,
                        }
                    )
                else:
                    # Keep existing format if already a dict
                    formatted_context_messages.append(msg)
            context_messages = formatted_context_messages

        # Generate response
        response = self.generate_response(
            prompt, context_messages[-10:]
        )  # Last 10 messages

        # Create message for Virtual Agora state tracking
        message = self.create_message(response, topic)

        # Create AI message for LangGraph
        ai_message = AIMessage(content=response, name=self.agent_id)

        # Update message count with thread safety
        with self._count_lock:
            self.message_count += 1

        # Prepare state updates - use AI message for LangGraph compatibility
        updates = {
            "messages": [ai_message],
            "messages_by_agent": {
                self.agent_id: state.get("messages_by_agent", {}).get(self.agent_id, 0)
                + 1
            },
        }

        if topic:
            updates["messages_by_topic"] = {
                topic: state.get("messages_by_topic", {}).get(topic, 0) + 1
            }

        # Stream if writer provided
        if writer:
            writer.write({"agent_response": response})

        return updates

    async def _handle_virtual_agora_state_async(
        self,
        state: Dict[str, Any],
        config: Optional[RunnableConfig],
        writer: Optional[StreamWriter],
        **kwargs,
    ) -> Dict[str, Any]:
        """Async handler for VirtualAgoraState.

        Args:
            state: Full Virtual Agora state
            config: LangGraph config
            writer: Stream writer
            **kwargs: Additional arguments

        Returns:
            State updates
        """
        # Extract relevant context
        phase = state.get("current_phase", 0)
        topic = state.get("active_topic")
        messages = state.get("messages", [])

        # Get prompt
        prompt = kwargs.get("prompt", "")
        if not prompt:
            if phase == 1:
                prompt = "Please propose 3-5 discussion topics for our session."
            elif phase == 2:
                prompt = f"Please share your thoughts on: {topic}"
            elif phase == 3:
                prompt = f"Should we conclude our discussion on '{topic}'? Please vote Yes or No with reasoning."
            else:
                prompt = "Please provide your input."

        # Use context_messages from kwargs (colleague messages) or filter from state as fallback
        context_messages = kwargs.get("context_messages", [])

        # If no context_messages provided via kwargs, fall back to state-based filtering
        if not context_messages:
            context_messages = []
            for msg in messages:
                # Handle both dict messages and BaseMessage objects
                if isinstance(msg, dict):
                    if msg.get("phase") == phase or (
                        topic and msg.get("topic") == topic
                    ):
                        context_messages.append(msg)
                elif hasattr(msg, "additional_kwargs"):
                    # BaseMessage objects might have metadata in additional_kwargs
                    metadata = getattr(msg, "additional_kwargs", {})
                    if metadata.get("phase") == phase or (
                        topic and metadata.get("topic") == topic
                    ):
                        # Convert to dict format for consistency
                        context_messages.append(
                            {
                                "id": getattr(msg, "id", str(uuid.uuid4())),
                                "speaker_id": getattr(msg, "name", "unknown"),
                                "speaker_role": "assistant",
                                "content": msg.content,
                                "timestamp": datetime.now(),
                                "phase": metadata.get("phase", -1),
                                "topic": metadata.get("topic"),
                            }
                        )
        else:
            # Convert HumanMessage objects from discussion to format expected by agent
            formatted_context_messages = []
            for msg in context_messages:
                if isinstance(msg, HumanMessage):
                    formatted_context_messages.append(
                        {
                            "id": str(uuid.uuid4()),
                            "speaker_id": getattr(msg, "name", "unknown"),
                            "speaker_role": "user",  # Colleague messages come as HumanMessage
                            "content": msg.content,
                            "timestamp": datetime.now(),
                            "phase": phase,
                            "topic": topic,
                        }
                    )
                else:
                    # Keep existing format if already a dict
                    formatted_context_messages.append(msg)
            context_messages = formatted_context_messages

        response = await self.generate_response_async(prompt, context_messages[-10:])

        # Create message for Virtual Agora state tracking
        message = self.create_message(response, topic)

        # Create AI message for LangGraph
        ai_message = AIMessage(content=response, name=self.agent_id)

        with self._count_lock:
            self.message_count += 1

        # Prepare updates - use AI message for LangGraph compatibility
        updates = {
            "messages": [ai_message],
            "messages_by_agent": {
                self.agent_id: state.get("messages_by_agent", {}).get(self.agent_id, 0)
                + 1
            },
        }

        if topic:
            updates["messages_by_topic"] = {
                topic: state.get("messages_by_topic", {}).get(topic, 0) + 1
            }

        # Stream if writer provided
        if writer:
            await writer.awrite({"agent_response": response})

        return updates

    def _handle_generic_state(
        self,
        state: Dict[str, Any],
        config: Optional[RunnableConfig],
        writer: Optional[StreamWriter],
        **kwargs,
    ) -> Dict[str, Any]:
        """Handle generic dictionary state.

        Args:
            state: Generic state dict
            config: LangGraph config
            writer: Stream writer
            **kwargs: Additional arguments

        Returns:
            State updates
        """
        # Try to extract a prompt
        prompt = kwargs.get("prompt", state.get("prompt", state.get("question", "")))
        if not prompt:
            raise ValueError("No prompt found in kwargs or state")

        # Generate response
        response = self.generate_response(prompt)

        # Return generic update
        return {"response": response, "agent_id": self.agent_id}

    async def _handle_generic_state_async(
        self,
        state: Dict[str, Any],
        config: Optional[RunnableConfig],
        writer: Optional[StreamWriter],
        **kwargs,
    ) -> Dict[str, Any]:
        """Async handler for generic state.

        Args:
            state: Generic state dict
            config: LangGraph config
            writer: Stream writer
            **kwargs: Additional arguments

        Returns:
            State updates
        """
        prompt = kwargs.get("prompt", state.get("prompt", state.get("question", "")))
        if not prompt:
            raise ValueError("No prompt found in kwargs or state")

        response = await self.generate_response_async(prompt)

        return {"response": response, "agent_id": self.agent_id}

    def stream_in_graph(
        self,
        state: Union[MessagesState, Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        stream_mode: str = "messages",
        **kwargs,
    ) -> Iterator[Union[str, Dict[str, Any]]]:
        """Stream responses within a LangGraph context.

        Args:
            state: Current state
            config: LangGraph config
            stream_mode: How to stream ('messages', 'updates', 'values')
            **kwargs: Additional arguments

        Yields:
            Streamed content based on mode
        """
        messages = state.get("messages", [])

        # Check if we need to handle tool calls
        if (
            messages
            and isinstance(messages[-1], AIMessage)
            and hasattr(messages[-1], "tool_calls")
        ):
            tool_calls = getattr(messages[-1], "tool_calls", [])
            if tool_calls and self._tool_node:
                # Execute tool calls
                tool_results = self._tool_node.invoke(state, config)

                # Yield tool results based on stream mode
                if stream_mode == "messages":
                    # Yield tool messages as text
                    for msg in tool_results.get("messages", []):
                        if isinstance(msg, ToolMessage):
                            yield f"Tool Result: {msg.content}"
                elif stream_mode == "updates":
                    yield tool_results
                elif stream_mode == "values":
                    current_messages = list(messages) + tool_results.get("messages", [])
                    yield {"messages": current_messages}
                return

        # Extract prompt
        if "prompt" in kwargs:
            prompt = kwargs["prompt"]
        elif messages and hasattr(messages[-1], "content"):
            prompt = messages[-1].content
        else:
            raise ValueError("No prompt available")

        # Prepare context
        context_messages = []
        if messages and not kwargs.get("prompt"):
            for msg in messages[:-1]:
                if hasattr(msg, "content"):
                    context_messages.append(
                        {
                            "id": str(uuid.uuid4()),
                            "speaker_id": getattr(msg, "name", "unknown"),
                            "speaker_role": (
                                "user"
                                if msg.__class__.__name__ == "HumanMessage"
                                else "assistant"
                            ),
                            "content": msg.content,
                            "timestamp": datetime.now(),
                            "phase": -1,
                            "topic": None,
                        }
                    )

        # If we have tools, check if we should stream tool calls
        if self.tools:
            # Format messages for LLM
            messages_for_llm = self.format_messages(prompt, context_messages)
            llm_to_use = self.get_bound_llm()

            # Stream from LLM
            full_response = ""
            tool_calls = []

            for chunk in llm_to_use.stream(messages_for_llm):
                # Check if chunk contains tool calls
                if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    tool_calls.extend(chunk.tool_calls)

                # Extract content
                content = chunk.content if hasattr(chunk, "content") else ""
                full_response += content

                if content:  # Only yield if there's content
                    if stream_mode == "messages":
                        yield content
                    elif stream_mode == "updates":
                        yield {
                            "messages": [AIMessage(content=content, name=self.agent_id)]
                        }
                    elif stream_mode == "values":
                        # Build current message
                        current_ai_msg = AIMessage(
                            content=full_response,
                            name=self.agent_id,
                            tool_calls=tool_calls if tool_calls else None,
                        )
                        current_messages = list(messages) + [current_ai_msg]
                        yield {"messages": current_messages}

            # Update message count
            with self._count_lock:
                self.message_count += 1
        else:
            # No tools, use regular streaming
            full_response = ""
            for chunk in self.stream_response(prompt, context_messages):
                full_response += chunk

                if stream_mode == "messages":
                    yield chunk
                elif stream_mode == "updates":
                    yield {"messages": [AIMessage(content=chunk, name=self.agent_id)]}
                elif stream_mode == "values":
                    current_messages = list(messages) + [
                        AIMessage(content=full_response, name=self.agent_id)
                    ]
                    yield {"messages": current_messages}

    async def stream_in_graph_async(
        self,
        state: Union[MessagesState, Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        stream_mode: str = "messages",
        **kwargs,
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Async streaming within LangGraph context.

        Args:
            state: Current state
            config: LangGraph config
            stream_mode: How to stream
            **kwargs: Additional arguments

        Yields:
            Streamed content based on mode
        """
        messages = state.get("messages", [])

        # Check if we need to handle tool calls
        if (
            messages
            and isinstance(messages[-1], AIMessage)
            and hasattr(messages[-1], "tool_calls")
        ):
            tool_calls = getattr(messages[-1], "tool_calls", [])
            if tool_calls and self._tool_node:
                # Execute tool calls (sync for now as ToolNode doesn't have async)
                tool_results = self._tool_node.invoke(state, config)

                # Yield tool results based on stream mode
                if stream_mode == "messages":
                    for msg in tool_results.get("messages", []):
                        if isinstance(msg, ToolMessage):
                            yield f"Tool Result: {msg.content}"
                elif stream_mode == "updates":
                    yield tool_results
                elif stream_mode == "values":
                    current_messages = list(messages) + tool_results.get("messages", [])
                    yield {"messages": current_messages}
                return

        # Extract prompt
        if "prompt" in kwargs:
            prompt = kwargs["prompt"]
        elif messages and hasattr(messages[-1], "content"):
            prompt = messages[-1].content
        else:
            raise ValueError("No prompt available")

        # Prepare context
        context_messages = []
        if messages and not kwargs.get("prompt"):
            for msg in messages[:-1]:
                if hasattr(msg, "content"):
                    context_messages.append(
                        {
                            "id": str(uuid.uuid4()),
                            "speaker_id": getattr(msg, "name", "unknown"),
                            "speaker_role": (
                                "user"
                                if msg.__class__.__name__ == "HumanMessage"
                                else "assistant"
                            ),
                            "content": msg.content,
                            "timestamp": datetime.now(),
                            "phase": -1,
                            "topic": None,
                        }
                    )

        # If we have tools, check if we should stream tool calls
        if self.tools:
            # Format messages for LLM
            messages_for_llm = self.format_messages(prompt, context_messages)
            llm_to_use = self.get_bound_llm()

            # Stream from LLM
            full_response = ""
            tool_calls = []

            async for chunk in llm_to_use.astream(messages_for_llm):
                # Check if chunk contains tool calls
                if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    tool_calls.extend(chunk.tool_calls)

                # Extract content
                content = chunk.content if hasattr(chunk, "content") else ""
                full_response += content

                if content:  # Only yield if there's content
                    if stream_mode == "messages":
                        yield content
                    elif stream_mode == "updates":
                        yield {
                            "messages": [AIMessage(content=content, name=self.agent_id)]
                        }
                    elif stream_mode == "values":
                        # Build current message
                        current_ai_msg = AIMessage(
                            content=full_response,
                            name=self.agent_id,
                            tool_calls=tool_calls if tool_calls else None,
                        )
                        current_messages = list(messages) + [current_ai_msg]
                        yield {"messages": current_messages}

            # Update message count
            with self._count_lock:
                self.message_count += 1
        else:
            # No tools, use regular streaming
            full_response = ""
            async for chunk in self.stream_response_async(prompt, context_messages):
                full_response += chunk

                if stream_mode == "messages":
                    yield chunk
                elif stream_mode == "updates":
                    yield {"messages": [AIMessage(content=chunk, name=self.agent_id)]}
                elif stream_mode == "values":
                    current_messages = list(messages) + [
                        AIMessage(content=full_response, name=self.agent_id)
                    ]
                    yield {"messages": current_messages}

    def as_langgraph_node(
        self, name: Optional[str] = None, retry_policy: Optional["RetryPolicy"] = None
    ) -> Dict[str, Any]:
        """Convert this agent to a LangGraph-compatible node configuration.

        This method returns a configuration that can be used with LangGraph's
        add_node() method, including error handling and retry configuration.

        Args:
            name: Node name (defaults to agent_id)
            retry_policy: Optional RetryPolicy (uses agent's default if None)

        Returns:
            Node configuration dict for LangGraph

        Example:
            ```python
            agent = LLMAgent("agent1", llm)
            graph.add_node(**agent.as_langgraph_node())
            ```
        """
        node_name = name or self.agent_id

        # Use agent's retry policy if not provided
        if retry_policy is None:
            retry_policy = self.get_retry_policy()

        # Create node configuration
        node_config = {
            "name": node_name,
            "func": self,  # The agent itself is callable
        }

        # Add retry policy if available
        if retry_policy is not None:
            node_config["retry_policy"] = retry_policy

        # Add metadata for debugging
        node_config["metadata"] = {
            "agent_id": self.agent_id,
            "role": self.role,
            "model": self.model,
            "provider": self.provider,
            "error_handling": self.enable_error_handling,
        }

        logger.debug(f"Created LangGraph node config for agent {self.agent_id}")
        return node_config

    @classmethod
    def create_with_fallback(
        cls,
        agent_id: str,
        primary_llm: BaseChatModel,
        fallback_llms: List[BaseChatModel],
        role: str = "participant",
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
    ) -> "LLMAgent":
        """Create an agent with built-in fallback LLMs.

        This factory method creates an agent that automatically falls back
        to alternative LLMs when the primary fails, using LangGraph patterns.

        Args:
            agent_id: Unique identifier for the agent
            primary_llm: Primary LLM to use
            fallback_llms: List of fallback LLMs in order of preference
            role: Agent role (moderator or participant)
            system_prompt: Optional system prompt
            max_retries: Max retries per LLM

        Returns:
            LLMAgent instance with fallback configuration

        Example:
            ```python
            agent = LLMAgent.create_with_fallback(
                "agent1",
                primary_llm=openai_llm,
                fallback_llms=[anthropic_llm, google_llm],
                max_retries=2
            )
            ```
        """
        # Import here to avoid circular dependency
        from virtual_agora.utils.langgraph_error_handler import (
            LangGraphErrorHandler,
            create_provider_error_chain,
        )

        # Create error-resilient chain with all providers
        all_providers = [primary_llm] + fallback_llms
        resilient_llm = create_provider_error_chain(
            all_providers, max_retries_per_provider=max_retries
        )

        # Create agent with the resilient LLM
        # Note: We don't pass fallback_llm here since the chain already handles it
        agent = cls(
            agent_id=agent_id,
            llm=resilient_llm,
            role=role,
            system_prompt=system_prompt,
            enable_error_handling=False,  # Already handled by the chain
            max_retries=max_retries,
        )

        # Store fallback info for debugging
        agent._fallback_llms = fallback_llms
        agent._fallback_configured = True

        logger.info(f"Created agent {agent_id} with {len(fallback_llms)} fallback LLMs")

        return agent

    @classmethod
    def create_with_tools(
        cls,
        agent_id: str,
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        role: str = "participant",
        system_prompt: Optional[str] = None,
        enable_error_handling: bool = True,
        max_retries: int = 3,
        fallback_llm: Optional[BaseChatModel] = None,
    ) -> "LLMAgent":
        """Create an agent with tools bound.

        This factory method creates an agent with tools automatically bound
        to the LLM, enabling tool calling capabilities in LangGraph workflows.

        Args:
            agent_id: Unique identifier for the agent
            llm: LangChain chat model instance
            tools: List of tools the agent can use
            role: Agent role (moderator or participant)
            system_prompt: Optional system prompt
            enable_error_handling: Whether to enable enhanced error handling
            max_retries: Maximum retry attempts
            fallback_llm: Optional fallback LLM

        Returns:
            LLMAgent instance with tools bound

        Example:
            ```python
            tools = [search_tool, calculator_tool]
            agent = LLMAgent.create_with_tools(
                "assistant",
                llm=openai_llm,
                tools=tools
            )
            ```
        """
        return cls(
            agent_id=agent_id,
            llm=llm,
            role=role,
            system_prompt=system_prompt,
            enable_error_handling=enable_error_handling,
            max_retries=max_retries,
            fallback_llm=fallback_llm,
            tools=tools,
        )
