"""LLM Agent wrapper for Virtual Agora.

This module provides a wrapper around LangChain chat models to add
Virtual Agora-specific functionality for discussion agents.
"""

from typing import Optional, List, Dict, Any, Union, AsyncIterator, Iterator, Annotated
from datetime import datetime
import uuid
import asyncio
from threading import Lock

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)
from langchain_core.outputs import ChatGeneration
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
from langgraph.types import StreamWriter

from virtual_agora.utils.logging import get_logger
from virtual_agora.state.schema import (
    AgentInfo, 
    Message,
    VirtualAgoraState,
    MessagesState
)


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
        system_prompt: Optional[str] = None
    ):
        """Initialize the LLM agent.
        
        Args:
            agent_id: Unique identifier for the agent
            llm: LangChain chat model instance
            role: Agent role (moderator or participant)
            system_prompt: Optional system prompt for the agent
        """
        self.agent_id = agent_id
        self.llm = llm
        self.role = role
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.message_count = 0
        self.created_at = datetime.now()
        
        # Extract model info from LLM
        self.model = getattr(llm, "model_name", "unknown")
        self.provider = self._extract_provider_name()
        
        # Thread safety
        self._count_lock = Lock()
        
        logger.info(
            f"Initialized agent {agent_id} with role={role}, "
            f"model={self.model}, provider={self.provider}"
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
                "You are a thoughtful participant in a structured discussion. "
                "You will be given a topic and context from previous turns. "
                "Your goal is to provide a well-reasoned, concise comment that builds upon the conversation. "
                "Stay strictly on the topic provided by the Moderator. "
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
            created_at=self.created_at
        )
    
    def format_messages(
        self,
        prompt: str,
        context_messages: Optional[List[Message]] = None,
        include_system: bool = True
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
        
        # Add current prompt
        messages.append(HumanMessage(content=prompt))
        
        return messages
    
    def generate_response(
        self,
        prompt: str,
        context_messages: Optional[List[Message]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
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
        
        # Generate response
        try:
            if llm_kwargs:
                # Use bind to set parameters
                response = self.llm.bind(**llm_kwargs).invoke(messages)
            else:
                response = self.llm.invoke(messages)
            
            # Extract text content
            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Update message count with thread safety
            with self._count_lock:
                self.message_count += 1
            
            logger.info(
                f"Agent {self.agent_id} generated response "
                f"({len(response_text)} chars)"
            )
            
            return response_text
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to generate response: {e}")
            raise
    
    async def generate_response_async(
        self,
        prompt: str,
        context_messages: Optional[List[Message]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
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
        
        logger.debug(f"Agent {self.agent_id} generating async response to: {prompt[:100]}...")
        
        try:
            if llm_kwargs:
                response = await self.llm.bind(**llm_kwargs).ainvoke(messages)
            else:
                response = await self.llm.ainvoke(messages)
            
            # Extract text content
            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Update message count with thread safety
            with self._count_lock:
                self.message_count += 1
            
            logger.info(
                f"Agent {self.agent_id} generated async response "
                f"({len(response_text)} chars)"
            )
            
            return response_text
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to generate async response: {e}")
            raise
    
    def stream_response(
        self,
        prompt: str,
        context_messages: Optional[List[Message]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
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
            if llm_kwargs:
                stream = self.llm.bind(**llm_kwargs).stream(messages)
            else:
                stream = self.llm.stream(messages)
            
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
        **kwargs
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
        
        logger.debug(f"Agent {self.agent_id} async streaming response to: {prompt[:100]}...")
        
        try:
            if llm_kwargs:
                stream = self.llm.bind(**llm_kwargs).astream(messages)
            else:
                stream = self.llm.astream(messages)
            
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
            topic=topic
        )
    
    # ===== LangGraph StateGraph Integration Methods =====
    
    def __call__(
        self,
        state: Union[MessagesState, VirtualAgoraState, Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        *,
        writer: Optional[StreamWriter] = None,
        **kwargs
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
        **kwargs
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
                return await self._handle_virtual_agora_state_async(state, config, writer, **kwargs)
            elif "messages" in state:
                return await self._handle_messages_state_async(state, config, writer, **kwargs)
            else:
                return await self._handle_generic_state_async(state, config, writer, **kwargs)
        else:
            raise TypeError(f"Unsupported state type: {type(state)}")
    
    def _handle_messages_state(
        self,
        state: Dict[str, Any],
        config: Optional[RunnableConfig],
        writer: Optional[StreamWriter],
        **kwargs
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
                    context_messages.append({
                        "id": str(uuid.uuid4()),
                        "speaker_id": getattr(msg, "name", "unknown"),
                        "speaker_role": "user" if msg.__class__.__name__ == "HumanMessage" else "assistant",
                        "content": msg.content,
                        "timestamp": datetime.now(),
                        "phase": -1,
                        "topic": None
                    })
        
        # Generate response
        response = self.generate_response(
            prompt,
            context_messages,
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens")
        )
        
        # Create AI message with agent name
        ai_message = AIMessage(content=response, name=self.agent_id)
        
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
        **kwargs
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
                    context_messages.append({
                        "id": str(uuid.uuid4()),
                        "speaker_id": getattr(msg, "name", "unknown"),
                        "speaker_role": "user" if msg.__class__.__name__ == "HumanMessage" else "assistant",
                        "content": msg.content,
                        "timestamp": datetime.now(),
                        "phase": -1,
                        "topic": None
                    })
        
        # Generate response asynchronously
        response = await self.generate_response_async(
            prompt,
            context_messages,
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens")
        )
        
        # Create AI message
        ai_message = AIMessage(content=response, name=self.agent_id)
        
        # Stream if writer is provided
        if writer:
            await writer.awrite({"messages": [ai_message]})
        
        return {"messages": [ai_message]}
    
    def _handle_virtual_agora_state(
        self,
        state: Dict[str, Any],
        config: Optional[RunnableConfig],
        writer: Optional[StreamWriter],
        **kwargs
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
        
        # Filter messages for current topic/phase
        context_messages = []
        for msg in messages:
            # Handle both dict messages and BaseMessage objects
            if isinstance(msg, dict):
                if msg.get("phase") == phase or (topic and msg.get("topic") == topic):
                    context_messages.append(msg)
            elif hasattr(msg, "additional_kwargs"):
                # BaseMessage objects might have metadata in additional_kwargs
                metadata = getattr(msg, "additional_kwargs", {})
                if metadata.get("phase") == phase or (topic and metadata.get("topic") == topic):
                    # Convert to dict format for consistency
                    context_messages.append({
                        "id": getattr(msg, "id", str(uuid.uuid4())),
                        "speaker_id": getattr(msg, "name", "unknown"),
                        "speaker_role": "assistant",
                        "content": msg.content,
                        "timestamp": datetime.now(),
                        "phase": metadata.get("phase", -1),
                        "topic": metadata.get("topic")
                    })
        
        # Generate response
        response = self.generate_response(prompt, context_messages[-10:])  # Last 10 messages
        
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
                self.agent_id: state.get("messages_by_agent", {}).get(self.agent_id, 0) + 1
            }
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
        **kwargs
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
        
        # Filter messages for current topic/phase
        context_messages = []
        for msg in messages:
            # Handle both dict messages and BaseMessage objects
            if isinstance(msg, dict):
                if msg.get("phase") == phase or (topic and msg.get("topic") == topic):
                    context_messages.append(msg)
            elif hasattr(msg, "additional_kwargs"):
                # BaseMessage objects might have metadata in additional_kwargs
                metadata = getattr(msg, "additional_kwargs", {})
                if metadata.get("phase") == phase or (topic and metadata.get("topic") == topic):
                    # Convert to dict format for consistency
                    context_messages.append({
                        "id": getattr(msg, "id", str(uuid.uuid4())),
                        "speaker_id": getattr(msg, "name", "unknown"),
                        "speaker_role": "assistant",
                        "content": msg.content,
                        "timestamp": datetime.now(),
                        "phase": metadata.get("phase", -1),
                        "topic": metadata.get("topic")
                    })
        
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
                self.agent_id: state.get("messages_by_agent", {}).get(self.agent_id, 0) + 1
            }
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
        **kwargs
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
        **kwargs
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
        **kwargs
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
        # Extract prompt
        messages = state.get("messages", [])
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
                    context_messages.append({
                        "id": str(uuid.uuid4()),
                        "speaker_id": getattr(msg, "name", "unknown"),
                        "speaker_role": "user" if msg.__class__.__name__ == "HumanMessage" else "assistant",
                        "content": msg.content,
                        "timestamp": datetime.now(),
                        "phase": -1,
                        "topic": None
                    })
        
        # Stream response
        full_response = ""
        for chunk in self.stream_response(prompt, context_messages):
            full_response += chunk
            
            if stream_mode == "messages":
                # Yield message chunks
                yield chunk
            elif stream_mode == "updates":
                # Yield state updates
                yield {"messages": [AIMessage(content=chunk, name=self.agent_id)]}
            elif stream_mode == "values":
                # Yield full state
                current_messages = list(messages) + [AIMessage(content=full_response, name=self.agent_id)]
                yield {"messages": current_messages}
    
    async def stream_in_graph_async(
        self,
        state: Union[MessagesState, Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        stream_mode: str = "messages",
        **kwargs
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
        # Extract prompt
        messages = state.get("messages", [])
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
                    context_messages.append({
                        "id": str(uuid.uuid4()),
                        "speaker_id": getattr(msg, "name", "unknown"),
                        "speaker_role": "user" if msg.__class__.__name__ == "HumanMessage" else "assistant",
                        "content": msg.content,
                        "timestamp": datetime.now(),
                        "phase": -1,
                        "topic": None
                    })
        
        # Stream response
        full_response = ""
        async for chunk in self.stream_response_async(prompt, context_messages):
            full_response += chunk
            
            if stream_mode == "messages":
                yield chunk
            elif stream_mode == "updates":
                yield {"messages": [AIMessage(content=chunk, name=self.agent_id)]}
            elif stream_mode == "values":
                current_messages = list(messages) + [AIMessage(content=full_response, name=self.agent_id)]
                yield {"messages": current_messages}

