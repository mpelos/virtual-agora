"""LLM Agent wrapper for Virtual Agora.

This module provides a wrapper around LangChain chat models to add
Virtual Agora-specific functionality for discussion agents.
"""

from typing import Optional, List, Dict, Any, Union, AsyncIterator, Iterator
from datetime import datetime
import uuid

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)
from langchain_core.outputs import ChatGeneration

from virtual_agora.utils.logging import get_logger
from virtual_agora.state.schema import AgentInfo, Message


logger = get_logger(__name__)


class LLMAgent:
    """Wrapper for LangChain chat models with Virtual Agora functionality.
    
    This class wraps a LangChain chat model and adds:
    - Agent identity management
    - Message formatting for discussions
    - Integration with Virtual Agora state
    - Logging and monitoring
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
            
            # Update message count
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
            
            # Update message count
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
            
            # Update message count
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
            
            # Update message count
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

