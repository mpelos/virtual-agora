"""Tests for LLM agent wrapper."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import uuid

from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.state.schema import AgentInfo, Message


class TestLLMAgent:
    """Test LLMAgent class."""
    
    def setup_method(self):
        """Set up test method."""
        # Create mock LLM instance
        self.mock_llm = Mock()
        self.mock_llm.__class__.__name__ = "ChatOpenAI"
        self.mock_llm.model_name = "gpt-4"
        
        # Create agent
        self.agent = LLMAgent(
            agent_id="test-agent",
            llm=self.mock_llm,
            role="participant",
            system_prompt="Test system prompt"
        )
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        assert self.agent.agent_id == "test-agent"
        assert self.agent.llm == self.mock_llm
        assert self.agent.role == "participant"
        assert self.agent.system_prompt == "Test system prompt"
        assert self.agent.message_count == 0
        assert self.agent.model == "gpt-4"
        assert self.agent.provider == "openai"
        assert isinstance(self.agent.created_at, datetime)
    
    def test_provider_extraction(self):
        """Test provider name extraction from LLM class."""
        # Test different provider classes
        test_cases = [
            ("ChatOpenAI", "openai"),
            ("ChatAnthropic", "anthropic"),
            ("ChatGoogleGenerativeAI", "google"),
            ("ChatGemini", "google"),
            ("UnknownChat", "unknown"),
        ]
        
        for class_name, expected_provider in test_cases:
            mock_llm = Mock()
            mock_llm.__class__.__name__ = class_name
            mock_llm.model_name = "test-model"
            
            agent = LLMAgent("test", mock_llm)
            assert agent.provider == expected_provider
    
    def test_default_system_prompts(self):
        """Test default system prompts for different roles."""
        # Test moderator role
        moderator_llm = Mock()
        moderator_llm.__class__.__name__ = "ChatOpenAI"
        moderator_llm.model_name = "gpt-4"
        
        moderator = LLMAgent("mod", moderator_llm, role="moderator")
        assert "impartial Moderator" in moderator.system_prompt
        assert "facilitate the creation" in moderator.system_prompt
        
        # Test participant role
        participant_llm = Mock()
        participant_llm.__class__.__name__ = "ChatOpenAI"
        participant_llm.model_name = "gpt-4"
        
        participant = LLMAgent("part", participant_llm, role="participant")
        assert "thoughtful participant" in participant.system_prompt
        assert "well-reasoned, concise comment" in participant.system_prompt
    
    def test_get_agent_info(self):
        """Test getting agent info."""
        info = self.agent.get_agent_info()
        
        assert isinstance(info, dict)  # AgentInfo is a TypedDict
        assert info["id"] == "test-agent"
        assert info["model"] == "gpt-4"
        assert info["provider"] == "openai"
        assert info["role"] == "participant"
        assert info["message_count"] == 0
        assert isinstance(info["created_at"], datetime)
    
    def test_format_messages(self):
        """Test message formatting."""
        # Test with no context
        messages = self.agent.format_messages("Hello world")
        assert len(messages) == 2  # System + Human
        assert messages[0].content == "Test system prompt"
        assert messages[1].content == "Hello world"
        
        # Test without system prompt
        messages = self.agent.format_messages("Hello", include_system=False)
        assert len(messages) == 1
        assert messages[0].content == "Hello"
        
        # Test with context messages
        context = [
            {
                "id": "1",
                "speaker_id": "user",
                "speaker_role": "user",
                "content": "Previous user message",
                "timestamp": datetime.now(),
                "phase": 1,
                "topic": "test"
            },
            {
                "id": "2",
                "speaker_id": "agent1",
                "speaker_role": "participant",
                "content": "Previous agent message",
                "timestamp": datetime.now(),
                "phase": 1,
                "topic": "test"
            }
        ]
        
        messages = self.agent.format_messages("Current prompt", context)
        assert len(messages) == 4  # System + 2 context + current
        assert messages[1].content == "Previous user message"
        assert messages[2].content == "Previous agent message"
        assert messages[3].content == "Current prompt"
    
    def test_generate_response(self):
        """Test generating response."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Generated response"
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm
        
        response = self.agent.generate_response("Test prompt")
        
        assert response == "Generated response"
        assert self.agent.message_count == 1
        self.mock_llm.invoke.assert_called_once()
    
    def test_generate_response_with_parameters(self):
        """Test generating response with custom parameters."""
        mock_response = Mock()
        mock_response.content = "Response with params"
        bound_llm = Mock()
        bound_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = bound_llm
        
        response = self.agent.generate_response(
            "Test prompt",
            temperature=0.5,
            max_tokens=1000
        )
        
        assert response == "Response with params"
        self.mock_llm.bind.assert_called_once_with(temperature=0.5, max_tokens=1000)
        bound_llm.invoke.assert_called_once()
    
    def test_generate_response_without_bind(self):
        """Test generating response without custom parameters."""
        mock_response = Mock()
        mock_response.content = "Simple response"
        self.mock_llm.invoke.return_value = mock_response
        
        response = self.agent.generate_response("Test prompt")
        
        assert response == "Simple response"
        # Should call invoke directly, not bind
        self.mock_llm.invoke.assert_called_once()
        self.mock_llm.bind.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_generate_response_async(self):
        """Test generating response asynchronously."""
        mock_response = Mock()
        mock_response.content = "Async response"
        self.mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        self.mock_llm.bind.return_value = self.mock_llm
        
        response = await self.agent.generate_response_async("Test prompt")
        
        assert response == "Async response"
        assert self.agent.message_count == 1
        self.mock_llm.ainvoke.assert_called_once()
    
    def test_stream_response(self):
        """Test streaming response."""
        # Mock streaming chunks
        mock_chunks = [
            Mock(content="Hello"),
            Mock(content=" world"),
            Mock(content="!")
        ]
        self.mock_llm.stream.return_value = iter(mock_chunks)
        self.mock_llm.bind.return_value = self.mock_llm
        
        # Collect streamed tokens
        tokens = list(self.agent.stream_response("Test prompt"))
        
        assert tokens == ["Hello", " world", "!"]
        assert self.agent.message_count == 1
        self.mock_llm.stream.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stream_response_async(self):
        """Test streaming response asynchronously."""
        # Mock async streaming chunks
        async def async_chunks():
            chunks = [
                Mock(content="Async"),
                Mock(content=" stream"),
                Mock(content="!")
            ]
            for chunk in chunks:
                yield chunk
        
        self.mock_llm.astream.return_value = async_chunks()
        self.mock_llm.bind.return_value = self.mock_llm
        
        # Collect streamed tokens
        tokens = []
        async for token in self.agent.stream_response_async("Test prompt"):
            tokens.append(token)
        
        assert tokens == ["Async", " stream", "!"]
        assert self.agent.message_count == 1
    
    def test_create_message(self):
        """Test creating message object."""
        with patch('uuid.uuid4') as mock_uuid:
            mock_uuid.return_value = uuid.UUID('12345678-1234-5678-1234-567812345678')
            
            message = self.agent.create_message("Test content", "Test topic")
            
            assert isinstance(message, dict)  # Message is a TypedDict
            assert message["id"] == "12345678-1234-5678-1234-567812345678"
            assert message["speaker_id"] == "test-agent"
            assert message["speaker_role"] == "participant"
            assert message["content"] == "Test content"
            assert message["topic"] == "Test topic"
            assert message["phase"] == -1
            assert isinstance(message["timestamp"], datetime)
    
    def test_error_handling_in_generate_response(self):
        """Test error handling in generate_response."""
        self.mock_llm.invoke.side_effect = Exception("LLM error")
        
        with pytest.raises(Exception) as exc_info:
            self.agent.generate_response("Test prompt")
        
        assert "LLM error" in str(exc_info.value)
        # Message count should not be incremented on error
        assert self.agent.message_count == 0
    
    @pytest.mark.asyncio
    async def test_error_handling_in_async_methods(self):
        """Test error handling in async methods."""
        self.mock_llm.ainvoke = AsyncMock(side_effect=Exception("Async error"))
        
        with pytest.raises(Exception) as exc_info:
            await self.agent.generate_response_async("Test prompt")
        
        assert "Async error" in str(exc_info.value)
        assert self.agent.message_count == 0
    
    def test_response_without_content_attribute(self):
        """Test handling response without content attribute."""
        mock_response = "String response"  # No .content attribute
        self.mock_llm.invoke.return_value = mock_response
        
        response = self.agent.generate_response("Test prompt")
        
        assert response == "String response"
        assert self.agent.message_count == 1
    
    def test_stream_chunks_without_content_attribute(self):
        """Test streaming with chunks without content attribute."""
        mock_chunks = ["chunk1", "chunk2", "chunk3"]  # String chunks
        self.mock_llm.stream.return_value = iter(mock_chunks)
        self.mock_llm.bind.return_value = self.mock_llm
        
        tokens = list(self.agent.stream_response("Test prompt"))
        
        assert tokens == ["chunk1", "chunk2", "chunk3"]
        assert self.agent.message_count == 1


class TestLLMAgentIntegration:
    """Integration tests for LLMAgent with real-like scenarios."""
    
    def test_full_conversation_flow(self):
        """Test a full conversation flow."""
        mock_llm = Mock()
        mock_llm.__class__.__name__ = "ChatAnthropic"
        mock_llm.model_name = "claude-3-opus"
        
        agent = LLMAgent("claude-agent", mock_llm, role="participant")
        
        # Mock responses for a conversation
        responses = [
            Mock(content="Hello! I'm ready to discuss."),
            Mock(content="That's an interesting point about AI ethics."),
            Mock(content="I agree, we need more transparency.")
        ]
        mock_llm.invoke.side_effect = responses
        
        # Simulate conversation
        context_messages = []
        
        # First message
        response1 = agent.generate_response("Let's discuss AI ethics")
        context_messages.append({
            "id": "1",
            "speaker_id": "user",
            "speaker_role": "user",
            "content": "Let's discuss AI ethics",
            "timestamp": datetime.now(),
            "phase": 1,
            "topic": "AI ethics"
        })
        context_messages.append(agent.create_message(response1, "AI ethics"))
        
        # Second message with context
        response2 = agent.generate_response(
            "What about transparency in AI systems?",
            context_messages
        )
        
        # Third message
        response3 = agent.generate_response("Do you think regulation is needed?")
        
        assert response1 == "Hello! I'm ready to discuss."
        assert response2 == "That's an interesting point about AI ethics."
        assert response3 == "I agree, we need more transparency."
        assert agent.message_count == 3
        assert agent.provider == "anthropic"
        
        # Check agent info
        info = agent.get_agent_info()
        assert info["message_count"] == 3
        assert info["provider"] == "anthropic"