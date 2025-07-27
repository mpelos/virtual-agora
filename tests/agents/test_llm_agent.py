"""Tests for LLM agent wrapper."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import uuid
import asyncio

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter

from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.state.schema import (
    AgentInfo,
    Message,
    MessagesState,
    VirtualAgoraState,
)


class TestLLMAgent:
    """Test LLMAgent class."""

    def setup_method(self):
        """Set up test method."""
        # Create mock LLM instance
        self.mock_llm = Mock()
        self.mock_llm.__class__.__name__ = "ChatOpenAI"
        self.mock_llm.model_name = "gpt-4"

        # Create agent with error handling disabled to test the raw LLM
        self.agent = LLMAgent(
            agent_id="test-agent",
            llm=self.mock_llm,
            role="participant",
            system_prompt="Test system prompt",
            enable_error_handling=False,
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
        assert "Facilitate the creation" in moderator.system_prompt

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
                "topic": "test",
            },
            {
                "id": "2",
                "speaker_id": "agent1",
                "speaker_role": "participant",
                "content": "Previous agent message",
                "timestamp": datetime.now(),
                "phase": 1,
                "topic": "test",
            },
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
            "Test prompt", temperature=0.5, max_tokens=1000
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
        mock_chunks = [Mock(content="Hello"), Mock(content=" world"), Mock(content="!")]
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
            chunks = [Mock(content="Async"), Mock(content=" stream"), Mock(content="!")]
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
        with patch("uuid.uuid4") as mock_uuid:
            mock_uuid.return_value = uuid.UUID("12345678-1234-5678-1234-567812345678")

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

        agent = LLMAgent(
            "claude-agent", mock_llm, role="participant", enable_error_handling=False
        )

        # Mock responses for a conversation
        responses = [
            Mock(content="Hello! I'm ready to discuss."),
            Mock(content="That's an interesting point about AI ethics."),
            Mock(content="I agree, we need more transparency."),
        ]
        mock_llm.invoke.side_effect = responses

        # Simulate conversation
        context_messages = []

        # First message
        response1 = agent.generate_response("Let's discuss AI ethics")
        context_messages.append(
            {
                "id": "1",
                "speaker_id": "user",
                "speaker_role": "user",
                "content": "Let's discuss AI ethics",
                "timestamp": datetime.now(),
                "phase": 1,
                "topic": "AI ethics",
            }
        )
        context_messages.append(agent.create_message(response1, "AI ethics"))

        # Second message with context
        response2 = agent.generate_response(
            "What about transparency in AI systems?", context_messages
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


class TestLLMAgentLangGraphIntegration:
    """Test LLMAgent's LangGraph integration features."""

    def setup_method(self):
        """Set up test method."""
        # Create mock LLM
        self.mock_llm = Mock()
        self.mock_llm.__class__.__name__ = "ChatOpenAI"
        self.mock_llm.model_name = "gpt-4"

        # Create agent with error handling disabled for testing
        self.agent = LLMAgent(
            agent_id="test-agent",
            llm=self.mock_llm,
            role="participant",
            enable_error_handling=False,
        )

    def test_agent_as_callable_node_with_messages_state(self):
        """Test agent can be used as a callable node with MessagesState."""
        # Setup mock response
        mock_response = Mock()
        mock_response.content = "Test response"
        self.mock_llm.invoke.return_value = mock_response

        # Create state with messages
        state = {"messages": [HumanMessage(content="Hello, how are you?")]}

        # Call agent as node
        result = self.agent(state)

        # Check result
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "Test response"
        assert result["messages"][0].name == "test-agent"

    def test_agent_with_virtual_agora_state(self):
        """Test agent with full VirtualAgoraState."""
        # Setup mock
        mock_response = Mock()
        mock_response.content = "I propose discussing AI ethics."
        self.mock_llm.invoke.return_value = mock_response

        # Create VirtualAgoraState
        state = {
            "session_id": "test-session",
            "current_phase": 1,  # Agenda setting
            "agents": {"test-agent": {"id": "test-agent"}},
            "messages": [],
            "active_topic": None,
            "messages_by_agent": {},
            "messages_by_topic": {},
        }

        # Call agent
        result = self.agent(state)

        # Check result - now returns AIMessage objects for LangGraph compatibility
        assert "messages" in result
        assert len(result["messages"]) == 1
        message = result["messages"][0]
        assert isinstance(message, AIMessage)
        assert message.name == "test-agent"
        assert message.content == "I propose discussing AI ethics."
        assert "messages_by_agent" in result
        assert result["messages_by_agent"]["test-agent"] == 1

    @pytest.mark.asyncio
    async def test_async_call_with_messages_state(self):
        """Test async __acall__ method."""
        # Setup mock
        mock_response = Mock()
        mock_response.content = "Async response"
        self.mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        # Create state
        state = {"messages": [HumanMessage(content="Test async")]}

        # Call async
        result = await self.agent.__acall__(state)

        # Verify
        assert "messages" in result
        assert result["messages"][0].content == "Async response"
        assert result["messages"][0].name == "test-agent"

    def test_thread_safety_of_message_count(self):
        """Test thread-safe message count updates."""
        mock_response = Mock()
        mock_response.content = "Response"
        self.mock_llm.invoke.return_value = mock_response

        # Simulate concurrent calls
        import threading

        def call_agent():
            state = {"messages": [HumanMessage(content="Test")]}
            self.agent(state)

        threads = []
        initial_count = self.agent.message_count

        # Create 10 threads
        for _ in range(10):
            t = threading.Thread(target=call_agent)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Check count incremented correctly
        assert self.agent.message_count == initial_count + 10

    def test_stream_in_graph_messages_mode(self):
        """Test streaming in graph with messages mode."""
        # Setup streaming chunks
        mock_chunks = [Mock(content="Hello"), Mock(content=" world"), Mock(content="!")]
        self.mock_llm.stream.return_value = iter(mock_chunks)
        self.mock_llm.bind.return_value = self.mock_llm

        state = {"messages": [HumanMessage(content="Say hello")]}

        # Stream in messages mode
        chunks = list(self.agent.stream_in_graph(state, stream_mode="messages"))

        assert chunks == ["Hello", " world", "!"]

    def test_stream_in_graph_updates_mode(self):
        """Test streaming in graph with updates mode."""
        # Setup
        mock_chunks = [Mock(content="Test")]
        self.mock_llm.stream.return_value = iter(mock_chunks)
        self.mock_llm.bind.return_value = self.mock_llm

        state = {"messages": [HumanMessage(content="Test")]}

        # Stream in updates mode
        updates = list(self.agent.stream_in_graph(state, stream_mode="updates"))

        assert len(updates) == 1
        assert "messages" in updates[0]
        assert isinstance(updates[0]["messages"][0], AIMessage)
        assert updates[0]["messages"][0].content == "Test"

    def test_stream_in_graph_values_mode(self):
        """Test streaming in graph with values mode."""
        # Setup
        mock_chunks = [Mock(content="Part1"), Mock(content="Part2")]
        self.mock_llm.stream.return_value = iter(mock_chunks)
        self.mock_llm.bind.return_value = self.mock_llm

        initial_message = HumanMessage(content="Test")
        state = {"messages": [initial_message]}

        # Stream in values mode
        values = list(self.agent.stream_in_graph(state, stream_mode="values"))

        # First value should have partial response
        assert len(values[0]["messages"]) == 2
        assert values[0]["messages"][0] == initial_message
        assert values[0]["messages"][1].content == "Part1"

        # Second value should have full response
        assert len(values[1]["messages"]) == 2
        assert values[1]["messages"][1].content == "Part1Part2"

    @pytest.mark.asyncio
    async def test_stream_in_graph_async(self):
        """Test async streaming in graph."""

        # Setup async streaming
        async def async_chunks():
            chunks = [Mock(content="Async1"), Mock(content="Async2")]
            for chunk in chunks:
                yield chunk

        self.mock_llm.astream.return_value = async_chunks()
        self.mock_llm.bind.return_value = self.mock_llm

        state = {"messages": [HumanMessage(content="Test")]}

        # Collect async stream
        chunks = []
        async for chunk in self.agent.stream_in_graph_async(
            state, stream_mode="messages"
        ):
            chunks.append(chunk)

        assert chunks == ["Async1", "Async2"]

    def test_handle_generic_state(self):
        """Test handling of generic state dictionary."""
        mock_response = Mock()
        mock_response.content = "Generic response"
        self.mock_llm.invoke.return_value = mock_response

        # Generic state with prompt
        state = {"prompt": "What is AI?"}

        result = self.agent(state)

        assert result == {"response": "Generic response", "agent_id": "test-agent"}

    def test_writer_integration(self):
        """Test StreamWriter integration."""
        mock_response = Mock()
        mock_response.content = "Written response"
        self.mock_llm.invoke.return_value = mock_response

        # Mock writer with write method
        mock_writer = Mock()
        mock_writer.write = Mock()

        state = {"messages": [HumanMessage(content="Test")]}

        # Call with writer
        result = self.agent(state, writer=mock_writer)

        # Check writer was called
        mock_writer.write.assert_called_once()
        written_data = mock_writer.write.call_args[0][0]
        assert "messages" in written_data
        assert written_data["messages"][0].content == "Written response"

    @pytest.mark.asyncio
    async def test_async_writer_integration(self):
        """Test async StreamWriter integration."""
        mock_response = Mock()
        mock_response.content = "Async written"
        self.mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        # Mock async writer
        mock_writer = AsyncMock()

        state = {"messages": [HumanMessage(content="Test")]}

        # Call with writer
        result = await self.agent.__acall__(state, writer=mock_writer)

        # Check async writer was called
        mock_writer.awrite.assert_called_once()

    def test_config_passthrough(self):
        """Test RunnableConfig is properly handled."""
        mock_response = Mock()
        mock_response.content = "Config test"
        self.mock_llm.invoke.return_value = mock_response

        state = {"messages": [HumanMessage(content="Test")]}
        config = RunnableConfig({"metadata": {"test": True}})

        # Call with config
        result = self.agent(state, config)

        assert "messages" in result
        assert result["messages"][0].content == "Config test"

    def test_error_handling_in_call(self):
        """Test error handling in __call__ method."""
        # Test with invalid state type
        with pytest.raises(TypeError) as exc_info:
            self.agent("invalid state type")

        assert "Unsupported state type" in str(exc_info.value)

        # Test with missing prompt
        state = {"unknown_field": "value"}
        with pytest.raises(ValueError) as exc_info:
            self.agent(state)

        assert "No prompt found" in str(exc_info.value)

    def test_phase_specific_prompts(self):
        """Test phase-specific prompt generation for VirtualAgoraState."""
        mock_response = Mock()
        self.mock_llm.invoke.return_value = mock_response

        # Test different phases
        test_cases = [
            (1, None, "Please propose 3-5 discussion topics"),
            (2, "AI Safety", "Please share your thoughts on: AI Safety"),
            (3, "Ethics", "Should we conclude our discussion on 'Ethics'?"),
            (0, None, "Please provide your input"),
        ]

        for phase, topic, expected_prompt_part in test_cases:
            mock_response.content = f"Response for phase {phase}"

            state = {
                "current_phase": phase,
                "active_topic": topic,
                "agents": {},
                "messages": [],
                "messages_by_agent": {},
                "messages_by_topic": {},
            }

            result = self.agent(state)

            # Verify the correct prompt was used
            call_args = self.mock_llm.invoke.call_args[0][0]
            human_messages = [msg for msg in call_args if isinstance(msg, HumanMessage)]
            assert len(human_messages) > 0
            assert expected_prompt_part in human_messages[-1].content

    def test_message_context_filtering(self):
        """Test that messages are properly filtered by phase and topic."""
        mock_response = Mock()
        mock_response.content = "Filtered response"
        self.mock_llm.invoke.return_value = mock_response

        # Create messages with proper Message structure
        messages = [
            {
                "id": "1",
                "speaker_id": "agent1",
                "speaker_role": "participant",
                "content": "Phase 1 message",
                "timestamp": datetime.now(),
                "phase": 1,
                "topic": None,
            },
            {
                "id": "2",
                "speaker_id": "agent2",
                "speaker_role": "participant",
                "content": "AI topic message",
                "timestamp": datetime.now(),
                "phase": 2,
                "topic": "AI",
            },
            {
                "id": "3",
                "speaker_id": "agent3",
                "speaker_role": "participant",
                "content": "Ethics topic message",
                "timestamp": datetime.now(),
                "phase": 2,
                "topic": "Ethics",
            },
            {
                "id": "4",
                "speaker_id": "agent1",
                "speaker_role": "participant",
                "content": "Phase 3 AI message",
                "timestamp": datetime.now(),
                "phase": 3,
                "topic": "AI",
            },
        ]

        state = {
            "current_phase": 2,
            "active_topic": "AI",
            "agents": {},
            "messages": messages,
            "messages_by_agent": {},
            "messages_by_topic": {},
        }

        result = self.agent(state)

        # The agent should have filtered messages for phase 2 and topic "AI"
        assert "messages" in result
        assert len(result["messages"]) == 1


class TestLLMAgentToolIntegration:
    """Test LLMAgent's tool integration features."""

    def setup_method(self):
        """Set up test method."""
        # Create mock LLM
        self.mock_llm = Mock()
        self.mock_llm.__class__.__name__ = "ChatOpenAI"
        self.mock_llm.model_name = "gpt-4"
        self.mock_llm.bind_tools = Mock(return_value=self.mock_llm)

        # Create proper tool functions that LangChain can accept
        from langchain_core.tools import Tool

        def test_func_1(input: str) -> str:
            """Test tool 1 function."""
            return f"Tool 1 result: {input}"

        def test_func_2(input: str) -> str:
            """Test tool 2 function."""
            return f"Tool 2 result: {input}"

        self.mock_tool1 = Tool(
            name="test_tool_1", description="Test tool 1", func=test_func_1
        )

        self.mock_tool2 = Tool(
            name="test_tool_2", description="Test tool 2", func=test_func_2
        )

        self.tools = [self.mock_tool1, self.mock_tool2]

    def test_agent_with_tools_initialization(self):
        """Test creating agent with tools."""
        agent = LLMAgent(
            agent_id="tool-agent",
            llm=self.mock_llm,
            tools=self.tools,
            enable_error_handling=False,
        )

        assert agent.tools == self.tools
        assert agent.has_tools()
        assert agent._tool_bound_llm is not None
        assert agent._tool_node is not None

        # Verify bind_tools was called
        self.mock_llm.bind_tools.assert_called_once_with(self.tools)

    def test_agent_bind_tools_after_creation(self):
        """Test binding tools after agent creation."""
        agent = LLMAgent(
            agent_id="test-agent", llm=self.mock_llm, enable_error_handling=False
        )

        # Initially no tools
        assert not agent.has_tools()
        assert agent.tools == []

        # Bind tools
        agent.bind_tools(self.tools)

        assert agent.tools == self.tools
        assert agent.has_tools()
        self.mock_llm.bind_tools.assert_called_with(self.tools)

    def test_create_with_tools_factory(self):
        """Test creating agent with tools using factory method."""
        agent = LLMAgent.create_with_tools(
            agent_id="factory-agent",
            llm=self.mock_llm,
            tools=self.tools,
            role="participant",
        )

        assert agent.agent_id == "factory-agent"
        assert agent.tools == self.tools
        assert agent.role == "participant"
        assert agent.has_tools()

    def test_agent_generates_tool_calls(self):
        """Test agent generating tool calls in response."""
        agent = LLMAgent(
            agent_id="tool-agent",
            llm=self.mock_llm,
            tools=self.tools,
            enable_error_handling=False,
        )

        # Mock LLM response with tool calls
        tool_call = {
            "id": "call_123",
            "name": "test_tool_1",
            "args": {"param": "value"},
        }

        mock_response = AIMessage(content="I'll use a tool.", tool_calls=[tool_call])
        self.mock_llm.invoke.return_value = mock_response

        # Create state
        state = {"messages": [HumanMessage(content="Please use a tool")]}

        # Call agent
        result = agent(state)

        # Verify result contains tool call
        assert "messages" in result
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert isinstance(msg, AIMessage)
        assert hasattr(msg, "tool_calls")
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "test_tool_1"

    def test_agent_executes_tool_calls(self):
        """Test agent executing tool calls in state."""
        # Create mock ToolNode
        mock_tool_node = Mock()
        mock_tool_results = {
            "messages": [
                ToolMessage(
                    content="Tool result", tool_call_id="call_123", name="test_tool_1"
                )
            ]
        }
        mock_tool_node.invoke.return_value = mock_tool_results

        agent = LLMAgent(
            agent_id="tool-agent",
            llm=self.mock_llm,
            tools=self.tools,
            enable_error_handling=False,
        )
        agent._tool_node = mock_tool_node

        # Create state with tool call
        tool_call = {
            "id": "call_123",
            "name": "test_tool_1",
            "args": {"param": "value"},
        }

        ai_msg = AIMessage(content="Using tool", tool_calls=[tool_call])

        state = {"messages": [HumanMessage(content="Use tool"), ai_msg]}

        # Call agent - should execute tool
        result = agent(state)

        # Verify tool node was called with invoke method
        mock_tool_node.invoke.assert_called_once()

        # Verify result
        assert result == mock_tool_results

    def test_agent_streaming_with_tools(self):
        """Test streaming with tool calls."""
        agent = LLMAgent(
            agent_id="stream-agent",
            llm=self.mock_llm,
            tools=self.tools,
            enable_error_handling=False,
        )

        # Mock streaming chunks with tool calls
        chunks = [
            Mock(content="I'll use ", tool_calls=[]),
            Mock(
                content="the tool.",
                tool_calls=[
                    {
                        "id": "call_stream",
                        "name": "test_tool_1",
                        "args": {"param": "value"},
                    }
                ],
            ),
        ]

        self.mock_llm.stream.return_value = iter(chunks)

        state = {"messages": [HumanMessage(content="Stream with tool")]}

        # Stream response
        streamed = list(agent.stream_in_graph(state, stream_mode="messages"))

        assert len(streamed) == 2
        assert streamed[0] == "I'll use "
        assert streamed[1] == "the tool."

    @pytest.mark.asyncio
    async def test_agent_async_with_tools(self):
        """Test async execution with tools."""
        agent = LLMAgent(
            agent_id="async-tool-agent",
            llm=self.mock_llm,
            tools=self.tools,
            enable_error_handling=False,
        )

        # Mock async response with tool call
        tool_call = {
            "id": "call_async",
            "name": "test_tool_2",
            "args": {"async_param": "async_value"},
        }

        mock_response = AIMessage(content="Async tool call", tool_calls=[tool_call])
        self.mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        state = {"messages": [HumanMessage(content="Async tool test")]}

        # Call async
        result = await agent.__acall__(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert msg.tool_calls[0]["name"] == "test_tool_2"

    def test_get_tool_node(self):
        """Test getting tool node from agent."""
        agent = LLMAgent(
            agent_id="test-agent",
            llm=self.mock_llm,
            tools=self.tools,
            enable_error_handling=False,
        )

        tool_node = agent.get_tool_node()
        assert tool_node is not None

        # Agent without tools
        agent_no_tools = LLMAgent(
            agent_id="no-tools", llm=self.mock_llm, enable_error_handling=False
        )

        assert agent_no_tools.get_tool_node() is None

    def test_tool_binding_failure_handling(self):
        """Test handling of tool binding failures."""
        # Mock bind_tools to raise exception
        self.mock_llm.bind_tools.side_effect = Exception("Binding failed")

        # Create agent with tools - should handle error gracefully
        agent = LLMAgent(
            agent_id="fail-agent",
            llm=self.mock_llm,
            tools=self.tools,
            enable_error_handling=False,
        )

        # Agent should still be created but without tools
        assert agent._tool_bound_llm is None
        assert agent._tool_node is None
        assert not agent.has_tools()

        # Agent should still work without tools
        self.mock_llm.invoke.return_value = Mock(content="Response without tools")
        response = agent.generate_response("Test prompt")
        assert response == "Response without tools"
