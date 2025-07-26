"""Integration tests for tool functionality in Virtual Agora.

These tests verify that tools work correctly with agents and StateGraph.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import uuid

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool

from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.agents.tool_enabled_factory import (
    create_tool_enabled_moderator,
    create_tool_enabled_participant,
    create_specialized_agent,
)
from virtual_agora.tools import (
    ProposalTool,
    VotingTool,
    SummaryTool,
    VirtualAgoraToolNode,
    create_discussion_tools,
)
from virtual_agora.state.schema import MessagesState, ToolEnabledState


class TestToolEnabledAgentCreation:
    """Test creation of tool-enabled agents."""

    def test_create_tool_enabled_moderator(self):
        """Test creating a moderator with tools."""
        # Create mock LLM
        mock_llm = Mock()
        mock_llm.__class__.__name__ = "ChatOpenAI"
        mock_llm.model_name = "gpt-4"
        mock_llm.bind_tools = Mock(return_value=mock_llm)

        # Create moderator
        moderator = create_tool_enabled_moderator(agent_id="moderator-1", llm=mock_llm)

        # Verify agent properties
        assert moderator.agent_id == "moderator-1"
        assert moderator.role == "moderator"
        assert len(moderator.tools) == 2  # SummaryTool and VotingTool

        # Verify tools
        tool_names = {tool.name for tool in moderator.tools}
        assert "summarize" in tool_names
        assert "vote" in tool_names

        # Verify tool binding was called
        mock_llm.bind_tools.assert_called_once()

    def test_create_tool_enabled_participant_phase_1(self):
        """Test creating a participant for phase 1 (agenda setting)."""
        mock_llm = Mock()
        mock_llm.__class__.__name__ = "ChatOpenAI"
        mock_llm.model_name = "gpt-4"
        mock_llm.bind_tools = Mock(return_value=mock_llm)

        participant = create_tool_enabled_participant(
            agent_id="participant-1", llm=mock_llm, phase=1
        )

        # Should have ProposalTool and SummaryTool
        assert len(participant.tools) == 2
        tool_names = {tool.name for tool in participant.tools}
        assert "propose_topics" in tool_names
        assert "summarize" in tool_names

    def test_create_tool_enabled_participant_phase_2(self):
        """Test creating a participant for phase 2 (discussion)."""
        mock_llm = Mock()
        mock_llm.__class__.__name__ = "ChatOpenAI"
        mock_llm.model_name = "gpt-4"
        mock_llm.bind_tools = Mock(return_value=mock_llm)

        participant = create_tool_enabled_participant(
            agent_id="participant-2", llm=mock_llm, phase=2
        )

        # Should have SummaryTool
        assert len(participant.tools) == 1
        assert participant.tools[0].name == "summarize"

    def test_create_tool_enabled_participant_phase_3(self):
        """Test creating a participant for phase 3 (consensus)."""
        mock_llm = Mock()
        mock_llm.__class__.__name__ = "ChatOpenAI"
        mock_llm.model_name = "gpt-4"
        mock_llm.bind_tools = Mock(return_value=mock_llm)

        participant = create_tool_enabled_participant(
            agent_id="participant-3", llm=mock_llm, phase=3
        )

        # Should have VotingTool and SummaryTool
        assert len(participant.tools) == 2
        tool_names = {tool.name for tool in participant.tools}
        assert "vote" in tool_names
        assert "summarize" in tool_names

    def test_create_specialized_agent(self):
        """Test creating specialized agents."""
        mock_llm = Mock()
        mock_llm.__class__.__name__ = "ChatOpenAI"
        mock_llm.model_name = "gpt-4"
        mock_llm.bind_tools = Mock(return_value=mock_llm)

        # Test different specializations
        specializations = {
            "analyst": ["summarize"],
            "facilitator": ["propose_topics", "vote"],
            "summarizer": ["summarize"],
            "generalist": ["propose_topics", "vote", "summarize"],
        }

        for spec, expected_tools in specializations.items():
            agent = create_specialized_agent(
                agent_id=f"{spec}-agent", llm=mock_llm, specialization=spec
            )

            tool_names = {tool.name for tool in agent.tools}
            for expected in expected_tools:
                assert expected in tool_names, f"{expected} not found for {spec}"


class TestAgentToolExecution:
    """Test agents executing tools in StateGraph context."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock LLM
        self.mock_llm = Mock()
        self.mock_llm.__class__.__name__ = "ChatOpenAI"
        self.mock_llm.model_name = "gpt-4"
        self.mock_llm.bind_tools = Mock(return_value=self.mock_llm)

        # Create tools
        self.tools = create_discussion_tools()

    def test_agent_generates_tool_call(self):
        """Test agent generating tool calls."""
        # Create agent with tools
        agent = LLMAgent.create_with_tools(
            agent_id="test-agent", llm=self.mock_llm, tools=self.tools
        )

        # Mock LLM response with tool call
        tool_call = {
            "id": "call_123",
            "name": "propose_topics",
            "args": {
                "topics": ["AI Ethics", "Climate Change", "Space Exploration"],
                "rationale": "These are important topics for discussion",
            },
        }

        mock_ai_message = AIMessage(
            content="I'll propose some topics.", tool_calls=[tool_call]
        )
        self.mock_llm.invoke.return_value = mock_ai_message

        # Create state
        state = {"messages": [HumanMessage(content="Please propose some topics")]}

        # Call agent
        result = agent(state)

        # Verify result contains tool call
        assert "messages" in result
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert isinstance(msg, AIMessage)
        assert hasattr(msg, "tool_calls")
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "propose_topics"

    def test_agent_executes_tool_calls(self):
        """Test agent executing tool calls from previous message."""
        # Create agent with tool node
        agent = LLMAgent.create_with_tools(
            agent_id="test-agent", llm=self.mock_llm, tools=self.tools
        )

        # Create state with AI message containing tool calls
        tool_call = {
            "id": "call_456",
            "name": "propose_topics",
            "args": {"topics": ["AI Safety", "Quantum Computing", "Biotechnology"]},
        }

        ai_msg = AIMessage(content="Let me propose topics.", tool_calls=[tool_call])

        state = {"messages": [HumanMessage(content="Propose topics"), ai_msg]}

        # Execute - should detect tool calls and execute them
        result = agent(state)

        # Should return tool messages
        assert "messages" in result
        # The result should contain tool execution results
        # Since we're using real tools, we should get actual results

    @pytest.mark.asyncio
    async def test_agent_async_tool_execution(self):
        """Test async tool execution."""
        # Create agent
        agent = LLMAgent.create_with_tools(
            agent_id="async-agent", llm=self.mock_llm, tools=self.tools
        )

        # Mock async LLM response
        tool_call = {
            "id": "call_789",
            "name": "summarize",
            "args": {
                "content": "This is a long discussion about AI ethics...",
                "max_length": 100,
                "style": "concise",
            },
        }

        mock_ai_message = AIMessage(
            content="I'll summarize the discussion.", tool_calls=[tool_call]
        )
        self.mock_llm.ainvoke = AsyncMock(return_value=mock_ai_message)

        state = {"messages": [HumanMessage(content="Please summarize")]}

        # Call async
        result = await agent.__acall__(state)

        assert "messages" in result
        assert len(result["messages"]) == 1

    def test_agent_streaming_with_tools(self):
        """Test streaming responses with tool calls."""
        agent = LLMAgent.create_with_tools(
            agent_id="stream-agent", llm=self.mock_llm, tools=self.tools
        )

        # Mock streaming with tool calls
        chunks = [
            Mock(content="I'll vote ", tool_calls=[]),
            Mock(
                content="on the topic.",
                tool_calls=[
                    {
                        "id": "call_stream",
                        "name": "vote",
                        "args": {"topic": "AI Ethics", "vote": "yes"},
                    }
                ],
            ),
        ]

        self.mock_llm.stream.return_value = iter(chunks)

        state = {"messages": [HumanMessage(content="Vote on AI Ethics")]}

        # Stream response
        streamed = list(agent.stream_in_graph(state, stream_mode="messages"))

        assert len(streamed) == 2
        assert streamed[0] == "I'll vote "
        assert streamed[1] == "on the topic."


class TestToolNodeIntegration:
    """Test VirtualAgoraToolNode integration."""

    def test_tool_node_creation(self):
        """Test creating tool node wrapper."""
        tools = create_discussion_tools()
        tool_node = VirtualAgoraToolNode(
            tools=tools, validate_inputs=True, track_metrics=True
        )

        assert len(tool_node.tools) == 3
        assert tool_node.validate_inputs
        assert tool_node.track_metrics
        assert tool_node.metrics["total_calls"] == 0

    def test_tool_node_execution(self):
        """Test tool node executing tool calls."""
        tools = create_discussion_tools()
        tool_node = VirtualAgoraToolNode(tools=tools)

        # Create state with tool call
        tool_call = {
            "id": "test_call",
            "name": "vote",
            "args": {
                "topic": "Climate Action",
                "vote": "yes",
                "reasoning": "This is critical for our future",
            },
        }

        ai_msg = AIMessage(content="I vote yes", tool_calls=[tool_call])

        state = {"messages": [ai_msg]}

        # Execute
        result = tool_node(state)

        # Should return tool messages
        assert "messages" in result
        assert len(result["messages"]) > 0
        assert isinstance(result["messages"][0], ToolMessage)

        # Check metrics
        assert tool_node.metrics["total_calls"] == 1
        assert tool_node.metrics["successful_calls"] == 1

    def test_tool_node_validation(self):
        """Test tool input validation."""
        tools = create_discussion_tools()
        tool_node = VirtualAgoraToolNode(tools=tools, validate_inputs=True)

        # Invalid tool call (missing required args)
        invalid_call = {
            "id": "invalid_call",
            "name": "vote",
            "args": {"topic": "Test"},  # Missing 'vote' field
        }

        ai_msg = AIMessage(content="Invalid vote", tool_calls=[invalid_call])

        state = {"messages": [ai_msg]}

        # Should raise validation error
        with pytest.raises(Exception):
            tool_node(state)

    def test_tool_node_metrics(self):
        """Test tool execution metrics tracking."""
        tools = create_discussion_tools()
        tool_node = VirtualAgoraToolNode(tools=tools, track_metrics=True)

        # Execute multiple tool calls
        calls = [
            {
                "id": f"call_{i}",
                "name": "summarize",
                "args": {"content": f"Content {i}", "max_length": 50},
            }
            for i in range(3)
        ]

        for call in calls:
            ai_msg = AIMessage(content="", tool_calls=[call])
            state = {"messages": [ai_msg]}
            tool_node(state)

        # Check metrics
        metrics = tool_node.get_metrics()
        assert metrics["total_calls"] == 3
        assert metrics["successful_calls"] == 3
        assert metrics["calls_by_tool"]["summarize"] == 3
        assert metrics["success_rate"] == 1.0


class TestToolErrorHandling:
    """Test error handling in tool execution."""

    def test_tool_validation_error(self):
        """Test handling of tool validation errors."""
        from virtual_agora.tools.tool_error_handling import (
            validate_tool_input,
            ToolValidationError,
        )

        # Create a tool with schema
        proposal_tool = ProposalTool()

        # Invalid args (not enough topics)
        invalid_args = {"topics": ["Only one topic"], "rationale": "Test"}  # Needs 3-5

        with pytest.raises(ToolValidationError):
            validate_tool_input(proposal_tool, invalid_args)

    def test_error_tool_message_creation(self):
        """Test creating error tool messages."""
        from virtual_agora.tools.tool_error_handling import create_error_tool_message

        error = ValueError("Test error")
        tool_msg = create_error_tool_message(
            tool_call_id="test_call",
            tool_name="test_tool",
            error=error,
            include_traceback=False,
        )

        assert isinstance(tool_msg, ToolMessage)
        assert tool_msg.tool_call_id == "test_call"
        assert tool_msg.name == "test_tool"
        assert tool_msg.status == "error"
        assert "ValueError" in tool_msg.content
        assert "Test error" in tool_msg.content

    def test_tool_error_recovery_strategy(self):
        """Test error recovery strategy."""
        from virtual_agora.tools.tool_error_handling import (
            ToolErrorRecoveryStrategy,
            ToolRetryableError,
        )

        strategy = ToolErrorRecoveryStrategy(max_retries=3, exponential_backoff=True)

        # Test retryable error
        retryable_error = ToolRetryableError("Network error", tool_name="test_tool")

        assert strategy.should_retry(retryable_error, attempt=0)
        assert strategy.should_retry(retryable_error, attempt=2)
        assert not strategy.should_retry(retryable_error, attempt=3)

        # Test wait time calculation
        assert strategy.get_wait_time(0, retryable_error) == 1.0
        assert strategy.get_wait_time(1, retryable_error) == 2.0
        assert strategy.get_wait_time(2, retryable_error) == 4.0


class TestToolSerialization:
    """Test tool call serialization."""

    def test_serialize_tool_call(self):
        """Test serializing a tool call."""
        from virtual_agora.tools.tool_serialization import (
            serialize_tool_call,
            deserialize_tool_call,
        )

        tool_call = {
            "id": "call_123",
            "name": "propose_topics",
            "args": {
                "topics": ["Topic 1", "Topic 2", "Topic 3"],
                "rationale": "Test rationale",
            },
        }

        # Serialize
        serialized = serialize_tool_call(tool_call)
        assert "timestamp" in serialized
        assert serialized["name"] == "propose_topics"

        # Deserialize
        deserialized = deserialize_tool_call(serialized)
        assert deserialized["name"] == tool_call["name"]
        assert deserialized["args"] == tool_call["args"]

    def test_serialize_ai_message_with_tools(self):
        """Test serializing AI message with tool calls."""
        from virtual_agora.tools.tool_serialization import (
            serialize_ai_message_with_tools,
            deserialize_ai_message_with_tools,
        )

        # Create AI message with tool calls
        tool_calls = [
            {"id": "call_1", "name": "vote", "args": {"topic": "Test", "vote": "yes"}}
        ]

        ai_msg = AIMessage(content="I vote yes", name="agent-1", tool_calls=tool_calls)

        # Serialize
        serialized = serialize_ai_message_with_tools(ai_msg)
        assert serialized["content"] == "I vote yes"
        assert serialized["name"] == "agent-1"
        assert "tool_calls" in serialized
        assert len(serialized["tool_calls"]) == 1

        # Deserialize
        deserialized = deserialize_ai_message_with_tools(serialized)
        assert isinstance(deserialized, AIMessage)
        assert deserialized.content == ai_msg.content
        assert deserialized.name == ai_msg.name
        assert hasattr(deserialized, "tool_calls")
        assert len(deserialized.tool_calls) == 1
