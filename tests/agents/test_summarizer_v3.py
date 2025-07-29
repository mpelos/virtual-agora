"""Unit tests for SummarizerAgent in Virtual Agora v1.3."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from virtual_agora.agents.summarizer import SummarizerAgent
from tests.helpers.fake_llm import SummarizerFakeLLM, FakeLLMBase


class TestSummarizerAgentV3:
    """Unit tests for SummarizerAgent v1.3 functionality."""

    @pytest.fixture
    def summarizer_llm(self):
        """Create a fake LLM for summarizer testing."""
        return SummarizerFakeLLM()

    @pytest.fixture
    def summarizer(self, summarizer_llm):
        """Create test summarizer instance."""
        return SummarizerAgent(
            agent_id="test_summarizer",
            llm=summarizer_llm,
            compression_ratio=0.3,
            max_summary_tokens=500,
            enable_error_handling=False,  # Disable error handling for tests
        )

    def test_initialization_v13(self, summarizer):
        """Test agent initialization with v1.3 prompt."""
        assert summarizer.agent_id == "test_summarizer"
        assert summarizer.compression_ratio == 0.3
        assert summarizer.max_summary_tokens == 500

        # Check v1.3 specific prompt elements
        assert "specialized text compression tool" in summarizer.system_prompt
        assert "agent-agnostic summary" in summarizer.system_prompt
        assert "Virtual Agora" in summarizer.system_prompt
        assert "concise" in summarizer.system_prompt
        assert "third person" in summarizer.system_prompt

    def test_summarize_round_basic(self, summarizer):
        """Test basic round summarization functionality."""
        messages = [
            {
                "speaker": "agent1",
                "content": "We need to prioritize security in our design.",
            },
            {
                "speaker": "agent2",
                "content": "I agree, but performance is also critical.",
            },
            {
                "speaker": "agent3",
                "content": "Let's find a balanced approach between both.",
            },
        ]

        summary = summarizer.summarize_round(
            messages=messages, topic="System Architecture", round_number=1
        )

        assert isinstance(summary, str)
        assert len(summary) > 0
        # Check that it contains expected content
        assert "discussion" in summary.lower()
        # Should contain key themes from the messages
        assert "technical" in summary or "implementation" in summary

    def test_summarize_round_with_context(self, summarizer):
        """Test round summarization with previous context."""
        messages = [
            {"content": "Building on previous points about scalability..."},
            {"content": "I'd like to add that microservices could help."},
            {"content": "But we need to consider operational complexity."},
        ]

        # Note: In v1.3, previous summaries are part of the context provided
        # to agents, not passed directly to summarize_round
        summary = summarizer.summarize_round(
            messages=messages, topic="Architecture Decisions", round_number=3
        )

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_compression_metrics(self, summarizer):
        """Test compression metric tracking."""
        # Create long messages to test compression
        long_messages = [
            {"content": "This is a very detailed technical discussion " * 20}
            for _ in range(5)
        ]

        with patch.object(summarizer, "_estimate_tokens") as mock_tokens:
            # Mock token counts: original=1000, compressed=300
            mock_tokens.side_effect = [1000, 300]

            summary = summarizer.summarize_round(long_messages, "Test Topic", 1)

            # Verify compression was tracked
            assert mock_tokens.call_count == 2
            # Should achieve target compression ratio
            # 300/1000 = 0.3 which matches our compression_ratio

    def test_max_token_enforcement(self, summarizer):
        """Test that max token limit is enforced."""
        summarizer.max_summary_tokens = 50  # Very low limit

        # Create extremely long input
        huge_messages = [
            {"content": "Extensive discussion point " * 100} for _ in range(10)
        ]

        summary = summarizer.summarize_round(huge_messages, "Test", 1)

        # Summary should still be generated but respect token limit
        assert isinstance(summary, str)
        # Check it's reasonably short (approximate check)
        assert len(summary.split()) < 100  # Rough approximation

    def test_generate_progressive_summary(self, summarizer):
        """Test progressive summary generation across rounds."""
        round_summaries = [
            "Round 1: Agents explored the problem space and identified key challenges.",
            "Round 2: Discussion focused on potential solutions and trade-offs.",
            "Round 3: Consensus began forming around a phased implementation approach.",
            "Round 4: Detailed planning of first phase requirements.",
            "Round 5: Risk assessment and mitigation strategies discussed.",
        ]

        progressive_summary = summarizer.generate_progressive_summary(
            summaries=round_summaries, topic="Project Implementation Strategy"
        )

        assert isinstance(progressive_summary, str)
        assert "discussion rounds" in progressive_summary
        assert "evolved" in progressive_summary
        # Should be more concise than all summaries combined
        combined_length = sum(len(s) for s in round_summaries)
        assert len(progressive_summary) < combined_length

    def test_extract_key_insights(self, summarizer):
        """Test key insight extraction from messages."""
        messages = [
            {
                "content": "The most critical factor is maintaining backward compatibility."
            },
            {"content": "We must ensure zero downtime during migration."},
            {
                "content": "Cost optimization should be a secondary concern after reliability."
            },
            {"content": "User experience must not degrade during the transition."},
        ]

        insights = summarizer.extract_key_insights(
            messages=messages, topic="System Migration Strategy"
        )

        assert isinstance(insights, list)
        assert len(insights) > 0
        assert all(isinstance(insight, str) for insight in insights)
        # Check that insights were extracted
        assert any("Technical implementation" in insight for insight in insights)
        assert any(
            "crucial" in insight or "monitoring" in insight for insight in insights
        )

    def test_summarize_empty_round(self, summarizer):
        """Test handling of empty message list."""
        summary = summarizer.summarize_round(
            messages=[], topic="Test Topic", round_number=1
        )

        # Should handle gracefully
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summarize_with_special_characters(self, summarizer):
        """Test summarization with special characters and formatting."""
        messages = [
            {"content": "Let's consider the API endpoint: `/api/v2/users/{id}`"},
            {
                "content": 'The JSON response would be: {"status": "success", "data": [...]}'
            },
            {"content": "We need to handle special chars like &, <, > properly."},
        ]

        summary = summarizer.summarize_round(messages, "API Design", 1)

        # Should handle special characters gracefully
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_error_handling(self, summarizer):
        """Test error handling in summarization."""
        # Test with invalid message format - should filter out invalid messages
        invalid_messages = [
            {"wrong_key": "This message has wrong structure"},
            {"content": None},  # Content is None
            {"content": "Valid message"},
        ]

        # Should handle errors gracefully by filtering invalid messages
        summary = summarizer.summarize_round(invalid_messages, "Test", 1)
        # Should still produce output with valid messages
        assert isinstance(summary, str)
        assert len(summary) > 0

        # Test with messages that would cause None to be in content_list
        messages_with_none = [
            {"content": ""},  # Empty content
            {"no_content": "Missing content key"},
            {"content": "   "},  # Whitespace only
        ]

        # Should handle gracefully
        summary = summarizer.summarize_round(messages_with_none, "Test", 1)
        assert isinstance(summary, str)
        # Should return the empty round message
        assert "No discussion content" in summary

    def test_v13_prompt_compliance(self, summarizer):
        """Test that output complies with v1.3 prompt requirements."""
        messages = [
            {
                "speaker": "gpt-4-1",
                "content": "Agent 1 believes we should use microservices.",
            },
            {
                "speaker": "claude-2",
                "content": "Agent 2 prefers a monolithic approach.",
            },
            {"speaker": "gemini-1", "content": "Agent 3 suggests a hybrid solution."},
        ]

        summary = summarizer.summarize_round(messages, "Architecture", 1)

        # Check v1.3 requirements
        # Should be agent-agnostic (no agent names)
        assert "gpt-4-1" not in summary
        assert "claude-2" not in summary
        assert "gemini-1" not in summary
        assert "Agent 1" not in summary.replace("Agents", "")  # Allow "Agents" plural

        # Should focus on content, not attribution
        assert "discussion" in summary.lower() or "points" in summary.lower()

    def test_integration_with_state(self, summarizer):
        """Test summarizer integration with Virtual Agora state."""
        # Simulate state messages format
        state_messages = [
            {
                "id": "msg-1",
                "agent_id": "agent_1",
                "content": "First point about implementation.",
                "timestamp": datetime.now(),
                "round_number": 1,
                "topic": "Implementation",
                "message_type": "discussion",
            },
            {
                "id": "msg-2",
                "agent_id": "agent_2",
                "content": "Response with alternative approach.",
                "timestamp": datetime.now(),
                "round_number": 1,
                "topic": "Implementation",
                "message_type": "discussion",
            },
        ]

        # Extract content for summarization
        messages_for_summary = [
            {"content": msg["content"], "speaker": msg["agent_id"]}
            for msg in state_messages
        ]

        summary = summarizer.summarize_round(
            messages=messages_for_summary, topic="Implementation", round_number=1
        )

        assert isinstance(summary, str)
        assert len(summary) > 0
        # Summary should capture the discussion essence
        assert "discussion" in summary.lower()

    def test_summarize_topic_conclusion_basic(self, summarizer):
        """Test basic topic conclusion summarization functionality."""
        round_summaries = [
            "Initial discussion focused on technical requirements and constraints.",
            "Second round examined implementation options and resource needs.",
            "Final round reached consensus on phased approach with monitoring.",
        ]

        final_considerations = [
            "We should prioritize security throughout implementation.",
            "Timeline needs to be realistic and account for testing phases.",
            "Stakeholder communication will be critical for success.",
        ]

        topic_summary = summarizer.summarize_topic_conclusion(
            round_summaries=round_summaries,
            final_considerations=final_considerations,
            topic="Technical Implementation Strategy",
        )

        assert isinstance(topic_summary, str)
        assert len(topic_summary) > 0
        # Should contain key elements from the prompt
        assert (
            "consensus" in topic_summary.lower() or "agreement" in topic_summary.lower()
        )

    def test_summarize_topic_conclusion_empty_inputs(self, summarizer):
        """Test topic conclusion summarization with empty inputs."""
        topic_summary = summarizer.summarize_topic_conclusion(
            round_summaries=[], final_considerations=[], topic="Empty Topic"
        )

        assert isinstance(topic_summary, str)
        assert len(topic_summary) > 0

    def test_summarize_topic_conclusion_with_complex_content(self, summarizer):
        """Test topic conclusion summarization with complex, realistic content."""
        round_summaries = [
            "Initial exploration of cloud vs on-premise deployment options, with security and cost considerations raised.",
            "Deep dive into technical architecture, including scalability patterns and data flow requirements.",
            "Final evaluation of hybrid approach, with consensus emerging on gradual migration strategy.",
        ]

        final_considerations = [
            "Security audit should be completed before any production deployment.",
            "Budget approval needed for additional infrastructure components.",
            "Team training on new technologies should begin immediately.",
            "Risk mitigation plan must address potential downtime scenarios.",
        ]

        topic_summary = summarizer.summarize_topic_conclusion(
            round_summaries=round_summaries,
            final_considerations=final_considerations,
            topic="Cloud Migration Strategy",
        )

        assert isinstance(topic_summary, str)
        assert len(topic_summary) > 50  # Should be substantial

        # Should be agent-agnostic (no specific agent references)
        assert "agent 1" not in topic_summary.lower()
        assert "participant" not in topic_summary.lower()

        # Should capture the essence of the discussion
        assert any(
            word in topic_summary.lower()
            for word in ["consensus", "agreement", "approach", "strategy"]
        )

    def test_topic_conclusion_prompt_structure(self, summarizer):
        """Test that topic conclusion follows the correct prompt structure."""
        round_summaries = ["Summary 1", "Summary 2"]
        final_considerations = ["Consideration 1", "Consideration 2"]

        # Mock the generate_response method to capture the prompt
        original_generate = summarizer.generate_response
        captured_prompt = None

        def mock_generate(prompt):
            nonlocal captured_prompt
            captured_prompt = prompt
            return "Test summary response"

        summarizer.generate_response = mock_generate

        try:
            summarizer.summarize_topic_conclusion(
                round_summaries=round_summaries,
                final_considerations=final_considerations,
                topic="Test Topic",
            )

            # Verify prompt structure
            assert captured_prompt is not None
            assert "Topic Conclusion Summary" in captured_prompt
            assert "Test Topic" in captured_prompt
            assert "Round Summaries" in captured_prompt
            assert "Final Considerations" in captured_prompt
            assert "single paragraph" in captured_prompt

        finally:
            # Restore original method
            summarizer.generate_response = original_generate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
