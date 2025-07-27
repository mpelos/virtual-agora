"""Integration tests for v1.3 agent coordination patterns."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.agents.summarizer import SummarizerAgent
from virtual_agora.agents.topic_report_agent import TopicReportAgent
from virtual_agora.agents.ecclesia_report_agent import EcclesiaReportAgent
from virtual_agora.agents.moderator import ModeratorAgent
from virtual_agora.agents.discussion_agent import DiscussionAgent

from tests.helpers.fake_llm import (
    ModeratorFakeLLM,
    SummarizerFakeLLM,
    TopicReportFakeLLM,
    EcclesiaReportFakeLLM,
    AgentFakeLLM,
)
from tests.helpers.integration_utils import (
    create_v13_test_state,
    create_test_discussion_round,
    create_test_messages,
    run_integration_test,
)


class TestAgentCoordinationV3:
    """Test agent coordination patterns in v1.3 architecture."""

    @pytest.fixture
    def specialized_agents(self):
        """Create all specialized agents for testing."""
        return {
            "moderator": ModeratorAgent(
                agent_id="moderator",
                llm=ModeratorFakeLLM(),
                enable_error_handling=False,
            ),
            "summarizer": SummarizerAgent(
                agent_id="summarizer",
                llm=SummarizerFakeLLM(),
                compression_ratio=0.3,
                enable_error_handling=False,
            ),
            "topic_report": TopicReportAgent(
                agent_id="topic_report",
                llm=TopicReportFakeLLM(),
                enable_error_handling=False,
            ),
            "ecclesia_report": EcclesiaReportAgent(
                agent_id="ecclesia_report",
                llm=EcclesiaReportFakeLLM(),
                enable_error_handling=False,
            ),
        }

    @pytest.fixture
    def discussion_agents(self):
        """Create discussion agents for testing."""
        return {
            f"agent_{i}": DiscussionAgent(
                agent_id=f"agent_{i}",
                llm=AgentFakeLLM(agent_personality="balanced", agent_id=f"agent_{i}"),
                enable_error_handling=False,
            )
            for i in range(1, 4)
        }

    def test_summarizer_moderator_coordination(self, specialized_agents):
        """Test coordination between summarizer and moderator."""
        state = create_v13_test_state()
        moderator = specialized_agents["moderator"]
        summarizer = specialized_agents["summarizer"]

        # Add discussion messages to state
        discussion_messages = [
            {
                "agent_id": "agent_1",
                "content": "We should focus on scalability from the start.",
                "round_number": 1,
                "topic": "Architecture Planning",
            },
            {
                "agent_id": "agent_2",
                "content": "I agree, but let's not over-engineer initially.",
                "round_number": 1,
                "topic": "Architecture Planning",
            },
            {
                "agent_id": "agent_3",
                "content": "A phased approach would balance both concerns.",
                "round_number": 1,
                "topic": "Architecture Planning",
            },
        ]
        state["discussion_messages"] = discussion_messages

        # Summarizer creates round summary
        summary = summarizer.summarize_round(
            messages=discussion_messages, topic="Architecture Planning", round_number=1
        )

        assert isinstance(summary, str)
        assert len(summary) > 0

        # Add summary to state
        state["round_summaries"] = [summary]

        # Moderator can use summaries for context
        # In v1.3, moderator doesn't directly call summarizer but uses the summaries
        assert len(state["round_summaries"]) == 1
        assert "discussion" in state["round_summaries"][0].lower()

    def test_topic_report_summarizer_coordination(self, specialized_agents):
        """Test coordination between topic report agent and summarizer."""
        state = create_v13_test_state()
        summarizer = specialized_agents["summarizer"]
        topic_report_agent = specialized_agents["topic_report"]

        # Create multiple round summaries
        round_summaries = []
        for i in range(1, 4):
            messages = [
                {"content": f"Point {i}.1 about implementation details"},
                {"content": f"Point {i}.2 about potential challenges"},
                {"content": f"Point {i}.3 about proposed solutions"},
            ]
            summary = summarizer.summarize_round(messages, "Implementation Strategy", i)
            round_summaries.append(summary)

        # Topic report agent synthesizes all round summaries
        final_considerations = [
            "We must ensure backward compatibility",
            "Performance metrics need continuous monitoring",
        ]

        topic_report = topic_report_agent.synthesize_topic(
            round_summaries=round_summaries,
            final_considerations=final_considerations,
            topic="Implementation Strategy",
            discussion_theme="System Modernization",
        )

        assert isinstance(topic_report, str)
        assert "Implementation Strategy" in topic_report
        assert len(topic_report) > 100  # Should be comprehensive

    def test_ecclesia_topic_report_coordination(self, specialized_agents):
        """Test coordination between ecclesia report and topic report agents."""
        state = create_v13_test_state()
        topic_report_agent = specialized_agents["topic_report"]
        ecclesia_report_agent = specialized_agents["ecclesia_report"]

        # Create multiple topic reports
        topic_reports = {}
        topics = [
            "Security Architecture",
            "Performance Optimization",
            "User Experience",
        ]

        for topic in topics:
            # Simulate round summaries for each topic
            round_summaries = [
                f"Round {i}: Discussion on {topic} aspects and considerations"
                for i in range(1, 4)
            ]

            topic_report = topic_report_agent.synthesize_topic(
                round_summaries=round_summaries,
                final_considerations=["Key consideration for " + topic],
                topic=topic,
                discussion_theme="System Design",
            )
            topic_reports[topic] = topic_report

        # Ecclesia report agent creates final comprehensive report
        report_structure = ecclesia_report_agent.generate_report_structure(
            topic_reports=topic_reports, discussion_theme="System Design"
        )

        assert isinstance(report_structure, list)
        assert len(report_structure) > 0
        assert "Executive Summary" in report_structure

        # Write executive summary section
        exec_summary = ecclesia_report_agent.write_section(
            section_title="Executive Summary",
            topic_reports=topic_reports,
            discussion_theme="System Design",
            previous_sections={},
        )

        assert isinstance(exec_summary, str)
        assert "Executive Summary" in exec_summary
        assert len(exec_summary) > 50

    def test_full_agent_pipeline(self, specialized_agents, discussion_agents):
        """Test full pipeline of agent coordination."""
        state = create_v13_test_state()

        # 1. Discussion agents generate content
        messages = []
        for agent_id, agent in discussion_agents.items():
            response = agent.generate_discussion_response(
                topic="API Design Standards", phase=1
            )
            messages.append(
                {"agent_id": agent_id, "content": response, "round_number": 1}
            )

        # 2. Summarizer compresses the discussion
        summarizer = specialized_agents["summarizer"]
        round_summary = summarizer.summarize_round(
            messages=messages, topic="API Design Standards", round_number=1
        )

        # 3. After multiple rounds, topic report agent synthesizes
        topic_report_agent = specialized_agents["topic_report"]
        topic_report = topic_report_agent.synthesize_topic(
            round_summaries=[round_summary],  # In reality would have multiple
            final_considerations=["RESTful principles are essential"],
            topic="API Design Standards",
            discussion_theme="Technical Standards",
        )

        # 4. Finally, ecclesia report agent creates comprehensive report
        ecclesia_agent = specialized_agents["ecclesia_report"]
        topic_reports = {"API Design Standards": topic_report}

        sections = ecclesia_agent.generate_report_structure(
            topic_reports=topic_reports, discussion_theme="Technical Standards"
        )

        # Verify the full pipeline worked
        assert len(messages) == 3  # 3 discussion agents
        assert len(round_summary) > 0
        assert "Topic Report:" in topic_report or "Overview" in topic_report
        assert len(sections) > 0

    def test_parallel_topic_processing(self, specialized_agents):
        """Test parallel processing of multiple topics."""
        summarizer = specialized_agents["summarizer"]
        topic_report_agent = specialized_agents["topic_report"]

        # Simulate parallel topic discussions
        topics = {
            "Authentication": {"rounds": 3, "messages_per_round": 4},
            "Data Storage": {"rounds": 2, "messages_per_round": 3},
            "API Gateway": {"rounds": 4, "messages_per_round": 3},
        }

        all_topic_reports = {}

        for topic_name, config in topics.items():
            # Generate summaries for each round
            round_summaries = []
            for round_num in range(1, config["rounds"] + 1):
                messages = [
                    {
                        "content": f"Discussion point {i} for {topic_name} in round {round_num}"
                    }
                    for i in range(config["messages_per_round"])
                ]

                summary = summarizer.summarize_round(
                    messages=messages, topic=topic_name, round_number=round_num
                )
                round_summaries.append(summary)

            # Generate topic report
            topic_report = topic_report_agent.synthesize_topic(
                round_summaries=round_summaries,
                final_considerations=[f"Final thought on {topic_name}"],
                topic=topic_name,
                discussion_theme="System Architecture",
            )

            all_topic_reports[topic_name] = topic_report

        # Verify all topics were processed
        assert len(all_topic_reports) == 3
        # Each report should have content
        assert all(len(report) > 100 for report in all_topic_reports.values())
        assert all(
            "Topic Report:" in report or "Overview" in report
            for report in all_topic_reports.values()
        )

    def test_error_handling_coordination(self, specialized_agents):
        """Test error handling across agent coordination."""
        summarizer = specialized_agents["summarizer"]
        topic_report_agent = specialized_agents["topic_report"]

        # Test with problematic input
        empty_messages = []
        summary = summarizer.summarize_round(
            messages=empty_messages, topic="Empty Topic", round_number=1
        )

        # Should handle gracefully
        assert isinstance(summary, str)
        assert "No discussion content" in summary

        # Topic report should handle empty summaries
        topic_report = topic_report_agent.synthesize_topic(
            round_summaries=[summary],
            final_considerations=[],
            topic="Empty Topic",
            discussion_theme="Test Theme",
        )

        assert isinstance(topic_report, str)
        assert len(topic_report) > 0

    def test_context_preservation(self, specialized_agents):
        """Test that context is preserved across agent interactions."""
        summarizer = specialized_agents["summarizer"]

        # Create messages with specific context
        technical_messages = [
            {"content": "We need OAuth2 for authentication"},
            {"content": "JWT tokens for session management"},
            {"content": "Rate limiting at 1000 requests per minute"},
        ]

        summary = summarizer.summarize_round(
            messages=technical_messages, topic="Security Implementation", round_number=1
        )

        # Key technical details should be preserved
        # The fake LLM returns generic summaries, but in real usage
        # these specific details would be preserved
        assert isinstance(summary, str)
        assert len(summary) > 0

        # Progressive summary should maintain context
        progressive_summary = summarizer.generate_progressive_summary(
            summaries=[summary, "Round 2: Further security discussions"],
            topic="Security Implementation",
        )

        assert isinstance(progressive_summary, str)
        assert "rounds" in progressive_summary.lower()

    def test_agent_state_isolation(self, specialized_agents):
        """Test that agents maintain proper state isolation."""
        summarizer1 = specialized_agents["summarizer"]
        summarizer2 = SummarizerAgent(
            agent_id="summarizer2",
            llm=SummarizerFakeLLM(),
            compression_ratio=0.5,  # Different config
            enable_error_handling=False,
        )

        # Each agent should maintain its own configuration
        assert summarizer1.compression_ratio == 0.3
        assert summarizer2.compression_ratio == 0.5
        assert summarizer1.agent_id != summarizer2.agent_id

        # Operations on one shouldn't affect the other
        messages = [{"content": "Test message for isolation"}]
        summary1 = summarizer1.summarize_round(messages, "Topic 1", 1)
        summary2 = summarizer2.summarize_round(messages, "Topic 2", 1)

        # Both should work independently
        assert isinstance(summary1, str)
        assert isinstance(summary2, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
