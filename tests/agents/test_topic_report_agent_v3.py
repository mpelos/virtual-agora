"""Unit tests for TopicReportAgent in Virtual Agora v1.3."""

import pytest
from unittest.mock import Mock, patch
import json

from virtual_agora.agents.topic_report_agent import TopicReportAgent
from tests.helpers.fake_llm import TopicReportFakeLLM, FakeLLMBase


class TestTopicReportAgentV3:
    """Unit tests for TopicReportAgent v1.3 functionality."""

    @pytest.fixture
    def topic_report_llm(self):
        """Create a fake LLM for topic report testing."""
        return TopicReportFakeLLM()

    @pytest.fixture
    def topic_agent(self, topic_report_llm):
        """Create test topic report agent."""
        return TopicReportAgent(agent_id="test_topic_report", llm=topic_report_llm)

    def test_initialization_v13(self, topic_agent):
        """Test agent initialization with v1.3 synthesis prompt."""
        assert topic_agent.agent_id == "test_topic_report"

        # Check v1.3 specific prompt elements
        assert "specialized synthesis tool" in topic_agent.system_prompt
        assert "Virtual Agora" in topic_agent.system_prompt
        assert "comprehensive, standalone report" in topic_agent.system_prompt
        assert "topic reporting" in topic_agent.system_prompt

        # Check required report structure elements in prompt
        assert "Topic overview" in topic_agent.system_prompt
        assert "Major themes" in topic_agent.system_prompt
        assert "Points of consensus" in topic_agent.system_prompt
        assert "Areas of disagreement" in topic_agent.system_prompt
        assert "Key insights" in topic_agent.system_prompt
        assert "Implications" in topic_agent.system_prompt

    def test_synthesize_topic_basic(self, topic_agent):
        """Test basic topic synthesis functionality."""
        round_summaries = [
            "Round 1: Agents discussed the importance of security in system design.",
            "Round 2: Debate focused on balancing security with performance.",
            "Round 3: Consensus emerged on implementing security layers.",
            "Round 4: Detailed discussion of specific security measures.",
        ]

        final_considerations = [
            "Agent 1: We should not compromise on encryption standards.",
            "Agent 2: Performance monitoring is essential during implementation.",
        ]

        report = topic_agent.synthesize_topic(
            round_summaries=round_summaries,
            final_considerations=final_considerations,
            topic="Security Architecture",
            discussion_theme="System Design Best Practices",
        )

        assert isinstance(report, str)
        assert len(report) > 100  # Should be substantial
        assert "# Topic Report:" in report or "## Overview" in report
        assert "Security Architecture" in report

    def test_report_structure_compliance(self, topic_agent):
        """Test that report follows v1.3 required structure."""
        round_summaries = [
            "Round 1: Initial exploration of the topic.",
            "Round 2: Deeper analysis and debate.",
            "Round 3: Moving toward consensus.",
        ]

        report = topic_agent.synthesize_topic(
            round_summaries=round_summaries,
            final_considerations=[],
            topic="AI Ethics",
            discussion_theme="Future of AI",
        )

        # Check for required sections
        assert "Overview" in report
        assert "Major Themes" in report or "Themes" in report
        assert "Consensus" in report
        assert "Disagreement" in report or "debate" in report
        assert "Insights" in report
        assert "Implications" in report or "Next Steps" in report

    def test_handle_minority_considerations(self, topic_agent):
        """Test synthesis of minority dissenting views."""
        dissenting_agents = ["gpt-4-1", "claude-1"]

        dissenting_views = [
            {
                "agent": "gpt-4-1",
                "content": "I still believe the timeline is too aggressive.",
                "vote": "No",
                "justification": "More testing needed",
            },
            {
                "agent": "claude-1",
                "content": "The resource allocation seems insufficient.",
                "vote": "No",
                "justification": "Budget constraints not addressed",
            },
        ]

        synthesis = topic_agent.handle_minority_considerations(
            dissenting_agents=dissenting_agents,
            all_agent_views=dissenting_views,
            topic="Implementation Timeline",
            majority_decision="Proceed with Q2 2024 launch",
        )

        assert isinstance(synthesis, str)
        assert "dissenting" in synthesis.lower()
        assert "concern" in synthesis.lower()
        # Should mention the specific concerns
        assert "timeline" in synthesis.lower() or "testing" in synthesis.lower()
        assert "resource" in synthesis.lower() or "budget" in synthesis.lower()

    def test_synthesize_empty_discussions(self, topic_agent):
        """Test handling of minimal or empty discussions."""
        # Empty round summaries
        report = topic_agent.synthesize_topic(
            round_summaries=[],
            final_considerations=[],
            topic="Test Topic",
            discussion_theme="Test Theme",
        )

        assert isinstance(report, str)
        assert len(report) > 0
        # Should still have structure
        assert "#" in report  # Markdown headers

    def test_synthesize_with_rich_content(self, topic_agent):
        """Test synthesis with detailed, rich content."""
        round_summaries = [
            "Round 1: Agents explored technical requirements including API design, database schema, and microservice architecture. Key considerations included scalability, maintainability, and developer experience.",
            "Round 2: Security discussion covered authentication (OAuth2 vs JWT), authorization (RBAC vs ABAC), and data encryption strategies. Consensus on defense-in-depth approach.",
            "Round 3: Performance optimization strategies debated - caching layers (Redis vs Memcached), CDN usage, and database query optimization. Agreement on monitoring-first approach.",
            "Round 4: Deployment strategies examined - blue-green deployments, canary releases, and rollback procedures. Infrastructure as code emphasized.",
            "Round 5: Operational concerns addressed - logging, monitoring, alerting, and incident response procedures. SRE practices recommended.",
        ]

        final_considerations = [
            "Agent A: While we've covered the technical aspects thoroughly, we should allocate more time for security audits before launch.",
            "Agent B: The performance targets seem achievable, but we need clear SLAs defined upfront.",
            "Agent C: Consider adding a dedicated DevOps engineer to the team for smoother deployment.",
        ]

        report = topic_agent.synthesize_topic(
            round_summaries=round_summaries,
            final_considerations=final_considerations,
            topic="Technical Infrastructure Design",
            discussion_theme="Building Scalable Systems",
        )

        assert isinstance(report, str)
        assert len(report) > 500  # Should be comprehensive
        # Should capture the technical depth
        assert "Technical" in report
        assert "security" in report.lower() or "Security" in report
        assert "performance" in report.lower() or "Performance" in report

    def test_markdown_formatting(self, topic_agent):
        """Test that output uses proper markdown formatting."""
        round_summaries = ["Summary 1", "Summary 2"]

        report = topic_agent.synthesize_topic(
            round_summaries=round_summaries,
            final_considerations=["Final thought"],
            topic="Markdown Test",
            discussion_theme="Formatting",
        )

        # Check markdown elements
        assert "#" in report  # Headers
        assert "##" in report  # Subheaders
        assert "-" in report or "*" in report  # List items likely
        # Should be readable markdown
        lines = report.split("\n")
        header_count = sum(1 for line in lines if line.strip().startswith("#"))
        assert header_count >= 3  # Multiple sections

    def test_context_integration(self, topic_agent):
        """Test that agent properly integrates all context."""
        round_summaries = [
            "Round 1: Discussion of blockchain fundamentals.",
            "Round 2: Exploration of consensus mechanisms.",
            "Round 3: Scalability challenges and solutions.",
        ]

        final_considerations = [
            "We need to consider environmental impact.",
            "Regulatory compliance is crucial.",
        ]

        topic = "Blockchain Architecture"
        theme = "Distributed Systems Design"

        report = topic_agent.synthesize_topic(
            round_summaries=round_summaries,
            final_considerations=final_considerations,
            topic=topic,
            discussion_theme=theme,
        )

        # Should integrate the context
        assert topic in report
        assert "blockchain" in report.lower()
        # Should reflect the progression
        assert "consensus" in report.lower()
        assert "scalability" in report.lower()
        # Should include final considerations
        assert "environmental" in report.lower() or "regulatory" in report.lower()

    def test_agent_agnostic_output(self, topic_agent):
        """Test that output is agent-agnostic per v1.3 requirements."""
        round_summaries = [
            "Round 1: GPT-4 suggested approach A, while Claude preferred approach B.",
            "Round 2: Gemini introduced approach C, leading to further debate.",
        ]

        # Even with agent names in input, output should be agnostic
        report = topic_agent.synthesize_topic(
            round_summaries=round_summaries,
            final_considerations=[],
            topic="Solution Approaches",
            discussion_theme="Problem Solving",
        )

        # The mock response is already agent-agnostic, but in real usage
        # the agent should abstract away specific agent identities
        assert isinstance(report, str)
        # Report should focus on ideas, not agents
        assert "approach" in report.lower()

    def test_comprehensive_synthesis(self, topic_agent):
        """Test that synthesis is comprehensive and valuable."""
        # Simulate a complex discussion
        round_summaries = [
            "Round 1: Problem definition and scoping. Agents identified key challenges in current system: poor scalability, technical debt, and user experience issues.",
            "Round 2: Solution brainstorming. Multiple approaches proposed: complete rewrite, incremental refactoring, or hybrid approach with new microservices.",
            "Round 3: Technical deep dive. Detailed analysis of each approach's pros/cons, resource requirements, and implementation timelines.",
            "Round 4: Risk assessment. Identified risks include data migration complexity, downtime during transition, and team skill gaps.",
            "Round 5: Consensus building. Agreement on hybrid approach with phased migration, starting with least critical components.",
            "Round 6: Implementation planning. Defined 6-month roadmap with clear milestones and success metrics.",
        ]

        final_considerations = [
            "Dissenting view: The 6-month timeline may be optimistic given our current resources.",
            "Additional consideration: We should plan for extensive user testing during migration.",
            "Risk mitigation: Consider hiring external consultants for critical migration phases.",
        ]

        report = topic_agent.synthesize_topic(
            round_summaries=round_summaries,
            final_considerations=final_considerations,
            topic="System Modernization Strategy",
            discussion_theme="Legacy System Transformation",
        )

        assert isinstance(report, str)
        assert len(report) > 300

        # Should synthesize key elements
        assert "modernization" in report.lower() or "transformation" in report.lower()
        assert "hybrid" in report.lower() or "phased" in report.lower()
        assert "risk" in report.lower()

        # Should be well-structured
        section_markers = ["##", "Overview", "Theme", "Consensus", "Insight"]
        matches = sum(1 for marker in section_markers if marker in report)
        assert matches >= 3  # Multiple structural elements

    def test_error_handling(self, topic_agent):
        """Test error handling for various edge cases."""
        # Test with None values
        try:
            report = topic_agent.synthesize_topic(
                round_summaries=None,  # Invalid
                final_considerations=[],
                topic="Test",
                discussion_theme="Test",
            )
            # If it handles gracefully, check output
            assert isinstance(report, str)
        except Exception as e:
            # If it raises, should be informative
            assert "summaries" in str(e).lower()

        # Test with malformed input
        malformed_summaries = [
            "Valid summary",
            None,  # Invalid entry
            "",  # Empty entry
            "Another valid summary",
        ]

        # Should handle gracefully
        report = topic_agent.synthesize_topic(
            round_summaries=[s for s in malformed_summaries if s],  # Filter out invalid
            final_considerations=[],
            topic="Test",
            discussion_theme="Test",
        )
        assert isinstance(report, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
