"""Tests for report section writing."""

import pytest
from datetime import datetime

from virtual_agora.reporting.report_writer import ReportSectionWriter


class TestReportSectionWriter:
    """Test ReportSectionWriter functionality."""

    def setup_method(self):
        """Set up test method."""
        self.writer = ReportSectionWriter()

        # Set up test context
        self.topic_summaries = {
            "AI Ethics": "Discussion on ethical implications of AI systems.",
            "Technical Implementation": "Technical details and architecture.",
            "Future Outlook": "Predictions and trends for the future.",
        }

        self.report_structure = [
            "Executive Summary",
            "Introduction",
            "Topic Analyses",
            "  - AI Ethics",
            "  - Technical Implementation",
            "  - Future Outlook",
            "Key Insights and Findings",
            "Conclusions and Recommendations",
            "Session Metadata",
            "Appendices",
        ]

        self.writer.set_context(
            self.topic_summaries, self.report_structure, main_topic="AI Development"
        )

    def test_initialization(self):
        """Test writer initialization."""
        writer = ReportSectionWriter()
        assert writer.topic_summaries == {}
        assert writer.report_structure == []
        assert writer.main_topic is None

    def test_set_context(self):
        """Test setting context."""
        assert self.writer.topic_summaries == self.topic_summaries
        assert self.writer.report_structure == self.report_structure
        assert self.writer.main_topic == "AI Development"

    def test_write_executive_summary(self):
        """Test executive summary generation."""
        content = self.writer.write_section("Executive Summary")

        assert "## Executive Summary" in content
        assert "3 interconnected topics" in content
        assert "AI Development" in content
        assert "**Topics Discussed:**" in content
        assert "1. AI Ethics" in content
        assert "2. Technical Implementation" in content
        assert "3. Future Outlook" in content

    def test_write_executive_summary_no_topics(self):
        """Test executive summary with no topics."""
        self.writer.set_context({}, [], None)
        content = self.writer.write_section("Executive Summary")

        assert "## Executive Summary" in content
        assert "No topics were discussed" in content

    def test_write_introduction(self):
        """Test introduction generation."""
        content = self.writer.write_section("Introduction")

        assert "## Introduction" in content
        assert "### Background" in content
        assert "Virtual Agora" in content
        assert "### Session Overview" in content
        assert "AI Development" in content
        assert "### Methodology" in content
        assert "### Report Structure" in content

    def test_write_topic_overview(self):
        """Test topic overview generation."""
        content = self.writer.write_section("Overview: AI Development")

        assert "## Overview: AI Development" in content
        assert "AI Development" in content
        assert "### Context and Significance" in content
        assert "### Approach" in content

    def test_write_discussion_overview(self):
        """Test discussion overview generation."""
        content = self.writer.write_section("Discussion Overview")

        assert "## Discussion Overview" in content
        assert "### Discussion Flow" in content
        assert "**1. AI Ethics**" in content
        assert "**2. Technical Implementation**" in content
        assert "**3. Future Outlook**" in content

    def test_write_topic_analyses(self):
        """Test topic analyses generation."""
        content = self.writer.write_section("Topic Analyses")

        assert "## Topic Analyses" in content
        assert "### AI Ethics" in content
        assert "Discussion on ethical implications" in content
        assert "### Technical Implementation" in content
        assert "### Future Outlook" in content
        assert content.count("---") >= 2  # Separators between topics

    def test_write_discussion_themes(self):
        """Test discussion themes generation."""
        content = self.writer.write_section("Discussion Themes")

        assert "## Discussion Themes" in content
        assert "### Emerging Patterns" in content
        assert "**Interconnectedness**" in content
        assert "**Complexity**" in content
        assert "**Innovation**" in content
        assert "**Collaboration**" in content

    def test_write_key_insights(self):
        """Test key insights generation."""
        content = self.writer.write_section("Key Insights and Findings")

        assert "## Key Insights and Findings" in content
        assert "### Major Findings" in content
        assert "**From the discussion on AI Ethics:**" in content
        assert "**From the discussion on Technical Implementation:**" in content
        assert "### Consensus Points" in content

    def test_write_cross_topic_synthesis(self):
        """Test cross-topic synthesis generation."""
        content = self.writer.write_section("Cross-Topic Synthesis")

        assert "## Cross-Topic Synthesis" in content
        assert "### Interconnections" in content
        assert "AI Ethics" in content
        assert "Technical Implementation" in content
        assert "↔" in content  # Connection symbol
        assert "### Integrated Perspective" in content

    def test_write_conclusions(self):
        """Test conclusions generation."""
        content = self.writer.write_section("Conclusions and Recommendations")

        assert "## Conclusions and Recommendations" in content
        assert "### Conclusions" in content
        assert "1. Regarding **AI Ethics**:" in content
        assert "2. Regarding **Technical Implementation**:" in content
        assert "### Recommendations" in content
        assert "### Next Steps" in content

    def test_write_session_metadata(self):
        """Test session metadata generation."""
        context = {
            "session_id": "test-123",
            "start_time": datetime.now().isoformat(),
            "duration": 60,
            "total_messages": 150,
            "participating_agents": 5,
        }

        content = self.writer.write_section("Session Metadata", context)

        assert "## Session Metadata" in content
        assert "### Session Information" in content
        assert "**Session ID**: test-123" in content
        assert "**Duration**: 60 minutes" in content
        assert "**Total Messages**: 150" in content
        assert "**Participating Agents**: 5" in content
        assert "### Discussion Statistics" in content
        assert "**Topics Discussed**: 3" in content

    def test_write_appendices(self):
        """Test appendices generation."""
        content = self.writer.write_section("Appendices")

        assert "## Appendices" in content
        assert "### Appendix A: Topic List" in content
        assert "1. AI Ethics" in content
        assert "2. Technical Implementation" in content
        assert "3. Future Outlook" in content
        assert "### Appendix B: Glossary" in content
        assert "Virtual Agora" in content
        assert "### Appendix C: Additional Resources" in content

    def test_write_custom_section(self):
        """Test custom section generation."""
        content = self.writer.write_section("Custom Analysis")

        assert "## Custom Analysis" in content
        assert "This section explores custom analysis" in content

    def test_write_custom_section_with_topic(self):
        """Test custom section with topic name."""
        content = self.writer.write_section("Deep Dive: AI Ethics")

        assert "## Deep Dive: AI Ethics" in content
        assert "### Analysis: AI Ethics" in content
        assert "Discussion on ethical implications" in content

    def test_write_section_with_error(self):
        """Test section writing with error handling."""
        # Create a writer that will encounter an error
        writer = ReportSectionWriter()
        # Don't set context - this should cause errors

        content = writer.write_section("Executive Summary")

        assert "## Executive Summary" in content
        assert "Section generation failed" in content

    def test_subsection_handling(self):
        """Test handling of subsection markers."""
        content = self.writer.write_section("  - AI Ethics")

        # Should clean the title
        assert "## AI Ethics" in content or "### Analysis: AI Ethics" in content

    def test_empty_topic_summaries(self):
        """Test handling empty topic summaries."""
        self.writer.set_context({}, ["Executive Summary"], None)

        content = self.writer.write_section("Executive Summary")

        assert "## Executive Summary" in content
        assert "No topics were discussed" in content

    def test_cross_topic_synthesis_single_topic(self):
        """Test cross-topic synthesis with single topic."""
        self.writer.set_context(
            {"Single Topic": "Summary"}, ["Cross-Topic Synthesis"], None
        )

        content = self.writer.write_section("Cross-Topic Synthesis")

        assert "## Cross-Topic Synthesis" in content
        # Should handle single topic gracefully
        assert "connections and relationships" in content
