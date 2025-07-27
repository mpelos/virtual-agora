"""Unit tests for EcclesiaReportAgent in Virtual Agora v1.3."""

import pytest
from unittest.mock import Mock, patch
import json

from virtual_agora.agents.ecclesia_report_agent import EcclesiaReportAgent
from tests.helpers.fake_llm import EcclesiaReportFakeLLM, FakeLLMBase


class TestEcclesiaReportAgentV3:
    """Unit tests for EcclesiaReportAgent v1.3 functionality."""

    @pytest.fixture
    def ecclesia_llm(self):
        """Create a fake LLM for ecclesia report testing."""
        return EcclesiaReportFakeLLM()

    @pytest.fixture
    def ecclesia_agent(self, ecclesia_llm):
        """Create test ecclesia report agent."""
        return EcclesiaReportAgent(
            agent_id="test_ecclesia_report",
            llm=ecclesia_llm,
            enable_error_handling=False,
        )

    def test_initialization_v13(self, ecclesia_agent):
        """Test agent initialization with v1.3 'The Writer' prompt."""
        assert ecclesia_agent.agent_id == "test_ecclesia_report"

        # Check v1.3 specific prompt elements
        assert (
            "'The Writer'" in ecclesia_agent.system_prompt
            or "The Writer" in ecclesia_agent.system_prompt
        )
        assert "Virtual Agora" in ecclesia_agent.system_prompt
        assert "final session analysis" in ecclesia_agent.system_prompt
        assert "multi-section final report" in ecclesia_agent.system_prompt

        # Check required capabilities in prompt
        assert "analyze all topic reports" in ecclesia_agent.system_prompt.lower()
        assert "logical structure" in ecclesia_agent.system_prompt
        assert "JSON list" in ecclesia_agent.system_prompt
        assert "synthesize content" in ecclesia_agent.system_prompt
        assert "executive summary" in ecclesia_agent.system_prompt

    def test_generate_report_structure_basic(self, ecclesia_agent):
        """Test basic report structure generation."""
        topic_reports = {
            "AI Safety": "Comprehensive discussion on AI safety measures...",
            "AI Ethics": "Ethical considerations in AI development...",
            "AI Governance": "Governance frameworks for AI systems...",
        }

        structure = ecclesia_agent.generate_report_structure(
            topic_reports=topic_reports,
            discussion_theme="Future of Artificial Intelligence",
        )

        assert isinstance(structure, list)
        assert len(structure) > 0
        assert all(isinstance(section, str) for section in structure)

        # Check for expected sections
        assert "Executive Summary" in structure
        assert any("Recommendation" in s for s in structure)

    def test_generate_report_structure_json_format(self, ecclesia_agent):
        """Test that structure is returned in proper JSON format."""
        topic_reports = {
            "Topic A": "Discussion content A",
            "Topic B": "Discussion content B",
        }

        # The fake LLM returns a JSON string
        structure = ecclesia_agent.generate_report_structure(
            topic_reports=topic_reports, discussion_theme="Test Theme"
        )

        # Should be parsed into a Python list
        assert isinstance(structure, list)
        assert len(structure) >= 3  # Minimum sections
        assert len(structure) <= 10  # Reasonable maximum

    def test_write_section_executive_summary(self, ecclesia_agent):
        """Test writing executive summary section."""
        topic_reports = {
            "Security": "Security discussion findings...",
            "Performance": "Performance optimization strategies...",
            "Scalability": "Scalability considerations...",
        }

        section = ecclesia_agent.write_section(
            section_title="Executive Summary",
            topic_reports=topic_reports,
            discussion_theme="System Architecture Review",
            previous_sections={},
        )

        assert isinstance(section, str)
        assert len(section) > 100  # Should be substantial
        assert "Executive Summary" in section
        assert "Virtual Agora" in section
        # Should mention achievements
        assert "Achievement" in section or "accomplish" in section.lower()

    def test_write_section_with_context(self, ecclesia_agent):
        """Test writing section with previous sections context."""
        topic_reports = {
            "Implementation": "Implementation strategies discussed...",
            "Testing": "Testing methodologies explored...",
        }

        previous_sections = {
            "Executive Summary": "This session explored key aspects...",
            "Key Findings": "Major findings include...",
        }

        section = ecclesia_agent.write_section(
            section_title="Strategic Recommendations",
            topic_reports=topic_reports,
            discussion_theme="Software Development Best Practices",
            previous_sections=previous_sections,
        )

        assert isinstance(section, str)
        assert "recommendation" in section.lower()
        # Should build on previous context
        assert "insights" in section.lower() or "finding" in section.lower()

    def test_cross_topic_synthesis(self, ecclesia_agent):
        """Test cross-topic analysis capabilities."""
        topic_reports = {
            "Frontend Architecture": "Discussion on React vs Vue vs Angular...",
            "Backend Architecture": "Microservices vs monolithic debate...",
            "Database Design": "SQL vs NoSQL considerations...",
            "DevOps Strategy": "CI/CD pipeline and deployment...",
        }

        section = ecclesia_agent.write_section(
            section_title="Cross-Topic Analysis",
            topic_reports=topic_reports,
            discussion_theme="Full-Stack Architecture Design",
            previous_sections={},
        )

        assert isinstance(section, str)
        # Should identify connections
        assert "interconnect" in section.lower() or "relationship" in section.lower()
        assert "pattern" in section.lower()

    def test_generate_full_report(self, ecclesia_agent):
        """Test generating a complete report with all sections."""
        topic_reports = {
            "Topic 1": "First topic detailed discussion...",
            "Topic 2": "Second topic comprehensive analysis...",
            "Topic 3": "Third topic strategic planning...",
        }

        # First generate structure
        structure = ecclesia_agent.generate_report_structure(
            topic_reports=topic_reports, discussion_theme="Strategic Planning Session"
        )

        # Then write each section
        full_report = {}
        previous_sections = {}

        for section_title in structure:
            section_content = ecclesia_agent.write_section(
                section_title=section_title,
                topic_reports=topic_reports,
                discussion_theme="Strategic Planning Session",
                previous_sections=previous_sections,
            )
            full_report[section_title] = section_content
            previous_sections[section_title] = section_content

        # Verify complete report
        assert len(full_report) == len(structure)
        assert all(len(content) > 50 for content in full_report.values())
        assert "Executive Summary" in full_report

    def test_handle_empty_topics(self, ecclesia_agent):
        """Test handling of empty or minimal topic reports."""
        topic_reports = {}  # No topics

        structure = ecclesia_agent.generate_report_structure(
            topic_reports=topic_reports, discussion_theme="Test Theme"
        )

        # Should still generate a structure
        assert isinstance(structure, list)
        assert len(structure) > 0

        # Should handle section writing gracefully
        section = ecclesia_agent.write_section(
            section_title="Executive Summary",
            topic_reports=topic_reports,
            discussion_theme="Test Theme",
            previous_sections={},
        )

        assert isinstance(section, str)
        assert len(section) > 0

    def test_markdown_formatting_quality(self, ecclesia_agent):
        """Test quality of markdown formatting in output."""
        topic_reports = {
            "Technical Discussion": "Deep technical analysis...",
            "Business Impact": "Business implications explored...",
        }

        section = ecclesia_agent.write_section(
            section_title="Key Findings",
            topic_reports=topic_reports,
            discussion_theme="Technology Strategy",
            previous_sections={},
        )

        # Check markdown quality
        lines = section.split("\n")

        # Should have headers
        header_lines = [l for l in lines if l.strip().startswith("#")]
        assert len(header_lines) >= 1

        # Should have proper formatting
        assert any("##" in line for line in lines)  # Subheaders
        assert any("-" in line for line in lines)  # List items likely

    def test_thematic_coherence(self, ecclesia_agent):
        """Test that output maintains thematic coherence."""
        discussion_theme = "Sustainable Technology Development"

        topic_reports = {
            "Green Computing": "Environmental impact of computing...",
            "Energy Efficiency": "Optimizing energy consumption...",
            "Circular Economy": "Hardware lifecycle management...",
        }

        # Generate multiple sections
        sections = []
        section_titles = [
            "Executive Summary",
            "Key Themes and Patterns",
            "Strategic Recommendations",
        ]

        for title in section_titles:
            section = ecclesia_agent.write_section(
                section_title=title,
                topic_reports=topic_reports,
                discussion_theme=discussion_theme,
                previous_sections={},
            )
            sections.append(section)

        # All sections should relate to the theme
        combined_text = " ".join(sections).lower()
        assert "sustainable" in combined_text or "environment" in combined_text

        # Should maintain consistent focus
        for section in sections:
            assert len(section) > 50  # Substantial content

    def test_professional_tone(self, ecclesia_agent):
        """Test that output maintains professional analytical tone."""
        topic_reports = {
            "Market Analysis": "Current market trends and opportunities...",
            "Competitive Landscape": "Competitor analysis and positioning...",
            "Growth Strategy": "Strategic growth initiatives...",
        }

        section = ecclesia_agent.write_section(
            section_title="Strategic Recommendations",
            topic_reports=topic_reports,
            discussion_theme="Business Strategy Review",
            previous_sections={},
        )

        # Check for professional language
        assert "recommend" in section.lower()
        assert any(
            word in section.lower()
            for word in ["strategic", "analysis", "insight", "finding"]
        )

        # Should avoid casual language (this is implicit in the mock responses)
        assert isinstance(section, str)

    def test_structure_flexibility(self, ecclesia_agent):
        """Test that agent can generate varied structures based on content."""
        # Technical topics
        technical_reports = {
            "API Design": "RESTful vs GraphQL analysis...",
            "Database Schema": "Normalization strategies...",
            "Caching Strategy": "Redis implementation...",
        }

        tech_structure = ecclesia_agent.generate_report_structure(
            topic_reports=technical_reports, discussion_theme="Technical Architecture"
        )

        # Business topics
        business_reports = {
            "Revenue Model": "Subscription vs licensing...",
            "Market Entry": "Geographic expansion strategy...",
            "Partnerships": "Strategic alliance opportunities...",
        }

        business_structure = ecclesia_agent.generate_report_structure(
            topic_reports=business_reports, discussion_theme="Business Strategy"
        )

        # Both should be valid but potentially different
        assert isinstance(tech_structure, list)
        assert isinstance(business_structure, list)
        assert len(tech_structure) >= 3
        assert len(business_structure) >= 3

        # Common sections should exist
        assert any("Summary" in s for s in tech_structure)
        assert any("Summary" in s for s in business_structure)

    def test_error_handling(self, ecclesia_agent):
        """Test error handling for edge cases."""
        # Test with None values
        try:
            structure = ecclesia_agent.generate_report_structure(
                topic_reports=None, discussion_theme="Test"  # Invalid
            )
            # If it handles gracefully, check output
            assert isinstance(structure, list)
        except Exception as e:
            # Should have informative error
            assert "topic" in str(e).lower() or "report" in str(e).lower()

        # Test section writing with invalid title
        topic_reports = {"Topic": "Content"}

        section = ecclesia_agent.write_section(
            section_title="",  # Empty title
            topic_reports=topic_reports,
            discussion_theme="Test",
            previous_sections={},
        )

        # Should handle gracefully
        assert isinstance(section, str)
        assert len(section) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
