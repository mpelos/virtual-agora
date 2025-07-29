"""Tests for ReportWriterAgent implementation."""

import json
import pytest
from unittest.mock import Mock, patch

from virtual_agora.agents.report_writer_agent import ReportWriterAgent, ReportStructure
from virtual_agora.utils.exceptions import ConfigurationError


class TestReportWriterAgent:
    """Test cases for ReportWriterAgent."""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        llm = Mock()
        llm.invoke.return_value = Mock(content="Test response")
        return llm

    @pytest.fixture
    def report_writer_agent(self, mock_llm):
        """Create ReportWriterAgent instance for testing."""
        return ReportWriterAgent(agent_id="test_report_writer", llm=mock_llm)

    def test_initialization(self, mock_llm):
        """Test ReportWriterAgent initialization."""
        agent = ReportWriterAgent(agent_id="test_agent", llm=mock_llm)

        assert agent.agent_id == "test_agent"
        assert agent.llm == mock_llm
        assert agent.PROMPT_VERSION == "1.3"

    def test_create_topic_report_structure(self, report_writer_agent, mock_llm):
        """Test topic report structure creation."""
        # Mock LLM response with valid JSON
        mock_response = """
        {
            "sections": [
                {"title": "Executive Summary", "description": "Overview of the topic"},
                {"title": "Key Arguments", "description": "Main arguments presented"},
                {"title": "Conclusions", "description": "Final thoughts and implications"}
            ]
        }
        """
        mock_llm.invoke.return_value = Mock(content=mock_response)

        source_material = {
            "topic": "Test Topic",
            "theme": "Test Theme",
            "round_summaries": ["Summary 1", "Summary 2"],
            "final_considerations": ["Final thought 1"],
        }

        sections, raw_response = report_writer_agent.create_report_structure(
            source_material, "topic"
        )

        assert len(sections) == 3
        assert sections[0]["title"] == "Executive Summary"
        assert sections[1]["title"] == "Key Arguments"
        assert sections[2]["title"] == "Conclusions"
        assert all("description" in section for section in sections)

    def test_create_session_report_structure(self, report_writer_agent, mock_llm):
        """Test session report structure creation."""
        # Mock LLM response with valid JSON
        mock_response = """
        {
            "sections": [
                {"title": "Executive Summary", "description": "High-level overview"},
                {"title": "Cross-Topic Themes", "description": "Overarching patterns"},
                {"title": "Synthesis", "description": "Integration of insights"}
            ]
        }
        """
        mock_llm.invoke.return_value = Mock(content=mock_response)

        source_material = {
            "theme": "Test Session Theme",
            "topic_reports": {
                "Topic A": "Report content A",
                "Topic B": "Report content B",
            },
        }

        sections, raw_response = report_writer_agent.create_report_structure(
            source_material, "session"
        )

        assert len(sections) == 3
        assert sections[0]["title"] == "Executive Summary"
        assert sections[1]["title"] == "Cross-Topic Themes"
        assert sections[2]["title"] == "Synthesis"

    def test_invalid_report_type(self, report_writer_agent):
        """Test handling of invalid report type."""
        source_material = {"topic": "Test"}

        with pytest.raises(ValueError, match="Unknown report type"):
            report_writer_agent.create_report_structure(source_material, "invalid")

    def test_structure_parsing_fallback(self, report_writer_agent, mock_llm):
        """Test fallback when JSON parsing fails."""
        # Mock LLM response with invalid JSON
        mock_llm.invoke.return_value = Mock(content="Invalid JSON response")

        source_material = {
            "topic": "Test Topic",
            "theme": "Test Theme",
            "round_summaries": ["Summary 1"],
            "final_considerations": [],
        }

        sections, _ = report_writer_agent.create_report_structure(
            source_material, "topic"
        )

        # Should fall back to default structure
        assert len(sections) == 4
        assert any(section["title"] == "Executive Summary" for section in sections)

    def test_write_topic_section(self, report_writer_agent, mock_llm):
        """Test writing a topic report section."""
        mock_llm.invoke.return_value = Mock(
            content="## Executive Summary\n\nTest section content"
        )

        section = {"title": "Executive Summary", "description": "Overview of the topic"}
        source_material = {
            "topic": "Test Topic",
            "theme": "Test Theme",
            "round_summaries": ["Summary 1", "Summary 2"],
            "final_considerations": ["Final thought"],
        }

        content = report_writer_agent.write_section(
            section, source_material, "topic", None
        )

        assert "Executive Summary" in content
        assert "Test section content" in content
        mock_llm.invoke.assert_called_once()

    def test_write_session_section(self, report_writer_agent, mock_llm):
        """Test writing a session report section."""
        mock_llm.invoke.return_value = Mock(
            content="## Cross-Topic Themes\n\nSession analysis content"
        )

        section = {"title": "Cross-Topic Themes", "description": "Overarching patterns"}
        source_material = {
            "theme": "Test Session Theme",
            "topic_reports": {
                "Topic A": "Report A content",
                "Topic B": "Report B content",
            },
        }

        content = report_writer_agent.write_section(
            section, source_material, "session", None
        )

        assert "Cross-Topic Themes" in content
        assert "Session analysis content" in content

    def test_write_section_with_previous_context(self, report_writer_agent, mock_llm):
        """Test section writing with previous sections context."""
        mock_llm.invoke.return_value = Mock(
            content="## Key Arguments\n\nArguments content"
        )

        section = {"title": "Key Arguments", "description": "Main arguments presented"}
        source_material = {
            "topic": "Test Topic",
            "theme": "Test Theme",
            "round_summaries": ["Summary 1"],
            "final_considerations": [],
        }
        previous_sections = ["Previous section content"]

        content = report_writer_agent.write_section(
            section, source_material, "topic", previous_sections
        )

        assert "Key Arguments" in content
        # Check that LLM was called with previous context
        call_args = mock_llm.invoke.call_args[0][0]
        assert "Previous Sections" in call_args

    def test_synthesize_topic_legacy_compatibility(self, report_writer_agent, mock_llm):
        """Test legacy synthesize_topic method for backward compatibility."""
        # Mock structure creation and section writing
        structure_response = """
        {
            "sections": [
                {"title": "Executive Summary", "description": "Overview"},
                {"title": "Analysis", "description": "Detailed analysis"}
            ]
        }
        """
        section_responses = [
            "## Executive Summary\n\nExecutive content",
            "## Analysis\n\nDetailed analysis content",
        ]

        # Mock multiple calls
        mock_llm.invoke.side_effect = [
            Mock(content=structure_response),
            Mock(content=section_responses[0]),
            Mock(content=section_responses[1]),
        ]

        report = report_writer_agent.synthesize_topic(
            round_summaries=["Summary 1", "Summary 2"],
            final_considerations=["Final thought"],
            topic="Test Topic",
            discussion_theme="Test Theme",
        )

        assert "Executive Summary" in report
        assert "Analysis" in report
        assert "Executive content" in report
        assert "Detailed analysis content" in report

        # Should have made 3 calls: structure + 2 sections
        assert mock_llm.invoke.call_count == 3

    def test_generate_report_structure_legacy_compatibility(
        self, report_writer_agent, mock_llm
    ):
        """Test legacy generate_report_structure method."""
        structure_response = """
        {
            "sections": [
                {"title": "Executive Summary", "description": "Overview"},
                {"title": "Key Themes", "description": "Main themes"}
            ]
        }
        """
        mock_llm.invoke.return_value = Mock(content=structure_response)

        topic_reports = {"Topic A": "Report A content", "Topic B": "Report B content"}

        sections = report_writer_agent.generate_report_structure(
            topic_reports, "Test Theme"
        )

        assert sections == ["Executive Summary", "Key Themes"]

    def test_write_section_legacy_compatibility(self, report_writer_agent, mock_llm):
        """Test legacy write_section_legacy method."""
        mock_llm.invoke.return_value = Mock(
            content="## Executive Summary\n\nLegacy content"
        )

        content = report_writer_agent.write_section_legacy(
            section_title="Executive Summary",
            topic_reports={"Topic A": "Report A"},
            discussion_theme="Test Theme",
            previous_sections={"Intro": "Intro content"},
        )

        assert "Executive Summary" in content
        assert "Legacy content" in content

    def test_error_handling_in_structure_creation(self, report_writer_agent, mock_llm):
        """Test error handling during structure creation."""
        # Mock LLM to raise an exception
        mock_llm.invoke.side_effect = Exception("LLM error")

        source_material = {
            "topic": "Test Topic",
            "theme": "Test Theme",
            "round_summaries": ["Summary 1"],
            "final_considerations": [],
        }

        # Should not raise exception, should fall back gracefully
        sections, _ = report_writer_agent.create_report_structure(
            source_material, "topic"
        )

        # Should return default structure
        assert len(sections) == 4
        assert any(section["title"] == "Executive Summary" for section in sections)

    def test_report_structure_validation(self):
        """Test ReportStructure Pydantic model validation."""
        # Valid structure
        valid_data = {
            "sections": [
                {"title": "Section 1", "description": "Description 1"},
                {"title": "Section 2", "description": "Description 2"},
            ]
        }
        structure = ReportStructure(**valid_data)
        assert len(structure.sections) == 2

        # Invalid structure - missing required fields
        with pytest.raises(Exception):  # Pydantic validation error
            ReportStructure(sections=[{"title": "Section 1"}])  # Missing description

    def test_complex_iterative_workflow(self, report_writer_agent, mock_llm):
        """Test complex iterative workflow with multiple sections."""
        # Mock structure response
        structure_response = """
        {
            "sections": [
                {"title": "Introduction", "description": "Introduction to topic"},
                {"title": "Main Analysis", "description": "Core analysis"},
                {"title": "Key Insights", "description": "Important findings"},
                {"title": "Conclusions", "description": "Final conclusions"}
            ]
        }
        """

        # Mock section responses
        section_responses = [
            "## Introduction\n\nIntroduction content",
            "## Main Analysis\n\nAnalysis content with insights",
            "## Key Insights\n\nImportant discoveries",
            "## Conclusions\n\nFinal thoughts and recommendations",
        ]

        # Set up mock responses
        mock_llm.invoke.side_effect = [
            Mock(content=structure_response),
            *[Mock(content=resp) for resp in section_responses],
        ]

        # Test full workflow
        source_material = {
            "topic": "Complex Topic",
            "theme": "Complex Theme",
            "round_summaries": ["Summary 1", "Summary 2", "Summary 3"],
            "final_considerations": ["Final consideration 1", "Final consideration 2"],
        }

        # Phase 1: Create structure
        sections, _ = report_writer_agent.create_report_structure(
            source_material, "topic"
        )
        assert len(sections) == 4

        # Phase 2: Write all sections
        report_parts = []
        previous_sections = []

        for section in sections:
            section_content = report_writer_agent.write_section(
                section, source_material, "topic", previous_sections
            )
            report_parts.append(section_content)
            previous_sections.append(section_content)

        # Verify all sections were generated
        full_report = "\n\n".join(report_parts)
        assert "Introduction" in full_report
        assert "Main Analysis" in full_report
        assert "Key Insights" in full_report
        assert "Conclusions" in full_report

        # Should have made 5 calls total: 1 structure + 4 sections
        assert mock_llm.invoke.call_count == 5

    @pytest.mark.parametrize(
        "report_type,expected_prompt_part",
        [("topic", "Topic Report"), ("session", "Session Report")],
    )
    def test_prompt_customization_by_type(
        self, report_writer_agent, mock_llm, report_type, expected_prompt_part
    ):
        """Test that prompts are customized based on report type."""
        mock_llm.invoke.return_value = Mock(
            content='{"sections": [{"title": "Test", "description": "Test"}]}'
        )

        source_material = {
            "topic": "Test Topic",
            "theme": "Test Theme",
            "round_summaries": ["Summary"],
            "final_considerations": [],
            "topic_reports": {"Topic A": "Report A"},
        }

        report_writer_agent.create_report_structure(source_material, report_type)

        # Check that the prompt includes the expected report type
        call_args = mock_llm.invoke.call_args[0][0]
        assert expected_prompt_part in call_args

    def test_section_isolation(self, report_writer_agent, mock_llm):
        """Test that each section is written independently."""
        mock_responses = [
            "## Section 1\n\nContent for section 1",
            "## Section 2\n\nContent for section 2",
        ]
        mock_llm.invoke.side_effect = [Mock(content=resp) for resp in mock_responses]

        sections = [
            {"title": "Section 1", "description": "First section"},
            {"title": "Section 2", "description": "Second section"},
        ]

        source_material = {
            "topic": "Test Topic",
            "theme": "Test Theme",
            "round_summaries": ["Summary"],
            "final_considerations": [],
        }

        # Write sections independently
        content1 = report_writer_agent.write_section(
            sections[0], source_material, "topic", []
        )
        content2 = report_writer_agent.write_section(
            sections[1], source_material, "topic", [content1]
        )

        assert "Section 1" in content1
        assert "Section 2" in content2
        assert mock_llm.invoke.call_count == 2

        # Second call should include context from first section
        second_call_args = mock_llm.invoke.call_args[0][0]
        assert "Previous Sections" in second_call_args
