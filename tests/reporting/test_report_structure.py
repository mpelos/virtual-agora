"""Tests for report structure management."""

import pytest
import json
from pathlib import Path
import tempfile

from virtual_agora.reporting.report_structure import ReportStructureManager


class TestReportStructureManager:
    """Test ReportStructureManager functionality."""

    def setup_method(self):
        """Set up test method."""
        self.manager = ReportStructureManager()

    def test_initialization(self):
        """Test manager initialization."""
        assert self.manager.structure_cache is None
        assert hasattr(self.manager, "STANDARD_SECTIONS")
        assert len(self.manager.STANDARD_SECTIONS) > 0

    def test_define_structure_no_topics(self):
        """Test structure definition with no topics."""
        topic_summaries = {}

        structure = self.manager.define_structure(topic_summaries)

        assert isinstance(structure, list)
        assert "Executive Summary" in structure
        assert "Introduction" in structure
        assert "Conclusions and Recommendations" in structure
        assert self.manager.structure_cache == structure

    def test_define_structure_single_topic(self):
        """Test structure definition with single topic."""
        topic_summaries = {"AI Ethics": "Summary of AI Ethics discussion"}

        structure = self.manager.define_structure(topic_summaries)

        assert "Topic Analysis: AI Ethics" in structure
        assert "Executive Summary" in structure
        assert "Introduction" in structure

    def test_define_structure_multiple_topics(self):
        """Test structure definition with multiple topics."""
        topic_summaries = {
            "Topic 1": "Summary 1",
            "Topic 2": "Summary 2",
            "Topic 3": "Summary 3",
        }

        structure = self.manager.define_structure(topic_summaries)

        assert "Topic Analyses" in structure
        # Check for subsections
        subsections = [s for s in structure if s.startswith("  -")]
        assert len(subsections) == 3

    def test_define_structure_many_topics(self):
        """Test structure definition with many topics."""
        topic_summaries = {f"Topic {i}": f"Summary {i}" for i in range(6)}

        structure = self.manager.define_structure(topic_summaries)

        assert "Discussion Themes" in structure
        # Should group into themes
        assert any("  -" in s for s in structure)

    def test_define_structure_with_main_topic(self):
        """Test structure definition with main topic."""
        topic_summaries = {"Subtopic 1": "Summary 1"}
        main_topic = "Main Topic Overview"

        structure = self.manager.define_structure(
            topic_summaries, main_topic=main_topic
        )

        assert f"Overview: {main_topic}" in structure

    def test_define_structure_with_custom_sections(self):
        """Test structure definition with custom sections."""
        topic_summaries = {"Topic": "Summary"}
        custom_sections = ["Custom Analysis", "Special Considerations"]

        structure = self.manager.define_structure(
            topic_summaries, custom_sections=custom_sections
        )

        assert "Custom Analysis" in structure
        assert "Special Considerations" in structure
        # Custom sections should be before conclusions
        custom_idx = structure.index("Custom Analysis")
        conclusions_idx = structure.index("Conclusions and Recommendations")
        assert custom_idx < conclusions_idx

    def test_identify_themes(self):
        """Test theme identification."""
        topic_summaries = {
            "Legal Framework for AI": "Legal discussion",
            "Technical Implementation": "Technical details",
            "Social Impact": "Social considerations",
            "Economic Benefits": "Economic analysis",
            "Future Trends": "Future outlook",
        }

        themes = self.manager._identify_themes(topic_summaries)

        assert len(themes) > 0
        assert any("Technical" in theme for theme in themes)
        assert any("Legal" in theme for theme in themes)
        assert any("Social" in theme for theme in themes)
        assert any("Economic" in theme for theme in themes)
        assert any("Future" in theme for theme in themes)

    def test_integrate_custom_sections(self):
        """Test custom section integration."""
        base_sections = [
            "Executive Summary",
            "Introduction",
            "Topic Analysis",
            "Conclusions and Recommendations",
            "Session Metadata",
            "Appendices",
        ]
        custom_sections = ["Risk Assessment", "Implementation Plan"]

        integrated = self.manager._integrate_custom_sections(
            base_sections.copy(), custom_sections
        )

        # Custom sections should be inserted before the last 3 sections
        risk_idx = integrated.index("Risk Assessment")
        conclusions_idx = integrated.index("Conclusions and Recommendations")
        assert risk_idx < conclusions_idx

    def test_validate_structure(self):
        """Test structure validation."""
        sections = [
            "Executive Summary",
            "",  # Empty section
            "Introduction",
            "  - Subsection 1",
            "  - Subsection 2",
            "Analysis",
            "Analysis",  # Duplicate
            "  - Sub Analysis",
            "Conclusions and Recommendations",
        ]

        validated = self.manager._validate_structure(sections)

        # Empty sections removed
        assert "" not in validated
        # Duplicates removed from main sections
        assert validated.count("Analysis") == 1
        # Subsections preserved
        assert "  - Subsection 1" in validated
        assert "  - Subsection 2" in validated
        assert "  - Sub Analysis" in validated
        # Required sections present
        assert "Executive Summary" in validated
        assert "Introduction" in validated
        assert "Conclusions and Recommendations" in validated

    def test_export_structure_json(self, tmp_path):
        """Test structure export to JSON."""
        # Define a structure first
        topic_summaries = {"Topic": "Summary"}
        structure = self.manager.define_structure(topic_summaries)

        # Export to JSON
        output_path = tmp_path / "structure.json"
        exported = self.manager.export_structure(output_path)

        assert output_path.exists()
        assert exported["report_structure"] == structure
        assert exported["section_count"] == len(structure)

        # Verify JSON content
        loaded = json.loads(output_path.read_text())
        assert loaded["report_structure"] == structure

    def test_export_structure_no_structure_defined(self):
        """Test export when no structure is defined."""
        with pytest.raises(ValueError, match="No structure defined"):
            self.manager.export_structure()

    def test_get_section_hierarchy(self):
        """Test getting section hierarchy."""
        # Define structure with subsections
        self.manager.structure_cache = [
            "Executive Summary",
            "Introduction",
            "Topic Analyses",
            "  - Topic 1",
            "  - Topic 2",
            "Key Insights",
            "Conclusions",
        ]

        hierarchy = self.manager.get_section_hierarchy()

        assert "Topic Analyses" in hierarchy
        assert hierarchy["Topic Analyses"] == ["Topic 1", "Topic 2"]
        assert hierarchy["Executive Summary"] == []
        assert hierarchy["Key Insights"] == []

    def test_error_handling_in_define_structure(self):
        """Test error handling in structure definition."""

        # Test with invalid input that causes an error
        class BadDict(dict):
            def keys(self):
                raise RuntimeError("Test error")

        bad_summaries = BadDict()

        # Should return standard sections on error
        structure = self.manager.define_structure(bad_summaries)
        assert structure == self.manager.STANDARD_SECTIONS

    def test_theme_identification_edge_cases(self):
        """Test theme identification with edge cases."""
        # No matching themes
        topic_summaries = {
            "Random Topic 1": "Summary",
            "Random Topic 2": "Summary",
        }

        themes = self.manager._identify_themes(topic_summaries)

        # Should return generic themes
        assert len(themes) > 0
        assert any("Primary" in theme or "Core" in theme for theme in themes)
