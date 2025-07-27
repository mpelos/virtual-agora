"""Integration tests for the reporting module."""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime, timedelta

from virtual_agora.reporting import (
    TopicSummaryGenerator,
    ReportStructureManager,
    ReportSectionWriter,
    ReportFileManager,
    ReportMetadataGenerator,
    ReportQualityValidator,
    ReportExporter,
    ReportTemplateManager,
    EnhancedSessionLogger,
)


class TestReportingIntegration:
    """Integration tests for the complete reporting workflow."""

    def setup_method(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_dir = Path(self.temp_dir)

        # Create components
        self.topic_generator = TopicSummaryGenerator(self.base_dir / "summaries")
        self.structure_manager = ReportStructureManager()
        self.section_writer = ReportSectionWriter()
        self.file_manager = ReportFileManager(self.base_dir / "reports")
        self.metadata_generator = ReportMetadataGenerator()
        self.quality_validator = ReportQualityValidator()
        self.exporter = ReportExporter()
        self.template_manager = ReportTemplateManager()
        self.session_logger = EnhancedSessionLogger(
            "integration-test", self.base_dir / "logs"
        )

    def teardown_method(self):
        """Clean up integration test environment."""
        self.session_logger.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_reporting_workflow(self):
        """Test the complete end-to-end reporting workflow."""
        # Step 1: Create test data
        state = self._create_test_state()

        # Step 2: Generate topic summaries
        topic_summaries = {}
        for topic in ["AI Ethics", "Technical Implementation", "Future Outlook"]:
            summary_content = f"Comprehensive discussion on {topic} with key insights."
            summary_path = self.topic_generator.generate_summary(
                topic, summary_content, state
            )
            topic_summaries[topic] = summary_content

            # Log the topic summary generation
            self.session_logger.log_system_event(
                f"Generated summary for {topic}", f"File: {summary_path.name}"
            )

        # Step 3: Define report structure
        report_structure = self.structure_manager.define_structure(
            topic_summaries,
            main_topic="AI Development",
        )

        assert len(report_structure) > 0
        assert "Executive Summary" in report_structure

        # Step 4: Set up section writer
        self.section_writer.set_context(
            topic_summaries, report_structure, main_topic="AI Development"
        )

        # Step 5: Create report directory and files
        report_dir = self.file_manager.create_report_directory(
            state["session_id"], "AI Development"
        )

        # Step 6: Generate and save report sections
        for i, section_title in enumerate(report_structure, 1):
            if section_title.startswith("  -"):
                continue  # Skip subsections for this test

            section_content = self.section_writer.write_section(section_title)
            self.file_manager.save_report_section(i, section_title, section_content)

        # Step 7: Generate metadata
        metadata = self.metadata_generator.generate_metadata(state, topic_summaries)
        metadata_path = self.file_manager.create_metadata_file(metadata)

        # Step 8: Create table of contents and manifest
        toc_path = self.file_manager.create_table_of_contents(report_structure)
        manifest_path = self.file_manager.create_report_manifest()

        # Step 9: Validate report quality
        validation_results = self.quality_validator.validate_report(report_dir)
        assert validation_results["valid"]
        assert validation_results["overall_score"] > 70

        # Step 10: Export in multiple formats
        exports = self.exporter.export_report(
            report_dir, self.base_dir / "exports", format="combined"
        )

        assert "markdown" in exports
        assert "html" in exports
        assert "archive" in exports
        assert all(path.exists() for path in exports.values())

        # Step 11: Verify session analytics
        analytics = self.session_logger.get_session_analytics()
        assert analytics["session_id"] == "integration-test"
        assert analytics["total_events"] > 0

        # Verify all outputs exist and are valid
        assert report_dir.exists()
        assert metadata_path.exists()
        assert toc_path.exists()
        assert manifest_path.exists()
        assert len(list(report_dir.glob("*.md"))) >= 3

    def test_template_integration(self):
        """Test template integration with the reporting workflow."""
        # Create custom template
        custom_template = self.template_manager.customize_template(
            base_template="default",
            styles={"heading_color": "#ff0000"},
            metadata={"include_header": True, "include_footer": True},
        )

        # Save template
        template_path = self.template_manager.save_template(
            custom_template, "integration_test"
        )

        # Generate a simple report
        state = self._create_test_state()
        topic_summaries = {"Test Topic": "Test summary content"}

        # Create report with custom template
        report_dir = self.file_manager.create_report_directory(
            state["session_id"], "Template Test"
        )

        # Generate section with template
        self.section_writer.set_context(topic_summaries, ["Executive Summary"])
        content = self.section_writer.write_section("Executive Summary")

        # Apply template formatting
        formatted_content = self.template_manager.apply_template_to_content(
            content, "integration_test"
        )

        # Save formatted content
        self.file_manager.save_report_section(1, "Executive Summary", formatted_content)

        # Verify template was applied
        assert "template: integration_test" in formatted_content
        assert "Generated using Virtual Agora" in formatted_content

    def test_quality_validation_workflow(self):
        """Test quality validation integrated with report generation."""
        # Generate a report with intentional quality issues
        state = self._create_test_state()
        topic_summaries = {"Test Topic": "Test content"}

        report_dir = self.file_manager.create_report_directory(
            state["session_id"], "Quality Test"
        )

        # Create report with issues
        bad_content = """
# Test Section

This has **unclosed bold formatting

This is a very long sentence with many words that exceeds the recommended maximum sentence length for good readability and should trigger a warning in the quality validator because it's just too long to read comfortably.

Too



Many blank lines above.
"""

        self.file_manager.save_report_section(1, "Test Section", bad_content)

        # Validate (should find issues)
        validation_results = self.quality_validator.validate_report(
            report_dir, check_completeness=False
        )

        assert not validation_results["valid"]
        assert len(validation_results["issues"]) > 0
        assert validation_results["overall_score"] < 90

        # Generate quality report
        quality_report = self.quality_validator.generate_quality_report()
        assert "Report Quality Validation Results" in quality_report
        assert "FAILED" in quality_report

    def test_export_integration(self):
        """Test export integration with various formats."""
        # Create a complete report
        state = self._create_test_state()
        topic_summaries = {
            "Topic 1": "First topic summary",
            "Topic 2": "Second topic summary",
        }

        # Generate complete report
        report_dir = self.file_manager.create_report_directory(
            state["session_id"], "Export Test"
        )

        report_structure = self.structure_manager.define_structure(topic_summaries)
        self.section_writer.set_context(topic_summaries, report_structure)

        # Generate key sections
        for i, section in enumerate(
            ["Executive Summary", "Introduction", "Topic Analyses"], 1
        ):
            content = self.section_writer.write_section(section)
            self.file_manager.save_report_section(i, section, content)

        # Create required files
        metadata = self.metadata_generator.generate_metadata(state, topic_summaries)
        self.file_manager.create_metadata_file(metadata)
        self.file_manager.create_table_of_contents(report_structure)
        self.file_manager.create_report_manifest()

        # Test different export formats
        formats = ["markdown", "html", "archive"]
        export_dir = self.base_dir / "multi_exports"
        export_dir.mkdir()

        for fmt in formats:
            output_path = export_dir / f"report.{fmt}"
            result = self.exporter.export_report(report_dir, output_path, format=fmt)
            assert result.exists()

            # Verify content based on format
            if fmt == "markdown":
                content = result.read_text()
                assert "# Executive Summary" in content
                assert "Topic 1" in content
            elif fmt == "html":
                content = result.read_text()
                assert "<h1>Executive Summary</h1>" in content
                assert "<!DOCTYPE html>" in content

    def test_error_handling_integration(self):
        """Test error handling across integrated components."""
        # Test with invalid state
        invalid_state = {"invalid": "data"}

        # Topic summary generation should handle errors
        try:
            self.topic_generator.generate_summary(
                "Invalid Topic", "Content", invalid_state
            )
        except Exception as e:
            self.session_logger.log_error("Topic summary generation failed", e)

        # Metadata generation should handle errors gracefully
        metadata = self.metadata_generator.generate_metadata(invalid_state)
        assert metadata == {}  # Should return empty dict on error

        # Quality validation should handle missing directories
        results = self.quality_validator.validate_report(Path("/non/existent/path"))
        assert not results["valid"]
        assert "does not exist" in results["issues"][0]

    def test_large_scale_simulation(self):
        """Test the system with a larger-scale simulation."""
        # Create state with more topics and complexity
        state = self._create_large_test_state()

        # Generate summaries for all topics
        topic_summaries = {}
        for i in range(10):
            topic = f"Complex Topic {i+1}"
            summary = (
                f"Detailed analysis of {topic} with multiple perspectives and insights."
            )

            summary_path = self.topic_generator.generate_summary(topic, summary, state)
            topic_summaries[topic] = summary

        # Test structure generation with many topics
        report_structure = self.structure_manager.define_structure(
            topic_summaries, main_topic="Complex Multi-Topic Discussion"
        )

        # Should handle many topics by grouping them
        assert "Discussion Themes" in report_structure

        # Generate metadata
        metadata = self.metadata_generator.generate_metadata(state, topic_summaries)

        # Verify analytics make sense
        assert metadata["topics_discussed"] == 10
        assert metadata["agent_metrics"]["total_agents"] == 5
        assert metadata["content_statistics"]["estimated_words"] > 0

    def _create_test_state(self):
        """Create a test state for integration testing."""
        start_time = datetime.now() - timedelta(hours=1)

        return {
            "session_id": "integration-test-123",
            "start_time": start_time,
            "current_phase": 5,
            "total_messages": 150,
            "main_topic": "AI Development",
            "completed_topics": [
                "AI Ethics",
                "Technical Implementation",
                "Future Outlook",
            ],
            "current_round": 10,
            "agents": {
                "gpt-4": {
                    "model": "gpt-4",
                    "provider": "OpenAI",
                    "role": "participant",
                },
                "claude-3": {
                    "model": "claude-3-opus",
                    "provider": "Anthropic",
                    "role": "participant",
                },
                "moderator": {
                    "model": "gemini-1.5-pro",
                    "provider": "Google",
                    "role": "moderator",
                },
            },
            "messages_by_agent": {
                "gpt-4": 50,
                "claude-3": 45,
                "moderator": 55,
            },
            "topics_info": {
                "AI Ethics": {
                    "start_time": start_time,
                    "end_time": start_time + timedelta(minutes=20),
                    "message_count": 50,
                    "proposed_by": "gpt-4",
                    "status": "completed",
                },
                "Technical Implementation": {
                    "start_time": start_time + timedelta(minutes=20),
                    "end_time": start_time + timedelta(minutes=35),
                    "message_count": 50,
                    "proposed_by": "claude-3",
                    "status": "completed",
                },
                "Future Outlook": {
                    "start_time": start_time + timedelta(minutes=35),
                    "end_time": start_time + timedelta(minutes=50),
                    "message_count": 50,
                    "proposed_by": "moderator",
                    "status": "completed",
                },
            },
            "messages": [
                {"speaker_id": "gpt-4", "content": "A" * 100, "topic": "AI Ethics"},
                {"speaker_id": "claude-3", "content": "B" * 150, "topic": "AI Ethics"},
                {
                    "speaker_id": "moderator",
                    "content": "C" * 200,
                    "topic": "Technical Implementation",
                },
            ],
            "vote_history": [
                {
                    "vote_type": "topic_selection",
                    "phase": 1,
                    "required_votes": 3,
                    "received_votes": 3,
                    "result": "AI Ethics",
                },
            ],
            "votes": [],
            "round_history": [],
            "turn_order_history": [],
            "proposed_topics": [
                "AI Ethics",
                "Technical Implementation",
                "Future Outlook",
                "Unused Topic",
            ],
        }

    def _create_large_test_state(self):
        """Create a larger test state for stress testing."""
        base_state = self._create_test_state()

        # Add more agents
        for i in range(5):
            agent_id = f"agent-{i}"
            base_state["agents"][agent_id] = {
                "model": f"model-{i}",
                "provider": f"provider-{i}",
                "role": "participant",
            }
            base_state["messages_by_agent"][agent_id] = 20

        # Add more topics
        for i in range(10):
            topic = f"Complex Topic {i+1}"
            base_state["topics_info"][topic] = {
                "start_time": base_state["start_time"] + timedelta(minutes=i * 5),
                "end_time": base_state["start_time"] + timedelta(minutes=(i + 1) * 5),
                "message_count": 15,
                "proposed_by": f"agent-{i % 5}",
                "status": "completed",
            }

        base_state["total_messages"] = 500
        base_state["completed_topics"] = [f"Complex Topic {i+1}" for i in range(10)]

        return base_state
