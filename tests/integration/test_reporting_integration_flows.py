"""Integration tests for Reporting & Documentation System flows.

This module tests the complete report generation workflow including
topic summaries, final reports, file management, and export functionality.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime
import uuid

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.flow.graph import VirtualAgoraFlow
from virtual_agora.reporting.report_writer import ReportSectionWriter
from virtual_agora.reporting.topic_summary import TopicSummaryGenerator
from virtual_agora.reporting.file_manager import FileManager
from virtual_agora.reporting.exporter import ReportExporter
from virtual_agora.reporting.metadata import MetadataCollector

from ..helpers.fake_llm import ModeratorFakeLLM, AgentFakeLLM
from ..helpers.integration_utils import (
    IntegrationTestHelper,
    TestStateBuilder,
    TestResponseValidator,
    patch_ui_components,
)


class TestTopicSummaryGeneration:
    """Test topic summary generation workflows."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")
        self.validator = TestResponseValidator()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test method."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.integration
    def test_topic_summary_generation_complete_flow(self):
        """Test complete topic summary generation."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add rich discussion data
            state = self._add_comprehensive_discussion_data(state)

            # Generate topic summary
            summary_generator = TopicSummaryGenerator()
            summary = summary_generator.generate_summary(state, topic_index=0)

            # Validate summary structure
            assert self.validator.validate_topic_summary(summary)
            assert len(summary) > 500  # Should be comprehensive
            assert "# Topic Summary:" in summary
            assert "## Overview" in summary
            assert "## Key Points" in summary
            assert "## Conclusions" in summary

    @pytest.mark.integration
    def test_topic_summary_with_minority_dissent(self):
        """Test topic summary including minority dissent."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add discussion with minority dissent
            state = self._add_discussion_with_minority_dissent(state)

            # Generate summary
            summary_generator = TopicSummaryGenerator()
            summary = summary_generator.generate_summary(state, topic_index=0)

            # Should include dissenting views
            summary_lower = summary.lower()
            assert any(
                word in summary_lower
                for word in ["dissent", "minority", "alternative", "concern"]
            )
            assert (
                "## Dissenting Views" in summary
                or "## Alternative Perspectives" in summary
            )

    @pytest.mark.integration
    def test_topic_summary_file_creation(self):
        """Test topic summary file creation and naming."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Set up file manager
            file_manager = FileManager(base_dir=self.temp_dir)
            summary_generator = TopicSummaryGenerator(file_manager=file_manager)

            # Generate and save summary
            topic_title = state["agenda"][0]["title"]
            summary = summary_generator.generate_summary(state, topic_index=0)
            filename = summary_generator.save_summary(summary, topic_title)

            # Verify file was created
            file_path = Path(self.temp_dir) / filename
            assert file_path.exists()
            assert file_path.suffix == ".md"
            assert "topic_summary_" in filename

            # Verify content
            content = file_path.read_text()
            assert content == summary

    @pytest.mark.integration
    def test_multiple_topic_summaries(self):
        """Test generating summaries for multiple topics."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add multiple completed topics
            state = self._add_multiple_completed_topics(state)

            file_manager = FileManager(base_dir=self.temp_dir)
            summary_generator = TopicSummaryGenerator(file_manager=file_manager)

            # Generate summaries for all topics
            summaries = []
            for i, topic in enumerate(state["agenda"]):
                summary = summary_generator.generate_summary(state, topic_index=i)
                filename = summary_generator.save_summary(summary, topic["title"])
                summaries.append((summary, filename))

            # Verify all summaries
            assert len(summaries) == len(state["agenda"])

            for summary, filename in summaries:
                file_path = Path(self.temp_dir) / filename
                assert file_path.exists()
                assert len(summary) > 200  # Each should be substantial

    def _add_comprehensive_discussion_data(
        self, state: VirtualAgoraState
    ) -> VirtualAgoraState:
        """Add comprehensive discussion data for testing."""
        # Add multiple rounds of detailed messages
        discussion_themes = [
            "technical implementation challenges",
            "security and privacy considerations",
            "user experience and accessibility",
            "performance and scalability requirements",
            "business impact and cost analysis",
        ]

        for round_num in range(1, 6):
            for i, agent_id in enumerate(state["speaking_order"]):
                theme = discussion_themes[i % len(discussion_themes)]
                message = {
                    "id": str(uuid.uuid4()),
                    "agent_id": agent_id,
                    "content": f"In round {round_num}, I want to address {theme}. "
                    f"Based on my analysis, we need to consider multiple factors including "
                    f"technical feasibility, resource allocation, timeline constraints, "
                    f"and stakeholder requirements. The implications for {theme} are "
                    f"particularly significant because they affect our overall strategy.",
                    "timestamp": datetime.now(),
                    "round_number": round_num,
                    "topic": state["agenda"][0]["title"],
                    "message_type": "discussion",
                }
                state["messages"].append(message)

        # Add round summaries
        for round_num in range(1, 6):
            theme = discussion_themes[(round_num - 1) % len(discussion_themes)]
            summary = (
                f"Round {round_num}: Comprehensive discussion of {theme} "
                f"with detailed analysis of implementation requirements and challenges."
            )
            state["round_summaries"].append(summary)

        # Add voting rounds
        voting_round = {
            "id": str(uuid.uuid4()),
            "phase": 5,
            "vote_type": "conclusion",
            "options": ["Yes", "No"],
            "start_time": datetime.now(),
            "end_time": datetime.now(),
            "required_votes": 3,
            "received_votes": 3,
            "votes": {
                "agent_1": "Yes. After thorough analysis, I'm satisfied with our conclusions.",
                "agent_2": "Yes. The technical requirements are well-defined.",
                "agent_3": "Yes. Business impact has been adequately assessed.",
            },
            "result": "Yes",
            "status": "completed",
        }
        state["voting_rounds"].append(voting_round)

        return state

    def _add_discussion_with_minority_dissent(
        self, state: VirtualAgoraState
    ) -> VirtualAgoraState:
        """Add discussion data with minority dissent."""
        # Add basic discussion
        state = self._add_comprehensive_discussion_data(state)

        # Add minority dissent in voting
        dissent_voting_round = {
            "id": str(uuid.uuid4()),
            "phase": 5,
            "vote_type": "conclusion",
            "options": ["Yes", "No"],
            "start_time": datetime.now(),
            "end_time": datetime.now(),
            "required_votes": 3,
            "received_votes": 3,
            "votes": {
                "agent_1": "Yes. I believe we've covered the essential points.",
                "agent_2": "Yes. Ready to conclude this topic.",
                "agent_3": "No. I have significant concerns about the security implications "
                "that haven't been adequately addressed. We need more analysis.",
            },
            "result": "Yes",  # Majority rules
            "status": "completed",
            "minority_voters": ["agent_3"],
        }

        # Replace the last voting round
        state["voting_rounds"][-1] = dissent_voting_round

        # Add minority statement
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["minority_statements"] = {
            "agent_3": "While I respect the majority decision, I remain concerned about "
            "the security vulnerabilities that I believe require additional "
            "analysis before implementation. The potential risks outweigh "
            "the proposed benefits in my assessment."
        }

        return state

    def _add_multiple_completed_topics(
        self, state: VirtualAgoraState
    ) -> VirtualAgoraState:
        """Add data for multiple completed topics."""
        # Expand agenda
        additional_topics = [
            {
                "title": "Implementation Timeline",
                "description": "Project timeline and milestones",
            },
            {
                "title": "Resource Allocation",
                "description": "Budget and staffing requirements",
            },
            {
                "title": "Risk Management",
                "description": "Risk assessment and mitigation strategies",
            },
        ]
        state["agenda"].extend(additional_topics)

        # Add messages for each topic
        for topic_index, topic in enumerate(state["agenda"]):
            for round_num in range(1, 4):
                for agent_id in state["speaking_order"]:
                    message = {
                        "id": str(uuid.uuid4()),
                        "agent_id": agent_id,
                        "content": f"Discussing {topic['title']} in round {round_num}. "
                        f"Key considerations include strategic alignment, "
                        f"resource requirements, and implementation challenges.",
                        "timestamp": datetime.now(),
                        "round_number": round_num,
                        "topic": topic["title"],
                        "message_type": "discussion",
                    }
                    state["messages"].append(message)

        # Add topic summaries for completed topics
        state["topic_summaries"] = [
            f"Comprehensive analysis of {topic['title']} with consensus reached on key requirements."
            for topic in state["agenda"]
        ]

        return state


class TestFinalReportGeneration:
    """Test final report generation workflows."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test method."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.integration
    def test_final_report_generation_complete(self):
        """Test complete final report generation."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add completed session data
            state = self._add_completed_session_data(state)

            # Generate final report
            file_manager = FileManager(base_dir=self.temp_dir)
            report_writer = ReportSectionWriter(file_manager=file_manager)

            report_content = report_writer.generate_final_report(state)
            report_path = report_writer.save_report(report_content, state["session_id"])

            # Validate report
            assert len(report_content) > 1000  # Should be comprehensive
            assert "# Final Discussion Report" in report_content
            assert "## Executive Summary" in report_content
            assert "## Session Overview" in report_content
            assert "## Topic Analysis" in report_content
            assert "## Conclusions and Recommendations" in report_content

            # Verify file was created
            file_path = Path(self.temp_dir) / report_path
            assert file_path.exists()
            assert file_path.suffix == ".md"

    @pytest.mark.integration
    def test_final_report_with_attachments(self):
        """Test final report generation with attachments."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add session data with attachments
            state = self._add_session_data_with_attachments(state)

            file_manager = FileManager(base_dir=self.temp_dir)
            report_writer = ReportSectionWriter(file_manager=file_manager)

            # Generate report with attachments
            report_content = report_writer.generate_final_report(state)
            report_path = report_writer.save_report(report_content, state["session_id"])

            # Should reference attachments
            assert (
                "## Supporting Documents" in report_content
                or "## Attachments" in report_content
            )
            assert (
                "topic_summary_" in report_content
            )  # Should reference topic summaries

    @pytest.mark.integration
    def test_final_report_metadata_inclusion(self):
        """Test final report includes comprehensive metadata."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add rich metadata
            state = self._add_rich_metadata(state)

            file_manager = FileManager(base_dir=self.temp_dir)
            report_writer = ReportSectionWriter(file_manager=file_manager)

            report_content = report_writer.generate_final_report(state)

            # Should include metadata
            assert "## Session Statistics" in report_content
            assert "**Duration:**" in report_content
            assert "**Total Messages:**" in report_content
            assert "**Participants:**" in report_content
            assert "**Topics Discussed:**" in report_content

    def _add_completed_session_data(
        self, state: VirtualAgoraState
    ) -> VirtualAgoraState:
        """Add data for a completed session."""
        # Set session as completed
        state["current_phase"] = 5
        state["session_start_time"] = datetime.now().replace(hour=9, minute=0)
        state["session_end_time"] = datetime.now().replace(hour=11, minute=30)

        # Add comprehensive discussion data
        topics = ["AI Ethics", "Climate Policy", "Digital Privacy"]
        state["agenda"] = [
            {
                "title": topic,
                "description": f"Discussion of {topic}",
                "status": "completed",
            }
            for topic in topics
        ]

        # Add messages for all topics
        for topic_index, topic in enumerate(topics):
            for round_num in range(1, 5):
                for agent_id in state["speaking_order"]:
                    message = {
                        "id": str(uuid.uuid4()),
                        "agent_id": agent_id,
                        "content": f"My perspective on {topic} in round {round_num}: "
                        f"This is a critical issue that requires careful consideration "
                        f"of multiple stakeholder perspectives and long-term implications.",
                        "timestamp": datetime.now(),
                        "round_number": round_num,
                        "topic": topic,
                        "message_type": "discussion",
                    }
                    state["messages"].append(message)

        # Add topic summaries
        state["topic_summaries"] = [
            f"# Topic Summary: {topic}\n\n## Overview\nComprehensive discussion "
            f"reached consensus on key principles and implementation approaches.\n\n"
            f"## Key Points\n- Strategic importance\n- Implementation challenges\n"
            f"- Stakeholder considerations\n\n## Conclusions\nReady to proceed with "
            f"recommended approach for {topic}."
            for topic in topics
        ]

        # Add voting rounds
        for topic in topics:
            voting_round = {
                "id": str(uuid.uuid4()),
                "phase": 3,
                "vote_type": "conclusion",
                "options": ["Yes", "No"],
                "start_time": datetime.now(),
                "end_time": datetime.now(),
                "required_votes": 3,
                "received_votes": 3,
                "votes": {
                    agent_id: "Yes. Ready to conclude."
                    for agent_id in state["speaking_order"]
                },
                "result": "Yes",
                "status": "completed",
            }
            state["voting_rounds"].append(voting_round)

        return state

    def _add_session_data_with_attachments(
        self, state: VirtualAgoraState
    ) -> VirtualAgoraState:
        """Add session data with attachments."""
        state = self._add_completed_session_data(state)

        # Add attachment metadata
        if "metadata" not in state:
            state["metadata"] = {}

        state["metadata"]["attachments"] = [
            {
                "filename": "topic_summary_ai_ethics.md",
                "type": "topic_summary",
                "topic": "AI Ethics",
                "created_at": datetime.now(),
            },
            {
                "filename": "topic_summary_climate_policy.md",
                "type": "topic_summary",
                "topic": "Climate Policy",
                "created_at": datetime.now(),
            },
            {
                "filename": "voting_results.json",
                "type": "voting_data",
                "description": "Complete voting records",
                "created_at": datetime.now(),
            },
        ]

        return state

    def _add_rich_metadata(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Add rich metadata for testing."""
        state = self._add_completed_session_data(state)

        # Calculate statistics
        total_messages = len(state["messages"])
        total_rounds = max(msg["round_number"] for msg in state["messages"])
        session_duration = (
            state["session_end_time"] - state["session_start_time"]
        ).total_seconds() / 3600

        # Add computed metadata
        if "metadata" not in state:
            state["metadata"] = {}

        state["metadata"]["session_statistics"] = {
            "total_messages": total_messages,
            "total_rounds": total_rounds,
            "duration_hours": session_duration,
            "topics_completed": len(state["topic_summaries"]),
            "voting_rounds": len(state["voting_rounds"]),
            "participants": len(state["speaking_order"]),
            "average_messages_per_round": (
                total_messages / total_rounds if total_rounds > 0 else 0
            ),
        }

        # Add performance metrics
        state["metadata"]["performance_metrics"] = {
            "consensus_rate": 1.0,  # All votes passed
            "participation_rate": 1.0,  # All agents participated
            "efficiency_score": 0.85,  # Based on rounds to conclusion
            "quality_indicators": {
                "substantive_contributions": 0.9,
                "topic_coverage": 0.95,
                "stakeholder_alignment": 0.8,
            },
        }

        return state


class TestFileManagementFlow:
    """Test file management and organization workflows."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test method."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.integration
    def test_file_organization_structure(self):
        """Test proper file organization and structure."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Create file manager
            file_manager = FileManager(base_dir=self.temp_dir)

            # Create session directory structure
            session_id = state["session_id"]
            session_dir = file_manager.create_session_directory(session_id)

            # Verify directory structure
            assert session_dir.exists()
            assert session_dir.name == f"session_{session_id}"

            # Create subdirectories
            topics_dir = file_manager.create_subdirectory(session_dir, "topics")
            reports_dir = file_manager.create_subdirectory(session_dir, "reports")

            assert topics_dir.exists()
            assert reports_dir.exists()

    @pytest.mark.integration
    def test_file_naming_conventions(self):
        """Test file naming conventions and collision handling."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            file_manager = FileManager(base_dir=self.temp_dir)

            # Test topic summary filename generation
            topic_title = "AI Ethics & Regulatory Compliance"
            filename1 = file_manager.generate_topic_summary_filename(topic_title)

            # Should sanitize special characters
            assert "&" not in filename1
            assert " " not in filename1
            assert filename1.endswith(".md")
            assert filename1.startswith("topic_summary_")

            # Test collision handling
            file_path1 = Path(self.temp_dir) / filename1
            file_path1.touch()  # Create file

            filename2 = file_manager.generate_topic_summary_filename(topic_title)
            # Should generate different name to avoid collision
            if file_path1.exists():
                assert filename2 != filename1

    @pytest.mark.integration
    def test_file_metadata_tracking(self):
        """Test file metadata tracking and indexing."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            file_manager = FileManager(base_dir=self.temp_dir)

            # Save files with metadata
            content1 = "# Topic Summary: AI Ethics\n\nDetailed analysis..."
            file_info1 = file_manager.save_file_with_metadata(
                content=content1,
                filename="topic_summary_ai_ethics.md",
                metadata={
                    "type": "topic_summary",
                    "topic": "AI Ethics",
                    "session_id": state["session_id"],
                    "created_by": "system",
                },
            )

            # Verify metadata was saved
            assert file_info1["metadata"]["type"] == "topic_summary"
            assert file_info1["metadata"]["topic"] == "AI Ethics"

            # Get file registry
            registry = file_manager.get_file_registry()
            assert len(registry) == 1
            assert registry[0]["filename"] == "topic_summary_ai_ethics.md"


class TestExportFunctionality:
    """Test report export functionality."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test method."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.integration
    def test_export_multiple_formats(self):
        """Test exporting reports in multiple formats."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add completed session data
            state = self._add_export_test_data(state)

            file_manager = FileManager(base_dir=self.temp_dir)
            exporter = ReportExporter(file_manager=file_manager)

            # Export in different formats
            formats = ["markdown", "json", "html"]
            exported_files = {}

            for format_type in formats:
                export_path = exporter.export_session_report(
                    state=state, format_type=format_type
                )
                exported_files[format_type] = export_path

                # Verify file was created
                file_path = Path(self.temp_dir) / export_path
                assert file_path.exists()
                assert file_path.suffix == f".{format_type}" or (
                    format_type == "html" and file_path.suffix == ".html"
                )

    @pytest.mark.integration
    def test_export_with_attachments(self):
        """Test exporting session with all attachments."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add session with attachments
            state = self._add_session_with_attachments(state)

            file_manager = FileManager(base_dir=self.temp_dir)
            exporter = ReportExporter(file_manager=file_manager)

            # Export complete session package
            package_path = exporter.export_complete_session_package(state)

            # Verify package was created
            package_file = Path(self.temp_dir) / package_path
            assert package_file.exists()
            assert package_file.suffix in [".zip", ".tar.gz"]

    @pytest.mark.integration
    def test_export_filtering_and_customization(self):
        """Test export with filtering and customization options."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            state = self._add_export_test_data(state)

            file_manager = FileManager(base_dir=self.temp_dir)
            exporter = ReportExporter(file_manager=file_manager)

            # Export with custom options
            export_options = {
                "include_metadata": True,
                "include_raw_messages": False,
                "include_voting_details": True,
                "include_statistics": True,
                "format_style": "executive_summary",
            }

            export_path = exporter.export_custom_report(
                state=state, options=export_options
            )

            # Verify customized export
            file_path = Path(self.temp_dir) / export_path
            assert file_path.exists()

            # Check content based on options
            content = file_path.read_text()
            assert "## Session Statistics" in content  # include_statistics=True
            assert "## Voting Results" in content  # include_voting_details=True

    def _add_export_test_data(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Add test data for export testing."""
        # Set up completed session
        state["current_phase"] = 5
        state["session_start_time"] = datetime.now().replace(hour=10, minute=0)
        state["session_end_time"] = datetime.now().replace(hour=12, minute=0)

        # Add agenda and messages
        state["agenda"] = [
            {
                "title": "Future of Work",
                "description": "Remote work trends",
                "status": "completed",
            },
            {
                "title": "Technology Impact",
                "description": "AI automation effects",
                "status": "completed",
            },
        ]

        # Add messages
        for topic in state["agenda"]:
            for round_num in range(1, 4):
                for agent_id in state["speaking_order"]:
                    message = {
                        "id": str(uuid.uuid4()),
                        "agent_id": agent_id,
                        "content": f"Analysis of {topic['title']} in round {round_num}",
                        "timestamp": datetime.now(),
                        "round_number": round_num,
                        "topic": topic["title"],
                        "message_type": "discussion",
                    }
                    state["messages"].append(message)

        # Add topic summaries and voting results
        state["topic_summaries"] = [
            f"Summary of {topic['title']}" for topic in state["agenda"]
        ]
        state["voting_rounds"] = [
            {
                "id": str(uuid.uuid4()),
                "vote_type": "conclusion",
                "result": "Yes",
                "votes": {
                    agent_id: "Yes. Ready to conclude."
                    for agent_id in state["speaking_order"]
                },
                "status": "completed",
            }
        ]

        return state

    def _add_session_with_attachments(
        self, state: VirtualAgoraState
    ) -> VirtualAgoraState:
        """Add session data with multiple attachments."""
        state = self._add_export_test_data(state)

        # Create actual attachment files
        attachments_dir = Path(self.temp_dir) / "attachments"
        attachments_dir.mkdir(exist_ok=True)

        # Create topic summary files
        for topic in state["agenda"]:
            summary_file = (
                attachments_dir
                / f"topic_summary_{topic['title'].lower().replace(' ', '_')}.md"
            )
            summary_file.write_text(
                f"# Topic Summary: {topic['title']}\n\nDetailed analysis..."
            )

        # Create voting data file
        voting_file = attachments_dir / "voting_results.json"
        voting_file.write_text(
            json.dumps(state["voting_rounds"], default=str, indent=2)
        )

        # Update state with attachment metadata
        if "metadata" not in state:
            state["metadata"] = {}

        state["metadata"]["attachments"] = [
            {
                "filename": f.name,
                "path": str(f),
                "type": "topic_summary" if "topic_summary" in f.name else "voting_data",
                "size": f.stat().st_size,
                "created_at": datetime.now(),
            }
            for f in attachments_dir.iterdir()
        ]

        return state


@pytest.mark.integration
class TestMetadataCollection:
    """Test metadata collection and analysis."""

    def test_comprehensive_metadata_collection(self):
        """Test collection of comprehensive session metadata."""
        helper = IntegrationTestHelper(num_agents=4, scenario="default")

        with patch_ui_components():
            flow = helper.create_test_flow()
            state = helper.create_discussion_state()

            # Add rich session data
            state = self._add_rich_session_data(state)

            # Collect metadata
            metadata_collector = MetadataCollector()
            metadata = metadata_collector.collect_session_metadata(state)

            # Verify comprehensive metadata
            assert "session_info" in metadata
            assert "participation_metrics" in metadata
            assert "discussion_metrics" in metadata
            assert "quality_indicators" in metadata
            assert "performance_stats" in metadata

            # Verify specific metrics
            session_info = metadata["session_info"]
            assert "duration" in session_info
            assert "participant_count" in session_info
            assert "topics_discussed" in session_info

    def test_performance_metrics_calculation(self):
        """Test calculation of performance metrics."""
        helper = IntegrationTestHelper(num_agents=3, scenario="default")

        with patch_ui_components():
            flow = helper.create_test_flow()
            state = helper.create_discussion_state()

            state = self._add_performance_test_data(state)

            metadata_collector = MetadataCollector()
            performance_metrics = metadata_collector.calculate_performance_metrics(
                state
            )

            # Verify performance calculations
            assert "consensus_rate" in performance_metrics
            assert "participation_rate" in performance_metrics
            assert "efficiency_score" in performance_metrics
            assert "discussion_quality" in performance_metrics

            # Values should be in valid ranges
            assert 0 <= performance_metrics["consensus_rate"] <= 1
            assert 0 <= performance_metrics["participation_rate"] <= 1
            assert 0 <= performance_metrics["efficiency_score"] <= 1

    def _add_rich_session_data(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Add rich session data for metadata testing."""
        # Set session timing
        state["session_start_time"] = datetime.now().replace(hour=9, minute=0)
        state["session_end_time"] = datetime.now().replace(hour=11, minute=45)

        # Add multiple topics and extensive discussion
        state["agenda"] = [
            {
                "title": f"Topic {i+1}",
                "description": f"Description {i+1}",
                "status": "completed",
            }
            for i in range(3)
        ]

        # Add messages from all participants
        for topic_index, topic in enumerate(state["agenda"]):
            for round_num in range(1, 6):  # 5 rounds per topic
                for agent_id in state["speaking_order"]:
                    message = {
                        "id": str(uuid.uuid4()),
                        "agent_id": agent_id,
                        "content": f"Substantive analysis of {topic['title']} round {round_num}",
                        "timestamp": datetime.now(),
                        "round_number": round_num,
                        "topic": topic["title"],
                        "message_type": "discussion",
                    }
                    state["messages"].append(message)

        # Add voting rounds with mixed results
        state["voting_rounds"] = [
            {
                "id": str(uuid.uuid4()),
                "vote_type": "conclusion",
                "result": "Yes" if i % 2 == 0 else "No",  # Mixed results
                "votes": {
                    agent_id: f"{'Yes' if i % 2 == 0 else 'No'}. Reasoning."
                    for agent_id in state["speaking_order"]
                },
                "status": "completed",
            }
            for i in range(6)  # Multiple voting rounds
        ]

        return state

    def _add_performance_test_data(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Add data for performance metrics testing."""
        state = self._add_rich_session_data(state)

        # Add specific performance indicators
        state["round_summaries"] = [
            f"Round {i+1}: High-quality discussion with substantive contributions"
            for i in range(15)  # Multiple rounds
        ]

        # Add consensus tracking
        state["consensus_tracking"] = [
            {"round": i + 1, "agreement_level": 0.8 + (i % 3) * 0.1} for i in range(15)
        ]

        return state
