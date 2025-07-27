"""Tests for report file management."""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from virtual_agora.reporting.file_manager import ReportFileManager


class TestReportFileManager:
    """Test ReportFileManager functionality."""

    def setup_method(self):
        """Set up test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_dir = Path(self.temp_dir) / "reports"
        self.manager = ReportFileManager(self.base_dir)

    def teardown_method(self):
        """Clean up test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test manager initialization."""
        assert self.manager.base_output_dir == self.base_dir
        assert self.base_dir.exists()
        assert self.manager.current_report_dir is None
        assert self.manager.file_manifest == []

    def test_create_report_directory(self):
        """Test report directory creation."""
        session_id = "test-session-123"
        main_topic = "AI Ethics Discussion"

        report_dir = self.manager.create_report_directory(session_id, main_topic)

        assert report_dir.exists()
        assert report_dir.parent == self.base_dir
        assert "AI_Ethics_Discussion" in report_dir.name
        assert session_id[:8] not in report_dir.name  # Should use topic name
        assert self.manager.current_report_dir == report_dir

    def test_create_report_directory_no_topic(self):
        """Test directory creation without main topic."""
        session_id = "test-session-456"

        report_dir = self.manager.create_report_directory(session_id)

        assert report_dir.exists()
        assert session_id[:8] in report_dir.name

    def test_create_report_directory_special_chars(self):
        """Test directory creation with special characters."""
        session_id = "test"
        main_topic = "Complex: Topic! With? Special/Characters"

        report_dir = self.manager.create_report_directory(session_id, main_topic)

        assert report_dir.exists()
        # Special characters should be removed
        assert ":" not in report_dir.name
        assert "!" not in report_dir.name
        assert "?" not in report_dir.name
        assert "/" not in report_dir.name

    def test_save_report_section(self):
        """Test saving a report section."""
        # Create report directory first
        self.manager.create_report_directory("test-session")

        # Save a section
        section_path = self.manager.save_report_section(
            1,
            "Executive Summary",
            "## Executive Summary\n\nThis is the executive summary.",
        )

        assert section_path.exists()
        assert section_path.name == "01_Executive_Summary.md"
        assert len(self.manager.file_manifest) == 1

        # Check manifest entry
        manifest_entry = self.manager.file_manifest[0]
        assert manifest_entry["section_number"] == 1
        assert manifest_entry["section_title"] == "Executive Summary"
        assert manifest_entry["filename"] == "01_Executive_Summary.md"

    def test_save_report_section_subsection(self):
        """Test saving a subsection."""
        self.manager.create_report_directory("test-session")

        section_path = self.manager.save_report_section(
            2, "  - AI Ethics", "### AI Ethics\n\nContent here."
        )

        assert section_path.exists()
        assert "sub_" in section_path.name

    def test_save_report_section_no_directory(self):
        """Test saving section without directory."""
        with pytest.raises(ValueError, match="No report directory"):
            self.manager.save_report_section(1, "Test", "Content")

    def test_create_table_of_contents(self):
        """Test table of contents creation."""
        # Set up report with sections
        self.manager.create_report_directory("test-session")

        # Save some sections
        self.manager.save_report_section(1, "Executive Summary", "Content 1")
        self.manager.save_report_section(2, "Introduction", "Content 2")
        self.manager.save_report_section(3, "  - Subsection", "Content 3")

        # Create table of contents
        report_structure = [
            "Executive Summary",
            "Introduction",
            "  - Subsection",
            "Missing Section",
        ]
        toc_path = self.manager.create_table_of_contents(report_structure)

        assert toc_path.exists()
        assert toc_path.name == "00_Table_of_Contents.md"

        # Check content
        toc_content = toc_path.read_text()
        assert "# Table of Contents" in toc_content
        assert "[Executive Summary]" in toc_content
        assert "[Introduction]" in toc_content
        assert "Missing Section *(pending)*" in toc_content
        assert "## File Listing" in toc_content

    def test_create_metadata_file(self):
        """Test metadata file creation."""
        self.manager.create_report_directory("test-session")
        self.manager.save_report_section(1, "Test Section", "Content")

        metadata = {
            "session_id": "test-session",
            "duration_minutes": 45,
            "topics_discussed": ["Topic 1", "Topic 2"],
        }

        metadata_path = self.manager.create_metadata_file(metadata)

        assert metadata_path.exists()
        assert metadata_path.name == "report_metadata.json"

        # Check JSON content
        loaded_metadata = json.loads(metadata_path.read_text())
        assert "report_info" in loaded_metadata
        assert loaded_metadata["report_info"]["total_sections"] == 1

        # Check README was created
        readme_path = self.manager.current_report_dir / "README.md"
        assert readme_path.exists()
        readme_content = readme_path.read_text()
        assert "# Report Information" in readme_content
        assert "**Session ID**: test-session" in readme_content
        assert "**Duration**: 45 minutes" in readme_content
        assert "Topic 1" in readme_content
        assert "Topic 2" in readme_content

    def test_create_report_manifest(self):
        """Test manifest creation."""
        self.manager.create_report_directory("test-session")

        # Create some files
        self.manager.save_report_section(1, "Section 1", "Content 1")
        self.manager.save_report_section(2, "Section 2", "Content 2")
        self.manager.create_metadata_file({"test": "data"})

        manifest_path = self.manager.create_report_manifest()

        assert manifest_path.exists()
        assert manifest_path.name == "manifest.json"

        # Check manifest content
        manifest = json.loads(manifest_path.read_text())
        assert manifest["manifest_version"] == "1.0"
        assert len(manifest["files"]) >= 4  # Sections + metadata + README + manifest

    def test_validate_report_structure(self):
        """Test report structure validation."""
        # Test with no directory
        results = self.manager.validate_report_structure()
        assert not results["valid"]
        assert results["error"] == "No report directory"

        # Create valid report
        self.manager.create_report_directory("test-session")
        self.manager.save_report_section(1, "Executive Summary", "Content")
        self.manager.save_report_section(2, "Introduction", "Content")
        self.manager.create_table_of_contents(["Executive Summary", "Introduction"])
        self.manager.create_metadata_file({})
        self.manager.create_report_manifest()

        results = self.manager.validate_report_structure()

        assert results["valid"]
        assert results["file_count"] >= 5
        assert results["section_count"] >= 2
        assert len(results["issues"]) == 0

    def test_validate_report_structure_missing_files(self):
        """Test validation with missing required files."""
        self.manager.create_report_directory("test-session")
        self.manager.save_report_section(1, "Test", "Content")

        results = self.manager.validate_report_structure()

        assert not results["valid"]
        assert any("Table_of_Contents" in issue for issue in results["issues"])
        assert any("README" in issue for issue in results["issues"])
        assert any("metadata" in issue for issue in results["issues"])

    def test_get_report_path(self):
        """Test getting report path."""
        assert self.manager.get_report_path() is None

        self.manager.create_report_directory("test-session")
        path = self.manager.get_report_path()

        assert path is not None
        assert path == self.manager.current_report_dir

    def test_archive_report(self):
        """Test report archiving."""
        # Create a report with content
        self.manager.create_report_directory("test-session")
        self.manager.save_report_section(1, "Section 1", "Content 1")
        self.manager.save_report_section(2, "Section 2", "Content 2")
        self.manager.create_metadata_file({})

        # Archive the report
        archive_path = self.manager.archive_report()

        assert archive_path.exists()
        assert archive_path.suffix == ".zip"
        assert archive_path.stem == self.manager.current_report_dir.name

    def test_archive_report_no_directory(self):
        """Test archiving without directory."""
        with pytest.raises(ValueError, match="No report directory"):
            self.manager.archive_report()

    def test_file_manifest_tracking(self):
        """Test file manifest tracking."""
        self.manager.create_report_directory("test-session")

        # Save multiple sections
        sections = [
            (1, "Executive Summary", "Content 1"),
            (2, "Introduction", "Content 2"),
            (3, "Analysis", "Content 3"),
        ]

        for num, title, content in sections:
            self.manager.save_report_section(num, title, content)

        assert len(self.manager.file_manifest) == 3

        # Check manifest entries
        for i, (num, title, _) in enumerate(sections):
            entry = self.manager.file_manifest[i]
            assert entry["section_number"] == num
            assert entry["section_title"] == title
            assert entry["size_bytes"] > 0
