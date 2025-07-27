"""Tests for report export functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path
import zipfile

from virtual_agora.reporting.exporter import ReportExporter


class TestReportExporter:
    """Test ReportExporter functionality."""

    def setup_method(self):
        """Set up test method."""
        self.exporter = ReportExporter()
        self.temp_dir = tempfile.mkdtemp()
        self.report_dir = Path(self.temp_dir) / "test_report"
        self.report_dir.mkdir()
        self._create_test_report()

    def teardown_method(self):
        """Clean up test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_report(self):
        """Create a test report structure."""
        files = {
            "00_Table_of_Contents.md": "# Table of Contents\n\n1. Executive Summary\n2. Introduction",
            "01_Executive_Summary.md": "# Executive Summary\n\nThis is the executive summary.",
            "02_Introduction.md": "# Introduction\n\n## Background\n\nThis is the introduction.",
            "03_Analysis.md": "# Analysis\n\nDetailed analysis here.",
            "README.md": "# Report Info\n\nGenerated report",
            "report_metadata.json": '{"session_id": "test-123", "duration": 60}',
        }

        for filename, content in files.items():
            (self.report_dir / filename).write_text(content)

    def test_initialization(self):
        """Test exporter initialization."""
        assert self.exporter.supported_formats == [
            "markdown",
            "html",
            "archive",
            "combined",
        ]

    def test_export_unsupported_format(self):
        """Test export with unsupported format."""
        output_path = Path(self.temp_dir) / "output.pdf"

        with pytest.raises(ValueError, match="Unsupported format"):
            self.exporter.export_report(self.report_dir, output_path, format="pdf")

    def test_export_markdown(self):
        """Test markdown export."""
        output_path = Path(self.temp_dir) / "combined.md"

        result = self.exporter._export_markdown(self.report_dir, output_path)

        assert result.exists()
        assert result.suffix == ".md"

        content = result.read_text()
        # Should include all sections except TOC
        assert "# Executive Summary" in content
        assert "# Introduction" in content
        assert "# Analysis" in content
        # Should not include TOC
        assert "# Table of Contents" not in content
        # Should have separators
        assert "---" in content

    def test_export_markdown_with_metadata(self):
        """Test markdown export with metadata."""
        output_path = Path(self.temp_dir) / "combined.md"

        result = self.exporter._export_markdown(
            self.report_dir, output_path, include_metadata=True
        )

        content = result.read_text()
        # Should have metadata header
        assert "---" == content.split("\n")[0]
        assert "title: Virtual Agora Report" in content
        assert "generated:" in content
        assert "source:" in content

    def test_export_html(self):
        """Test HTML export."""
        output_path = Path(self.temp_dir) / "report.html"

        result = self.exporter._export_html(self.report_dir, output_path)

        assert result.exists()
        assert result.suffix == ".html"

        content = result.read_text()
        # Check HTML structure
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "<head>" in content
        assert "<body>" in content
        assert "Virtual Agora Report" in content
        # Check content conversion
        assert "<h1>Executive Summary</h1>" in content
        assert "<h2>Background</h2>" in content

    def test_markdown_to_html_conversion(self):
        """Test markdown to HTML conversion."""
        markdown = """
# Heading 1
## Heading 2

This is **bold** and this is *italic*.

This is `inline code`.

```
Code block
```

- List item 1
- List item 2

1. Numbered item 1
2. Numbered item 2
"""

        html = self.exporter._markdown_to_html(markdown)

        assert "<h1>Heading 1</h1>" in html
        assert "<h2>Heading 2</h2>" in html
        assert "<strong>bold</strong>" in html
        assert "<em>italic</em>" in html
        assert "<code>inline code</code>" in html
        assert "<pre><code>" in html
        assert "<ul>" in html
        assert "<li>List item 1</li>" in html
        assert "<ol>" in html
        assert "<li>Numbered item 1</li>" in html

    def test_export_archive(self):
        """Test archive export."""
        output_path = Path(self.temp_dir) / "report_archive"

        result = self.exporter._export_archive(self.report_dir, output_path)

        assert result.exists()
        assert result.suffix == ".zip"

        # Verify archive contents
        with zipfile.ZipFile(result, "r") as zf:
            files = zf.namelist()
            assert any("Executive_Summary.md" in f for f in files)
            assert any("Introduction.md" in f for f in files)
            assert any("report_metadata.json" in f for f in files)

    def test_export_combined(self):
        """Test combined export."""
        output_dir = Path(self.temp_dir) / "combined_output"

        results = self.exporter._export_combined(self.report_dir, output_dir)

        assert isinstance(results, dict)
        assert "markdown" in results
        assert "html" in results
        assert "archive" in results

        # Check all files exist
        assert results["markdown"].exists()
        assert results["html"].exists()
        assert results["archive"].exists()

        # Check they're in the output directory
        assert results["markdown"].parent == output_dir
        assert results["html"].parent == output_dir
        assert results["archive"].parent == output_dir

    def test_export_report_markdown(self):
        """Test main export method with markdown format."""
        output_path = Path(self.temp_dir) / "export.md"

        result = self.exporter.export_report(
            self.report_dir, output_path, format="markdown"
        )

        assert result == output_path
        assert result.exists()

    def test_export_selective(self):
        """Test selective export."""
        output_path = Path(self.temp_dir) / "selective.md"
        sections = ["Executive Summary", "Analysis"]

        result = self.exporter.export_selective(self.report_dir, output_path, sections)

        assert result.exists()

        content = result.read_text()
        # Should include selected sections
        assert "Executive Summary" in content
        assert "Analysis" in content
        # Should not include non-selected sections
        assert "Introduction" not in content

    def test_create_shareable_link(self):
        """Test shareable link creation (placeholder)."""
        test_path = Path("test_report.md")

        link = self.exporter.create_shareable_link(test_path)

        assert "upload" in link
        assert "test_report.md" in link
        assert "file sharing service" in link

    def test_export_error_handling(self):
        """Test error handling in export."""
        # Try to export non-existent directory
        bad_dir = Path(self.temp_dir) / "non_existent"
        output_path = Path(self.temp_dir) / "output.md"

        with pytest.raises(Exception):
            self.exporter.export_report(bad_dir, output_path)

    def test_html_template_formatting(self):
        """Test HTML template is properly formatted."""
        output_path = Path(self.temp_dir) / "formatted.html"

        result = self.exporter._export_html(self.report_dir, output_path)

        content = result.read_text()

        # Check CSS is included
        assert "font-family:" in content
        assert "max-width:" in content
        assert "background-color:" in content

        # Check responsive design
        assert "@media print" in content

        # Check footer
        assert "Generated by Virtual Agora" in content

    def test_export_with_extension_handling(self):
        """Test extension handling in export methods."""
        # Test markdown export adds .md
        output_path = Path(self.temp_dir) / "no_extension"
        result = self.exporter._export_markdown(self.report_dir, output_path)
        assert result.suffix == ".md"

        # Test HTML export adds .html
        output_path = Path(self.temp_dir) / "no_extension"
        result = self.exporter._export_html(self.report_dir, output_path)
        assert result.suffix == ".html"

        # Test archive removes .zip if provided
        output_path = Path(self.temp_dir) / "archive.zip"
        result = self.exporter._export_archive(self.report_dir, output_path)
        assert result.suffix == ".zip"
        assert "archive.zip" in str(result)

    def test_combined_export_directory_creation(self):
        """Test that combined export creates output directory."""
        output_dir = Path(self.temp_dir) / "new_dir" / "nested"

        results = self.exporter._export_combined(self.report_dir, output_dir)

        assert output_dir.exists()
        assert all(r.exists() for r in results.values())
