"""Tests for report template management."""

import pytest
import json
import tempfile
import shutil
from pathlib import Path

from virtual_agora.reporting.templates import ReportTemplateManager


class TestReportTemplateManager:
    """Test ReportTemplateManager functionality."""

    def setup_method(self):
        """Set up test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.template_dir = Path(self.temp_dir) / "templates"
        self.template_dir.mkdir()
        self.manager = ReportTemplateManager(self.template_dir)

    def teardown_method(self):
        """Clean up test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test manager initialization."""
        assert self.manager.template_dir == self.template_dir
        assert self.manager.custom_templates == {}
        assert self.manager.current_template == self.manager.DEFAULT_TEMPLATE

    def test_load_custom_templates(self):
        """Test loading custom templates from directory."""
        # Create a custom template file
        custom_template = {
            "name": "custom",
            "version": "1.0",
            "sections": [
                {"id": "summary", "title": "Summary", "required": True},
            ],
        }

        template_file = self.template_dir / "custom.json"
        template_file.write_text(json.dumps(custom_template))

        # Reload manager to trigger loading
        manager = ReportTemplateManager(self.template_dir)

        assert "custom" in manager.custom_templates
        assert manager.custom_templates["custom"]["name"] == "custom"

    def test_get_template_default(self):
        """Test getting default template."""
        template = self.manager.get_template("default")

        assert template["name"] == "default"
        assert "sections" in template
        assert len(template["sections"]) > 0
        assert template["sections"][0]["title"] == "Executive Summary"

    def test_get_template_language(self):
        """Test getting language-specific template."""
        # Test Spanish template
        es_template = self.manager.get_template("es")
        assert es_template["name"] == "spanish"
        assert es_template["sections"][0]["title"] == "Resumen Ejecutivo"

        # Test French template
        fr_template = self.manager.get_template("fr")
        assert fr_template["name"] == "french"
        assert fr_template["sections"][0]["title"] == "Résumé Exécutif"

    def test_get_template_custom(self):
        """Test getting custom template."""
        # Add custom template
        self.manager.custom_templates["mytemplate"] = {
            "name": "mytemplate",
            "sections": [],
        }

        template = self.manager.get_template("mytemplate")
        assert template["name"] == "mytemplate"

    def test_get_template_not_found(self):
        """Test getting non-existent template."""
        template = self.manager.get_template("non_existent")
        # Should return default
        assert template == self.manager.DEFAULT_TEMPLATE

    def test_set_template_by_name(self):
        """Test setting template by name."""
        self.manager.set_template("es")
        assert self.manager.current_template["name"] == "spanish"

    def test_set_template_by_dict(self):
        """Test setting template by dictionary."""
        custom = {"name": "custom", "sections": []}
        self.manager.set_template(custom)
        assert self.manager.current_template == custom

    def test_get_section_structure(self):
        """Test getting section structure."""
        sections = self.manager.get_section_structure()

        assert isinstance(sections, list)
        assert len(sections) > 0
        assert all("id" in s and "title" in s for s in sections)

    def test_get_section_titles(self):
        """Test getting section titles."""
        # English titles
        en_titles = self.manager.get_section_titles("en")
        assert en_titles["executive_summary"] == "Executive Summary"
        assert en_titles["introduction"] == "Introduction"

        # Spanish titles
        es_titles = self.manager.get_section_titles("es")
        assert es_titles["executive_summary"] == "Resumen Ejecutivo"
        assert es_titles["introduction"] == "Introducción"

    def test_customize_template(self):
        """Test template customization."""
        custom_sections = [
            {"id": "custom1", "title": "Custom Section 1", "required": True},
            {"id": "custom2", "title": "Custom Section 2", "required": False},
        ]

        custom_styles = {
            "heading_color": "#ff0000",
            "font_family": "Comic Sans MS",
        }

        custom_metadata = {
            "include_page_numbers": True,
            "include_logo": True,
        }

        customized = self.manager.customize_template(
            base_template="default",
            sections=custom_sections,
            styles=custom_styles,
            metadata=custom_metadata,
        )

        assert customized["name"] == "custom"
        assert customized["sections"] == custom_sections
        assert customized["styles"]["heading_color"] == "#ff0000"
        assert customized["styles"]["font_family"] == "Comic Sans MS"
        assert customized["metadata"]["include_page_numbers"] is True
        assert customized["metadata"]["include_logo"] is True
        assert "customized_at" in customized

    def test_save_template(self):
        """Test saving a template."""
        template = {
            "name": "test",
            "version": "1.0",
            "sections": [],
        }

        saved_path = self.manager.save_template(template, "test_template")

        assert saved_path.exists()
        assert saved_path.name == "test_template.json"

        # Verify content
        loaded = json.loads(saved_path.read_text())
        assert loaded["name"] == "test_template"
        assert "saved_at" in loaded

        # Check it was added to custom templates
        assert "test_template" in self.manager.custom_templates

    def test_save_template_overwrite_protection(self):
        """Test template overwrite protection."""
        template = {"name": "test", "sections": []}

        # Save once
        self.manager.save_template(template, "protected")

        # Try to save again without overwrite
        with pytest.raises(ValueError, match="already exists"):
            self.manager.save_template(template, "protected", overwrite=False)

        # Should work with overwrite=True
        saved = self.manager.save_template(template, "protected", overwrite=True)
        assert saved.exists()

    def test_apply_template_to_content(self):
        """Test applying template formatting to content."""
        content = """## Executive Summary

This is the summary.

## Introduction

This is the introduction."""

        # Apply Spanish template
        formatted = self.manager.apply_template_to_content(content, "es")

        assert "## Resumen Ejecutivo" in formatted
        assert "## Introducción" in formatted
        assert "This is the summary." in formatted  # Content preserved

    def test_apply_template_with_header_footer(self):
        """Test applying template with header and footer."""
        template = self.manager.DEFAULT_TEMPLATE.copy()
        template["metadata"]["include_header"] = True
        template["metadata"]["include_footer"] = True
        template["name"] = "with_header"

        self.manager.custom_templates["with_header"] = template

        content = "# Test Content"
        formatted = self.manager.apply_template_to_content(content, "with_header")

        # Check header
        assert "---" in formatted
        assert "template: with_header" in formatted

        # Check footer
        assert "Generated using Virtual Agora" in formatted

    def test_get_css_styles(self):
        """Test CSS generation from template."""
        css = self.manager.get_css_styles("default")

        assert "body {" in css
        assert "font-family:" in css
        assert "#2c3e50" in css  # Default heading color
        assert "#3498db" in css  # Default accent color

    def test_list_available_templates(self):
        """Test listing available templates."""
        # Add a custom template
        self.manager.custom_templates["custom1"] = {"name": "custom1"}

        available = self.manager.list_available_templates()

        assert "default" in available
        assert "languages" in available
        assert "custom" in available

        assert "default" in available["default"]
        assert "en" in available["languages"]
        assert "es" in available["languages"]
        assert "fr" in available["languages"]
        assert "custom1" in available["custom"]

    def test_validate_template_valid(self):
        """Test template validation with valid template."""
        template = {
            "name": "valid",
            "version": "1.0",
            "sections": [
                {"id": "section1", "title": "Section 1", "required": True},
            ],
            "styles": {},
        }

        results = self.manager.validate_template(template)

        assert results["valid"]
        assert len(results["errors"]) == 0
        assert len(results["warnings"]) == 0

    def test_validate_template_invalid(self):
        """Test template validation with invalid template."""
        # Missing required fields
        template = {
            "version": "1.0",
        }

        results = self.manager.validate_template(template)

        assert not results["valid"]
        assert any("name" in error for error in results["errors"])
        assert any("sections" in error for error in results["errors"])

    def test_validate_template_invalid_sections(self):
        """Test template validation with invalid sections."""
        template = {
            "name": "invalid",
            "sections": [
                {"title": "Missing ID"},  # Missing required 'id'
                "not a dict",  # Wrong type
            ],
        }

        results = self.manager.validate_template(template)

        assert len(results["errors"]) >= 2
        assert any("missing required fields" in error for error in results["errors"])
        assert any("must be a dictionary" in error for error in results["errors"])

    def test_export_template_preview(self):
        """Test template preview export."""
        output_path = Path(self.temp_dir) / "preview.md"

        result = self.manager.export_template_preview("default", output_path)

        assert result.exists()

        content = result.read_text()
        assert "# Template Preview: default" in content
        assert "## Section Structure" in content
        assert "Executive Summary" in content
        assert "## Styles" in content
        assert "```css" in content
        assert "## Metadata Settings" in content

    def test_error_handling_in_load_templates(self):
        """Test error handling when loading invalid templates."""
        # Create invalid JSON file
        bad_file = self.template_dir / "bad.json"
        bad_file.write_text("{ invalid json")

        # Should handle error gracefully
        manager = ReportTemplateManager(self.template_dir)
        assert "bad" not in manager.custom_templates

    def test_generate_header_with_timestamps(self):
        """Test header generation with timestamps."""
        template = {
            "name": "test",
            "metadata": {
                "include_timestamps": True,
            },
        }

        header = self.manager._generate_header(template)

        assert "timestamp:" in header
        assert "generated:" in header

        # Without timestamps
        template["metadata"]["include_timestamps"] = False
        header = self.manager._generate_header(template)
        assert "timestamp:" not in header
