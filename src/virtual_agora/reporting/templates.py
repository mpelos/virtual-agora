"""Report template management for Virtual Agora.

This module provides functionality to manage report templates,
customize formats, and support internationalization.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import re
from copy import deepcopy

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ReportTemplateManager:
    """Manage report templates and customization."""

    # Default template structure
    DEFAULT_TEMPLATE = {
        "name": "default",
        "version": "1.0",
        "sections": [
            {"id": "executive_summary", "title": "Executive Summary", "required": True},
            {"id": "introduction", "title": "Introduction", "required": True},
            {
                "id": "discussion_overview",
                "title": "Discussion Overview",
                "required": False,
            },
            {"id": "topic_analyses", "title": "Topic Analyses", "required": True},
            {
                "id": "key_insights",
                "title": "Key Insights and Findings",
                "required": True,
            },
            {
                "id": "conclusions",
                "title": "Conclusions and Recommendations",
                "required": True,
            },
            {"id": "metadata", "title": "Session Metadata", "required": False},
            {"id": "appendices", "title": "Appendices", "required": False},
        ],
        "styles": {
            "font_family": "Arial, sans-serif",
            "heading_color": "#2c3e50",
            "text_color": "#333333",
            "background_color": "#ffffff",
            "accent_color": "#3498db",
        },
        "metadata": {
            "include_header": True,
            "include_footer": True,
            "include_page_numbers": False,
            "include_timestamps": True,
        },
    }

    # Language-specific templates
    LANGUAGE_TEMPLATES = {
        "en": DEFAULT_TEMPLATE,
        "es": {
            "name": "spanish",
            "version": "1.0",
            "sections": [
                {
                    "id": "executive_summary",
                    "title": "Resumen Ejecutivo",
                    "required": True,
                },
                {"id": "introduction", "title": "Introducción", "required": True},
                {
                    "id": "discussion_overview",
                    "title": "Resumen de la Discusión",
                    "required": False,
                },
                {
                    "id": "topic_analyses",
                    "title": "Análisis de Temas",
                    "required": True,
                },
                {"id": "key_insights", "title": "Hallazgos Clave", "required": True},
                {
                    "id": "conclusions",
                    "title": "Conclusiones y Recomendaciones",
                    "required": True,
                },
                {
                    "id": "metadata",
                    "title": "Metadatos de la Sesión",
                    "required": False,
                },
                {"id": "appendices", "title": "Apéndices", "required": False},
            ],
            "styles": DEFAULT_TEMPLATE["styles"],
            "metadata": DEFAULT_TEMPLATE["metadata"],
        },
        "fr": {
            "name": "french",
            "version": "1.0",
            "sections": [
                {
                    "id": "executive_summary",
                    "title": "Résumé Exécutif",
                    "required": True,
                },
                {"id": "introduction", "title": "Introduction", "required": True},
                {
                    "id": "discussion_overview",
                    "title": "Aperçu de la Discussion",
                    "required": False,
                },
                {
                    "id": "topic_analyses",
                    "title": "Analyses des Sujets",
                    "required": True,
                },
                {"id": "key_insights", "title": "Conclusions Clés", "required": True},
                {
                    "id": "conclusions",
                    "title": "Conclusions et Recommandations",
                    "required": True,
                },
                {
                    "id": "metadata",
                    "title": "Métadonnées de Session",
                    "required": False,
                },
                {"id": "appendices", "title": "Annexes", "required": False},
            ],
            "styles": DEFAULT_TEMPLATE["styles"],
            "metadata": DEFAULT_TEMPLATE["metadata"],
        },
    }

    def __init__(self, template_dir: Optional[Path] = None):
        """Initialize ReportTemplateManager.

        Args:
            template_dir: Directory for custom templates.
        """
        self.template_dir = template_dir or Path("templates")
        self.custom_templates = {}
        self.current_template = deepcopy(self.DEFAULT_TEMPLATE)

        # Load custom templates if directory exists
        if self.template_dir.exists():
            self._load_custom_templates()

    def _load_custom_templates(self):
        """Load custom templates from directory."""
        for template_file in self.template_dir.glob("*.json"):
            try:
                template_data = json.loads(template_file.read_text(encoding="utf-8"))
                self.custom_templates[template_file.stem] = template_data
                logger.info(f"Loaded custom template: {template_file.stem}")
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")

    def get_template(self, template_name: str = "default") -> Dict[str, Any]:
        """Get a specific template.

        Args:
            template_name: Name of the template.

        Returns:
            Template dictionary.
        """
        if template_name in self.custom_templates:
            return self.custom_templates[template_name]
        elif template_name in self.LANGUAGE_TEMPLATES:
            return self.LANGUAGE_TEMPLATES[template_name]
        else:
            logger.warning(f"Template '{template_name}' not found, using default")
            return self.DEFAULT_TEMPLATE

    def set_template(self, template: Union[str, Dict[str, Any]]):
        """Set the current template.

        Args:
            template: Template name or dictionary.
        """
        if isinstance(template, str):
            self.current_template = self.get_template(template)
        else:
            self.current_template = template

    def get_section_structure(self) -> List[Dict[str, Any]]:
        """Get the section structure from current template.

        Returns:
            List of section definitions.
        """
        return self.current_template.get("sections", [])

    def get_section_titles(self, language: str = "en") -> Dict[str, str]:
        """Get section titles for a specific language.

        Args:
            language: Language code.

        Returns:
            Dictionary mapping section IDs to titles.
        """
        template = self.get_template(language)
        return {
            section["id"]: section["title"] for section in template.get("sections", [])
        }

    def customize_template(
        self,
        base_template: str = "default",
        sections: Optional[List[Dict[str, Any]]] = None,
        styles: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Customize a template.

        Args:
            base_template: Base template to customize.
            sections: Custom section structure.
            styles: Custom styles.
            metadata: Custom metadata settings.

        Returns:
            Customized template dictionary.
        """
        # Start with base template - use deepcopy to avoid modifying original
        custom = deepcopy(self.get_template(base_template))

        # Update sections if provided
        if sections:
            custom["sections"] = sections

        # Update styles if provided
        if styles:
            custom["styles"].update(styles)

        # Update metadata if provided
        if metadata:
            custom["metadata"].update(metadata)

        # Update name and timestamp
        custom["name"] = "custom"
        custom["customized_at"] = datetime.now().isoformat()

        return custom

    def save_template(
        self,
        template: Dict[str, Any],
        name: str,
        overwrite: bool = False,
    ) -> Path:
        """Save a custom template.

        Args:
            template: Template dictionary.
            name: Template name.
            overwrite: Whether to overwrite existing template.

        Returns:
            Path to saved template file.
        """
        # Ensure template directory exists
        self.template_dir.mkdir(parents=True, exist_ok=True)

        # Check if template already exists
        template_path = self.template_dir / f"{name}.json"
        if template_path.exists() and not overwrite:
            raise ValueError(f"Template '{name}' already exists")

        # Add metadata
        template["name"] = name
        template["saved_at"] = datetime.now().isoformat()

        # Save template
        template_path.write_text(json.dumps(template, indent=2), encoding="utf-8")

        # Update custom templates cache
        self.custom_templates[name] = template

        logger.info(f"Saved template: {name}")
        return template_path

    def apply_template_to_content(
        self,
        content: str,
        template_name: str = "default",
    ) -> str:
        """Apply template formatting to content.

        Args:
            content: Raw content.
            template_name: Template to apply.

        Returns:
            Formatted content.
        """
        template = self.get_template(template_name)

        # Apply section renaming
        for section in template.get("sections", []):
            old_title = section.get("id", "").replace("_", " ").title()
            new_title = section.get("title", old_title)

            # Replace section headings
            content = re.sub(
                f"^##\\s+{re.escape(old_title)}$",
                f"## {new_title}",
                content,
                flags=re.MULTILINE,
            )

        # Apply metadata formatting if requested
        if template.get("metadata", {}).get("include_header"):
            header = self._generate_header(template)
            content = header + "\n\n" + content

        if template.get("metadata", {}).get("include_footer"):
            footer = self._generate_footer(template)
            content = content + "\n\n" + footer

        return content

    def _generate_header(self, template: Dict[str, Any]) -> str:
        """Generate header based on template.

        Args:
            template: Template dictionary.

        Returns:
            Header content.
        """
        lines = [
            "---",
            f"template: {template.get('name', 'default')}",
            f"generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        if template.get("metadata", {}).get("include_timestamps"):
            lines.append(f"timestamp: {datetime.now().isoformat()}")

        lines.append("---")
        return "\n".join(lines)

    def _generate_footer(self, template: Dict[str, Any]) -> str:
        """Generate footer based on template.

        Args:
            template: Template dictionary.

        Returns:
            Footer content.
        """
        lines = ["---"]

        if template.get("metadata", {}).get("include_timestamps"):
            lines.append(
                f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                f"using template '{template.get('name', 'default')}'*"
            )
        else:
            lines.append(f"*Generated using Virtual Agora*")

        return "\n".join(lines)

    def get_css_styles(self, template_name: str = "default") -> str:
        """Get CSS styles from template.

        Args:
            template_name: Template name.

        Returns:
            CSS styles string.
        """
        template = self.get_template(template_name)
        styles = template.get("styles", {})

        css_lines = [
            "body {",
            f"    font-family: {styles.get('font_family', 'Arial, sans-serif')};",
            f"    color: {styles.get('text_color', '#333333')};",
            f"    background-color: {styles.get('background_color', '#ffffff')};",
            "}",
            "",
            "h1, h2, h3, h4, h5, h6 {",
            f"    color: {styles.get('heading_color', '#2c3e50')};",
            "}",
            "",
            "a {",
            f"    color: {styles.get('accent_color', '#3498db')};",
            "}",
            "",
            "blockquote {",
            f"    border-left: 4px solid {styles.get('accent_color', '#3498db')};",
            "}",
        ]

        return "\n".join(css_lines)

    def list_available_templates(self) -> Dict[str, List[str]]:
        """List all available templates.

        Returns:
            Dictionary categorizing available templates.
        """
        return {
            "default": ["default"],
            "languages": list(self.LANGUAGE_TEMPLATES.keys()),
            "custom": list(self.custom_templates.keys()),
        }

    def validate_template(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a template structure.

        Args:
            template: Template to validate.

        Returns:
            Validation results.
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        # Check required fields
        required_fields = ["name", "sections"]
        for field in required_fields:
            if field not in template:
                results["valid"] = False
                results["errors"].append(f"Missing required field: {field}")

        # Validate sections
        if "sections" in template:
            if not isinstance(template["sections"], list):
                results["valid"] = False
                results["errors"].append("Sections must be a list")
            else:
                for i, section in enumerate(template["sections"]):
                    if not isinstance(section, dict):
                        results["errors"].append(f"Section {i} must be a dictionary")
                    elif "id" not in section or "title" not in section:
                        results["errors"].append(
                            f"Section {i} missing required fields (id, title)"
                        )

        # Check for recommended fields
        if "version" not in template:
            results["warnings"].append("Template missing version field")

        if "styles" not in template:
            results["warnings"].append("Template missing styles configuration")

        return results

    def export_template_preview(
        self,
        template_name: str,
        output_path: Path,
    ) -> Path:
        """Export a preview of a template.

        Args:
            template_name: Template to preview.
            output_path: Output file path.

        Returns:
            Path to preview file.
        """
        template = self.get_template(template_name)

        # Generate preview content
        preview_lines = [
            f"# Template Preview: {template.get('name', 'Unknown')}",
            "",
            f"**Version**: {template.get('version', 'N/A')}",
            "",
            "## Section Structure",
            "",
        ]

        for section in template.get("sections", []):
            required = "Required" if section.get("required", False) else "Optional"
            preview_lines.append(f"- **{section['title']}** ({required})")

        preview_lines.extend(
            [
                "",
                "## Styles",
                "",
                "```css",
                self.get_css_styles(template_name),
                "```",
                "",
                "## Metadata Settings",
                "",
            ]
        )

        metadata = template.get("metadata", {})
        for key, value in metadata.items():
            preview_lines.append(f"- {key.replace('_', ' ').title()}: {value}")

        # Save preview
        output_path.write_text("\n".join(preview_lines), encoding="utf-8")

        logger.info(f"Exported template preview to {output_path}")
        return output_path
