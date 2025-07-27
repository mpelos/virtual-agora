"""Report structure definition for Virtual Agora final reports.

This module provides functionality to analyze topic summaries and define
a logical structure for the final multi-section report.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ReportStructureManager:
    """Manage the structure definition for final reports."""

    # Standard report sections
    STANDARD_SECTIONS = [
        "Executive Summary",
        "Introduction",
        "Discussion Overview",
        "Key Insights and Findings",
        "Conclusions and Recommendations",
        "Appendices",
    ]

    def __init__(self):
        """Initialize ReportStructureManager."""
        self.structure_cache = None

    def define_structure(
        self,
        topic_summaries: Dict[str, str],
        main_topic: Optional[str] = None,
        custom_sections: Optional[List[str]] = None,
    ) -> List[str]:
        """Define report structure based on topic summaries.

        Args:
            topic_summaries: Dictionary mapping topic names to summary content.
            main_topic: Main discussion topic if available.
            custom_sections: Optional custom sections to include.

        Returns:
            Ordered list of section titles.
        """
        try:
            # Analyze content to determine appropriate sections
            sections = self._analyze_content_for_sections(topic_summaries, main_topic)

            # Add custom sections if provided
            if custom_sections:
                sections = self._integrate_custom_sections(sections, custom_sections)

            # Validate and finalize structure
            final_structure = self._validate_structure(sections)

            # Cache the structure
            self.structure_cache = final_structure

            logger.info(
                f"Defined report structure with {len(final_structure)} sections"
            )
            return final_structure

        except Exception as e:
            logger.error(f"Error defining report structure: {e}")
            # Return a copy of default structure on error
            return list(self.STANDARD_SECTIONS)

    def _analyze_content_for_sections(
        self,
        topic_summaries: Dict[str, str],
        main_topic: Optional[str] = None,
    ) -> List[str]:
        """Analyze content to determine appropriate sections.

        Args:
            topic_summaries: Dictionary of topic summaries.
            main_topic: Main discussion topic.

        Returns:
            List of section titles.
        """
        sections = []

        # Always start with executive summary and introduction
        sections.extend(["Executive Summary", "Introduction"])

        # Add main topic overview if available
        if main_topic:
            sections.append(f"Overview: {main_topic}")

        # Analyze topics to determine structure
        topic_count = len(topic_summaries)

        if topic_count == 0:
            # No topics discussed
            sections.append("Discussion Summary")
        elif topic_count == 1:
            # Single topic - simple structure
            topic = list(topic_summaries.keys())[0]
            sections.append(f"Topic Analysis: {topic}")
        elif topic_count <= 3:
            # Few topics - individual sections
            sections.append("Topic Analyses")
            for topic in topic_summaries.keys():
                sections.append(f"  - {topic}")
        else:
            # Many topics - grouped structure
            sections.append("Discussion Themes")

            # Group topics by theme if possible
            themes = self._identify_themes(topic_summaries)
            for theme in themes:
                sections.append(f"  - {theme}")

        # Add insights and conclusions
        sections.extend(
            [
                "Key Insights and Findings",
                "Cross-Topic Synthesis",
                "Conclusions and Recommendations",
            ]
        )

        # Add metadata and appendices
        sections.extend(
            [
                "Session Metadata",
                "Appendices",
            ]
        )

        return sections

    def _identify_themes(self, topic_summaries: Dict[str, str]) -> List[str]:
        """Identify common themes across topics.

        Args:
            topic_summaries: Dictionary of topic summaries.

        Returns:
            List of identified themes.
        """
        # Simple theme identification based on topic names
        # In a more sophisticated implementation, this could use NLP
        topics = list(topic_summaries.keys())

        # Group by common keywords
        themes = []

        # Check for common patterns
        if any("technical" in t.lower() or "technology" in t.lower() for t in topics):
            themes.append("Technical Aspects")

        if any("legal" in t.lower() or "regulation" in t.lower() for t in topics):
            themes.append("Legal and Regulatory Considerations")

        if any("social" in t.lower() or "community" in t.lower() for t in topics):
            themes.append("Social and Community Impact")

        if any("economic" in t.lower() or "financial" in t.lower() for t in topics):
            themes.append("Economic Implications")

        if any("future" in t.lower() or "trend" in t.lower() for t in topics):
            themes.append("Future Outlook and Trends")

        # If no themes identified, use generic grouping
        if not themes:
            if len(topics) <= 5:
                themes = ["Primary Topics", "Secondary Considerations"]
            else:
                themes = [
                    "Core Discussions",
                    "Supporting Topics",
                    "Additional Considerations",
                ]

        return themes

    def _integrate_custom_sections(
        self, base_sections: List[str], custom_sections: List[str]
    ) -> List[str]:
        """Integrate custom sections into the base structure.

        Args:
            base_sections: Base section list.
            custom_sections: Custom sections to add.

        Returns:
            Integrated section list.
        """
        # Insert custom sections before conclusions
        insert_index = -3  # Before "Conclusions", "Metadata", "Appendices"

        for section in custom_sections:
            if section not in base_sections:
                base_sections.insert(insert_index, section)
                insert_index += 1

        return base_sections

    def _validate_structure(self, sections: List[str]) -> List[str]:
        """Validate and clean up the report structure.

        Args:
            sections: Proposed section list.

        Returns:
            Validated section list.
        """
        # Ensure no duplicates while preserving order
        seen = set()
        validated = []

        for section in sections:
            # Skip empty sections
            if not section or not section.strip():
                continue

            # Handle sub-sections (indented with -)
            if section.startswith("  -"):
                # Always include sub-sections (preserve original formatting)
                validated.append(section)
            else:
                # Clean up section names for main sections only
                clean_section = section.strip()
                # Check for duplicates in main sections
                if clean_section not in seen:
                    seen.add(clean_section)
                    validated.append(clean_section)

        # Ensure minimum required sections
        required = [
            "Executive Summary",
            "Introduction",
            "Conclusions and Recommendations",
        ]
        for req in required:
            if req not in validated:
                if req == "Conclusions and Recommendations":
                    validated.append(req)
                else:
                    validated.insert(0, req)

        return validated

    def export_structure(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Export the report structure as JSON.

        Args:
            output_path: Optional path to save the structure.

        Returns:
            Dictionary representation of the structure.
        """
        if not self.structure_cache:
            raise ValueError("No structure defined yet")

        structure_data = {
            "report_structure": self.structure_cache,
            "section_count": len(self.structure_cache),
            "has_subsections": any(s.startswith("  -") for s in self.structure_cache),
        }

        if output_path:
            output_path.write_text(
                json.dumps(structure_data, indent=2), encoding="utf-8"
            )
            logger.info(f"Exported report structure to {output_path}")

        return structure_data

    def get_section_hierarchy(self) -> Dict[str, List[str]]:
        """Get the hierarchical structure of sections.

        Returns:
            Dictionary mapping main sections to their subsections.
        """
        if not self.structure_cache:
            raise ValueError("No structure defined yet")

        hierarchy = {}
        current_main = None

        for section in self.structure_cache:
            if section.startswith("  -"):
                # This is a subsection
                if current_main:
                    if current_main not in hierarchy:
                        hierarchy[current_main] = []
                    hierarchy[current_main].append(section.strip("  - "))
            else:
                # This is a main section
                current_main = section
                hierarchy[current_main] = []

        return hierarchy
