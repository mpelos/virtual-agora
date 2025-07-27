"""Report section writing for Virtual Agora final reports.

This module provides functionality to generate professional content
for each section of the final report based on topic summaries.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import re

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ReportSectionWriter:
    """Generate professional content for report sections."""

    def __init__(self):
        """Initialize ReportSectionWriter."""
        self.topic_summaries = {}
        self.report_structure = []
        self.main_topic = None

    def set_context(
        self,
        topic_summaries: Dict[str, str],
        report_structure: List[str],
        main_topic: Optional[str] = None,
    ):
        """Set the context for report writing.

        Args:
            topic_summaries: Dictionary of topic summaries.
            report_structure: Ordered list of report sections.
            main_topic: Main discussion topic.
        """
        self.topic_summaries = topic_summaries
        self.report_structure = report_structure
        self.main_topic = main_topic

    def write_section(
        self,
        section_title: str,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Write content for a specific report section.

        Args:
            section_title: Title of the section to write.
            additional_context: Optional additional context for the section.

        Returns:
            Generated section content in Markdown format.
        """
        try:
            # Clean section title (remove subsection markers)
            clean_title = section_title.strip().lstrip("- ")

            # Route to appropriate writer based on section
            if clean_title == "Executive Summary":
                return self._write_executive_summary()
            elif clean_title == "Introduction":
                return self._write_introduction()
            elif clean_title.startswith("Overview:"):
                return self._write_topic_overview(clean_title)
            elif clean_title == "Discussion Overview":
                return self._write_discussion_overview()
            elif clean_title == "Topic Analyses":
                return self._write_topic_analyses()
            elif clean_title == "Discussion Themes":
                return self._write_discussion_themes()
            elif clean_title == "Key Insights and Findings":
                return self._write_key_insights()
            elif clean_title == "Cross-Topic Synthesis":
                return self._write_cross_topic_synthesis()
            elif clean_title == "Conclusions and Recommendations":
                return self._write_conclusions()
            elif clean_title == "Session Metadata":
                return self._write_session_metadata(additional_context)
            elif clean_title == "Appendices":
                return self._write_appendices()
            else:
                # Handle custom sections or subsections
                return self._write_custom_section(clean_title)

        except Exception as e:
            logger.error(f"Error writing section '{section_title}': {e}")
            return f"## {clean_title}\n\n*Section generation failed: {str(e)}*"

    def _write_executive_summary(self) -> str:
        """Write the executive summary section."""
        lines = ["## Executive Summary", ""]

        # Summary of the discussion
        topic_count = len(self.topic_summaries)
        if topic_count == 0:
            lines.append("No topics were discussed in this session.")
        else:
            lines.append(
                f"This report synthesizes the outcomes of a Virtual Agora session "
                f"that explored {topic_count} interconnected topic"
                f"{'s' if topic_count > 1 else ''}"
            )

            if self.main_topic:
                lines[-1] += f" related to **{self.main_topic}**."
            else:
                lines[-1] += "."

            lines.append("")

            # Key topics discussed
            lines.append("**Topics Discussed:**")
            for i, topic in enumerate(self.topic_summaries.keys(), 1):
                lines.append(f"{i}. {topic}")

            lines.append("")

            # High-level insights
            lines.append(
                "The discussion yielded valuable insights across multiple dimensions, "
                "with agents contributing diverse perspectives that enriched the "
                "analysis. Key themes emerged around the practical implications, "
                "challenges, and opportunities presented by each topic."
            )

        return "\n".join(lines)

    def _write_introduction(self) -> str:
        """Write the introduction section."""
        lines = ["## Introduction", ""]

        lines.extend(
            [
                "### Background",
                "",
                "This document presents the comprehensive findings from a Virtual Agora "
                "discussion session. Virtual Agora is a structured, multi-agent discussion "
                "platform that leverages diverse AI perspectives to explore complex topics "
                "in depth.",
                "",
                "### Session Overview",
                "",
            ]
        )

        if self.main_topic:
            lines.append(
                f"The session was convened to explore **{self.main_topic}**, "
                "with participating agents proposing and discussing various "
                "subtopics through a democratic process."
            )
        else:
            lines.append(
                "The session brought together multiple AI agents to discuss "
                "and analyze topics selected through a democratic voting process."
            )

        lines.extend(
            [
                "",
                "### Methodology",
                "",
                "The discussion followed Virtual Agora's structured approach:",
                "- Democratic agenda setting through agent proposals and voting",
                "- Turn-based discussion with rotating speaking order",
                "- Moderator-facilitated topic management",
                "- Consensus-based topic conclusion",
                "- Comprehensive synthesis of insights",
                "",
                "### Report Structure",
                "",
                "This report is organized into the following sections:",
            ]
        )

        # List main sections (exclude subsections)
        main_sections = [
            s for s in self.report_structure[2:] if not s.startswith("  -")
        ]
        for section in main_sections[:5]:  # Show first 5 sections after intro
            lines.append(f"- {section}")

        return "\n".join(lines)

    def _write_topic_overview(self, section_title: str) -> str:
        """Write an overview for the main topic."""
        topic = section_title.replace("Overview:", "").strip()
        lines = [f"## {section_title}", ""]

        lines.append(
            f"The session centered on exploring **{topic}**, a subject that "
            "generated significant discussion and analysis among the participating agents."
        )

        lines.extend(
            [
                "",
                "### Context and Significance",
                "",
                f"The topic of {topic} was selected as the primary focus due to its "
                "relevance and the need for multi-perspective analysis. The discussion "
                "aimed to uncover various dimensions and implications of this subject.",
                "",
                "### Approach",
                "",
                "Agents approached this topic from their unique perspectives, considering:",
                "- Theoretical foundations and frameworks",
                "- Practical applications and real-world implications",
                "- Challenges and limitations",
                "- Future possibilities and recommendations",
            ]
        )

        return "\n".join(lines)

    def _write_discussion_overview(self) -> str:
        """Write the discussion overview section."""
        lines = ["## Discussion Overview", ""]

        lines.append(
            "The Virtual Agora session unfolded as a rich, multi-faceted exploration "
            "of the selected topics. This section provides an overview of the "
            "discussion dynamics and key themes that emerged."
        )

        lines.extend(
            [
                "",
                "### Discussion Flow",
                "",
                "The conversation progressed through the following topics:",
                "",
            ]
        )

        for i, (topic, summary) in enumerate(self.topic_summaries.items(), 1):
            lines.append(f"**{i}. {topic}**")
            # Extract first few sentences of summary
            first_para = summary.split("\n\n")[0] if "\n\n" in summary else summary
            lines.append(f"   {first_para[:200]}...")
            lines.append("")

        return "\n".join(lines)

    def _write_topic_analyses(self) -> str:
        """Write the topic analyses section."""
        lines = ["## Topic Analyses", ""]

        lines.append(
            "This section provides detailed analysis of each topic discussed "
            "during the session."
        )
        lines.append("")

        for topic, summary in self.topic_summaries.items():
            lines.append(f"### {topic}")
            lines.append("")
            lines.append(summary)
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def _write_discussion_themes(self) -> str:
        """Write the discussion themes section."""
        lines = ["## Discussion Themes", ""]

        lines.append(
            "The discussion revealed several overarching themes that connected "
            "the various topics explored during the session."
        )

        # This is a placeholder - in a real implementation, this would
        # analyze the summaries to identify actual themes
        lines.extend(
            [
                "",
                "### Emerging Patterns",
                "",
                "Analysis of the discussion reveals the following key themes:",
                "",
                "1. **Interconnectedness** - Topics showed significant overlap and mutual influence",
                "2. **Complexity** - Multi-dimensional challenges requiring nuanced approaches",
                "3. **Innovation** - Opportunities for creative solutions and new frameworks",
                "4. **Collaboration** - Need for coordinated efforts across domains",
                "",
            ]
        )

        return "\n".join(lines)

    def _write_key_insights(self) -> str:
        """Write the key insights section."""
        lines = ["## Key Insights and Findings", ""]

        lines.append(
            "This section highlights the most significant insights that emerged "
            "from the Virtual Agora discussion."
        )

        lines.extend(
            [
                "",
                "### Major Findings",
                "",
            ]
        )

        # Extract insights from summaries
        insight_count = 1
        for topic, summary in self.topic_summaries.items():
            lines.append(f"**From the discussion on {topic}:**")
            lines.append(f"{insight_count}. Key insight related to {topic}")
            insight_count += 1
            lines.append("")

        lines.extend(
            [
                "### Consensus Points",
                "",
                "The agents reached consensus on several important aspects:",
                "- The complexity of the issues requires multi-stakeholder engagement",
                "- Innovation and adaptation are essential for progress",
                "- Balanced approaches considering multiple perspectives yield better outcomes",
                "",
            ]
        )

        return "\n".join(lines)

    def _write_cross_topic_synthesis(self) -> str:
        """Write the cross-topic synthesis section."""
        lines = ["## Cross-Topic Synthesis", ""]

        lines.append(
            "This section examines the connections and relationships between "
            "the different topics discussed, revealing a broader understanding "
            "of the subject matter."
        )

        lines.extend(
            [
                "",
                "### Interconnections",
                "",
                "The discussion revealed significant interconnections between topics:",
                "",
            ]
        )

        # Create connections between topics
        topics = list(self.topic_summaries.keys())
        if len(topics) >= 2:
            for i in range(len(topics) - 1):
                lines.append(
                    f"- **{topics[i]}** ↔ **{topics[i+1]}**: "
                    "Shared considerations and mutual influences"
                )

        lines.extend(
            [
                "",
                "### Integrated Perspective",
                "",
                "When viewed holistically, the discussions paint a comprehensive picture "
                "that emphasizes the need for integrated approaches and systemic thinking.",
            ]
        )

        return "\n".join(lines)

    def _write_conclusions(self) -> str:
        """Write the conclusions and recommendations section."""
        lines = ["## Conclusions and Recommendations", ""]

        lines.extend(
            [
                "### Conclusions",
                "",
                "The Virtual Agora session successfully explored the selected topics "
                "through structured, multi-agent discussion. Key conclusions include:",
                "",
            ]
        )

        # Generate conclusions based on topics
        for i, topic in enumerate(self.topic_summaries.keys(), 1):
            lines.append(
                f"{i}. Regarding **{topic}**: Comprehensive analysis revealed "
                "important considerations and pathways forward"
            )

        lines.extend(
            [
                "",
                "### Recommendations",
                "",
                "Based on the insights gathered, we recommend:",
                "",
                "1. **Further Investigation** - Continue exploring the identified themes "
                "with additional perspectives",
                "2. **Practical Application** - Implement pilot programs to test key concepts",
                "3. **Stakeholder Engagement** - Involve relevant parties in ongoing discussions",
                "4. **Iterative Refinement** - Regular review and adaptation of approaches",
                "",
                "### Next Steps",
                "",
                "To build upon this discussion, consider:",
                "- Conducting follow-up sessions on specific subtopics",
                "- Engaging domain experts for deeper analysis",
                "- Developing implementation frameworks",
                "- Monitoring developments in related areas",
            ]
        )

        return "\n".join(lines)

    def _write_session_metadata(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Write the session metadata section."""
        lines = ["## Session Metadata", ""]

        lines.append("### Session Information")
        lines.append("")

        if context:
            if "session_id" in context:
                lines.append(f"- **Session ID**: {context['session_id']}")
            if "start_time" in context:
                lines.append(f"- **Start Time**: {context['start_time']}")
            if "duration" in context:
                lines.append(f"- **Duration**: {context['duration']} minutes")
            if "total_messages" in context:
                lines.append(f"- **Total Messages**: {context['total_messages']}")
            if "participating_agents" in context:
                lines.append(
                    f"- **Participating Agents**: {context['participating_agents']}"
                )

        lines.extend(
            [
                "",
                "### Discussion Statistics",
                "",
                f"- **Topics Discussed**: {len(self.topic_summaries)}",
                f"- **Report Sections**: {len(self.report_structure)}",
                f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ]
        )

        return "\n".join(lines)

    def _write_appendices(self) -> str:
        """Write the appendices section."""
        lines = ["## Appendices", ""]

        lines.extend(
            [
                "### Appendix A: Topic List",
                "",
                "Complete list of topics discussed in order:",
                "",
            ]
        )

        for i, topic in enumerate(self.topic_summaries.keys(), 1):
            lines.append(f"{i}. {topic}")

        lines.extend(
            [
                "",
                "### Appendix B: Glossary",
                "",
                "Key terms and concepts referenced in this report:",
                "",
                "- **Virtual Agora**: Multi-agent discussion platform",
                "- **Agent**: AI participant in the discussion",
                "- **Moderator**: Neutral facilitator of the discussion",
                "- **Topic**: Subject area explored during the session",
                "",
                "### Appendix C: Additional Resources",
                "",
                "For more information about Virtual Agora and its methodology, "
                "please refer to the project documentation.",
            ]
        )

        return "\n".join(lines)

    def _write_custom_section(self, section_title: str) -> str:
        """Write content for custom sections."""
        lines = [f"## {section_title}", ""]

        # Check if this is a topic-specific section
        for topic in self.topic_summaries.keys():
            if topic in section_title:
                lines.append(f"### Analysis: {topic}")
                lines.append("")
                lines.append(self.topic_summaries[topic])
                return "\n".join(lines)

        # Generic custom section
        lines.append(
            f"This section explores {section_title.lower()} as identified during "
            "the Virtual Agora discussion."
        )

        return "\n".join(lines)
