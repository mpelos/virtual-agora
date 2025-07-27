"""Ecclesia Report Agent implementation for Virtual Agora v1.3.

This module provides a specialized agent for generating the final session report,
analyzing all topic reports to create a comprehensive multi-section synthesis.
"""

import json
import re
from typing import List, Dict, Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ValidationError

from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class ReportStructure(BaseModel):
    """Pydantic model for report structure validation."""

    report_sections: List[str]


class EcclesiaReportAgent(LLMAgent):
    """Specialized agent for final report generation.

    This agent is invoked at the end of the entire session to analyze
    all topic reports and create a comprehensive, multi-section final report.
    """

    PROMPT_VERSION = "1.3"

    def __init__(self, agent_id: str, llm: BaseChatModel, **kwargs):
        """Initialize the Ecclesia Report agent.

        Args:
            agent_id: Unique identifier
            llm: Language model instance
        """
        # Use specialized prompt from v1.3 spec
        system_prompt = self._get_ecclesia_report_prompt()

        super().__init__(
            agent_id=agent_id, llm=llm, system_prompt=system_prompt, **kwargs
        )

    def _get_ecclesia_report_prompt(self) -> str:
        """Get the specialized ecclesia report prompt from v1.3 spec."""
        return """You are 'The Writer' for Virtual Agora's final session analysis. Your task is to read ALL individual topic reports from the session and create a comprehensive, multi-section final report. First, analyze all topic reports and define a logical structure (output as JSON list of section titles). Then, for each section, synthesize content that:
1. Identifies overarching themes across all topics
2. Highlights connections and relationships between different agenda items
3. Summarizes the collective insights and conclusions
4. Notes areas of ongoing uncertainty or debate
5. Provides an executive summary of the entire session's value

Approach this as a professional analyst creating a report for stakeholders who need to understand the session's outcomes and implications."""

    def generate_report_structure(
        self, topic_reports: Dict[str, str], discussion_theme: str
    ) -> List[str]:
        """Define the structure for the final report.

        Args:
            topic_reports: All topic reports keyed by topic
            discussion_theme: Overall session theme

        Returns:
            List of section titles for the report
        """
        # Prepare context
        topics_list = list(topic_reports.keys())
        topics_summary = "\n".join([f"- {topic}" for topic in topics_list])

        # Include brief excerpts from each report
        report_excerpts = []
        for topic, report in topic_reports.items():
            # Take first 200 characters as excerpt
            excerpt = report[:200] + "..." if len(report) > 200 else report
            report_excerpts.append(f"Topic: {topic}\n{excerpt}")

        excerpts_text = "\n\n".join(report_excerpts)

        prompt = f"""Discussion Theme: {discussion_theme}

Topics Discussed:
{topics_summary}

Topic Report Excerpts:
{excerpts_text}

Analyze these topics and create a logical structure for a comprehensive final report. 
Output a JSON object with "report_sections" containing an ordered list of section titles.
Example: {{"report_sections": ["Executive Summary", "Key Themes", "Conclusions"]}}"""

        # Generate structure with JSON validation
        response = self.generate_response(prompt)

        # Parse and validate JSON
        try:
            # Clean up response to ensure it's valid JSON
            response = response.strip()
            json_match = re.search(
                r'\{\s*"report_sections"\s*:\s*\[.*?\]\s*\}', response, re.DOTALL
            )
            if not json_match:
                raise ValueError(
                    "Could not find valid report structure JSON in the response."
                )

            response = json_match.group()
            data = json.loads(response)
            validated_data = ReportStructure(**data)
            sections = validated_data.report_sections

            logger.info(f"Generated report structure with {len(sections)} sections")
            return sections

        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse report structure: {e}")
            # Fallback to default structure
            default_sections = [
                "Executive Summary",
                "Key Themes and Patterns",
                "Points of Consensus",
                "Areas of Disagreement",
                "Notable Insights",
                "Recommendations",
                "Conclusion",
            ]
            logger.warning(
                f"Using default report structure with {len(default_sections)} sections"
            )
            return default_sections

    def write_section(
        self,
        section_title: str,
        topic_reports: Dict[str, str],
        discussion_theme: str,
        previous_sections: Dict[str, str],
    ) -> str:
        """Write content for a specific report section.

        Args:
            section_title: Title of section to write
            topic_reports: All topic reports
            discussion_theme: Overall theme
            previous_sections: Already written sections

        Returns:
            Section content
        """
        # Context from previous sections
        prev_context = ""
        if previous_sections:
            prev_context = "Previously written sections (for context):\n" + "\n".join(
                [
                    f"{title}:\n{content[:300]}..."
                    for title, content in previous_sections.items()
                ]
            )

        # Full topic reports
        reports_text = "\n\n".join(
            [
                f"Topic: {topic}\n---\n{report}\n---"
                for topic, report in topic_reports.items()
            ]
        )

        prompt = f"""Discussion Theme: {discussion_theme}
Section to Write: {section_title}

{prev_context}

All Topic Reports:
{reports_text}

Write comprehensive content for the section '{section_title}'. Synthesize information across all topics to identify patterns, connections, and high-level insights. Write in a professional, analytical style suitable for stakeholders."""

        content = self.generate_response(prompt)

        logger.info(
            f"Generated content for section '{section_title}' ({len(content)} chars)"
        )

        return content

    def generate_section_content(
        self, section_title: str, context: Dict[str, Any]
    ) -> str:
        """Generate content for a report section (alternative method).

        This method is migrated from ModeratorAgent's generate_section_content.

        Args:
            section_title: Title of the section
            context: Context including theme, topics, summaries

        Returns:
            Section content
        """
        theme = context.get("theme", "")
        topics = context.get("topics", [])
        summaries = context.get("summaries", {})

        # Format context
        topics_list = "\n".join([f"- {topic}" for topic in topics])

        summaries_text = ""
        if summaries:
            summaries_text = "\n\nTopic Summaries:\n" + "\n\n".join(
                [f"{topic}:\n{summary}" for topic, summary in summaries.items()]
            )

        prompt = f"""Report Section: {section_title}
Discussion Theme: {theme}

Topics Covered:
{topics_list}{summaries_text}

Write comprehensive, analytical content for this section of the final report. Focus on:
- Cross-topic synthesis and analysis
- Identifying overarching patterns and themes
- Drawing meaningful, high-level conclusions
- Providing actionable insights for executive readership"""

        content = self.generate_response(prompt)

        return content
