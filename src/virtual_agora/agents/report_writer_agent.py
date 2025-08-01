"""Report Writer Agent implementation for Virtual Agora v1.3.

This module provides a unified agent for generating long-form reports through
an iterative process that overcomes LLM token output limitations. It handles
both topic-specific reports and comprehensive session reports.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, ValidationError

from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.context.agent_mixin import ContextAwareMixin
from virtual_agora.context.builders import ReportWriterContextBuilder
from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class ReportStructure(BaseModel):
    """Pydantic model for report structure validation."""

    sections: List[Dict[str, str]]


class ReportWriterAgent(ContextAwareMixin, LLMAgent):
    """Specialized agent for long-form report generation through iterative process.

    This agent consolidates the functionality of TopicReportAgent and EcclesiaReportAgent,
    working in two phases:
    1. Structure Creation: Analyzes source material and creates detailed outline
    2. Iterative Writing: Writes one section at a time based on the outline

    Due to LLM output token limitations, this approach ensures comprehensive reports
    while maintaining quality and avoiding truncation.

    Uses ReportWriterContextBuilder to ensure ONLY filtered discussion data is received,
    not context directory files or other extraneous information.
    """

    PROMPT_VERSION = "1.3"

    def __init__(self, agent_id: str, llm: BaseChatModel, **kwargs):
        """Initialize the Report Writer agent.

        Args:
            agent_id: Unique identifier
            llm: Language model instance
        """
        # Use specialized prompt from v1.3 spec
        system_prompt = self._get_report_writer_prompt()

        # Initialize with ReportWriterContextBuilder for proper context isolation
        super().__init__(
            context_builder=ReportWriterContextBuilder(),
            agent_id=agent_id,
            llm=llm,
            system_prompt=system_prompt,
            **kwargs,
        )

    def _get_report_writer_prompt(self) -> str:
        """Get the specialized report writer prompt from v1.3 spec."""
        return """You are a specialized long-form report writer for Virtual Agora. Due to LLM output token limitations, you must work iteratively to produce comprehensive reports. Your process has two phases:

**Phase 1 - Structure Creation:** When given source material (round summaries, topic discussions, or multiple topic reports), analyze the content and create a detailed outline. Output this as a JSON structure with section titles and brief descriptions of what each section will cover.

**Phase 2 - Iterative Writing:** You will be called multiple times, each time to write ONE section from your outline. Each section must be:
- Comprehensive and standalone (readable without other sections)
- Objective and analytical, not participatory  
- Thorough enough that readers understand the full scope of that aspect
- Well-organized with clear subsections where appropriate

For **Topic Reports**: Structure should include topic overview, major themes, consensus points, disagreements, key insights, and implications.

For **Session Reports**: Structure should include executive summary, overarching themes across topics, connections between agenda items, collective insights, areas of uncertainty, and session value assessment.

You must ensure no key points are missed while organizing complex information into readable, concise sections. Write as a professional analyst creating reports for stakeholders who need to understand the outcomes and implications."""

    def create_report_structure_with_state(
        self,
        state: VirtualAgoraState,
        report_type: str = "topic",
        topic: Optional[str] = None,
    ) -> Tuple[List[Dict[str, str]], str]:
        """Create detailed structure for a report using the new context system (Phase 1).

        Args:
            state: Application state for context building
            report_type: Type of report ("topic" or "session")
            topic: Topic for topic reports (optional)

        Returns:
            Tuple of (sections list, raw response)
        """
        if report_type == "topic":
            return self._create_topic_structure_with_state(state, topic)
        elif report_type == "session":
            return self._create_session_structure_with_state(state)
        else:
            raise ValueError(f"Unknown report type: {report_type}")

    def create_report_structure(
        self,
        source_material: Dict[str, Any],
        report_type: str = "topic",
    ) -> Tuple[List[Dict[str, str]], str]:
        """Create detailed structure for a report (Phase 1) - Legacy method.

        Args:
            source_material: Dictionary containing source material
            report_type: Type of report ("topic" or "session")

        Returns:
            Tuple of (sections list, raw response)

        Note:
            This is the legacy method. Use create_report_structure_with_state for proper context isolation.
        """
        if report_type == "topic":
            return self._create_topic_structure(source_material)
        elif report_type == "session":
            return self._create_session_structure(source_material)
        else:
            raise ValueError(f"Unknown report type: {report_type}")

    def _create_topic_structure(
        self, source_material: Dict[str, Any]
    ) -> Tuple[List[Dict[str, str]], str]:
        """Create structure for topic report."""
        topic = source_material.get("topic", "Unknown Topic")
        theme = source_material.get("theme", "")
        round_summaries = source_material.get("round_summaries", [])
        final_considerations = source_material.get("final_considerations", [])

        # Format source material
        summaries_text = "\n\n".join(
            [
                f"Round {i+1} Summary:\n{summary}"
                for i, summary in enumerate(round_summaries)
            ]
        )

        considerations_text = ""
        if final_considerations:
            considerations_text = "\n\n**Final Considerations:**\n" + "\n".join(
                [f"- {consideration}" for consideration in final_considerations]
            )

        # Use JsonOutputParser to get proper format instructions
        parser = JsonOutputParser(pydantic_object=ReportStructure)
        format_instructions = parser.get_format_instructions()

        prompt = f"""**PHASE 1: STRUCTURE CREATION**

**Report Type:** Topic Report
**Discussion Theme:** {theme}
**Topic:** {topic}

**Source Material:**
{summaries_text}{considerations_text}

Analyze this source material and create a detailed structure for a comprehensive topic report.

{format_instructions}

Your JSON should contain a "sections" array with objects having "title" and "description" fields.

Example:
{{
  "sections": [
    {{"title": "Executive Summary", "description": "Brief overview of the topic and key findings"}},
    {{"title": "Discussion Evolution", "description": "How the conversation developed over time"}},
    {{"title": "Key Arguments", "description": "Major arguments and perspectives presented"}}
  ]
}}

Ensure the structure covers: topic overview, discussion narrative, consensus points, disagreements, key insights, and implications."""

        response = self.generate_response(prompt)
        sections = self._parse_structure_response(response)

        logger.info(f"Created topic report structure with {len(sections)} sections")
        return sections, response

    def _create_session_structure(
        self, source_material: Dict[str, Any]
    ) -> Tuple[List[Dict[str, str]], str]:
        """Create structure for session report."""
        theme = source_material.get("theme", "")
        topic_reports = source_material.get("topic_reports", {})

        # Format topic reports
        topics_list = list(topic_reports.keys())
        topics_summary = "\n".join([f"- {topic}" for topic in topics_list])

        # Include excerpts from each report
        report_excerpts = []
        for topic, report in topic_reports.items():
            excerpt = report[:300] + "..." if len(report) > 300 else report
            report_excerpts.append(f"**Topic: {topic}**\n{excerpt}")

        excerpts_text = "\n\n".join(report_excerpts)

        # Use JsonOutputParser to get proper format instructions
        parser = JsonOutputParser(pydantic_object=ReportStructure)
        format_instructions = parser.get_format_instructions()

        prompt = f"""**PHASE 1: STRUCTURE CREATION**

**Report Type:** Session Report
**Discussion Theme:** {theme}

**Topics Discussed:**
{topics_summary}

**Topic Report Excerpts:**
{excerpts_text}

Analyze these topic reports and create a detailed structure for a comprehensive final session report.

{format_instructions}

Your JSON should contain a "sections" array with objects having "title" and "description" fields.

Example:
{{
  "sections": [
    {{"title": "Executive Summary", "description": "High-level overview of the entire session"}},
    {{"title": "Cross-Topic Themes", "description": "Overarching patterns across all discussions"}},
    {{"title": "Synthesis", "description": "Integration of insights from all topics"}}
  ]
}}

Ensure the structure covers: executive summary, overarching themes, connections between topics, collective insights, areas of uncertainty, and session value."""

        response = self.generate_response(prompt)
        sections = self._parse_structure_response(response)

        logger.info(f"Created session report structure with {len(sections)} sections")
        return sections, response

    def _parse_structure_response(self, response: str) -> List[Dict[str, str]]:
        """Parse and validate JSON structure response using LangChain JsonOutputParser."""
        try:
            # Check for empty response first
            if not response or response.strip() == "":
                logger.warning("Empty response received from LLM")
                return self._get_default_structure()

            logger.debug(f"Parsing structure response: {response[:200]}...")

            # Use LangChain's JsonOutputParser with our Pydantic model
            parser = JsonOutputParser(pydantic_object=ReportStructure)

            # The JsonOutputParser can handle various formats including markdown-wrapped JSON
            try:
                parsed_data = parser.parse(response)
                # Handle both Pydantic object and dictionary returns from JsonOutputParser
                if hasattr(parsed_data, "sections"):
                    return parsed_data.sections
                elif isinstance(parsed_data, dict) and "sections" in parsed_data:
                    return parsed_data["sections"]
                else:
                    raise ValueError(
                        f"Unexpected parsed_data format: {type(parsed_data)}"
                    )
            except Exception as parse_error:
                logger.warning(
                    f"JsonOutputParser failed: {parse_error}, trying direct JSON parsing"
                )

                # Fallback: try to extract JSON manually and use the parser
                # Handle both markdown-wrapped JSON (```json ... ```) and plain JSON
                json_match = re.search(
                    r'```json\s*(\{.*?\})\s*```|(\{.*?"sections"\s*:\s*\[.*?\]\s*\})',
                    response,
                    re.DOTALL,
                )
                if json_match:
                    # Get the first non-None group (either markdown-wrapped or plain)
                    json_str = json_match.group(1) or json_match.group(2)
                    data = json.loads(json_str)
                    validated_data = ReportStructure(**data)
                    return validated_data.sections
                else:
                    logger.warning(
                        f"Could not find valid structure JSON in response: {response[:500]}..."
                    )
                    raise ValueError("Could not find valid structure JSON in response")

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            logger.error(f"Failed to parse report structure: {e}")
            logger.error(f"Raw response was: {response}")
            # Fallback to default structures
            return self._get_default_structure()

    def _get_default_structure(self) -> List[Dict[str, str]]:
        """Get default report structure as fallback."""
        return [
            {"title": "Executive Summary", "description": "Overview and key findings"},
            {
                "title": "Main Analysis",
                "description": "Detailed analysis of key points",
            },
            {
                "title": "Key Insights",
                "description": "Important discoveries and perspectives",
            },
            {"title": "Conclusions", "description": "Final thoughts and implications"},
        ]

    def _create_topic_structure_with_state(
        self, state: VirtualAgoraState, topic: Optional[str] = None
    ) -> Tuple[List[Dict[str, str]], str]:
        """Create structure for topic report using context system."""

        prompt = """**PHASE 1: STRUCTURE CREATION**

**Report Type:** Topic Report

Analyze the provided context and create a detailed structure for a comprehensive topic report.

Create 3-6 logical sections that organize the discussion content effectively. Each section must have:
- A clear, specific title that accurately reflects the content
- A comprehensive description explaining the section's purpose, key points to cover, and its role in the overall narrative

The structure should ensure complete coverage of:
- Topic overview and context
- Evolution of the discussion 
- Major arguments and perspectives presented
- Areas of consensus and agreement
- Points of disagreement or debate
- Key insights and discoveries
- Implications and future considerations

Use JsonOutputParser format with a "sections" array containing objects with "title" and "description" fields.

Example:
{
  "sections": [
    {"title": "Executive Summary", "description": "Brief overview of the topic and key findings"},
    {"title": "Discussion Evolution", "description": "How the conversation developed over time"},
    {"title": "Key Arguments", "description": "Major arguments and perspectives presented"}
  ]
}

Make the structure logical, comprehensive, and suitable for stakeholder review."""

        # Use the new context-aware message formatting
        messages = self.format_messages_with_context(
            state=state, prompt=prompt, report_type="topic", topic=topic
        )

        # Generate response using the formatted messages
        response = self.llm.invoke(messages).content
        sections = self._parse_structure_response(response)

        logger.info(
            f"Created topic report structure with {len(sections)} sections using context system"
        )
        return sections, response

    def _create_session_structure_with_state(
        self, state: VirtualAgoraState
    ) -> Tuple[List[Dict[str, str]], str]:
        """Create structure for session report using context system."""

        prompt = """**PHASE 1: STRUCTURE CREATION**

**Report Type:** Session Report

Analyze the provided topic reports and create a detailed structure for a comprehensive final session report.

Create 4-6 logical sections that synthesize insights across all topics discussed. Each section must have:
- A clear, specific title that reflects the synthesized content
- A comprehensive description explaining how this section will connect insights across topics

The structure should ensure complete coverage of:
- Executive summary of key outcomes
- Overarching themes that emerged across multiple topics  
- Connections and relationships between different agenda items
- Collective insights that only become apparent when viewing all topics together
- Areas of uncertainty or questions that require further exploration
- Assessment of overall session value and impact
- Implications for future discussions or decisions

Use JsonOutputParser format with a "sections" array containing objects with "title" and "description" fields.

Example:
{
  "sections": [
    {"title": "Executive Summary", "description": "High-level overview of the entire session"},
    {"title": "Cross-Topic Themes", "description": "Overarching patterns across all discussions"},
    {"title": "Synthesis", "description": "Integration of insights from all topics"}
  ]
}

Make the structure comprehensive and suitable for stakeholders who need to understand the collective outcomes of the entire session."""

        # Use the new context-aware message formatting
        messages = self.format_messages_with_context(
            state=state, prompt=prompt, report_type="session"
        )

        # Generate response using the formatted messages
        response = self.llm.invoke(messages).content
        sections = self._parse_structure_response(response)

        logger.info(
            f"Created session report structure with {len(sections)} sections using context system"
        )
        return sections, response

    def write_section_with_state(
        self,
        state: VirtualAgoraState,
        section: Dict[str, str],
        report_type: str = "topic",
        topic: Optional[str] = None,
        previous_sections: Optional[List[str]] = None,
    ) -> str:
        """Write content for a specific report section using context system (Phase 2).

        Args:
            state: Application state for context building
            section: Section dict with title and description
            report_type: Type of report ("topic" or "session")
            topic: Topic for topic reports (optional)
            previous_sections: Previously written sections for context

        Returns:
            Section content in Markdown format
        """

        # Build the section writing prompt
        section_title = section.get("title", "Unknown Section")
        section_description = section.get("description", "No description provided")

        # Context from previous sections
        prev_context = ""
        if previous_sections:
            prev_context = "\n\n**Previous Sections (for context):**\n" + "\n".join(
                [f"- {prev[:100]}..." for prev in previous_sections[-2:]]
            )

        prompt = f"""**PHASE 2: ITERATIVE WRITING**

**Report Type:** {report_type.title()} Report
**Section to Write:** {section_title}
**Section Description:** {section_description}

{prev_context}

Write comprehensive, standalone content for the section "{section_title}". This section should be:
- Complete and readable on its own
- Objective and analytical in tone
- Well-organized with clear subsections if needed
- Thorough enough for full understanding

Use Markdown formatting and ensure no key points from the context are missed."""

        # Use the new context-aware message formatting
        messages = self.format_messages_with_context(
            state=state, prompt=prompt, report_type=report_type, topic=topic
        )

        # Generate response using the formatted messages
        content = self.llm.invoke(messages).content

        # Handle empty responses with a meaningful fallback
        if (
            not content
            or not content.strip()
            or "[Agent" in content
            and "empty response" in content
        ):
            logger.warning(
                f"Empty or fallback response for section '{section_title}', providing default content"
            )
            content = f"""# {section_title}

*[This section could not be generated due to an LLM response issue. Please regenerate the report or check the system logs for details.]*

**Section Purpose**: {section_description}

**Report Type**: {report_type.title()} Report

**Available Context**: Context provided via new context isolation system.
"""

        logger.info(
            f"Generated {report_type} section '{section_title}' ({len(content)} chars) using context system"
        )

        return content

    def write_section(
        self,
        section: Dict[str, str],
        source_material: Dict[str, Any],
        report_type: str = "topic",
        previous_sections: Optional[List[str]] = None,
    ) -> str:
        """Write content for a specific report section (Phase 2).

        Args:
            section: Section dict with title and description
            source_material: Original source material
            report_type: Type of report ("topic" or "session")
            previous_sections: Previously written sections for context

        Returns:
            Section content in Markdown format
        """
        if report_type == "topic":
            return self._write_topic_section(
                section, source_material, previous_sections
            )
        elif report_type == "session":
            return self._write_session_section(
                section, source_material, previous_sections
            )
        else:
            raise ValueError(f"Unknown report type: {report_type}")

    def _write_topic_section(
        self,
        section: Dict[str, str],
        source_material: Dict[str, Any],
        previous_sections: Optional[List[str]] = None,
    ) -> str:
        """Write a section for topic report."""
        topic = source_material.get("topic", "Unknown Topic")
        theme = source_material.get("theme", "")
        round_summaries = source_material.get("round_summaries", [])
        final_considerations = source_material.get("final_considerations", [])

        # Format source material
        summaries_text = "\n\n".join(
            [
                f"Round {i+1} Summary:\n{summary}"
                for i, summary in enumerate(round_summaries)
            ]
        )

        considerations_text = ""
        if final_considerations:
            considerations_text = "\n\n**Final Considerations:**\n" + "\n".join(
                [f"- {consideration}" for consideration in final_considerations]
            )

        # Context from previous sections
        prev_context = ""
        if previous_sections:
            prev_context = "\n\n**Previous Sections (for context):**\n" + "\n".join(
                [f"- {prev[:100]}..." for prev in previous_sections[-2:]]
            )

        prompt = f"""**PHASE 2: ITERATIVE WRITING**

**Report Type:** Topic Report
**Section to Write:** {section['title']}
**Section Description:** {section['description']}

**Discussion Theme:** {theme}
**Topic:** {topic}

**Source Material:**
{summaries_text}{considerations_text}{prev_context}

Write comprehensive, standalone content for the section "{section['title']}". This section should be:
- Complete and readable on its own
- Objective and analytical in tone
- Well-organized with clear subsections if needed
- Thorough enough for full understanding

Use Markdown formatting and ensure no key points from the source material are missed."""

        content = self.generate_response(prompt)

        # Handle empty responses with a meaningful fallback
        if (
            not content
            or not content.strip()
            or "[Agent" in content
            and "empty response" in content
        ):
            logger.warning(
                f"Empty or fallback response for section '{section['title']}', providing default content"
            )
            content = f"""# {section['title']}

*[This section could not be generated due to an LLM response issue. Please regenerate the report or check the system logs for details.]*

**Section Purpose**: {section['description']}

**Topic**: {source_material.get('topic', 'Unknown Topic')}

**Available Data**: {len(source_material.get('round_summaries', []))} round summaries, {len(source_material.get('final_considerations', []))} final considerations.
"""

        logger.info(
            f"Generated topic section '{section['title']}' ({len(content)} chars)"
        )

        return content

    def _write_session_section(
        self,
        section: Dict[str, str],
        source_material: Dict[str, Any],
        previous_sections: Optional[List[str]] = None,
    ) -> str:
        """Write a section for session report."""
        theme = source_material.get("theme", "")
        topic_reports = source_material.get("topic_reports", {})

        # Format topic reports
        reports_text = "\n\n".join(
            [
                f"**Topic: {topic}**\n---\n{report}\n---"
                for topic, report in topic_reports.items()
            ]
        )

        # Context from previous sections
        prev_context = ""
        if previous_sections:
            prev_context = "\n\n**Previous Sections (for context):**\n" + "\n".join(
                [f"- {prev[:100]}..." for prev in previous_sections[-2:]]
            )

        prompt = f"""**PHASE 2: ITERATIVE WRITING**

**Report Type:** Session Report
**Section to Write:** {section['title']}
**Section Description:** {section['description']}

**Discussion Theme:** {theme}

**All Topic Reports:**
{reports_text}{prev_context}

Write comprehensive, standalone content for the section "{section['title']}". This section should:
- Synthesize information across all topics
- Identify patterns and connections
- Provide high-level insights suitable for stakeholders
- Be complete and readable on its own

Use Markdown formatting and focus on cross-topic analysis and synthesis."""

        content = self.generate_response(prompt)

        # Handle empty responses with a meaningful fallback
        if (
            not content
            or not content.strip()
            or "[Agent" in content
            and "empty response" in content
        ):
            logger.warning(
                f"Empty or fallback response for session section '{section['title']}', providing default content"
            )
            content = f"""# {section['title']}

*[This section could not be generated due to an LLM response issue. Please regenerate the report or check the system logs for details.]*

**Section Purpose**: {section['description']}

**Discussion Theme**: {source_material.get('theme', 'Unknown Theme')}

**Available Topics**: {len(source_material.get('topic_reports', {}))} topic reports available for analysis.
"""

        logger.info(
            f"Generated session section '{section['title']}' ({len(content)} chars)"
        )

        return content

    # Legacy compatibility methods (migrated from TopicReportAgent and EcclesiaReportAgent)

    def synthesize_topic(
        self,
        round_summaries: List[str],
        final_considerations: List[str],
        topic: str,
        discussion_theme: str,
    ) -> str:
        """Create comprehensive report for concluded topic (legacy compatibility).

        Args:
            round_summaries: All round summaries for this topic
            final_considerations: Final thoughts from agents
            topic: The concluded topic
            discussion_theme: Overall session theme

        Returns:
            Comprehensive topic report
        """
        # Use new iterative approach
        source_material = {
            "topic": topic,
            "theme": discussion_theme,
            "round_summaries": round_summaries,
            "final_considerations": final_considerations,
        }

        # Create structure
        sections, _ = self.create_report_structure(source_material, "topic")

        # Write all sections
        report_parts = []
        previous_sections = []

        for section in sections:
            section_content = self.write_section(
                section, source_material, "topic", previous_sections
            )
            report_parts.append(section_content)
            previous_sections.append(section_content)

        report = "\n\n".join(report_parts)

        logger.info(
            f"Generated complete topic report for '{topic}' with {len(sections)} sections"
        )

        return report

    def generate_report_structure(
        self, topic_reports: Dict[str, str], discussion_theme: str
    ) -> List[str]:
        """Define the structure for the final report (legacy compatibility).

        Args:
            topic_reports: All topic reports keyed by topic
            discussion_theme: Overall session theme

        Returns:
            List of section titles for the report
        """
        source_material = {
            "theme": discussion_theme,
            "topic_reports": topic_reports,
        }

        sections, _ = self.create_report_structure(source_material, "session")
        return [section["title"] for section in sections]

    def write_section_legacy(
        self,
        section_title: str,
        topic_reports: Dict[str, str],
        discussion_theme: str,
        previous_sections: Dict[str, str],
    ) -> str:
        """Write content for a specific report section (legacy compatibility).

        Args:
            section_title: Title of section to write
            topic_reports: All topic reports
            discussion_theme: Overall theme
            previous_sections: Already written sections

        Returns:
            Section content
        """
        # Convert to new format
        section = {
            "title": section_title,
            "description": f"Content for {section_title}",
        }
        source_material = {
            "theme": discussion_theme,
            "topic_reports": topic_reports,
        }

        prev_sections_list = (
            list(previous_sections.values()) if previous_sections else None
        )

        return self.write_section(
            section, source_material, "session", prev_sections_list
        )
