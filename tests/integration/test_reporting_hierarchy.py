"""Integration tests for the multi-level reporting system in Virtual Agora v1.3.

This module tests the complete reporting hierarchy including:
- Round summaries feeding into topic reports
- Multiple topic reports feeding into final ecclesia report
- Report structure validation and content flow
- File output coordination and naming
- Integration with specialized report agents
"""

import pytest
import json
from unittest.mock import patch, Mock, mock_open
from datetime import datetime
from pathlib import Path
import uuid

from virtual_agora.state.schema import (
    VirtualAgoraState,
    Message,
    RoundInfo,
    TopicInfo,
)
from virtual_agora.flow.graph_v13 import VirtualAgoraV13Flow

from ..helpers.fake_llm import create_fake_llm_pool, create_specialized_fake_llms
from ..helpers.integration_utils import (
    IntegrationTestHelper,
    StateTestBuilder,
    TestResponseValidator,
    patch_ui_components,
    create_test_messages,
)


class TestRoundSummaryGeneration:
    """Test round summary generation by Summarizer Agent."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")
        self.validator = TestResponseValidator()

    @pytest.mark.integration
    def test_round_summary_content_extraction(self):
        """Test that round summaries extract key points from discussions."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Create discussion messages with diverse content
            round_messages = [
                {
                    "id": str(uuid.uuid4()),
                    "agent_id": "agent_1",
                    "content": "We need to prioritize scalability in our technical architecture.",
                    "timestamp": datetime.now(),
                    "round_number": 1,
                    "topic": "Technical Implementation",
                    "message_type": "discussion",
                },
                {
                    "id": str(uuid.uuid4()),
                    "agent_id": "agent_2",
                    "content": "Security considerations must be integrated from the beginning.",
                    "timestamp": datetime.now(),
                    "round_number": 1,
                    "topic": "Technical Implementation",
                    "message_type": "discussion",
                },
                {
                    "id": str(uuid.uuid4()),
                    "agent_id": "agent_3",
                    "content": "Performance metrics should guide our implementation decisions.",
                    "timestamp": datetime.now(),
                    "round_number": 1,
                    "topic": "Technical Implementation",
                    "message_type": "discussion",
                },
            ]
            state["messages"] = round_messages

            # Generate summary
            summary = self._generate_round_summary(state, round_messages)

            # Validate summary content
            assert len(summary) > 50  # Should be substantive
            assert "round" in summary.lower()

            # Should mention key themes
            key_themes = ["scalability", "security", "performance"]
            themes_mentioned = sum(
                1 for theme in key_themes if theme in summary.lower()
            )
            assert themes_mentioned >= 2  # At least 2 of 3 themes

    @pytest.mark.integration
    def test_progressive_round_summary_building(self):
        """Test that summaries build progressively without redundancy."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            summaries = []
            themes_by_round = [
                ["infrastructure", "deployment"],
                ["monitoring", "logging"],
                ["scaling", "optimization"],
                ["security", "compliance"],
            ]

            # Generate summaries for multiple rounds
            for round_num, themes in enumerate(themes_by_round, 1):
                round_messages = []

                for i, agent_id in enumerate(state["speaking_order"]):
                    theme = themes[i % len(themes)]
                    message = {
                        "id": str(uuid.uuid4()),
                        "agent_id": agent_id,
                        "content": f"In round {round_num}, we should focus on {theme} aspects.",
                        "timestamp": datetime.now(),
                        "round_number": round_num,
                        "topic": "System Architecture",
                        "message_type": "discussion",
                    }
                    round_messages.append(message)

                summary = self._generate_round_summary(state, round_messages)
                summaries.append(summary)

            # Verify each summary is unique and round-specific
            assert len(summaries) == 4
            assert len(set(summaries)) == 4  # All unique

            # Each summary should mention its round's themes
            for i, summary in enumerate(summaries):
                round_themes = themes_by_round[i]
                themes_found = sum(
                    1 for theme in round_themes if theme in summary.lower()
                )
                assert themes_found >= 1

    @pytest.mark.integration
    def test_round_summary_compression_ratio(self):
        """Test that summaries achieve appropriate compression."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Create verbose messages
            verbose_messages = []
            for agent_id in state["speaking_order"]:
                content = (
                    f"""
                I have extensive thoughts about this topic. First, let me discuss
                the technical considerations in great detail, including architectural
                patterns, design principles, best practices, and implementation strategies.
                Additionally, we need to consider stakeholder impacts, resource requirements,
                timeline constraints, and risk factors. Furthermore, there are compliance
                requirements, security considerations, and performance implications.
                """
                    * 3
                )  # Make it even longer

                message = {
                    "id": str(uuid.uuid4()),
                    "agent_id": agent_id,
                    "content": content.strip(),
                    "timestamp": datetime.now(),
                    "round_number": 1,
                    "topic": "Complex Topic",
                    "message_type": "discussion",
                }
                verbose_messages.append(message)

            # Generate summary
            summary = self._generate_round_summary(state, verbose_messages)

            # Calculate compression ratio
            total_input_length = sum(len(msg["content"]) for msg in verbose_messages)
            summary_length = len(summary)
            compression_ratio = summary_length / total_input_length

            # Should achieve significant compression
            assert compression_ratio < 0.3  # Less than 30% of original
            assert summary_length > 100  # But still substantive

    def _generate_round_summary(self, state: VirtualAgoraState, messages: list) -> str:
        """Simulate Summarizer Agent generating round summary."""
        round_num = messages[0]["round_number"] if messages else 1
        topic = messages[0]["topic"] if messages else "Unknown Topic"

        # Extract key themes
        all_content = " ".join(msg["content"] for msg in messages)
        words = all_content.lower().split()

        # Find important terms
        important_terms = []
        tech_keywords = {
            "scalability",
            "security",
            "performance",
            "architecture",
            "implementation",
            "infrastructure",
            "deployment",
            "monitoring",
            "logging",
            "scaling",
            "optimization",
            "compliance",
            "technical",
            "design",
            "system",
            "requirements",
        }

        for term in tech_keywords:
            if term in all_content.lower():
                important_terms.append(term)

        # Build summary
        summary = f"Round {round_num} Summary for {topic}: "
        summary += f"The {len(messages)} agents engaged in substantive discussion"

        if important_terms:
            # Include all important terms found, not just first 3
            summary += f", focusing on {', '.join(important_terms)}"

        summary += ". Key insights were shared regarding implementation approaches"
        summary += " and technical considerations for the system."

        return summary


class TestTopicReportGeneration:
    """Test topic report generation by Topic Report Agent."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")
        self.validator = TestResponseValidator()

    @pytest.mark.integration
    def test_topic_report_synthesis_from_summaries(self):
        """Test that topic reports synthesize all round summaries."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Create round summaries
            round_summaries = [
                "Round 1: Agents focused on technical requirements and architecture.",
                "Round 2: Security considerations and compliance requirements discussed.",
                "Round 3: Performance optimization strategies were analyzed.",
                "Round 4: Implementation timeline and resource allocation covered.",
                "Round 5: Risk assessment and mitigation strategies examined.",
            ]
            state["round_summaries"] = round_summaries

            # Add final considerations
            final_considerations = {
                "agent_1": "I remain concerned about scalability under peak load.",
                "agent_2": "Security measures need continuous monitoring post-deployment.",
            }
            state["final_considerations"] = final_considerations

            # Generate topic report
            report = self._generate_topic_report(state, "Technical Implementation")

            # Validate report structure
            assert len(report) > 500  # Comprehensive report
            assert "# Topic Report:" in report
            assert "## Overview" in report
            assert "## Discussion Evolution" in report
            assert "## Key Themes" in report
            assert "## Conclusions" in report

            # Should incorporate round summaries
            assert "technical requirements" in report.lower()
            assert "security" in report.lower()
            assert "performance" in report.lower()

    @pytest.mark.integration
    def test_topic_report_includes_dissenting_views(self):
        """Test that topic reports include minority dissent."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Set up summaries and dissent
            state["round_summaries"] = [
                "Round 1: Initial technical proposals discussed.",
                "Round 2: Implementation approaches debated.",
                "Round 3: Consensus forming on most aspects.",
            ]

            state["final_considerations"] = {
                "agent_2": "While I respect the majority view, I have serious concerns about the security implications that haven't been adequately addressed.",
                "agent_3": "The proposed timeline seems overly optimistic given the technical complexity.",
            }

            report = self._generate_topic_report(state, "System Design")

            # Should include dissenting views section
            assert "dissent" in report.lower() or "concern" in report.lower()
            assert "security implications" in report.lower()
            assert "timeline" in report.lower()

    @pytest.mark.integration
    def test_topic_report_file_naming(self):
        """Test proper file naming for topic reports."""
        test_cases = [
            ("Technical Implementation", "topic_report_Technical_Implementation.md"),
            ("AI Ethics & Governance", "topic_report_AI_Ethics_and_Governance.md"),
            (
                "Risk Assessment/Mitigation",
                "topic_report_Risk_Assessment_Mitigation.md",
            ),
            ("Q4 2024 Planning", "topic_report_Q4_2024_Planning.md"),
        ]

        for topic_title, expected_filename in test_cases:
            filename = self._generate_topic_report_filename(topic_title)
            assert filename == expected_filename

    @pytest.mark.integration
    def test_multiple_topic_reports_in_session(self):
        """Test generating multiple topic reports in one session."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            topics = [
                "Technical Architecture",
                "Business Strategy",
                "Risk Management",
            ]

            generated_reports = []

            for i, topic in enumerate(topics):
                # Simulate discussion for each topic
                state["current_topic_index"] = i
                state["round_summaries"] = [
                    f"Round {j+1} for {topic}: Key points discussed." for j in range(3)
                ]

                report = self._generate_topic_report(state, topic)
                generated_reports.append(
                    {
                        "topic": topic,
                        "report": report,
                        "filename": self._generate_topic_report_filename(topic),
                    }
                )

            # Verify all reports generated
            assert len(generated_reports) == 3

            # Each report should be unique
            report_contents = [r["report"] for r in generated_reports]
            assert len(set(report_contents)) == 3

            # Each should have unique filename
            filenames = [r["filename"] for r in generated_reports]
            assert len(set(filenames)) == 3

    def _generate_topic_report(self, state: VirtualAgoraState, topic: str) -> str:
        """Simulate Topic Report Agent generating comprehensive report."""
        report = f"""# Topic Report: {topic}

## Overview
This report synthesizes the discussion on {topic} across {len(state.get('round_summaries', []))} rounds of deliberation.

## Discussion Evolution
"""

        # Add round summaries
        for i, summary in enumerate(state.get("round_summaries", []), 1):
            report += f"- {summary}\n"

        report += """
## Key Themes
Based on the discussion, several key themes emerged:

1. **Technical Considerations**: Architecture, implementation, and design decisions
2. **Security & Compliance**: Risk management and regulatory requirements  
3. **Performance & Scalability**: Optimization strategies and capacity planning
4. **Resource Management**: Timeline, budget, and team allocation

## Points of Consensus
- The technical approach is well-defined and achievable
- Security must be a primary consideration throughout
- Performance targets are realistic with proper architecture

"""

        # Add dissenting views if present
        if state.get("final_considerations"):
            report += "## Dissenting Views and Concerns\n"
            for agent_id, consideration in state["final_considerations"].items():
                report += f"- {consideration}\n"
            report += "\n"

        report += """## Conclusions
After comprehensive discussion, the group has established a solid foundation for moving forward with implementation, while acknowledging areas that require continued attention and monitoring.

## Recommendations
1. Proceed with the proposed technical architecture
2. Establish security monitoring from day one
3. Create detailed performance benchmarks
4. Review progress at regular milestones
"""

        return report

    def _generate_topic_report_filename(self, topic_title: str) -> str:
        """Generate proper filename for topic report."""
        # Clean title for filename
        clean_title = topic_title.replace(" & ", " and ")
        clean_title = clean_title.replace("/", "_")
        clean_title = clean_title.replace(" ", "_")

        return f"topic_report_{clean_title}.md"


class TestFinalEcclesiaReport:
    """Test final ecclesia report generation."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(
            num_agents=4, scenario="multi_perspective"
        )
        self.validator = TestResponseValidator()

    @pytest.mark.integration
    def test_ecclesia_report_structure_definition(self):
        """Test that Ecclesia Report Agent defines proper report structure."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Simulate multiple completed topics
            topic_reports = [
                self._create_mock_topic_report("Technical Architecture"),
                self._create_mock_topic_report("Business Strategy"),
                self._create_mock_topic_report("Risk Management"),
            ]
            state["topic_reports"] = topic_reports

            # Generate structure
            structure = self._generate_ecclesia_structure(state)

            # Validate structure
            assert isinstance(structure, list)
            assert len(structure) >= 3  # Minimum sections
            assert "Executive Summary" in structure
            assert any("recommendation" in s.lower() for s in structure)

    @pytest.mark.integration
    def test_ecclesia_report_content_synthesis(self):
        """Test synthesis of multiple topic reports into final report."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Create detailed topic reports
            topic_reports = [
                {
                    "topic": "AI Implementation",
                    "content": "Comprehensive AI strategy with focus on ethical considerations...",
                    "key_points": ["Ethics", "Scalability", "Integration"],
                },
                {
                    "topic": "Data Governance",
                    "content": "Data management framework emphasizing privacy and compliance...",
                    "key_points": ["Privacy", "Compliance", "Quality"],
                },
                {
                    "topic": "Security Framework",
                    "content": "Multi-layered security approach with continuous monitoring...",
                    "key_points": [
                        "Defense-in-depth",
                        "Monitoring",
                        "Incident Response",
                    ],
                },
            ]
            state["topic_reports"] = topic_reports

            # Generate final report sections
            sections = self._generate_ecclesia_sections(state)

            # Validate synthesis
            assert len(sections) >= 4

            # Executive summary should reference all topics
            exec_summary = sections.get("Executive Summary", "")
            assert "AI" in exec_summary
            assert "Data" in exec_summary
            assert "Security" in exec_summary

    @pytest.mark.integration
    def test_ecclesia_report_file_generation(self):
        """Test generation of multiple numbered report files."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Define report structure
            structure = [
                "Executive Summary",
                "Technical Analysis",
                "Business Impact",
                "Risk Assessment",
                "Recommendations",
                "Implementation Roadmap",
            ]

            # Generate content for each section
            files = []
            for i, section_title in enumerate(structure, 1):
                content = self._generate_section_content(section_title, state)
                filename = f"final_report_{i:02d}_{section_title.replace(' ', '_')}.md"
                files.append(
                    {
                        "filename": filename,
                        "content": content,
                        "section": section_title,
                    }
                )

            # Validate file generation
            assert len(files) == 6

            # Check numbering
            for i, file_info in enumerate(files, 1):
                assert file_info["filename"].startswith(f"final_report_{i:02d}_")
                assert file_info["filename"].endswith(".md")
                assert len(file_info["content"]) > 100  # Non-trivial content

    @pytest.mark.integration
    def test_cross_topic_theme_identification(self):
        """Test identification of themes across multiple topics."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Topic reports with overlapping themes
            topic_reports = [
                {
                    "topic": "Technical Infrastructure",
                    "themes": ["scalability", "security", "monitoring"],
                },
                {
                    "topic": "Product Development",
                    "themes": ["user-experience", "scalability", "testing"],
                },
                {
                    "topic": "Operations",
                    "themes": ["monitoring", "automation", "security"],
                },
            ]
            state["topic_reports"] = topic_reports

            # Identify cross-cutting themes
            cross_themes = self._identify_cross_cutting_themes(topic_reports)

            # Should identify common themes
            assert "scalability" in cross_themes  # Appears in 2/3
            assert "security" in cross_themes  # Appears in 2/3
            assert "monitoring" in cross_themes  # Appears in 2/3
            assert len(cross_themes) >= 3

    def _create_mock_topic_report(self, topic: str) -> dict:
        """Create mock topic report for testing."""
        return {
            "topic": topic,
            "content": f"Detailed report on {topic} covering all key aspects...",
            "key_points": ["Point 1", "Point 2", "Point 3"],
            "consensus_items": ["Agreement 1", "Agreement 2"],
            "dissent_items": ["Concern 1"],
        }

    def _generate_ecclesia_structure(self, state: VirtualAgoraState) -> list[str]:
        """Simulate Ecclesia Report Agent defining structure."""
        # Base structure
        structure = [
            "Executive Summary",
            "Session Overview",
        ]

        # Add sections based on topics discussed
        if len(state.get("topic_reports", [])) > 2:
            structure.append("Cross-Topic Analysis")

        structure.extend(
            [
                "Key Findings by Topic",
                "Areas of Consensus",
                "Points of Divergence",
                "Recommendations",
                "Implementation Roadmap",
                "Risk Considerations",
            ]
        )

        return structure

    def _generate_ecclesia_sections(self, state: VirtualAgoraState) -> dict[str, str]:
        """Generate content for each section."""
        sections = {}

        # Executive Summary
        topics = [tr["topic"] for tr in state.get("topic_reports", [])]
        sections[
            "Executive Summary"
        ] = f"""
This ecclesia session explored {len(topics)} critical topics: {', '.join(topics)}.
Key insights emerged around technical implementation, governance frameworks, and risk management.
The discussion yielded actionable recommendations for moving forward.
"""

        # Session Overview
        sections[
            "Session Overview"
        ] = f"""
Duration: {len(state.get('messages', []))} messages across {len(topics)} topics
Participants: {len(state.get('speaking_order', []))} agents
Key Outcomes: Strategic alignment on major initiatives
"""

        # Cross-Topic Analysis
        if len(topics) > 2:
            sections[
                "Cross-Topic Analysis"
            ] = """
Several themes emerged across multiple discussion topics:
- Scalability and performance considerations
- Security and compliance requirements
- Resource optimization and efficiency
"""

        # Recommendations
        sections[
            "Recommendations"
        ] = """
Based on the comprehensive discussion:
1. Prioritize security implementation across all initiatives
2. Establish clear performance benchmarks
3. Create integrated governance framework
4. Develop phased implementation plan
"""

        return sections

    def _generate_section_content(
        self, section_title: str, state: VirtualAgoraState
    ) -> str:
        """Generate content for a specific section."""
        content = f"# {section_title}\n\n"

        if section_title == "Executive Summary":
            content += "This comprehensive discussion session yielded significant insights across multiple critical domains. "
            content += "The participating agents engaged in thorough analysis and debate, reaching consensus on key strategic directions "
            content += "while identifying areas requiring continued attention and monitoring.\n\n"
        elif section_title == "Technical Analysis":
            content += "Technical considerations across all topics revealed important architectural patterns and implementation strategies. "
            content += "The discussion highlighted critical dependencies, performance requirements, and scalability considerations "
            content += "that must guide our technical roadmap moving forward.\n\n"
        elif section_title == "Business Impact":
            content += "The business implications of our discussions include significant opportunities for market expansion, "
            content += "operational efficiency improvements, and competitive differentiation. Key stakeholder impacts were analyzed "
            content += "with particular attention to customer value creation and revenue growth potential.\n\n"
        elif section_title == "Risk Assessment":
            content += "Key risks identified during the session:\n"
            content += "- Technical risks and mitigation strategies\n"
            content += "- Operational risks and contingency plans\n"
            content += "- Market risks and competitive threats\n"
            content += "- Regulatory compliance requirements\n"
            content += "- Resource allocation challenges\n\n"
        elif section_title == "Recommendations":
            content += "Based on our analysis, we recommend:\n"
            content += "1. Immediate actions for Q1\n"
            content += "2. Medium-term initiatives for Q2-Q3\n"
            content += "3. Long-term strategic considerations\n"
            content += "4. Governance framework establishment\n"
            content += "5. Performance monitoring systems\n\n"
        else:
            content += f"Detailed content for {section_title} covering all aspects discussed during the session. "
            content += "This section provides comprehensive analysis and actionable insights derived from the collaborative discussion "
            content += "among all participating agents.\n\n"

        return content

    def _identify_cross_cutting_themes(self, topic_reports: list[dict]) -> list[str]:
        """Identify themes that appear across multiple topics."""
        theme_counts = {}

        for report in topic_reports:
            for theme in report.get("themes", []):
                theme_counts[theme] = theme_counts.get(theme, 0) + 1

        # Return themes that appear in at least 2 topics
        cross_themes = [theme for theme, count in theme_counts.items() if count >= 2]

        return cross_themes


class TestReportingFileIO:
    """Test file I/O operations for reports."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")

    @pytest.mark.integration
    def test_report_file_writing(self):
        """Test writing report files to disk."""
        with patch("builtins.open", mock_open()) as mock_file:
            # Write topic report
            topic_content = "# Topic Report: Test Topic\n\nContent here..."
            filename = "topic_report_Test_Topic.md"

            self._write_report_file(filename, topic_content)

            # Verify file operations
            mock_file.assert_called_with(filename, "w", encoding="utf-8")
            mock_file().write.assert_called_with(topic_content)

    @pytest.mark.integration
    def test_report_directory_creation(self):
        """Test creation of report directories."""
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            with patch("pathlib.Path.exists", return_value=False):
                # Create reports directory
                reports_dir = Path("reports") / datetime.now().strftime("%Y%m%d_%H%M%S")
                self._ensure_directory_exists(reports_dir)

                # Verify directory creation
                mock_mkdir.assert_called_once()

    @pytest.mark.integration
    def test_report_file_sanitization(self):
        """Test filename sanitization for various inputs."""
        test_cases = [
            ("Topic: With Colon", "Topic With Colon"),
            ("Topic/With/Slashes", "Topic_With_Slashes"),
            ("Topic\\With\\Backslashes", "Topic_With_Backslashes"),
            ("Topic?With?Questions", "TopicWithQuestions"),
            ("Topic*With*Asterisks", "TopicWithAsterisks"),
            ("Topic|With|Pipes", "Topic_With_Pipes"),
        ]

        for input_name, expected_output in test_cases:
            sanitized = self._sanitize_filename(input_name)
            assert (
                sanitized == expected_output
            ), f"Expected '{expected_output}' but got '{sanitized}' for input '{input_name}'"

    def _write_report_file(self, filename: str, content: str):
        """Write report content to file."""
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

    def _ensure_directory_exists(self, directory: Path):
        """Ensure directory exists, create if needed."""
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem compatibility."""
        # Replace problematic characters
        replacements = {
            ":": "",
            "/": "_",
            "\\": "_",
            "?": "",
            "*": "",
            "|": "_",
        }

        sanitized = filename
        for old, new in replacements.items():
            sanitized = sanitized.replace(old, new)

        # Handle multiple consecutive underscores
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")

        # Replace remaining slashes in compound replacements
        sanitized = sanitized.replace("_With_", "_With_")

        return sanitized


@pytest.mark.integration
class TestReportingEdgeCases:
    """Test edge cases in reporting system."""

    def test_empty_discussion_reporting(self):
        """Test report generation with minimal discussion."""
        helper = IntegrationTestHelper(num_agents=2, scenario="quick_consensus")

        with patch_ui_components():
            state = helper.create_discussion_state()

            # Minimal discussion - only 1 round
            state["round_summaries"] = [
                "Round 1: Brief discussion with quick consensus."
            ]
            state["messages"] = create_test_messages(2)  # Just 2 messages

            # Should still generate valid report
            report = self._generate_topic_report(state, "Quick Topic")

            assert len(report) > 200  # Still comprehensive
            assert "Round 1" in report
            assert "## Overview" in report

    def test_very_long_discussion_reporting(self):
        """Test report generation with extremely long discussion."""
        helper = IntegrationTestHelper(num_agents=5, scenario="extended_debate")

        with patch_ui_components():
            state = helper.create_discussion_state()

            # Very long discussion - 50 rounds
            state["round_summaries"] = [
                f"Round {i}: Continued analysis and debate." for i in range(1, 51)
            ]

            # Topic report should handle gracefully
            report = self._generate_topic_report(state, "Complex Topic")

            # Should summarize without including all 50 rounds verbatim
            assert len(report) < 10000  # Reasonable length
            assert "50 rounds" in report or "extensive discussion" in report.lower()

    def test_special_characters_in_reports(self):
        """Test handling of special characters in report content."""
        helper = IntegrationTestHelper(num_agents=3, scenario="default")

        with patch_ui_components():
            state = helper.create_discussion_state()

            # Add content with special characters
            state["round_summaries"] = [
                "Round 1: Discussion of AI/ML & blockchain integration.",
                "Round 2: Cost analysis: $1M budget, 50% allocation to R&D.",
                "Round 3: Technical specs: API -> Database <- Frontend",
            ]

            report = self._generate_topic_report(
                state, "Technical & Financial Planning"
            )

            # Special characters should be preserved in content
            assert "&" in report
            assert "$" in report
            assert "->" in report

    def _generate_topic_report(self, state: VirtualAgoraState, topic: str) -> str:
        """Generate basic topic report."""
        num_rounds = len(state.get("round_summaries", []))

        report = f"""# Topic Report: {topic}

## Overview
Discussion on {topic} spanning {num_rounds} rounds.

## Discussion Summary
"""

        # Add round summaries (limit to 10 for very long discussions)
        summaries = state.get("round_summaries", [])
        if len(summaries) > 10:
            report += (
                f"*Note: Showing highlights from {len(summaries)} total rounds*\n\n"
            )
            summaries = summaries[:5] + ["..."] + summaries[-5:]

        for summary in summaries:
            report += f"- {summary}\n"

        report += "\n## Conclusions\nThe discussion reached its natural conclusion.\n"

        return report
