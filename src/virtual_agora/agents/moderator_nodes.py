"""LangGraph node functions for Moderator Agent operations.

This module provides LangGraph-compatible node functions that wrap
ModeratorAgent functionality for use in state-based workflows.

Each node function:
- Accepts VirtualAgoraState as input
- Uses the ModeratorAgent instance methods
- Returns state updates as dictionaries
- Handles errors appropriately with logging
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from ..state.schema import VirtualAgoraState, VoteRound, Message
from .moderator import ModeratorAgent
from ..reporting import (
    TopicSummaryGenerator,
    ReportStructureManager,
    ReportSectionWriter,
    ReportFileManager,
    ReportMetadataGenerator,
    ReportQualityValidator,
    ReportExporter,
    ReportTemplateManager,
    EnhancedSessionLogger,
)

logger = logging.getLogger(__name__)


class ModeratorNodes:
    """LangGraph node functions for moderator operations."""

    def __init__(
        self,
        moderator_agent: ModeratorAgent,
        reports_dir: Optional[Path] = None,
        summaries_dir: Optional[Path] = None,
        logs_dir: Optional[Path] = None,
    ):
        """Initialize with a ModeratorAgent instance.

        Args:
            moderator_agent: The ModeratorAgent instance to use
            reports_dir: Directory for report files (default: ./reports)
            summaries_dir: Directory for topic summaries (default: ./summaries)
            logs_dir: Directory for session logs (default: ./logs)
        """
        self.moderator = moderator_agent

        # Initialize reporting components
        self.reports_dir = reports_dir or Path("./reports")
        self.summaries_dir = summaries_dir or Path("./summaries")
        self.logs_dir = logs_dir or Path("./logs")

        # Create directories if they don't exist
        for dir_path in [self.reports_dir, self.summaries_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize reporting components
        self.topic_generator = TopicSummaryGenerator(self.summaries_dir)
        self.structure_manager = ReportStructureManager()
        self.section_writer = ReportSectionWriter()
        self.file_manager = ReportFileManager(self.reports_dir)
        self.metadata_generator = ReportMetadataGenerator()
        self.quality_validator = ReportQualityValidator()
        self.exporter = ReportExporter()
        self.template_manager = ReportTemplateManager()

        # Session logger will be initialized when session starts
        self.session_logger: Optional[EnhancedSessionLogger] = None

    def initialize_session_logging(self, session_id: str) -> None:
        """Initialize session logging for a new session.

        Args:
            session_id: Unique identifier for the session
        """
        if self.session_logger:
            self.session_logger.close()

        self.session_logger = EnhancedSessionLogger(session_id, self.logs_dir)
        logger.info(f"Initialized session logging for session {session_id}")

    # Story 3.7: Minority Considerations Management

    async def identify_minority_voters_node(
        self, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """LangGraph node to identify minority voters from the last vote.

        Analyzes the most recent vote round to identify agents who voted
        for the losing option (typically "No" in topic conclusion votes).

        Args:
            state: Current VirtualAgoraState

        Returns:
            Dict with minority_voters list and updated vote round
        """
        try:
            if not state.get("active_vote"):
                logger.warning("No active vote found for minority identification")
                return {
                    "warnings": ["No active vote found for minority identification"]
                }

            active_vote = state["active_vote"]

            # Get votes for this round
            round_votes = [
                vote
                for vote in state.get("votes", [])
                if vote["vote_type"] == active_vote["vote_type"]
                and vote["phase"] == active_vote["phase"]
            ]

            if not round_votes:
                logger.warning("No votes found for active voting round")
                return {"warnings": ["No votes found for active voting round"]}

            # Count votes by choice
            vote_counts = {}
            voter_choices = {}

            for vote in round_votes:
                choice = vote["choice"].lower()
                vote_counts[choice] = vote_counts.get(choice, 0) + 1
                voter_choices[vote["voter_id"]] = choice

            # Determine winning and losing choices
            if not vote_counts:
                return {"warnings": ["No valid votes to analyze"]}

            winning_choice = max(vote_counts.keys(), key=lambda k: vote_counts[k])

            # Identify minority voters (those who didn't vote for winning choice)
            minority_voters = [
                voter_id
                for voter_id, choice in voter_choices.items()
                if choice != winning_choice
            ]

            logger.info(
                f"Identified {len(minority_voters)} minority voters: {minority_voters}"
            )

            # Update the active vote with minority voter information
            updated_vote = active_vote.copy()
            updated_vote["minority_voters"] = minority_voters

            return {
                "active_vote": updated_vote,
                "warnings": (
                    []
                    if minority_voters
                    else [
                        "No minority voters identified - all voted for winning choice"
                    ]
                ),
            }

        except Exception as e:
            logger.error(f"Error identifying minority voters: {e}")
            return {
                "last_error": f"Failed to identify minority voters: {str(e)}",
                "error_count": state.get("error_count", 0) + 1,
            }

    async def collect_minority_considerations_node(
        self, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """LangGraph node to collect final considerations from minority voters.

        Prompts agents who voted against the majority for their final thoughts
        on the topic before it is closed.

        Args:
            state: Current VirtualAgoraState

        Returns:
            Dict with minority considerations and updated vote round
        """
        try:
            active_vote = state.get("active_vote")
            if not active_vote or not active_vote.get("minority_voters"):
                logger.info("No minority voters to collect considerations from")
                return {"warnings": ["No minority voters found"]}

            minority_voters = active_vote["minority_voters"]
            active_topic = state.get("active_topic")

            if not active_topic:
                logger.warning("No active topic for minority considerations")
                return {"warnings": ["No active topic found"]}

            # Collect considerations from each minority voter
            minority_considerations = []

            for voter_id in minority_voters:
                try:
                    # Use the existing ModeratorAgent method
                    consideration = await self.moderator.collect_minority_consideration(
                        voter_id, active_topic, state
                    )

                    if consideration:
                        minority_considerations.append(consideration)
                        logger.info(f"Collected consideration from {voter_id}")
                    else:
                        logger.warning(f"No consideration received from {voter_id}")

                except Exception as e:
                    logger.error(
                        f"Failed to collect consideration from {voter_id}: {e}"
                    )
                    minority_considerations.append(
                        f"[Error collecting from {voter_id}]"
                    )

            # Update the vote round with collected considerations
            updated_vote = active_vote.copy()
            updated_vote["minority_considerations"] = minority_considerations

            logger.info(
                f"Collected {len(minority_considerations)} minority considerations"
            )

            return {
                "active_vote": updated_vote,
                "warnings": (
                    []
                    if minority_considerations
                    else ["No minority considerations collected"]
                ),
            }

        except Exception as e:
            logger.error(f"Error collecting minority considerations: {e}")
            return {
                "last_error": f"Failed to collect minority considerations: {str(e)}",
                "error_count": state.get("error_count", 0) + 1,
            }

    async def incorporate_minority_views_node(
        self, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """LangGraph node to incorporate minority views into topic summary.

        Ensures that minority considerations are properly integrated into
        the final topic summary before the topic is marked as completed.

        Args:
            state: Current VirtualAgoraState

        Returns:
            Dict with updated topic summary including minority views
        """
        try:
            active_vote = state.get("active_vote")
            active_topic = state.get("active_topic")

            if not active_vote or not active_topic:
                logger.warning(
                    "Missing active vote or topic for minority view incorporation"
                )
                return {"warnings": ["Missing active vote or topic"]}

            minority_considerations = active_vote.get("minority_considerations", [])

            if not minority_considerations:
                logger.info("No minority considerations to incorporate")
                return {}

            # Get existing topic summary or create new one
            topic_summaries = state.get("topic_summaries", {})
            existing_summary = topic_summaries.get(active_topic, "")

            # Use ModeratorAgent to incorporate minority views
            updated_summary = await self.moderator.incorporate_minority_views(
                existing_summary, minority_considerations, active_topic
            )

            if updated_summary:
                topic_summaries[active_topic] = updated_summary
                logger.info(
                    f"Successfully incorporated minority views into summary for {active_topic}"
                )

                return {"topic_summaries": topic_summaries}
            else:
                logger.warning("Failed to generate updated summary with minority views")
                return {
                    "warnings": ["Failed to incorporate minority views into summary"]
                }

        except Exception as e:
            logger.error(f"Error incorporating minority views: {e}")
            return {
                "last_error": f"Failed to incorporate minority views: {str(e)}",
                "error_count": state.get("error_count", 0) + 1,
            }

    # Story 3.8: Report Writer Mode Implementation

    async def initialize_report_structure_node(
        self, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """LangGraph node to initialize report structure as 'The Writer'.

        Analyzes all topic summaries and defines a logical structure
        for the final report as an ordered list of section titles.

        Args:
            state: Current VirtualAgoraState

        Returns:
            Dict with report structure and updated generation status
        """
        try:
            topic_summaries = state.get("topic_summaries", {})

            if not topic_summaries:
                logger.warning("No topic summaries available for report structure")
                return {
                    "report_generation_status": "failed",
                    "warnings": ["No topic summaries available for report generation"],
                }

            # Use ModeratorAgent in Writer mode to define report structure
            report_structure = await self.moderator.define_report_structure(
                topic_summaries
            )

            if not report_structure:
                logger.error("Failed to generate report structure")
                return {
                    "report_generation_status": "failed",
                    "last_error": "Failed to generate report structure",
                }

            logger.info(
                f"Generated report structure with {len(report_structure)} sections"
            )

            return {
                "report_structure": report_structure,
                "report_sections": {},  # Initialize empty sections dict
                "report_generation_status": "structuring",
            }

        except Exception as e:
            logger.error(f"Error initializing report structure: {e}")
            return {
                "report_generation_status": "failed",
                "last_error": f"Failed to initialize report structure: {str(e)}",
                "error_count": state.get("error_count", 0) + 1,
            }

    async def generate_report_section_node(
        self, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """LangGraph node to generate content for a single report section.

        This node should be called iteratively for each section in the
        report structure. It generates professional content for one section.

        Args:
            state: Current VirtualAgoraState

        Returns:
            Dict with updated report sections and generation status
        """
        try:
            report_structure = state.get("report_structure", [])
            report_sections = state.get("report_sections", {})
            topic_summaries = state.get("topic_summaries", {})

            if not report_structure:
                logger.error("No report structure available for section generation")
                return {
                    "report_generation_status": "failed",
                    "last_error": "No report structure available",
                }

            # Find next section to generate
            next_section = None
            for section in report_structure:
                if section not in report_sections:
                    next_section = section
                    break

            if not next_section:
                # All sections completed
                logger.info("All report sections generated successfully")
                return {
                    "report_generation_status": "completed",
                    "final_report": "Report generation completed - sections ready for file output",
                }

            # Generate content for this section
            section_content = await self.moderator.generate_report_section(
                next_section, topic_summaries, report_structure
            )

            if not section_content:
                logger.error(f"Failed to generate content for section: {next_section}")
                return {
                    "report_generation_status": "failed",
                    "last_error": f"Failed to generate content for section: {next_section}",
                }

            # Update report sections
            updated_sections = report_sections.copy()
            updated_sections[next_section] = section_content

            logger.info(f"Generated content for section: {next_section}")

            return {
                "report_sections": updated_sections,
                "report_generation_status": "writing",
            }

        except Exception as e:
            logger.error(f"Error generating report section: {e}")
            return {
                "report_generation_status": "failed",
                "last_error": f"Failed to generate report section: {str(e)}",
                "error_count": state.get("error_count", 0) + 1,
            }

    async def finalize_report_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """LangGraph node to finalize the complete report.

        Combines all generated sections into a complete final report
        and marks the report generation as completed.

        Args:
            state: Current VirtualAgoraState

        Returns:
            Dict with finalized report and completion status
        """
        try:
            report_structure = state.get("report_structure", [])
            report_sections = state.get("report_sections", {})

            if not report_structure or not report_sections:
                logger.error("Missing report structure or sections for finalization")
                return {
                    "report_generation_status": "failed",
                    "last_error": "Missing report structure or sections",
                }

            # Verify all sections are generated
            missing_sections = [s for s in report_structure if s not in report_sections]
            if missing_sections:
                logger.error(f"Missing sections for finalization: {missing_sections}")
                return {
                    "report_generation_status": "failed",
                    "last_error": f"Missing sections: {missing_sections}",
                }

            # Create final report summary
            total_sections = len(report_structure)
            report_summary = f"Final report completed with {total_sections} sections: {', '.join(report_structure)}"

            logger.info("Report generation finalized successfully")

            return {
                "final_report": report_summary,
                "report_generation_status": "completed",
            }

        except Exception as e:
            logger.error(f"Error finalizing report: {e}")
            return {
                "report_generation_status": "failed",
                "last_error": f"Failed to finalize report: {str(e)}",
                "error_count": state.get("error_count", 0) + 1,
            }

    # Story 3.9: Agenda Modification Facilitation

    async def request_agenda_modifications_node(
        self, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """LangGraph node to request agenda modifications from agents.

        After a topic concludes, asks all agents if they want to add
        or remove topics from the remaining agenda based on insights.

        Args:
            state: Current VirtualAgoraState

        Returns:
            Dict with collected agenda modification suggestions
        """
        try:
            topic_queue = state.get("topic_queue", [])
            completed_topics = state.get("completed_topics", [])
            agents = state.get("agents", {})

            if not topic_queue:
                logger.info("No remaining topics - agenda modification not needed")
                return {"warnings": ["No remaining topics for modification"]}

            # Collect modification suggestions from all agents
            modification_suggestions = []

            for agent_id in agents.keys():
                if agent_id == state.get("moderator_id"):
                    continue  # Skip moderator

                try:
                    suggestion = await self.moderator.request_agenda_modification(
                        agent_id, topic_queue, completed_topics, state
                    )

                    if suggestion:
                        modification_suggestions.append(suggestion)
                        logger.info(f"Collected agenda modification from {agent_id}")

                except Exception as e:
                    logger.error(
                        f"Failed to collect agenda modification from {agent_id}: {e}"
                    )

            logger.info(
                f"Collected {len(modification_suggestions)} agenda modification suggestions"
            )

            return {"pending_agenda_modifications": modification_suggestions}

        except Exception as e:
            logger.error(f"Error requesting agenda modifications: {e}")
            return {
                "last_error": f"Failed to request agenda modifications: {str(e)}",
                "error_count": state.get("error_count", 0) + 1,
            }

    async def synthesize_agenda_changes_node(
        self, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """LangGraph node to synthesize agenda change proposals.

        Analyzes all modification suggestions and creates a new proposed
        agenda that incorporates the changes, ready for voting.

        Args:
            state: Current VirtualAgoraState

        Returns:
            Dict with synthesized agenda changes
        """
        try:
            modifications = state.get("pending_agenda_modifications", [])
            current_queue = state.get("topic_queue", [])

            if not modifications:
                logger.info("No agenda modifications to synthesize")
                return {"warnings": ["No agenda modifications received"]}

            # Use ModeratorAgent to synthesize changes
            synthesized_agenda = await self.moderator.synthesize_agenda_modifications(
                modifications, current_queue
            )

            if synthesized_agenda is None:
                logger.warning("Failed to synthesize agenda modifications")
                return {"warnings": ["Failed to synthesize agenda modifications"]}

            logger.info(f"Synthesized agenda with {len(synthesized_agenda)} topics")

            # Create new proposed topics list for re-voting
            return {
                "proposed_topics": synthesized_agenda,
                "pending_agenda_modifications": [],  # Clear pending modifications
            }

        except Exception as e:
            logger.error(f"Error synthesizing agenda changes: {e}")
            return {
                "last_error": f"Failed to synthesize agenda changes: {str(e)}",
                "error_count": state.get("error_count", 0) + 1,
            }

    async def facilitate_agenda_revote_node(
        self, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """LangGraph node to facilitate re-voting on modified agenda.

        Presents the modified agenda to agents for voting and synthesizes
        the results into a new ordered topic queue.

        Args:
            state: Current VirtualAgoraState

        Returns:
            Dict with updated topic queue from re-vote
        """
        try:
            proposed_topics = state.get("proposed_topics", [])
            agents = state.get("agents", {})

            if not proposed_topics:
                logger.warning("No proposed topics for re-voting")
                return {"warnings": ["No proposed topics for re-voting"]}

            # Collect votes from all agents
            agenda_votes = {}

            for agent_id in agents.keys():
                if agent_id == state.get("moderator_id"):
                    continue  # Skip moderator

                try:
                    vote = await self.moderator.collect_agenda_vote(
                        agent_id, proposed_topics, state
                    )

                    if vote:
                        agenda_votes[agent_id] = vote
                        logger.info(f"Collected agenda vote from {agent_id}")

                except Exception as e:
                    logger.error(f"Failed to collect agenda vote from {agent_id}: {e}")

            if not agenda_votes:
                logger.error("No agenda votes collected")
                return {
                    "last_error": "No agenda votes collected",
                    "error_count": state.get("error_count", 0) + 1,
                }

            # Synthesize votes into ordered agenda
            new_topic_queue = await self.moderator.synthesize_agenda_votes(
                agenda_votes, proposed_topics
            )

            if not new_topic_queue:
                logger.error("Failed to synthesize agenda votes")
                return {
                    "last_error": "Failed to synthesize agenda votes",
                    "error_count": state.get("error_count", 0) + 1,
                }

            logger.info(f"New agenda synthesized with {len(new_topic_queue)} topics")

            return {
                "topic_queue": new_topic_queue,
                "agenda_modification_votes": agenda_votes,
            }

        except Exception as e:
            logger.error(f"Error facilitating agenda re-vote: {e}")
            return {
                "last_error": f"Failed to facilitate agenda re-vote: {str(e)}",
                "error_count": state.get("error_count", 0) + 1,
            }

    # Epic 8: Reporting & Documentation - Integration Methods

    async def generate_topic_summary_with_reporting_node(
        self, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """Enhanced topic summary generation using Epic 8 reporting functionality.

        Generates topic summaries using both the moderator agent and the
        new TopicSummaryGenerator with proper file management.

        Args:
            state: Current VirtualAgoraState

        Returns:
            Dict with updated topic summaries and summary file paths
        """
        try:
            active_topic = state.get("active_topic")
            if not active_topic:
                logger.warning("No active topic for summary generation")
                return {"warnings": ["No active topic found"]}

            # Get all messages for this topic
            all_messages = state.get("messages", [])
            topic_messages = [
                msg for msg in all_messages if msg.get("topic") == active_topic
            ]

            if not topic_messages:
                logger.warning(f"No messages found for topic '{active_topic}'")
                return {"warnings": [f"No messages found for topic '{active_topic}'"]}

            # Generate summary using moderator agent
            summary_content = self.moderator.generate_topic_summary(
                active_topic, all_messages
            )

            # Use Epic 8 TopicSummaryGenerator to create structured summary file
            summary_path = self.topic_generator.generate_summary(
                active_topic, summary_content, state
            )

            # Log the summary generation
            if self.session_logger:
                self.session_logger.log_topic_event(
                    f"Generated summary for topic: {active_topic}",
                    active_topic,
                    metadata={
                        "summary_file": str(summary_path),
                        "message_count": len(topic_messages),
                    },
                )

            # Update state with both the summary content and file path
            topic_summaries = state.get("topic_summaries", {})
            topic_summaries[active_topic] = summary_content

            summary_files = state.get("topic_summary_files", {})
            summary_files[active_topic] = str(summary_path)

            logger.info(
                f"Generated structured topic summary for '{active_topic}' at {summary_path}"
            )

            return {
                "topic_summaries": topic_summaries,
                "topic_summary_files": summary_files,
            }

        except Exception as e:
            logger.error(f"Error generating topic summary with reporting: {e}")
            return {
                "last_error": f"Failed to generate topic summary: {str(e)}",
                "error_count": state.get("error_count", 0) + 1,
            }

    async def generate_session_report_node(
        self, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """Generate a complete session report using Epic 8 reporting functionality.

        Creates a comprehensive multi-file report with metadata, quality validation,
        and export capabilities.

        Args:
            state: Current VirtualAgoraState

        Returns:
            Dict with report generation results and file paths
        """
        try:
            topic_summaries = state.get("topic_summaries", {})
            session_id = state.get("session_id", "unknown-session")
            main_topic = state.get("main_topic", "Virtual Agora Discussion")

            if not topic_summaries:
                logger.warning("No topic summaries available for report generation")
                return {
                    "warnings": ["No topic summaries available for report generation"]
                }

            # Step 1: Define report structure
            report_structure = self.structure_manager.define_structure(
                topic_summaries, main_topic=main_topic
            )

            # Step 2: Set up section writer context
            self.section_writer.set_context(
                topic_summaries, report_structure, main_topic=main_topic
            )

            # Step 3: Create report directory
            report_dir = self.file_manager.create_report_directory(
                session_id, main_topic
            )

            # Step 4: Generate report sections
            generated_sections = []
            for i, section_title in enumerate(report_structure, 1):
                if section_title.startswith("  -"):
                    continue  # Skip subsections

                section_content = self.section_writer.write_section(section_title)
                section_path = self.file_manager.save_report_section(
                    i, section_title, section_content
                )
                generated_sections.append(
                    {"title": section_title, "path": str(section_path)}
                )

            # Step 5: Generate metadata
            metadata = self.metadata_generator.generate_metadata(state, topic_summaries)
            metadata_path = self.file_manager.create_metadata_file(metadata)

            # Step 6: Create supporting files
            toc_path = self.file_manager.create_table_of_contents(report_structure)
            manifest_path = self.file_manager.create_report_manifest()

            # Step 7: Validate report quality
            validation_results = self.quality_validator.validate_report(report_dir)

            # Step 8: Export in multiple formats
            export_results = {}
            try:
                exports = self.exporter.export_report(
                    report_dir, report_dir.parent / "exports", format="combined"
                )
                export_results = {
                    "markdown": str(exports.get("markdown", "")),
                    "html": str(exports.get("html", "")),
                    "archive": str(exports.get("archive", "")),
                }
            except Exception as e:
                logger.warning(f"Export failed: {e}")
                export_results = {"error": f"Export failed: {str(e)}"}

            # Log the report generation
            if self.session_logger:
                self.session_logger.log_system_event(
                    "Generated complete session report",
                    metadata={
                        "report_dir": str(report_dir),
                        "sections_count": len(generated_sections),
                        "validation_score": validation_results.get("overall_score", 0),
                        "exports": list(export_results.keys()),
                    },
                )

            logger.info(
                f"Generated complete session report at {report_dir} with "
                f"{len(generated_sections)} sections (score: {validation_results.get('overall_score', 0)})"
            )

            return {
                "report_directory": str(report_dir),
                "report_structure": report_structure,
                "generated_sections": generated_sections,
                "metadata_file": str(metadata_path),
                "table_of_contents": str(toc_path),
                "manifest_file": str(manifest_path),
                "validation_results": validation_results,
                "export_results": export_results,
                "report_generation_status": "completed",
            }

        except Exception as e:
            logger.error(f"Error generating session report: {e}")
            return {
                "report_generation_status": "failed",
                "last_error": f"Failed to generate session report: {str(e)}",
                "error_count": state.get("error_count", 0) + 1,
            }

    async def log_discussion_event_node(
        self, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """Log discussion events using Enhanced Session Logger.

        Logs various discussion events with proper categorization and metadata.

        Args:
            state: Current VirtualAgoraState

        Returns:
            Dict with logging status
        """
        try:
            if not self.session_logger:
                logger.warning("Session logger not initialized")
                return {"warnings": ["Session logger not initialized"]}

            # Log recent messages
            recent_messages = state.get("recent_messages", [])
            for msg in recent_messages:
                self.session_logger.log_agent_event(
                    f"Message from {msg.get('speaker_id', 'unknown')}",
                    msg.get("speaker_id", "unknown"),
                    msg.get("content", ""),
                    metadata={
                        "topic": msg.get("topic"),
                        "timestamp": msg.get("timestamp"),
                        "round": state.get("current_round"),
                    },
                )

            # Log voting events
            recent_votes = state.get("recent_votes", [])
            for vote in recent_votes:
                self.session_logger.log_vote_event(
                    f"Vote: {vote.get('choice')} for {vote.get('vote_type')}",
                    vote.get("voter_id", "unknown"),
                    vote.get("choice", "unknown"),
                    metadata={
                        "vote_type": vote.get("vote_type"),
                        "phase": vote.get("phase"),
                    },
                )

            # Log system events
            if state.get("active_topic"):
                self.session_logger.log_topic_event(
                    f"Topic active: {state['active_topic']}",
                    state["active_topic"],
                    metadata={"round": state.get("current_round")},
                )

            return {"logging_status": "completed"}

        except Exception as e:
            logger.error(f"Error logging discussion events: {e}")
            return {
                "logging_status": "failed",
                "last_error": f"Failed to log discussion events: {str(e)}",
                "error_count": state.get("error_count", 0) + 1,
            }

    async def finalize_session_reporting_node(
        self, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """Finalize all session reporting and close logging.

        Performs final reporting tasks and closes the session logger.

        Args:
            state: Current VirtualAgoraState

        Returns:
            Dict with finalization results
        """
        try:
            session_id = state.get("session_id", "unknown-session")

            # Get session analytics if logger is available
            analytics = {}
            if self.session_logger:
                analytics = self.session_logger.get_session_analytics()

                # Generate analytics summary
                summary = self.metadata_generator.generate_analytics_summary()
                if summary and summary != "No metadata available":
                    analytics["summary"] = summary

                # Close the session logger
                self.session_logger.close()
                self.session_logger = None
                logger.info(f"Closed session logging for session {session_id}")

            return {
                "session_analytics": analytics,
                "reporting_finalized": True,
                "session_closed": True,
            }

        except Exception as e:
            logger.error(f"Error finalizing session reporting: {e}")
            return {
                "reporting_finalized": False,
                "last_error": f"Failed to finalize session reporting: {str(e)}",
                "error_count": state.get("error_count", 0) + 1,
            }


# Convenience functions for creating node instances


def create_moderator_nodes(
    moderator_agent: ModeratorAgent,
    reports_dir: Optional[Path] = None,
    summaries_dir: Optional[Path] = None,
    logs_dir: Optional[Path] = None,
) -> ModeratorNodes:
    """Create a ModeratorNodes instance with the given moderator agent.

    Args:
        moderator_agent: The ModeratorAgent instance
        reports_dir: Directory for report files (optional)
        summaries_dir: Directory for topic summaries (optional)
        logs_dir: Directory for session logs (optional)

    Returns:
        ModeratorNodes instance ready for use in LangGraph with Epic 8 reporting
    """
    return ModeratorNodes(moderator_agent, reports_dir, summaries_dir, logs_dir)


# Node function aliases for direct LangGraph usage


async def identify_minority_voters(
    state: VirtualAgoraState, moderator_agent: ModeratorAgent
) -> Dict[str, Any]:
    """Direct node function for minority voter identification."""
    nodes = ModeratorNodes(moderator_agent)
    return await nodes.identify_minority_voters_node(state)


async def collect_minority_considerations(
    state: VirtualAgoraState, moderator_agent: ModeratorAgent
) -> Dict[str, Any]:
    """Direct node function for minority consideration collection."""
    nodes = ModeratorNodes(moderator_agent)
    return await nodes.collect_minority_considerations_node(state)


async def incorporate_minority_views(
    state: VirtualAgoraState, moderator_agent: ModeratorAgent
) -> Dict[str, Any]:
    """Direct node function for minority view incorporation."""
    nodes = ModeratorNodes(moderator_agent)
    return await nodes.incorporate_minority_views_node(state)


async def initialize_report_structure(
    state: VirtualAgoraState, moderator_agent: ModeratorAgent
) -> Dict[str, Any]:
    """Direct node function for report structure initialization."""
    nodes = ModeratorNodes(moderator_agent)
    return await nodes.initialize_report_structure_node(state)


async def generate_report_section(
    state: VirtualAgoraState, moderator_agent: ModeratorAgent
) -> Dict[str, Any]:
    """Direct node function for report section generation."""
    nodes = ModeratorNodes(moderator_agent)
    return await nodes.generate_report_section_node(state)


async def finalize_report(
    state: VirtualAgoraState, moderator_agent: ModeratorAgent
) -> Dict[str, Any]:
    """Direct node function for report finalization."""
    nodes = ModeratorNodes(moderator_agent)
    return await nodes.finalize_report_node(state)


async def request_agenda_modifications(
    state: VirtualAgoraState, moderator_agent: ModeratorAgent
) -> Dict[str, Any]:
    """Direct node function for agenda modification requests."""
    nodes = ModeratorNodes(moderator_agent)
    return await nodes.request_agenda_modifications_node(state)


async def synthesize_agenda_changes(
    state: VirtualAgoraState, moderator_agent: ModeratorAgent
) -> Dict[str, Any]:
    """Direct node function for agenda change synthesis."""
    nodes = ModeratorNodes(moderator_agent)
    return await nodes.synthesize_agenda_changes_node(state)


async def facilitate_agenda_revote(
    state: VirtualAgoraState, moderator_agent: ModeratorAgent
) -> Dict[str, Any]:
    """Direct node function for agenda re-voting facilitation."""
    nodes = ModeratorNodes(moderator_agent)
    return await nodes.facilitate_agenda_revote_node(state)


# Epic 8: Reporting & Documentation Node Functions


async def generate_topic_summary_with_reporting(
    state: VirtualAgoraState, moderator_agent: ModeratorAgent
) -> Dict[str, Any]:
    """Direct node function for enhanced topic summary generation."""
    nodes = ModeratorNodes(moderator_agent)
    return await nodes.generate_topic_summary_with_reporting_node(state)


async def generate_session_report(
    state: VirtualAgoraState, moderator_agent: ModeratorAgent
) -> Dict[str, Any]:
    """Direct node function for complete session report generation."""
    nodes = ModeratorNodes(moderator_agent)
    return await nodes.generate_session_report_node(state)


async def log_discussion_event(
    state: VirtualAgoraState, moderator_agent: ModeratorAgent
) -> Dict[str, Any]:
    """Direct node function for discussion event logging."""
    nodes = ModeratorNodes(moderator_agent)
    return await nodes.log_discussion_event_node(state)


async def finalize_session_reporting(
    state: VirtualAgoraState, moderator_agent: ModeratorAgent
) -> Dict[str, Any]:
    """Direct node function for session reporting finalization."""
    nodes = ModeratorNodes(moderator_agent)
    return await nodes.finalize_session_reporting_node(state)
