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

from ..state.schema import VirtualAgoraState, VoteRound, Message
from .moderator import ModeratorAgent

logger = logging.getLogger(__name__)


class ModeratorNodes:
    """LangGraph node functions for moderator operations."""
    
    def __init__(self, moderator_agent: ModeratorAgent):
        """Initialize with a ModeratorAgent instance.
        
        Args:
            moderator_agent: The ModeratorAgent instance to use
        """
        self.moderator = moderator_agent
    
    # Story 3.7: Minority Considerations Management
    
    async def identify_minority_voters_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
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
                return {"warnings": ["No active vote found for minority identification"]}
            
            active_vote = state["active_vote"]
            
            # Get votes for this round
            round_votes = [
                vote for vote in state.get("votes", [])
                if vote["vote_type"] == active_vote["vote_type"] and vote["phase"] == active_vote["phase"]
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
                voter_id for voter_id, choice in voter_choices.items()
                if choice != winning_choice
            ]
            
            logger.info(f"Identified {len(minority_voters)} minority voters: {minority_voters}")
            
            # Update the active vote with minority voter information
            updated_vote = active_vote.copy()
            updated_vote["minority_voters"] = minority_voters
            
            return {
                "active_vote": updated_vote,
                "warnings": [] if minority_voters else ["No minority voters identified - all voted for winning choice"]
            }
            
        except Exception as e:
            logger.error(f"Error identifying minority voters: {e}")
            return {
                "last_error": f"Failed to identify minority voters: {str(e)}",
                "error_count": state.get("error_count", 0) + 1
            }
    
    async def collect_minority_considerations_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
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
                    logger.error(f"Failed to collect consideration from {voter_id}: {e}")
                    minority_considerations.append(f"[Error collecting from {voter_id}]")
            
            # Update the vote round with collected considerations
            updated_vote = active_vote.copy()
            updated_vote["minority_considerations"] = minority_considerations
            
            logger.info(f"Collected {len(minority_considerations)} minority considerations")
            
            return {
                "active_vote": updated_vote,
                "warnings": [] if minority_considerations else ["No minority considerations collected"]
            }
            
        except Exception as e:
            logger.error(f"Error collecting minority considerations: {e}")
            return {
                "last_error": f"Failed to collect minority considerations: {str(e)}",
                "error_count": state.get("error_count", 0) + 1
            }
    
    async def incorporate_minority_views_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
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
                logger.warning("Missing active vote or topic for minority view incorporation")
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
                logger.info(f"Successfully incorporated minority views into summary for {active_topic}")
                
                return {
                    "topic_summaries": topic_summaries
                }
            else:
                logger.warning("Failed to generate updated summary with minority views")
                return {"warnings": ["Failed to incorporate minority views into summary"]}
                
        except Exception as e:
            logger.error(f"Error incorporating minority views: {e}")
            return {
                "last_error": f"Failed to incorporate minority views: {str(e)}",
                "error_count": state.get("error_count", 0) + 1
            }
    
    # Story 3.8: Report Writer Mode Implementation
    
    async def initialize_report_structure_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
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
                    "warnings": ["No topic summaries available for report generation"]
                }
            
            # Use ModeratorAgent in Writer mode to define report structure
            report_structure = await self.moderator.define_report_structure(topic_summaries)
            
            if not report_structure:
                logger.error("Failed to generate report structure")
                return {
                    "report_generation_status": "failed",
                    "last_error": "Failed to generate report structure"
                }
            
            logger.info(f"Generated report structure with {len(report_structure)} sections")
            
            return {
                "report_structure": report_structure,
                "report_sections": {},  # Initialize empty sections dict
                "report_generation_status": "structuring"
            }
            
        except Exception as e:
            logger.error(f"Error initializing report structure: {e}")
            return {
                "report_generation_status": "failed",
                "last_error": f"Failed to initialize report structure: {str(e)}",
                "error_count": state.get("error_count", 0) + 1
            }
    
    async def generate_report_section_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
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
                    "last_error": "No report structure available"
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
                    "final_report": "Report generation completed - sections ready for file output"
                }
            
            # Generate content for this section
            section_content = await self.moderator.generate_report_section(
                next_section, topic_summaries, report_structure
            )
            
            if not section_content:
                logger.error(f"Failed to generate content for section: {next_section}")
                return {
                    "report_generation_status": "failed",
                    "last_error": f"Failed to generate content for section: {next_section}"
                }
            
            # Update report sections
            updated_sections = report_sections.copy()
            updated_sections[next_section] = section_content
            
            logger.info(f"Generated content for section: {next_section}")
            
            return {
                "report_sections": updated_sections,
                "report_generation_status": "writing"
            }
            
        except Exception as e:
            logger.error(f"Error generating report section: {e}")
            return {
                "report_generation_status": "failed",
                "last_error": f"Failed to generate report section: {str(e)}",
                "error_count": state.get("error_count", 0) + 1
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
                    "last_error": "Missing report structure or sections"
                }
            
            # Verify all sections are generated
            missing_sections = [s for s in report_structure if s not in report_sections]
            if missing_sections:
                logger.error(f"Missing sections for finalization: {missing_sections}")
                return {
                    "report_generation_status": "failed",
                    "last_error": f"Missing sections: {missing_sections}"
                }
            
            # Create final report summary
            total_sections = len(report_structure)
            report_summary = f"Final report completed with {total_sections} sections: {', '.join(report_structure)}"
            
            logger.info("Report generation finalized successfully")
            
            return {
                "final_report": report_summary,
                "report_generation_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error finalizing report: {e}")
            return {
                "report_generation_status": "failed",
                "last_error": f"Failed to finalize report: {str(e)}",
                "error_count": state.get("error_count", 0) + 1
            }
    
    # Story 3.9: Agenda Modification Facilitation
    
    async def request_agenda_modifications_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
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
                    logger.error(f"Failed to collect agenda modification from {agent_id}: {e}")
            
            logger.info(f"Collected {len(modification_suggestions)} agenda modification suggestions")
            
            return {
                "pending_agenda_modifications": modification_suggestions
            }
            
        except Exception as e:
            logger.error(f"Error requesting agenda modifications: {e}")
            return {
                "last_error": f"Failed to request agenda modifications: {str(e)}",
                "error_count": state.get("error_count", 0) + 1
            }
    
    async def synthesize_agenda_changes_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
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
                "pending_agenda_modifications": []  # Clear pending modifications
            }
            
        except Exception as e:
            logger.error(f"Error synthesizing agenda changes: {e}")
            return {
                "last_error": f"Failed to synthesize agenda changes: {str(e)}",
                "error_count": state.get("error_count", 0) + 1
            }
    
    async def facilitate_agenda_revote_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
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
                    "error_count": state.get("error_count", 0) + 1
                }
            
            # Synthesize votes into ordered agenda
            new_topic_queue = await self.moderator.synthesize_agenda_votes(
                agenda_votes, proposed_topics
            )
            
            if not new_topic_queue:
                logger.error("Failed to synthesize agenda votes")
                return {
                    "last_error": "Failed to synthesize agenda votes",
                    "error_count": state.get("error_count", 0) + 1
                }
            
            logger.info(f"New agenda synthesized with {len(new_topic_queue)} topics")
            
            return {
                "topic_queue": new_topic_queue,
                "agenda_modification_votes": agenda_votes
            }
            
        except Exception as e:
            logger.error(f"Error facilitating agenda re-vote: {e}")
            return {
                "last_error": f"Failed to facilitate agenda re-vote: {str(e)}",
                "error_count": state.get("error_count", 0) + 1
            }


# Convenience functions for creating node instances

def create_moderator_nodes(moderator_agent: ModeratorAgent) -> ModeratorNodes:
    """Create a ModeratorNodes instance with the given moderator agent.
    
    Args:
        moderator_agent: The ModeratorAgent instance
        
    Returns:
        ModeratorNodes instance ready for use in LangGraph
    """
    return ModeratorNodes(moderator_agent)


# Node function aliases for direct LangGraph usage

async def identify_minority_voters(state: VirtualAgoraState, moderator_agent: ModeratorAgent) -> Dict[str, Any]:
    """Direct node function for minority voter identification."""
    nodes = ModeratorNodes(moderator_agent)
    return await nodes.identify_minority_voters_node(state)


async def collect_minority_considerations(state: VirtualAgoraState, moderator_agent: ModeratorAgent) -> Dict[str, Any]:
    """Direct node function for minority consideration collection."""
    nodes = ModeratorNodes(moderator_agent)
    return await nodes.collect_minority_considerations_node(state)


async def incorporate_minority_views(state: VirtualAgoraState, moderator_agent: ModeratorAgent) -> Dict[str, Any]:
    """Direct node function for minority view incorporation."""
    nodes = ModeratorNodes(moderator_agent)
    return await nodes.incorporate_minority_views_node(state)


async def initialize_report_structure(state: VirtualAgoraState, moderator_agent: ModeratorAgent) -> Dict[str, Any]:
    """Direct node function for report structure initialization."""
    nodes = ModeratorNodes(moderator_agent)
    return await nodes.initialize_report_structure_node(state)


async def generate_report_section(state: VirtualAgoraState, moderator_agent: ModeratorAgent) -> Dict[str, Any]:
    """Direct node function for report section generation."""
    nodes = ModeratorNodes(moderator_agent)
    return await nodes.generate_report_section_node(state)


async def finalize_report(state: VirtualAgoraState, moderator_agent: ModeratorAgent) -> Dict[str, Any]:
    """Direct node function for report finalization."""
    nodes = ModeratorNodes(moderator_agent)
    return await nodes.finalize_report_node(state)


async def request_agenda_modifications(state: VirtualAgoraState, moderator_agent: ModeratorAgent) -> Dict[str, Any]:
    """Direct node function for agenda modification requests."""
    nodes = ModeratorNodes(moderator_agent)
    return await nodes.request_agenda_modifications_node(state)


async def synthesize_agenda_changes(state: VirtualAgoraState, moderator_agent: ModeratorAgent) -> Dict[str, Any]:
    """Direct node function for agenda change synthesis."""
    nodes = ModeratorNodes(moderator_agent)
    return await nodes.synthesize_agenda_changes_node(state)


async def facilitate_agenda_revote(state: VirtualAgoraState, moderator_agent: ModeratorAgent) -> Dict[str, Any]:
    """Direct node function for agenda re-voting facilitation."""
    nodes = ModeratorNodes(moderator_agent)
    return await nodes.facilitate_agenda_revote_node(state)