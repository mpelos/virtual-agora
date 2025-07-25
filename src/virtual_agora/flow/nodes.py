"""Node implementations for the Virtual Agora discussion flow.

This module contains all the node functions that represent different phases
and steps in the discussion workflow.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from langgraph.types import interrupt, Command
from langchain_core.runnables import RunnableConfig

from virtual_agora.state.schema import VirtualAgoraState, RoundInfo, PhaseTransition
from virtual_agora.state.manager import StateManager
from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.flow.context_window import ContextWindowManager
from virtual_agora.flow.cycle_detection import CyclePreventionManager
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class FlowNodes:
    """Container for all flow node implementations."""

    def __init__(self, agents: Dict[str, LLMAgent], state_manager: StateManager):
        """Initialize with agents and state manager.

        Args:
            agents: Dictionary of initialized agents
            state_manager: State manager instance
        """
        self.agents = agents
        self.state_manager = state_manager
        self.context_manager = ContextWindowManager()
        self.cycle_manager = CyclePreventionManager()

    def initialization_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Phase 0: Initialize the discussion session.

        Args:
            state: Current state

        Returns:
            State updates
        """
        logger.info("Starting Phase 0: Initialization")

        # Get main topic from user if not already provided
        main_topic = state.get("main_topic")
        if not main_topic:
            main_topic = interrupt(
                {
                    "type": "user_input",
                    "message": "Please enter the topic you would like the agents to discuss:",
                    "input_type": "text",
                }
            )

        # Initialize speaking order from agent IDs (excluding moderator)
        participant_ids = [
            aid for aid, agent in self.agents.items() if agent.role == "participant"
        ]

        updates = {
            "current_phase": 0,
            "main_topic": main_topic,
            "speaking_order": participant_ids,
            "moderator_id": "moderator",
            "phase_history": [
                {
                    "from_phase": -1,
                    "to_phase": 0,
                    "timestamp": datetime.now(),
                    "reason": "Session started",
                    "triggered_by": "system",
                }
            ],
            "phase_start_time": datetime.now(),
        }

        logger.info(f"Initialized session with topic: {main_topic}")
        logger.info(f"Speaking order: {participant_ids}")

        return updates

    def agenda_setting_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Phase 1: Agenda setting through agent proposals and voting.

        Args:
            state: Current state

        Returns:
            State updates
        """
        logger.info("Starting Phase 1: Agenda Setting")

        main_topic = state["main_topic"]
        moderator = self.agents["moderator"]

        # Step 1: Request topic proposals from all agents
        proposal_prompt = f"""
        You are facilitating a discussion on: "{main_topic}"
        
        Please request each agent to propose 3-5 sub-topics for structured discussion.
        Ask each agent individually and collect their proposals.
        """

        # Get proposals from each participant
        proposals = []
        for agent_id in state["speaking_order"]:
            agent = self.agents[agent_id]

            # Ask agent for proposals
            agent_response = agent(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Based on the main topic '{main_topic}', please propose 3-5 specific sub-topics you think would be valuable to discuss.",
                        }
                    ]
                }
            )

            # Extract proposals (simplified - in real implementation would parse response)
            agent_proposals = agent_response.get("proposals", [])
            proposals.extend(agent_proposals)

        # Step 2: Moderator collates and deduplicates proposals
        collation_prompt = f"""
        Here are all the proposed sub-topics: {proposals}
        
        Please create a unified, deduplicated list of discussion topics and present them
        to the agents for voting. Format as a clear numbered list.
        """

        collated_topics = moderator(
            {"messages": [{"role": "user", "content": collation_prompt}]}
        ).get("topics", [])

        # Step 3: Collect votes from agents
        votes = {}
        for agent_id in state["speaking_order"]:
            agent = self.agents[agent_id]

            vote_prompt = f"""
            Here are the proposed discussion topics:
            {chr(10).join(f"{i+1}. {topic}" for i, topic in enumerate(collated_topics))}
            
            Please rank these topics in order of preference (1 being most preferred).
            Respond with your ranking.
            """

            vote_response = agent(
                {"messages": [{"role": "user", "content": vote_prompt}]}
            )

            votes[agent_id] = vote_response.get(
                "ranking", list(range(len(collated_topics)))
            )

        # Step 4: Moderator synthesizes votes into final agenda
        synthesis_prompt = f"""
        Based on these votes from agents: {votes}
        For topics: {collated_topics}
        
        Analyze the votes and create a final ranked agenda. You have authority to break ties.
        
        Respond with ONLY a JSON object in this format:
        {{"proposed_agenda": ["Topic 1", "Topic 2", "Topic 3"]}}
        """

        agenda_response = moderator(
            {"messages": [{"role": "user", "content": synthesis_prompt}]}
        )

        # Parse agenda (simplified - would use proper JSON parsing)
        proposed_agenda = agenda_response.get("proposed_agenda", collated_topics[:3])

        updates = {
            "current_phase": 1,
            "phase_start_time": datetime.now(),
            "proposed_topics": collated_topics,
            "topic_queue": proposed_agenda,
            "proposed_agenda": proposed_agenda,  # For HITL approval
            "phase_history": [
                {
                    "from_phase": 0,
                    "to_phase": 1,
                    "timestamp": datetime.now(),
                    "reason": "Agenda setting initiated",
                    "triggered_by": "system",
                }
            ],
        }

        logger.info(f"Generated proposed agenda: {proposed_agenda}")

        return updates

    def agenda_approval_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """HITL: Human approval of the proposed agenda.

        Args:
            state: Current state

        Returns:
            State updates
        """
        logger.info("Requesting human approval of agenda")

        proposed_agenda = state["proposed_agenda"]

        # Present agenda to user for approval
        user_response = interrupt(
            {
                "type": "agenda_approval",
                "message": "Please review and approve the proposed discussion agenda:",
                "agenda": proposed_agenda,
                "options": ["approve", "edit", "reject"],
            }
        )

        if user_response == "approve":
            updates = {
                "agenda_approved": True,
                "topic_queue": proposed_agenda,
                "hitl_state": {
                    **state["hitl_state"],
                    "awaiting_approval": False,
                    "approval_history": state["hitl_state"]["approval_history"]
                    + [
                        {
                            "type": "agenda_approval",
                            "result": "approved",
                            "timestamp": datetime.now(),
                        }
                    ],
                },
            }
        elif user_response == "edit":
            # Get edited agenda from user
            edited_agenda = interrupt(
                {
                    "type": "agenda_edit",
                    "message": "Please provide your edited agenda:",
                    "current_agenda": proposed_agenda,
                }
            )

            updates = {
                "agenda_approved": True,
                "topic_queue": edited_agenda,
                "proposed_agenda": edited_agenda,
                "hitl_state": {
                    **state["hitl_state"],
                    "awaiting_approval": False,
                    "approval_history": state["hitl_state"]["approval_history"]
                    + [
                        {
                            "type": "agenda_approval",
                            "result": "edited",
                            "timestamp": datetime.now(),
                            "original": proposed_agenda,
                            "edited": edited_agenda,
                        }
                    ],
                },
            }
        else:  # reject
            updates = {
                "agenda_approved": False,
                "hitl_state": {
                    **state["hitl_state"],
                    "awaiting_approval": False,
                    "approval_history": state["hitl_state"]["approval_history"]
                    + [
                        {
                            "type": "agenda_approval",
                            "result": "rejected",
                            "timestamp": datetime.now(),
                        }
                    ],
                },
            }

        return updates

    def discussion_round_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Phase 2: Execute a discussion round with rotating speakers.

        Args:
            state: Current state

        Returns:
            State updates
        """
        logger.info("Executing discussion round")

        # Get current topic
        if not state["topic_queue"]:
            return {"error": "No topics in queue"}

        current_topic = (
            state["topic_queue"][0]
            if not state.get("active_topic")
            else state["active_topic"]
        )
        current_round = state["current_round"] + 1

        # Initialize or rotate speaking order
        speaking_order = state["speaking_order"]
        if current_round > 1:
            # Rotate: [A,B,C] -> [B,C,A]
            speaking_order = speaking_order[1:] + [speaking_order[0]]

        round_id = str(uuid.uuid4())
        round_start = datetime.now()

        # Execute the round - each agent speaks
        round_messages = []
        participants = []

        for i, agent_id in enumerate(speaking_order):
            agent = self.agents[agent_id]

            # Prepare context for agent
            context_messages = state["messages"][-10:]  # Last 10 messages for context

            agent_prompt = f"""
            Topic: {current_topic}
            Round: {current_round}
            Your turn: {i + 1}/{len(speaking_order)}
            
            Please provide your thoughts on this topic. Build upon the previous discussion.
            """

            agent_response = agent(
                {
                    "messages": context_messages
                    + [{"role": "user", "content": agent_prompt}]
                }
            )

            round_messages.append(
                {
                    "id": str(uuid.uuid4()),
                    "speaker_id": agent_id,
                    "speaker_role": "participant",
                    "content": agent_response["messages"][-1]["content"],
                    "timestamp": datetime.now(),
                    "phase": 2,
                    "topic": current_topic,
                    "round": current_round,
                }
            )

            participants.append(agent_id)

        # Create round info
        round_info = RoundInfo(
            round_id=round_id,
            round_number=current_round,
            topic=current_topic,
            start_time=round_start,
            end_time=datetime.now(),
            participants=participants,
            message_count=len(round_messages),
            summary=None,  # Will be filled by round_summary_node
        )

        updates = {
            "current_phase": 2,
            "current_round": current_round,
            "active_topic": current_topic,
            "speaking_order": speaking_order,
            "messages": round_messages,
            "round_history": [round_info],
            "turn_order_history": [speaking_order],
            "rounds_per_topic": {
                **state.get("rounds_per_topic", {}),
                current_topic: state.get("rounds_per_topic", {}).get(current_topic, 0)
                + 1,
            },
        }

        logger.info(f"Completed round {current_round} on topic: {current_topic}")

        return updates

    def round_summary_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Generate moderator summary of the discussion round.

        Args:
            state: Current state

        Returns:
            State updates
        """
        logger.info("Generating round summary")

        moderator = self.agents["moderator"]
        current_round = state["current_round"]
        current_topic = state["active_topic"]

        # Get messages from current round
        round_messages = [
            msg
            for msg in state["messages"]
            if msg.get("round") == current_round and msg.get("topic") == current_topic
        ]

        summary_prompt = f"""
        Please provide a concise summary of Round {current_round} discussion on "{current_topic}".
        
        Round messages:
        {chr(10).join(f"{msg['speaker_id']}: {msg['content']}" for msg in round_messages)}
        
        Summarize the key points, agreements, disagreements, and progression of ideas.
        Keep it concise but comprehensive.
        """

        summary_response = moderator(
            {"messages": [{"role": "user", "content": summary_prompt}]}
        )

        round_summary = summary_response["messages"][-1]["content"]

        # Update the latest round info with summary
        updated_round_history = state["round_history"].copy()
        if updated_round_history:
            updated_round_history[-1] = {
                **updated_round_history[-1],
                "summary": round_summary,
            }

        updates = {
            "round_history": updated_round_history,
            "phase_summaries": {
                **state.get("phase_summaries", {}),
                f"round_{current_round}": round_summary,
            },
        }

        logger.info(f"Generated summary for round {current_round}")

        return updates

    def conclusion_poll_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Phase 3: Poll agents on whether to conclude current topic.

        Args:
            state: Current state

        Returns:
            State updates
        """
        logger.info("Conducting topic conclusion poll")

        moderator = self.agents["moderator"]
        current_topic = state["active_topic"]
        current_round = state["current_round"]

        # Poll each agent
        votes = {}
        moderator_prompt = f"""
        We have completed {current_round} rounds on "{current_topic}".
        Should we conclude the discussion on this topic?
        Please ask each agent to vote 'Yes' or 'No' with a brief justification.
        """

        for agent_id in state["speaking_order"]:
            agent = self.agents[agent_id]

            poll_prompt = f"""
            We have discussed "{current_topic}" for {current_round} rounds.
            Should we conclude the discussion on '{current_topic}'?
            
            Please respond with 'Yes' or 'No' and provide a short justification.
            """

            vote_response = agent(
                {"messages": [{"role": "user", "content": poll_prompt}]}
            )

            # Parse vote (simplified - would use proper parsing)
            vote_content = vote_response["messages"][-1]["content"].lower()
            vote = "yes" if "yes" in vote_content else "no"

            votes[agent_id] = {
                "vote": vote,
                "justification": vote_response["messages"][-1]["content"],
                "timestamp": datetime.now(),
            }

        # Tally votes
        yes_votes = sum(1 for vote_info in votes.values() if vote_info["vote"] == "yes")
        total_votes = len(votes)
        majority_threshold = total_votes // 2 + 1

        conclusion_passed = yes_votes >= majority_threshold

        # Identify minority voters (those who voted against the majority)
        if conclusion_passed:
            minority_voters = [
                agent_id
                for agent_id, vote_info in votes.items()
                if vote_info["vote"] == "no"
            ]
        else:
            minority_voters = [
                agent_id
                for agent_id, vote_info in votes.items()
                if vote_info["vote"] == "yes"
            ]

        updates = {
            "current_phase": 3,
            "conclusion_vote": {
                "topic": current_topic,
                "round": current_round,
                "votes": votes,
                "yes_votes": yes_votes,
                "total_votes": total_votes,
                "passed": conclusion_passed,
                "minority_voters": minority_voters,
                "timestamp": datetime.now(),
            },
        }

        logger.info(
            f"Conclusion poll: {yes_votes}/{total_votes} votes to conclude. Passed: {conclusion_passed}"
        )

        return updates

    def minority_considerations_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Allow minority voters to provide final considerations.

        Args:
            state: Current state

        Returns:
            State updates
        """
        logger.info("Collecting minority considerations")

        conclusion_vote = state["conclusion_vote"]
        minority_voters = conclusion_vote["minority_voters"]
        current_topic = state["active_topic"]

        minority_considerations = []

        for agent_id in minority_voters:
            agent = self.agents[agent_id]

            consideration_prompt = f"""
            The majority has voted to conclude the discussion on "{current_topic}",
            but you voted against this. Please provide your final considerations
            on this topic before we move on.
            """

            consideration_response = agent(
                {"messages": [{"role": "user", "content": consideration_prompt}]}
            )

            minority_considerations.append(
                {
                    "agent_id": agent_id,
                    "content": consideration_response["messages"][-1]["content"],
                    "timestamp": datetime.now(),
                }
            )

        updates = {"minority_considerations": minority_considerations}

        logger.info(f"Collected {len(minority_considerations)} minority considerations")

        return updates

    def topic_summary_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Generate comprehensive topic summary and save to file.

        Args:
            state: Current state

        Returns:
            State updates
        """
        logger.info("Generating topic summary")

        moderator = self.agents["moderator"]
        current_topic = state["active_topic"]

        # Gather all discussion content for this topic
        topic_messages = [
            msg for msg in state["messages"] if msg.get("topic") == current_topic
        ]

        # Get round summaries for this topic
        topic_rounds = [
            round_info
            for round_info in state["round_history"]
            if round_info["topic"] == current_topic
        ]

        minority_considerations = state.get("minority_considerations", [])

        summary_prompt = f"""
        Please create a comprehensive, agent-agnostic summary of the entire discussion on "{current_topic}".
        
        Discussion included {len(topic_rounds)} rounds with the following progression:
        {chr(10).join(f"Round {r['round_number']}: {r.get('summary', 'No summary')}" for r in topic_rounds)}
        
        Minority considerations:
        {chr(10).join(f"- {mc['content']}" for mc in minority_considerations)}
        
        Create a synthesis that captures:
        1. Key points discussed
        2. Areas of agreement and disagreement
        3. Evolution of ideas through the rounds
        4. Final considerations from dissenting voices
        
        Write this as a comprehensive summary suitable for inclusion in a final report.
        """

        summary_response = moderator(
            {"messages": [{"role": "user", "content": summary_prompt}]}
        )

        topic_summary = summary_response["messages"][-1]["content"]

        # Save summary to file (simplified - would use proper file I/O)
        filename = f"topic_summary_{current_topic.replace(' ', '_')}.md"

        updates = {
            "topic_summaries": {
                **state.get("topic_summaries", {}),
                current_topic: topic_summary,
            },
            "completed_topics": [current_topic],
            "topic_summary_files": {
                **state.get("topic_summary_files", {}),
                current_topic: filename,
            },
            # Remove completed topic from queue
            "topic_queue": [t for t in state["topic_queue"] if t != current_topic],
            "active_topic": None,
            "current_round": 0,  # Reset for next topic
        }

        logger.info(f"Generated summary for topic: {current_topic}")

        return updates

    def continuation_approval_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """HITL: Get user approval to continue with next topic.

        Args:
            state: Current state

        Returns:
            State updates
        """
        logger.info("Requesting continuation approval")

        completed_topic = state["completed_topics"][-1]
        remaining_topics = state["topic_queue"]

        user_response = interrupt(
            {
                "type": "continuation_approval",
                "message": f"The topic '{completed_topic}' is now concluded and its summary has been saved. Shall we re-evaluate the agenda and continue?",
                "completed_topic": completed_topic,
                "remaining_topics": remaining_topics,
                "options": ["yes", "no"],
            }
        )

        continue_session = user_response.lower() == "yes"

        updates = {
            "continue_session": continue_session,
            "hitl_state": {
                **state["hitl_state"],
                "approval_history": state["hitl_state"]["approval_history"]
                + [
                    {
                        "type": "continuation_approval",
                        "result": user_response,
                        "timestamp": datetime.now(),
                    }
                ],
            },
        }

        if not continue_session:
            updates["current_phase"] = 5  # Move to report generation

        return updates

    def agenda_modification_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Phase 4: Allow agents to modify remaining agenda.

        Args:
            state: Current state

        Returns:
            State updates
        """
        logger.info("Processing agenda modifications")

        moderator = self.agents["moderator"]
        remaining_topics = state["topic_queue"]

        # Ask agents for modifications
        modifications = {}
        modification_prompt = f"""
        Based on our last discussion, we have these remaining topics:
        {chr(10).join(f"- {topic}" for topic in remaining_topics)}
        
        Should we add any new topics to our agenda, or remove any of the remaining ones?
        Please provide your suggestions.
        """

        for agent_id in state["speaking_order"]:
            agent = self.agents[agent_id]

            mod_response = agent(
                {"messages": [{"role": "user", "content": modification_prompt}]}
            )

            modifications[agent_id] = mod_response["messages"][-1]["content"]

        # Moderator synthesizes modifications
        synthesis_prompt = f"""
        Here are the agents' suggestions for agenda modifications:
        {chr(10).join(f"{agent_id}: {suggestion}" for agent_id, suggestion in modifications.items())}
        
        Current remaining topics: {remaining_topics}
        
        Based on these suggestions, create a revised agenda. Consider:
        - Adding genuinely valuable new topics
        - Removing topics that may no longer be relevant
        - Maintaining reasonable scope
        
        Respond with a JSON object: {{"revised_agenda": ["Topic 1", "Topic 2"]}}
        """

        synthesis_response = moderator(
            {"messages": [{"role": "user", "content": synthesis_prompt}]}
        )

        # Parse revised agenda (simplified)
        revised_agenda = synthesis_response.get("revised_agenda", remaining_topics)

        updates = {
            "current_phase": 4,
            "topic_queue": revised_agenda,
            "agenda_modifications": {
                "suggestions": modifications,
                "revised_agenda": revised_agenda,
                "timestamp": datetime.now(),
            },
        }

        logger.info(f"Agenda modified. New queue: {revised_agenda}")

        return updates

    def report_generation_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Phase 5: Generate final comprehensive report.

        Args:
            state: Current state

        Returns:
            State updates
        """
        logger.info("Generating final report")

        moderator = self.agents["moderator"]
        topic_summaries = state.get("topic_summaries", {})

        # Step 1: Define report structure
        structure_prompt = f"""
        You are now "The Writer" and must create a final report structure.
        
        We have discussed these topics with summaries:
        {chr(10).join(f"- {topic}" for topic in topic_summaries.keys())}
        
        Define a logical structure for the final report as a JSON list of section titles.
        Consider: Executive Summary, Topic Analysis, Key Insights, Conclusions, etc.
        
        Respond with: {{"report_sections": ["Section 1", "Section 2", ...]}}
        """

        structure_response = moderator(
            {"messages": [{"role": "user", "content": structure_prompt}]}
        )

        report_sections = structure_response.get(
            "report_sections",
            [
                "Executive Summary",
                "Discussion Overview",
                "Topic Analysis",
                "Key Insights",
                "Conclusions",
            ],
        )

        # Step 2: Generate content for each section
        report_content = {}

        for i, section in enumerate(report_sections):
            section_prompt = f"""
            Write the content for report section: "{section}"
            
            Available topic summaries:
            {chr(10).join(f"{topic}: {summary[:200]}..." for topic, summary in topic_summaries.items())}
            
            Create comprehensive content for this section based on the entire discussion.
            """

            section_response = moderator(
                {"messages": [{"role": "user", "content": section_prompt}]}
            )

            report_content[section] = section_response["messages"][-1]["content"]

            # Save to individual file (simplified)
            filename = f"final_report_{i+1:02d}_{section.replace(' ', '_')}.md"
            report_content[f"{section}_file"] = filename

        updates = {
            "current_phase": 5,
            "report_structure": report_sections,
            "report_sections": report_content,
            "report_generation_status": "completed",
            "final_report": report_content,
            "session_completed": True,
            "phase_history": [
                {
                    "from_phase": 4,
                    "to_phase": 5,
                    "timestamp": datetime.now(),
                    "reason": "Final report generation",
                    "triggered_by": "system",
                }
            ],
        }

        logger.info("Final report generation completed")

        return updates

    def context_compression_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Compress context when approaching token limits.

        Args:
            state: Current state

        Returns:
            State updates with compressed context
        """
        logger.info("Starting context compression")

        # Get current context statistics
        stats = self.context_manager.get_context_stats(state)
        logger.info(
            f"Context stats before compression: {stats['total_tokens']} tokens "
            f"({stats['usage_percent']:.1f}% of {stats['limit']})"
        )

        # Perform compression
        compression_updates = self.context_manager.compress_context(state)

        if compression_updates:
            new_stats = self.context_manager.get_context_stats(
                {**state, **compression_updates}
            )
            logger.info(
                f"Context compressed: {stats['total_tokens']} -> {new_stats['total_tokens']} tokens"
            )

            # Add compression metadata
            compression_updates["context_compressions_count"] = (
                compression_updates.get("context_compressions_count", 0) + 1
            )
            compression_updates["last_compression_time"] = datetime.now()

            return compression_updates
        else:
            logger.info("No compression needed")
            return {}

    def cycle_intervention_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Intervene to break detected cycles.

        Args:
            state: Current state

        Returns:
            State updates to break cycles
        """
        logger.info("Starting cycle intervention")

        # Analyze current state for cycles
        cycles = self.cycle_manager.analyze_state(state)

        if not cycles:
            logger.info("No cycles detected during intervention")
            return {}

        # Get intervention strategy
        intervention = self.cycle_manager.get_intervention_strategy(cycles)

        if not intervention["intervention_needed"]:
            logger.info("No intervention needed for detected cycles")
            return {}

        updates = {}

        for strategy in intervention["strategies"]:
            logger.info(
                f"Applying intervention strategy: {strategy['type']} - {strategy['reason']}"
            )

            if strategy["type"] == "phase_intervention":
                # Force phase progression
                updates.update(
                    {
                        "current_phase": strategy["target_phase"],
                        "phase_start_time": datetime.now(),
                        "phase_history": state["phase_history"]
                        + [
                            {
                                "from_phase": state["current_phase"],
                                "to_phase": strategy["target_phase"],
                                "timestamp": datetime.now(),
                                "reason": "Cycle intervention: " + strategy["reason"],
                                "triggered_by": "cycle_detector",
                            }
                        ],
                        "warnings": state["warnings"]
                        + [
                            f"Cycle intervention: Forced progression to phase {strategy['target_phase']}"
                        ],
                    }
                )

            elif strategy["type"] == "voting_intervention":
                # Moderator makes decision to break voting cycle
                updates.update(
                    {
                        "active_vote": None,  # Clear any active vote
                        "moderator_decision": True,
                        "warnings": state["warnings"]
                        + [
                            f"Cycle intervention: Moderator decision - {strategy['reason']}"
                        ],
                    }
                )

            elif strategy["type"] == "speaker_intervention":
                # Enforce turn rotation, skip monopolizing speaker
                excluded_speaker = strategy["excluded_speaker"]
                speaking_order = [
                    agent_id
                    for agent_id in state["speaking_order"]
                    if agent_id != excluded_speaker
                ]

                if speaking_order:
                    updates.update(
                        {
                            "current_speaker_id": speaking_order[0],
                            "next_speaker_index": 1 % len(speaking_order),
                            "warnings": state["warnings"]
                            + [
                                f"Cycle intervention: Enforced turn rotation, skipping {excluded_speaker}"
                            ],
                        }
                    )

            elif strategy["type"] == "topic_intervention":
                # Force topic conclusion
                if state["active_topic"]:
                    updates.update(
                        {
                            "completed_topics": state["completed_topics"]
                            + [state["active_topic"]],
                            "active_topic": None,
                            "warnings": state["warnings"]
                            + [
                                f"Cycle intervention: Forced topic conclusion - {strategy['reason']}"
                            ],
                        }
                    )

        # Record intervention in state
        updates.update(
            {
                "cycle_interventions_count": updates.get("cycle_interventions_count", 0)
                + 1,
                "last_intervention_time": datetime.now(),
                "last_intervention_reason": (
                    intervention["strategies"][0]["reason"]
                    if intervention["strategies"]
                    else "Unknown"
                ),
            }
        )

        logger.info(
            f"Cycle intervention completed with {len(intervention['strategies'])} strategies applied"
        )

        return updates
