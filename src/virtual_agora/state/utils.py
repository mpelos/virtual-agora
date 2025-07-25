"""Utility functions for state operations.

This module provides helper functions for working with Virtual Agora state,
including formatting, statistics, and export utilities.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

from virtual_agora.state.schema import (
    VirtualAgoraState,
    Message,
    Vote,
    PhaseTransition,
)
from virtual_agora.state.validators import StateValidator


def format_state_summary(state: VirtualAgoraState) -> str:
    """Format a human-readable summary of the current state.
    
    Args:
        state: Current state
        
    Returns:
        Formatted summary string
    """
    lines = [
        "=== Virtual Agora State Summary ===",
        f"Session ID: {state['session_id']}",
        f"Started: {state['start_time'].strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Current Phase: {StateValidator.PHASE_NAMES[state['current_phase']]} "
        f"(Phase {state['current_phase']})",
        f"Active Topic: {state['active_topic'] or 'None'}",
        f"Current Speaker: {state['current_speaker_id'] or 'None'}",
        "",
        f"Agents: {len(state['agents'])} total",
        f"  - Moderator: {state['moderator_id']}",
        f"  - Participants: {len(state['speaking_order'])}",
        "",
        f"Messages: {state['total_messages']} total",
    ]
    
    # Add phase breakdown
    if state['total_messages'] > 0:
        lines.append("  By phase:")
        for phase, count in state['messages_by_phase'].items():
            if count > 0:
                lines.append(f"    - {StateValidator.PHASE_NAMES[phase]}: {count}")
    
    # Add topic information
    if state['proposed_topics']:
        lines.extend([
            "",
            f"Topics: {len(state['proposed_topics'])} proposed",
            f"  - Completed: {len(state['completed_topics'])}",
            f"  - Remaining: {len(state['topic_queue'])}",
        ])
    
    # Add voting information
    total_votes = len(state['vote_history'])
    if total_votes > 0:
        avg_participation = sum(
            state['vote_participation_rate'].values()
        ) / len(state['vote_participation_rate'])
        lines.extend([
            "",
            f"Votes: {total_votes} completed",
            f"  - Average participation: {avg_participation:.1%}",
        ])
    
    return "\n".join(lines)


def calculate_statistics(state: VirtualAgoraState) -> Dict[str, Any]:
    """Calculate detailed statistics from the state.
    
    Args:
        state: Current state
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        "session": {
            "id": state["session_id"],
            "duration_seconds": (
                datetime.now() - state["start_time"]
            ).total_seconds(),
            "current_phase": state["current_phase"],
            "phase_name": StateValidator.PHASE_NAMES[state["current_phase"]],
        },
        "agents": {
            "total": len(state["agents"]),
            "moderators": len([a for a in state["agents"].values() 
                             if a["role"] == "moderator"]),
            "participants": len([a for a in state["agents"].values() 
                               if a["role"] == "participant"]),
        },
        "messages": {
            "total": state["total_messages"],
            "by_phase": dict(state["messages_by_phase"]),
            "by_role": {
                "moderator": sum(
                    count for agent_id, count in state["messages_by_agent"].items()
                    if state["agents"][agent_id]["role"] == "moderator"
                ),
                "participant": sum(
                    count for agent_id, count in state["messages_by_agent"].items()
                    if state["agents"][agent_id]["role"] == "participant"
                ),
            },
            "average_per_agent": (
                state["total_messages"] / len(state["agents"])
                if state["agents"] else 0
            ),
        },
        "topics": {
            "proposed": len(state["proposed_topics"]),
            "completed": len(state["completed_topics"]),
            "remaining": len(state["topic_queue"]),
            "active": state["active_topic"] is not None,
        },
        "voting": {
            "total_rounds": len(state["vote_history"]),
            "total_votes_cast": len(state["votes"]),
            "average_participation": (
                sum(state["vote_participation_rate"].values()) / 
                len(state["vote_participation_rate"])
                if state["vote_participation_rate"] else 0
            ),
        },
        "consensus": {
            "topics_with_consensus": sum(state["consensus_reached"].values()),
            "consensus_rate": (
                sum(state["consensus_reached"].values()) / 
                len(state["completed_topics"])
                if state["completed_topics"] else 0
            ),
        },
    }
    
    # Add phase durations
    phase_durations = calculate_phase_durations(state)
    stats["phase_durations_seconds"] = phase_durations
    
    # Add message timing statistics
    if state["messages"]:
        message_times = [msg["timestamp"] for msg in state["messages"]]
        stats["messages"]["first_message_time"] = min(message_times).isoformat()
        stats["messages"]["last_message_time"] = max(message_times).isoformat()
        
        # Calculate message frequency
        if len(message_times) > 1:
            total_duration = (max(message_times) - min(message_times)).total_seconds()
            stats["messages"]["messages_per_minute"] = (
                (len(message_times) - 1) / (total_duration / 60)
                if total_duration > 0 else 0
            )
    
    return stats


def calculate_phase_durations(state: VirtualAgoraState) -> Dict[int, float]:
    """Calculate duration spent in each phase.
    
    Args:
        state: Current state
        
    Returns:
        Dictionary mapping phase to duration in seconds
    """
    durations: Dict[int, float] = {i: 0.0 for i in range(5)}
    
    # Process phase history
    current_start = state["start_time"]
    current_phase = 0
    
    for transition in state["phase_history"]:
        # Add duration for the phase we're leaving
        duration = (transition["timestamp"] - current_start).total_seconds()
        durations[current_phase] += duration
        
        # Update for next iteration
        current_start = transition["timestamp"]
        current_phase = transition["to_phase"]
    
    # Add duration for current phase
    duration = (datetime.now() - current_start).total_seconds()
    durations[state["current_phase"]] += duration
    
    return durations


def get_phase_messages(
    state: VirtualAgoraState,
    phase: int
) -> List[Message]:
    """Get all messages from a specific phase.
    
    Args:
        state: Current state
        phase: Phase number
        
    Returns:
        List of messages from that phase
    """
    return [msg for msg in state["messages"] if msg["phase"] == phase]


def get_topic_messages(
    state: VirtualAgoraState,
    topic: str
) -> List[Message]:
    """Get all messages for a specific topic.
    
    Args:
        state: Current state
        topic: Topic name
        
    Returns:
        List of messages for that topic
    """
    return [msg for msg in state["messages"] if msg.get("topic") == topic]


def get_agent_messages(
    state: VirtualAgoraState,
    agent_id: str
) -> List[Message]:
    """Get all messages from a specific agent.
    
    Args:
        state: Current state
        agent_id: Agent ID
        
    Returns:
        List of messages from that agent
    """
    return [msg for msg in state["messages"] if msg["speaker_id"] == agent_id]


def get_vote_results(
    state: VirtualAgoraState,
    vote_round_id: str
) -> Dict[str, int]:
    """Get vote counts for a specific voting round.
    
    Args:
        state: Current state
        vote_round_id: Vote round ID
        
    Returns:
        Dictionary mapping choices to vote counts
    """
    vote_counts: Dict[str, int] = defaultdict(int)
    
    for vote in state["votes"]:
        if vote.get("metadata", {}).get("vote_round_id") == vote_round_id:
            vote_counts[vote["choice"]] += 1
    
    return dict(vote_counts)


def format_discussion_log(state: VirtualAgoraState) -> str:
    """Format the discussion as a readable log.
    
    Args:
        state: Current state
        
    Returns:
        Formatted discussion log
    """
    lines = []
    
    # Header
    lines.extend([
        "=== Virtual Agora Discussion Log ===",
        f"Session: {state['session_id']}",
        f"Date: {state['start_time'].strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ])
    
    # Process messages with phase/topic headers
    current_phase = -1
    current_topic = None
    
    for msg in state["messages"]:
        # Add phase header if changed
        if msg["phase"] != current_phase:
            current_phase = msg["phase"]
            lines.extend([
                "",
                f"--- {StateValidator.PHASE_NAMES[current_phase]} ---",
                "",
            ])
        
        # Add topic header if changed
        if msg.get("topic") != current_topic:
            current_topic = msg.get("topic")
            if current_topic:
                lines.extend([
                    f"[Topic: {current_topic}]",
                    "",
                ])
        
        # Format message
        timestamp = msg["timestamp"].strftime("%H:%M:%S")
        speaker = msg["speaker_id"]
        role = f"[{msg['speaker_role'].upper()}]"
        lines.append(f"{timestamp} {role} {speaker}: {msg['content']}")
    
    # Add summary section if available
    if state["final_report"]:
        lines.extend([
            "",
            "--- Final Report ---",
            "",
            state["final_report"],
        ])
    
    return "\n".join(lines)


def export_for_analysis(state: VirtualAgoraState) -> Dict[str, Any]:
    """Export state in a format suitable for data analysis.
    
    Args:
        state: Current state
        
    Returns:
        Dictionary with analysis-friendly data
    """
    # Convert messages to simple format
    messages = []
    for msg in state["messages"]:
        messages.append({
            "id": msg["id"],
            "speaker_id": msg["speaker_id"],
            "speaker_role": msg["speaker_role"],
            "content": msg["content"],
            "timestamp": msg["timestamp"].isoformat(),
            "phase": msg["phase"],
            "phase_name": StateValidator.PHASE_NAMES[msg["phase"]],
            "topic": msg.get("topic"),
            "word_count": len(msg["content"].split()),
            "char_count": len(msg["content"]),
        })
    
    # Convert votes to simple format
    votes = []
    for vote in state["votes"]:
        votes.append({
            "voter_id": vote["voter_id"],
            "vote_type": vote["vote_type"],
            "choice": vote["choice"],
            "timestamp": vote["timestamp"].isoformat(),
            "phase": vote["phase"],
            "round_id": vote.get("metadata", {}).get("vote_round_id"),
        })
    
    # Get statistics
    stats = calculate_statistics(state)
    
    return {
        "session_info": {
            "id": state["session_id"],
            "start_time": state["start_time"].isoformat(),
            "config_hash": state["config_hash"],
        },
        "agents": [
            {
                "id": agent["id"],
                "model": agent["model"],
                "provider": agent["provider"],
                "role": agent["role"],
                "message_count": agent["message_count"],
            }
            for agent in state["agents"].values()
        ],
        "messages": messages,
        "votes": votes,
        "topics": {
            "proposed": state["proposed_topics"],
            "completed": state["completed_topics"],
            "summaries": state["topic_summaries"],
            "consensus": state["consensus_summaries"],
        },
        "statistics": stats,
        "phase_transitions": [
            {
                "from": t["from_phase"],
                "to": t["to_phase"],
                "timestamp": t["timestamp"].isoformat(),
                "reason": t["reason"],
            }
            for t in state["phase_history"]
        ],
    }


def debug_state(state: VirtualAgoraState) -> str:
    """Create a debug representation of the state.
    
    Args:
        state: Current state
        
    Returns:
        Debug string with detailed state information
    """
    lines = ["=== DEBUG STATE ==="]
    
    # Basic info
    lines.extend([
        f"Session: {state['session_id']}",
        f"Phase: {state['current_phase']} ({StateValidator.PHASE_NAMES[state['current_phase']]})",
        f"Speaker: {state['current_speaker_id']}",
        f"Topic: {state['active_topic']}",
        "",
    ])
    
    # Agents
    lines.append("Agents:")
    for agent_id, agent in state["agents"].items():
        lines.append(
            f"  {agent_id}: {agent['role']} ({agent['model']}) - "
            f"{agent['message_count']} messages"
        )
    
    # Recent messages
    lines.extend(["", "Recent messages:"])
    for msg in state["messages"][-5:]:
        lines.append(
            f"  [{msg['timestamp'].strftime('%H:%M:%S')}] "
            f"{msg['speaker_id']}: {msg['content'][:50]}..."
        )
    
    # Active vote
    if state["active_vote"]:
        vote = state["active_vote"]
        lines.extend([
            "",
            f"Active vote: {vote['vote_type']}",
            f"  Options: {', '.join(vote['options'])}",
            f"  Progress: {vote['received_votes']}/{vote['required_votes']}",
        ])
    
    # Warnings
    if state["warnings"]:
        lines.extend(["", "Warnings:"])
        for warning in state["warnings"][-5:]:
            lines.append(f"  - {warning}")
    
    return "\n".join(lines)