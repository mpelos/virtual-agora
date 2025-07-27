"""Custom state reducers for Virtual Agora state management.

This module provides custom reducer functions for complex state updates
that go beyond simple append operations.
"""

from typing import Any, List, Dict, Optional
from datetime import datetime


def merge_hitl_state(
    existing: Optional[Dict[str, Any]], updates: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Merge HITL (Human-in-the-Loop) state updates.

    Args:
        existing: Current HITL state
        updates: New HITL state updates

    Returns:
        Merged HITL state
    """
    if existing is None:
        existing = {
            "awaiting_approval": False,
            "approval_type": None,
            "prompt_message": None,
            "options": None,
            "approval_history": [],
            # v1.3 additions
            "last_periodic_stop_round": None,
            "periodic_stop_responses": [],
        }

    if updates is None:
        return existing

    # Merge with special handling for approval_history and periodic_stop_responses
    merged = existing.copy()
    merged.update(updates)

    # Append to approval_history if provided
    if "approval_history" in updates and isinstance(updates["approval_history"], list):
        if len(updates["approval_history"]) == 1 and isinstance(
            updates["approval_history"][0], dict
        ):
            # Single new entry - append it
            merged["approval_history"] = (
                existing["approval_history"] + updates["approval_history"]
            )
        else:
            # Replace entire history
            merged["approval_history"] = updates["approval_history"]
    
    # Handle periodic_stop_responses similarly
    if "periodic_stop_responses" in updates and isinstance(updates["periodic_stop_responses"], list):
        if len(updates["periodic_stop_responses"]) == 1 and isinstance(
            updates["periodic_stop_responses"][0], dict
        ):
            # Single new entry - append it
            merged["periodic_stop_responses"] = (
                existing.get("periodic_stop_responses", []) + updates["periodic_stop_responses"]
            )
        else:
            # Replace entire list
            merged["periodic_stop_responses"] = updates["periodic_stop_responses"]

    return merged


def merge_flow_control(
    existing: Optional[Dict[str, Any]], updates: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Merge flow control parameter updates.

    Args:
        existing: Current flow control settings
        updates: New flow control updates

    Returns:
        Merged flow control settings
    """
    if existing is None:
        existing = {
            "max_rounds_per_topic": 10,
            "auto_conclude_threshold": 3,
            "context_window_limit": 8000,
            "cycle_detection_enabled": True,
            "max_iterations_per_phase": 5,
        }

    if updates is None:
        return existing

    # Simple merge for flow control
    merged = existing.copy()
    merged.update(updates)
    return merged


def update_rounds_per_topic(
    existing: Optional[Dict[str, int]], updates: Optional[Dict[str, int]]
) -> Dict[str, int]:
    """Update rounds per topic counter.

    Args:
        existing: Current rounds per topic
        updates: Updates to rounds per topic

    Returns:
        Updated rounds per topic
    """
    if existing is None:
        existing = {}

    if updates is None:
        return existing

    merged = existing.copy()

    # Add or increment counters
    for topic, count in updates.items():
        if topic in merged:
            merged[topic] += count
        else:
            merged[topic] = count

    return merged


def merge_topic_summaries(
    existing: Optional[Dict[str, str]], updates: Optional[Dict[str, str]]
) -> Dict[str, str]:
    """Merge topic summaries.

    Args:
        existing: Current topic summaries
        updates: New topic summaries

    Returns:
        Merged topic summaries
    """
    if existing is None:
        existing = {}

    if updates is None:
        return existing

    # Simple merge - updates replace existing
    merged = existing.copy()
    merged.update(updates)
    return merged


def merge_phase_summaries(
    existing: Optional[Dict[str, str]], updates: Optional[Dict[str, str]]
) -> Dict[str, str]:
    """Merge phase summaries.

    Args:
        existing: Current phase summaries
        updates: New phase summaries

    Returns:
        Merged phase summaries
    """
    if existing is None:
        existing = {}

    if updates is None:
        return existing

    # Simple merge - updates replace existing
    merged = existing.copy()
    merged.update(updates)
    return merged


def increment_counter(existing: Optional[int], increment: Optional[int]) -> int:
    """Increment a counter value.

    Args:
        existing: Current counter value
        increment: Value to add

    Returns:
        New counter value
    """
    if existing is None:
        existing = 0

    if increment is None:
        return existing

    return existing + increment


def merge_statistics(
    existing: Optional[Dict[str, int]], updates: Optional[Dict[str, int]]
) -> Dict[str, int]:
    """Merge runtime statistics.

    Args:
        existing: Current statistics
        updates: New statistics

    Returns:
        Merged statistics
    """
    if existing is None:
        existing = {}

    if updates is None:
        return existing

    merged = existing.copy()

    # Add or increment statistics
    for key, value in updates.items():
        if key in merged:
            merged[key] += value
        else:
            merged[key] = value

    return merged


def merge_agent_info(
    existing: Optional[Dict[str, Dict[str, Any]]],
    updates: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Merge agent information.

    Args:
        existing: Current agent info
        updates: New agent info

    Returns:
        Merged agent info
    """
    if existing is None:
        existing = {}

    if updates is None:
        return existing

    merged = existing.copy()

    # Merge agent info with special handling for message_count
    for agent_id, info in updates.items():
        if agent_id in merged:
            merged_info = merged[agent_id].copy()
            merged_info.update(info)
            # Increment message_count if provided
            if "message_count" in info and "message_count" in merged[agent_id]:
                merged_info["message_count"] = (
                    merged[agent_id]["message_count"] + info["message_count"]
                )
            merged[agent_id] = merged_info
        else:
            merged[agent_id] = info

    return merged


def update_topic_info(
    existing: Optional[Dict[str, Dict[str, Any]]],
    updates: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Update topic information.

    Args:
        existing: Current topic info
        updates: New topic info

    Returns:
        Updated topic info
    """
    if existing is None:
        existing = {}

    if updates is None:
        return existing

    merged = existing.copy()

    # Merge topic info with special handling for message_count
    for topic, info in updates.items():
        if topic in merged:
            merged_info = merged[topic].copy()
            merged_info.update(info)
            # Increment message_count if provided
            if "message_count" in info and "message_count" in merged[topic]:
                merged_info["message_count"] = (
                    merged[topic]["message_count"] + info["message_count"]
                )
            merged[topic] = merged_info
        else:
            merged[topic] = info

    return merged


def merge_specialized_agents(
    existing: Optional[Dict[str, str]], updates: Optional[Dict[str, str]]
) -> Dict[str, str]:
    """Merge specialized agent mappings.

    Args:
        existing: Current agent type to ID mappings
        updates: New agent mappings

    Returns:
        Merged agent mappings
    """
    if existing is None:
        existing = {}

    if updates is None:
        return existing

    # Simple merge - updates replace existing
    merged = existing.copy()
    merged.update(updates)
    return merged
