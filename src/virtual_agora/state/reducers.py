"""Custom state reducers for Virtual Agora state management.

This module provides custom reducer functions for complex state updates
that go beyond simple append operations.
"""

from typing import Any, List, Dict, Optional
from datetime import datetime


def safe_list_append(current_list: Optional[List[Any]], new_item: Any) -> List[Any]:
    """Safely append to a list, handling None values and validating inputs.

    This is a robust version of list.append that ensures the list is never None
    and prevents common corruption patterns.

    Args:
        current_list: Current list value (may be None)
        new_item: Item to append (should NOT be a list for most use cases)

    Returns:
        List with the new item appended

    Raises:
        ValueError: If invalid input patterns are detected
    """
    # LOG: Debug reducer calls for completed_topics corruption investigation
    from virtual_agora.utils.logging import get_logger

    logger = get_logger(__name__)

    logger.info(f"=== SAFE_LIST_APPEND REDUCER CALLED ===")
    logger.info(f"Current list: {current_list} (Type: {type(current_list)})")
    logger.info(f"New item: {new_item} (Type: {type(new_item)})")

    # VALIDATION: Handle empty lists being passed as new_item
    if isinstance(new_item, list) and len(new_item) == 0:
        # Count empty list occurrences to avoid spam
        empty_list_count = getattr(safe_list_append, "_empty_list_count", 0) + 1
        safe_list_append._empty_list_count = empty_list_count

        # Only log detailed stack trace for first few occurrences
        if empty_list_count <= 3:
            import traceback

            logger.error(
                f"REJECTED: Empty list passed as new_item: {new_item} (occurrence #{empty_list_count})"
            )
            logger.error(
                f"This indicates a bug in node return values or LangGraph processing"
            )

            if empty_list_count == 1:
                logger.error(
                    f"Call stack for debugging (showing only for first occurrence):"
                )
                # Get the call stack to help identify the source
                stack = traceback.format_stack()
                for frame in stack[-5:]:  # Show last 5 frames
                    logger.error(f"  {frame.strip()}")
        elif empty_list_count == 4:
            logger.error(
                f"SUPPRESSING further empty list errors to reduce log noise (total: {empty_list_count})"
            )
        elif empty_list_count % 10 == 0:
            logger.warning(
                f"Empty list rejections continue (total: {empty_list_count})"
            )

        # Return current list unchanged to prevent corruption
        return current_list if current_list is not None else []

    # VALIDATION: Warn about lists being passed (may be intentional in some cases)
    if isinstance(new_item, list):
        # Check if this might be an intentional list case (like turn_order_history)
        is_intentional = False

        # turn_order_history expects List[str] items (agent IDs)
        if (
            len(new_item) > 0
            and all(isinstance(item, str) for item in new_item)
            and len(new_item) <= 10
        ):  # Reasonable number of agents
            is_intentional = True
            logger.debug(
                f"Valid turn order list for turn_order_history: {len(new_item)} agents"
            )

        # phase_history field - handle single-item lists (common bug)
        elif (
            len(new_item) == 1
            and isinstance(new_item[0], dict)
            and "from_phase" in new_item[0]
            and "to_phase" in new_item[0]
        ):
            logger.warning(f"SINGLE-ITEM LIST detected for phase_history")
            logger.warning(
                f"Extracting phase transition from list wrapper - this indicates a bug in LangGraph processing"
            )
            # Extract the actual item from the list wrapper
            actual_item = new_item[0]
            is_intentional = True  # Mark as handled to avoid further warnings

            # Continue with normal processing using the extracted item
            if current_list is None:
                result = [actual_item]
                logger.info(
                    f"Extracted phase transition from list wrapper: {actual_item.get('from_phase')} → {actual_item.get('to_phase')}"
                )
                return result
            else:
                result = current_list + [actual_item]
                logger.info(
                    f"Extracted and appended phase transition: {len(result)} total transitions"
                )
                return result

        # round_history field - handle single-item lists (common bug)
        elif (
            len(new_item) == 1
            and isinstance(new_item[0], dict)
            and "round_id" in new_item[0]
            and "round_number" in new_item[0]
        ):
            logger.warning(f"SINGLE-ITEM LIST detected for round_history")
            logger.warning(
                f"Extracting round info from list wrapper - this indicates a bug in LangGraph processing"
            )
            # Extract the actual item from the list wrapper
            actual_item = new_item[0]
            is_intentional = True  # Mark as handled to avoid further warnings

            # Continue with normal processing using the extracted item
            if current_list is None:
                result = [actual_item]
                logger.info(
                    f"Extracted round info from list wrapper: round {actual_item.get('round_number')}"
                )
                return result
            else:
                result = current_list + [actual_item]
                logger.info(
                    f"Extracted and appended round info: {len(result)} total rounds"
                )
                return result

        # round_summaries field - handle single-item lists (common bug)
        elif (
            len(new_item) == 1
            and isinstance(new_item[0], dict)
            and "round_number" in new_item[0]
            and "summary_text" in new_item[0]
        ):
            logger.warning(f"SINGLE-ITEM LIST detected for round summaries")
            logger.warning(
                f"Extracting item from list wrapper - this indicates a bug in node return values"
            )
            # Extract the actual item from the list wrapper
            actual_item = new_item[0]
            is_intentional = True  # Mark as handled to avoid further warnings

            # Continue with normal processing using the extracted item
            if current_list is None:
                result = [actual_item]
                logger.info(
                    f"Extracted round summary from list wrapper: round {actual_item.get('round_number')}"
                )
                return result
            else:
                result = current_list + [actual_item]
                logger.info(
                    f"Extracted and appended round summary: {len(result)} total summaries"
                )
                return result

        # turn_order_history field - handle nested lists (double wrapping bug)
        elif (
            len(new_item) == 1
            and isinstance(new_item[0], list)
            and len(new_item[0]) > 0
            and all(isinstance(item, str) for item in new_item[0])
            and len(new_item[0]) <= 10
        ):
            logger.warning(f"NESTED LIST detected for turn_order_history")
            logger.warning(
                f"Extracting turn order from nested list wrapper - this indicates a bug in LangGraph processing"
            )
            # Extract the actual turn order from the nested list
            actual_turn_order = new_item[0]
            is_intentional = True  # Mark as handled to avoid further warnings

            # Continue with normal processing using the extracted turn order
            if current_list is None:
                result = [actual_turn_order]
                logger.info(
                    f"Extracted turn order from nested list: {len(actual_turn_order)} agents"
                )
                return result
            else:
                result = current_list + [actual_turn_order]
                logger.info(
                    f"Extracted and appended turn order: {len(result)} total turn orders"
                )
                return result

        # votes field - handle vote batches properly
        elif len(new_item) > 0 and all(
            isinstance(item, dict) and "voter_id" in item for item in new_item
        ):
            # Check if this is a reasonable batch size from agenda voting
            if len(new_item) <= 10:  # Reasonable number of agents
                logger.info(
                    f"VOTE BATCH PASSED as expected: {len(new_item)} votes from agenda voting"
                )
                logger.info(f"Processing batch from collect_agenda_votes_node")
                # Handle vote batch by extending the list instead of appending as single item
                if current_list is None:
                    result = new_item.copy()
                    logger.info(
                        f"Vote batch processed - created new list: {len(result)} votes"
                    )
                    return result
                else:
                    result = current_list + new_item
                    logger.info(
                        f"Vote batch processed - extended list: {len(result)} total votes"
                    )
                    return result
            else:
                logger.error(
                    f"VOTES LIST PASSED incorrectly as new_item: {len(new_item)} votes"
                )
                logger.error(
                    f"Batch size too large - should be handled by adding votes individually"
                )
                logger.error(
                    f"First vote sample: {new_item[0] if new_item else 'none'}"
                )
        else:
            # Only warn for non-intentional list patterns
            if not is_intentional:
                logger.warning(f"LIST PASSED as new_item: {new_item}")
                logger.warning(
                    f"Verify this is intentional - reducers typically expect individual items"
                )

    if current_list is None:
        result = [new_item]
        logger.info(f"None list case - created new: {result}")
        return result

    # Validate current_list is actually a list
    if not isinstance(current_list, list):
        logger.error(
            f"CRITICAL: current_list is not a list! Type: {type(current_list)}, Value: {current_list}"
        )
        # Try to recover by converting to list
        if current_list:
            current_list = [current_list]
        else:
            current_list = []
        logger.info(f"Recovered current_list: {current_list}")

    result = current_list + [new_item]
    logger.info(
        f"Append result: {result} (Type: {type(result)}, Length: {len(result)})"
    )
    return result


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
    if "periodic_stop_responses" in updates and isinstance(
        updates["periodic_stop_responses"], list
    ):
        if len(updates["periodic_stop_responses"]) == 1 and isinstance(
            updates["periodic_stop_responses"][0], dict
        ):
            # Single new entry - append it
            merged["periodic_stop_responses"] = (
                existing.get("periodic_stop_responses", [])
                + updates["periodic_stop_responses"]
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
