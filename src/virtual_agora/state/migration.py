"""State migration utilities for Virtual Agora.

This module provides utilities to migrate state from v1.1 format
to v1.3 format with specialized agent tracking and enhanced HITL.
"""

from typing import Dict, Any, List
from copy import deepcopy
from datetime import datetime

from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


def migrate_state_v1_to_v3(old_state: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate v1.1 state to v1.3 format.
    
    This function adds the required fields for v1.3 specialized agent
    tracking, enhanced HITL controls, and periodic stop functionality.
    
    Args:
        old_state: The v1.1 state dictionary
        
    Returns:
        A v1.3 compatible state dictionary
    """
    # Deep copy to avoid modifying the original
    new_state = deepcopy(old_state)
    
    # Initialize specialized agent tracking fields
    if 'specialized_agents' not in new_state:
        logger.info("Initializing specialized agent tracking fields")
        new_state['specialized_agents'] = {}  # agent_type -> agent_id mapping
    
    if 'agent_invocations' not in new_state:
        new_state['agent_invocations'] = []
    
    # Initialize enhanced context flow fields
    if 'round_summaries' not in new_state:
        logger.info("Initializing round summary tracking")
        new_state['round_summaries'] = []
    
    if 'agent_contexts' not in new_state:
        new_state['agent_contexts'] = []
    
    # Initialize periodic HITL stop fields
    if 'periodic_stop_counter' not in new_state:
        logger.info("Initializing periodic stop tracking")
        new_state['periodic_stop_counter'] = 0
    
    if 'user_stop_history' not in new_state:
        new_state['user_stop_history'] = []
    
    # Migrate HITL state for enhanced approval types
    if 'hitl_state' in new_state:
        logger.info("Migrating HITL state for v1.3 approval types")
        if 'last_periodic_stop_round' not in new_state['hitl_state']:
            new_state['hitl_state']['last_periodic_stop_round'] = None
        
        if 'periodic_stop_responses' not in new_state['hitl_state']:
            new_state['hitl_state']['periodic_stop_responses'] = []
    
    return new_state


def detect_state_version(state: Dict[str, Any]) -> str:
    """Detect the version of a state dictionary.
    
    Args:
        state: The state dictionary
        
    Returns:
        The detected version string ('1.1' or '1.3')
    """
    # v1.3 has specialized agent tracking fields
    v13_fields = [
        'specialized_agents',
        'agent_invocations',
        'round_summaries',
        'periodic_stop_counter',
        'user_stop_history'
    ]
    
    if all(field in state for field in v13_fields):
        return '1.3'
    
    # Check for basic v1.1 fields
    v11_fields = ['session_id', 'current_phase', 'agents', 'messages']
    if all(field in state for field in v11_fields):
        return '1.1'
    
    return 'unknown'


def is_migration_needed(state: Dict[str, Any]) -> bool:
    """Check if a state needs migration from v1.1 to v1.3.
    
    Args:
        state: The state dictionary
        
    Returns:
        True if migration is needed, False otherwise
    """
    version = detect_state_version(state)
    return version == '1.1'


def validate_migrated_state(state: Dict[str, Any]) -> bool:
    """Validate that a migrated state has all required v1.3 fields.
    
    Args:
        state: The state dictionary
        
    Returns:
        True if the state is valid for v1.3, False otherwise
    """
    # Check for v1.3 specific fields
    required_v13_fields = [
        'specialized_agents',
        'agent_invocations',
        'round_summaries',
        'agent_contexts',
        'periodic_stop_counter',
        'user_stop_history'
    ]
    
    for field in required_v13_fields:
        if field not in state:
            logger.error(f"Missing required v1.3 field in state: {field}")
            return False
    
    # Validate HITL state enhancements
    if 'hitl_state' in state:
        hitl_state = state['hitl_state']
        if 'last_periodic_stop_round' not in hitl_state:
            logger.error("Missing last_periodic_stop_round in HITL state")
            return False
        if 'periodic_stop_responses' not in hitl_state:
            logger.error("Missing periodic_stop_responses in HITL state")
            return False
    
    # Validate types
    if not isinstance(state['specialized_agents'], dict):
        logger.error("specialized_agents must be a dictionary")
        return False
    
    if not isinstance(state['agent_invocations'], list):
        logger.error("agent_invocations must be a list")
        return False
    
    if not isinstance(state['periodic_stop_counter'], int):
        logger.error("periodic_stop_counter must be an integer")
        return False
    
    return True


def migrate_state_with_validation(old_state: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate a state with full validation.
    
    Args:
        old_state: The state dictionary (v1.1 or v1.3)
        
    Returns:
        A validated v1.3 state dictionary
        
    Raises:
        ValueError: If the state cannot be migrated or validated
    """
    # Check version
    version = detect_state_version(old_state)
    if version == 'unknown':
        raise ValueError(f"Unknown state version, cannot migrate. State keys: {list(old_state.keys())}")
    
    # Check if migration is needed
    if not is_migration_needed(old_state):
        logger.info("State is already v1.3, no migration needed")
        return old_state
    
    # Perform migration
    logger.info("Migrating state from v1.1 to v1.3")
    new_state = migrate_state_v1_to_v3(old_state)
    
    # Validate the result
    if not validate_migrated_state(new_state):
        raise ValueError("Migration failed: resulting state is invalid")
    
    logger.info("State migration completed successfully")
    return new_state


def create_initial_v13_state() -> Dict[str, Any]:
    """Create a fresh v1.3 state with all required fields initialized.
    
    This is useful for new sessions that start directly with v1.3.
    
    Returns:
        A properly initialized v1.3 state dictionary
    """
    return {
        # v1.3 specific fields
        'specialized_agents': {},
        'agent_invocations': [],
        'round_summaries': [],
        'agent_contexts': [],
        'periodic_stop_counter': 0,
        'user_stop_history': [],
        
        # Enhanced HITL state
        'hitl_state': {
            'awaiting_approval': False,
            'approval_type': None,
            'prompt_message': None,
            'options': None,
            'approval_history': [],
            'last_periodic_stop_round': None,
            'periodic_stop_responses': []
        }
    }


def extract_summary_from_messages(messages: List[Dict[str, Any]], round_number: int) -> Dict[str, Any]:
    """Extract a round summary from v1.1 message history.
    
    This is a helper function to create RoundSummary entries from
    existing message data during migration.
    
    Args:
        messages: List of messages from the round
        round_number: The round number
        
    Returns:
        A RoundSummary-compatible dictionary
    """
    # Concatenate all messages from the round
    summary_text = "\n".join([msg.get('content', '') for msg in messages])
    
    # Create a basic summary (in real implementation, this would use a summarizer)
    return {
        'round_number': round_number,
        'topic': messages[0].get('topic', 'Unknown') if messages else 'Unknown',
        'summary_text': f"Round {round_number} discussion summary (migration placeholder)",
        'created_by': 'migration_utility',
        'timestamp': datetime.now(),
        'token_count': len(summary_text.split()),  # Rough estimate
        'compression_ratio': 1.0  # Placeholder
    }