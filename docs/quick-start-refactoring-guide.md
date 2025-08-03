# Quick Start: Architecture Refactoring Implementation

## Immediate Next Steps (Start Here)

### Prerequisites
- [ ] Read `architecture-refactoring-overview.md`
- [ ] Review current `nodes_v13.py` structure
- [ ] Understand existing round numbering issue
- [ ] Set up development branch: `git checkout -b architecture/refactor-phase1`

### Step 1: Create Round Manager (Start Here - 2 hours)

**File**: `src/virtual_agora/flow/round_manager.py`

```python
"""Centralized round state and transition management."""

from typing import Dict, Any
from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)

class RoundManager:
    """Manages round state and transitions consistently across the system."""
    
    def get_current_round(self, state: VirtualAgoraState) -> int:
        """Get current round number with consistent logic.
        
        This replaces scattered round logic throughout the codebase.
        """
        return state.get("current_round", 0)
    
    def get_next_round_number(self, state: VirtualAgoraState) -> int:
        """Get the next round number for new discussions."""
        return self.get_current_round(state) + 1
    
    def start_new_round(self, state: VirtualAgoraState) -> int:
        """Increment round and return new round number.
        
        Use this when actually starting a new discussion round.
        """
        new_round = self.get_next_round_number(state)
        logger.info(f"Starting discussion round {new_round}")
        return new_round
    
    def should_show_user_participation(self, state: VirtualAgoraState) -> bool:
        """Determine if user participation should be shown.
        
        User participation is shown from round 1 onwards.
        """
        current_round = self.get_current_round(state)
        return current_round >= 1
    
    def create_round_metadata(self, state: VirtualAgoraState, round_num: int) -> Dict[str, Any]:
        """Create round-specific metadata for context building."""
        return {
            "round_number": round_num,
            "topic": state.get("active_topic"),
            "theme": state.get("main_topic"),
            "speaking_order": state.get("speaking_order", []),
        }
```

**Test File**: `tests/flow/test_round_manager.py`

```python
"""Test round manager functionality."""

import pytest
from virtual_agora.flow.round_manager import RoundManager

class TestRoundManager:
    
    def setup_method(self):
        self.round_manager = RoundManager()
    
    def test_get_current_round_default(self):
        """Test getting current round with default state."""
        state = {}
        assert self.round_manager.get_current_round(state) == 0
    
    def test_get_current_round_with_value(self):
        """Test getting current round with existing value."""
        state = {"current_round": 3}
        assert self.round_manager.get_current_round(state) == 3
    
    def test_get_next_round_number(self):
        """Test getting next round number."""
        state = {"current_round": 2}
        assert self.round_manager.get_next_round_number(state) == 3
    
    def test_start_new_round(self):
        """Test starting new round."""
        state = {"current_round": 1}
        new_round = self.round_manager.start_new_round(state)
        assert new_round == 2
    
    def test_should_show_user_participation(self):
        """Test user participation logic."""
        # Round 0 - no participation
        state = {"current_round": 0}
        assert not self.round_manager.should_show_user_participation(state)
        
        # Round 1+ - show participation
        state = {"current_round": 1}
        assert self.round_manager.should_show_user_participation(state)
        
        state = {"current_round": 5}
        assert self.round_manager.should_show_user_participation(state)
    
    def test_create_round_metadata(self):
        """Test round metadata creation."""
        state = {
            "current_round": 2,
            "active_topic": "Test Topic",
            "main_topic": "Test Theme",
            "speaking_order": ["agent1", "agent2"]
        }
        
        metadata = self.round_manager.create_round_metadata(state, 2)
        
        assert metadata["round_number"] == 2
        assert metadata["topic"] == "Test Topic"
        assert metadata["theme"] == "Test Theme"
        assert metadata["speaking_order"] == ["agent1", "agent2"]
```

### Step 2: Create Message Coordinator (Next - 3 hours)

**File**: `src/virtual_agora/flow/message_coordinator.py`

```python
"""Message coordination and assembly for discussion flow."""

from typing import List, Tuple, Dict, Any
from datetime import datetime
from langchain_core.messages import HumanMessage, BaseMessage

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.context.message_processor import MessageProcessor, ProcessedMessage
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)

class MessageCoordinator:
    """Coordinates message assembly and routing for agents."""
    
    def store_user_message(
        self, 
        content: str, 
        round_num: int, 
        topic: str,
        state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """Store user participation message with correct round number.
        
        This fixes the round numbering issue by storing user messages
        with the round they're intended for, not the next round.
        """
        user_msg = HumanMessage(
            content=content,
            additional_kwargs={
                "speaker_id": "user",
                "speaker_role": "user", 
                "round": round_num,  # Store with intended round number
                "topic": topic,
                "timestamp": datetime.now().isoformat(),
                "participation_type": "user_turn_participation",
            },
        )
        
        logger.info(f"Storing user message for round {round_num}: {content[:100]}...")
        
        return {
            "messages": [user_msg],
            "user_participation_message": content,
            "user_participated_in_round": round_num,
        }
    
    def get_context_for_agent(
        self,
        agent_id: str,
        round_num: int,
        topic: str,
        state: VirtualAgoraState
    ) -> Tuple[List[ProcessedMessage], List[ProcessedMessage]]:
        """Get user messages and colleague messages for agent context.
        
        Returns:
            Tuple of (user_messages, colleague_messages)
        """
        all_messages = state.get("messages", [])
        
        # Get user messages from previous rounds (not excluding current round)
        # This allows user messages stored with current round to be included
        user_messages = MessageProcessor.filter_user_participation_messages(
            all_messages, topic, exclude_round=None  # Don't exclude any rounds
        )
        
        # Filter user messages to only include those for previous rounds
        # User message for current round should be included
        relevant_user_messages = [
            msg for msg in user_messages 
            if msg.round_number <= round_num
        ]
        
        # Get current round colleague messages
        colleague_messages = MessageProcessor.filter_messages_by_round(
            all_messages, round_num, topic
        )
        
        logger.debug(f"Agent {agent_id}: {len(relevant_user_messages)} user messages, "
                    f"{len(colleague_messages)} colleague messages")
        
        return relevant_user_messages, colleague_messages
    
    def create_context_messages_for_agent(
        self,
        agent_id: str,
        round_num: int,
        topic: str,
        state: VirtualAgoraState
    ) -> List[BaseMessage]:
        """Create formatted context messages for agent."""
        user_messages, colleague_messages = self.get_context_for_agent(
            agent_id, round_num, topic, state
        )
        
        return MessageProcessor.create_context_messages_for_agent(
            colleague_messages, user_messages, agent_id
        )
```

### Step 3: Quick Integration Test (30 minutes)

**File**: `tests/integration/test_quick_refactor_validation.py`

```python
"""Quick validation that refactored components work correctly."""

import pytest
from virtual_agora.flow.round_manager import RoundManager
from virtual_agora.flow.message_coordinator import MessageCoordinator

class TestQuickRefactorValidation:
    
    def setup_method(self):
        self.round_manager = RoundManager()
        self.message_coordinator = MessageCoordinator()
    
    def test_user_message_round_numbering_fix(self):
        """Test that user messages get correct round numbers."""
        # Simulate user participating for round 2
        state = {"current_round": 1, "active_topic": "Test Topic"}
        target_round = 2
        
        # Store user message for round 2
        updates = self.message_coordinator.store_user_message(
            "User guidance for round 2", 
            target_round, 
            "Test Topic",
            state
        )
        
        # Verify message is stored with correct round
        user_msg = updates["messages"][0]
        assert user_msg.additional_kwargs["round"] == 2
        assert user_msg.additional_kwargs["participation_type"] == "user_turn_participation"
    
    def test_round_manager_consistency(self):
        """Test that round manager provides consistent round numbering."""
        state = {"current_round": 3}
        
        assert self.round_manager.get_current_round(state) == 3
        assert self.round_manager.get_next_round_number(state) == 4
        assert self.round_manager.start_new_round(state) == 4
    
    def test_integration_user_participation_flow(self):
        """Test integrated user participation flow."""
        # Initial state
        state = {
            "current_round": 1,
            "active_topic": "Test Topic",
            "messages": []
        }
        
        # User wants to participate in round 2
        target_round = self.round_manager.get_next_round_number(state)
        
        # Store user message
        updates = self.message_coordinator.store_user_message(
            "Please focus on implementation details",
            target_round,
            state["active_topic"],
            state
        )
        
        # Update state with user message
        updated_state = state.copy()
        updated_state["messages"] = updates["messages"]
        updated_state["current_round"] = target_round
        
        # Get context for agent in round 2
        user_messages, colleague_messages = self.message_coordinator.get_context_for_agent(
            "agent1", target_round, state["active_topic"], updated_state
        )
        
        # Verify user message is available
        assert len(user_messages) == 1
        assert user_messages[0].content == "Please focus on implementation details"
        assert user_messages[0].round_number == target_round
```

### Step 4: Immediate Integration (1 hour)

**Modify**: `src/virtual_agora/flow/nodes_v13.py` (temporary integration)

Add this at the top of the file:
```python
from virtual_agora.flow.round_manager import RoundManager
from virtual_agora.flow.message_coordinator import MessageCoordinator

# Add to V13FlowNodes.__init__():
def __init__(self, ...):
    # ... existing code ...
    
    # Add new components
    self.round_manager = RoundManager()
    self.message_coordinator = MessageCoordinator()
```

Update `user_turn_participation_node` method:
```python
def user_turn_participation_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
    """HITL node for Round Moderator between rounds."""
    logger.info("=== USER_TURN_PARTICIPATION NODE START ===")
    
    current_round = self.round_manager.get_current_round(state)
    next_round = self.round_manager.get_next_round_number(state)
    current_topic = state.get("active_topic", "Unknown Topic")
    
    # ... existing HITL code ...
    
    if action == "participate" and user_message:
        # Use message coordinator for proper round numbering
        updates.update(self.message_coordinator.store_user_message(
            user_message, 
            next_round,  # Store for the round user is providing guidance for
            current_topic,
            state
        ))
    
    # ... rest of existing code ...
```

### Step 5: Run Tests (15 minutes)

```bash
# Run new tests
pytest tests/flow/test_round_manager.py -v
pytest tests/integration/test_quick_refactor_validation.py -v

# Run existing tests to ensure no regression
pytest tests/test_discussion_flow_end_to_end.py -v
pytest tests/test_user_participation_flow.py -v
```

## Expected Results After Step 5

1. **User messages stored with correct round numbers**
2. **Consistent round numbering across components**
3. **Foundation for configurable user participation timing**
4. **No functional regression**

## Configuration-Based User Participation (Future Extension)

After completing the full refactoring, changing user participation timing will be as simple as:

```python
# In configuration file - SINGLE LINE CHANGE
config.user_participation_timing = ParticipationTiming.START_OF_ROUND
# Alternative: ParticipationTiming.END_OF_ROUND

# No graph changes, no node restructuring, no flow modifications needed
```

This addresses the architectural requirement that **"it must be easy to change user discussion participation from end of round to start of round"**.

## Next Steps After Quick Start

1. **Continue with Step 1.3** from `architecture-refactoring-steps.md`
2. **Create pull request** for Phase 1 components
3. **Begin Phase 2** flow restructuring
4. **Gradually migrate** remaining components

## Troubleshooting

### Common Issues

**Import errors**:
```bash
# Make sure new files are in Python path
export PYTHONPATH="${PYTHONPATH}:src"
```

**Test failures**:
- Check that new components integrate correctly
- Verify round numbering logic matches expectations
- Ensure message metadata is preserved

**Integration issues**:
- Start with small changes
- Test each component independently
- Use existing test scenarios to validate

### Rollback Plan

If issues arise:
```bash
git stash  # Save current work
git checkout main  # Return to stable state
git checkout -b architecture/debug-issues  # Create debug branch
git stash pop  # Restore work for debugging
```

---

**Time Investment**: ~6 hours for immediate improvement  
**Payoff**: Foundation for architectural improvements and immediate fix for user participation  
**Next Milestone**: Complete Phase 1 refactoring (1 week)