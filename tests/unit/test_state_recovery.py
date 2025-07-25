"""Unit tests for state recovery functionality."""

import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from virtual_agora.state.recovery import StateRecoveryManager, StateCheckpoint
from virtual_agora.state.manager import StateManager
from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.config.models import Config as VirtualAgoraConfig, ModeratorConfig, Provider
from virtual_agora.utils.exceptions import StateError, StateCorruptionError


@pytest.fixture
def sample_config():
    """Create a sample configuration."""
    return VirtualAgoraConfig(
        moderator=ModeratorConfig(
            provider=Provider.OPENAI,
            model="gpt-4o"
        ),
        agents=[
            {
                "provider": Provider.OPENAI,
                "model": "gpt-4o-mini",
                "count": 3,
            }
        ]
    )


@pytest.fixture
def state_manager(sample_config):
    """Create a state manager."""
    return StateManager(sample_config)


@pytest.fixture
def initialized_state(state_manager):
    """Create an initialized state."""
    return state_manager.initialize_state("test_session")


@pytest.fixture
def recovery_manager(tmp_path):
    """Create a recovery manager with temp directory."""
    return StateRecoveryManager(
        checkpoint_dir=tmp_path / "checkpoints",
        max_checkpoints=5,
        auto_checkpoint=True
    )


class TestStateCheckpoint:
    """Test StateCheckpoint class."""
    
    def test_checkpoint_creation(self):
        """Test checkpoint creation."""
        now = datetime.now()
        checkpoint = StateCheckpoint(
            checkpoint_id="cp_001",
            state_snapshot={"key": "value"},
            timestamp=now,
            operation="test_op",
            metadata={"user": "test"}
        )
        
        assert checkpoint.checkpoint_id == "cp_001"
        assert checkpoint.state_snapshot == {"key": "value"}
        assert checkpoint.timestamp == now
        assert checkpoint.operation == "test_op"
        assert checkpoint.metadata["user"] == "test"
    
    def test_checkpoint_to_dict(self):
        """Test checkpoint dictionary conversion."""
        now = datetime.now()
        checkpoint = StateCheckpoint(
            checkpoint_id="cp_001",
            state_snapshot={"key": "value"},
            timestamp=now,
            operation="test_op"
        )
        
        checkpoint_dict = checkpoint.to_dict()
        
        assert checkpoint_dict["checkpoint_id"] == "cp_001"
        assert checkpoint_dict["timestamp"] == now.isoformat()
        assert checkpoint_dict["operation"] == "test_op"
        assert checkpoint_dict["metadata"] == {}


class TestStateRecoveryManager:
    """Test StateRecoveryManager functionality."""
    
    def test_initialization(self, tmp_path):
        """Test recovery manager initialization."""
        checkpoint_dir = tmp_path / "checkpoints"
        manager = StateRecoveryManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=10,
            auto_checkpoint=False
        )
        
        assert manager.checkpoint_dir == checkpoint_dir
        assert manager.max_checkpoints == 10
        assert manager.auto_checkpoint is False
        assert checkpoint_dir.exists()
    
    def test_create_checkpoint(self, recovery_manager, initialized_state):
        """Test checkpoint creation."""
        checkpoint = recovery_manager.create_checkpoint(
            initialized_state,
            operation="test_checkpoint"
        )
        
        assert checkpoint.checkpoint_id == "checkpoint_000001"
        assert checkpoint.operation == "test_checkpoint"
        assert checkpoint.state_snapshot["session_id"] == "test_session"
        assert len(recovery_manager.checkpoints) == 1
    
    def test_checkpoint_persistence(self, recovery_manager, initialized_state):
        """Test checkpoint persistence to disk."""
        checkpoint = recovery_manager.create_checkpoint(
            initialized_state,
            operation="test_save",
            save_to_disk=True
        )
        
        # Check file exists
        checkpoint_file = recovery_manager.checkpoint_dir / f"{checkpoint.checkpoint_id}.pkl"
        assert checkpoint_file.exists()
        
        # Load and verify
        with open(checkpoint_file, "rb") as f:
            loaded_checkpoint = pickle.load(f)
        
        assert loaded_checkpoint.checkpoint_id == checkpoint.checkpoint_id
        assert loaded_checkpoint.state_snapshot["session_id"] == "test_session"
    
    def test_max_checkpoints_limit(self, recovery_manager, initialized_state):
        """Test that max checkpoints limit is enforced."""
        # Create more checkpoints than the limit
        for i in range(7):
            recovery_manager.create_checkpoint(
                initialized_state,
                operation=f"checkpoint_{i}"
            )
        
        # Should only keep max_checkpoints (5)
        assert len(recovery_manager.checkpoints) == 5
        
        # Oldest checkpoints should be removed
        checkpoint_ids = [cp.checkpoint_id for cp in recovery_manager.checkpoints]
        assert "checkpoint_000001" not in checkpoint_ids
        assert "checkpoint_000002" not in checkpoint_ids
        assert "checkpoint_000007" in checkpoint_ids
    
    def test_rollback_to_checkpoint(self, recovery_manager, state_manager):
        """Test state rollback functionality."""
        # Initialize and modify state
        state1 = state_manager.initialize_state("session_1")
        checkpoint1 = recovery_manager.create_checkpoint(state1, "initial")
        
        # Modify state
        state_manager.state["session_id"] = "modified_session"
        state_manager.state["total_messages"] = 10
        
        # Create another checkpoint
        checkpoint2 = recovery_manager.create_checkpoint(
            state_manager.state,
            "modified"
        )
        
        # Rollback to first checkpoint
        restored_state = recovery_manager.rollback_to_checkpoint(
            checkpoint1.checkpoint_id,
            state_manager
        )
        
        assert restored_state["session_id"] == "session_1"
        assert restored_state["total_messages"] == 0
        assert state_manager.state["session_id"] == "session_1"
    
    def test_rollback_to_nonexistent_checkpoint(self, recovery_manager, state_manager):
        """Test rollback to non-existent checkpoint."""
        with pytest.raises(StateError, match="Checkpoint not found"):
            recovery_manager.rollback_to_checkpoint(
                "nonexistent_checkpoint",
                state_manager
            )
    
    def test_get_latest_checkpoint(self, recovery_manager, initialized_state):
        """Test getting the latest checkpoint."""
        # No checkpoints
        assert recovery_manager.get_latest_checkpoint() is None
        
        # Create checkpoints
        cp1 = recovery_manager.create_checkpoint(initialized_state, "first")
        cp2 = recovery_manager.create_checkpoint(initialized_state, "second")
        cp3 = recovery_manager.create_checkpoint(initialized_state, "third")
        
        latest = recovery_manager.get_latest_checkpoint()
        assert latest.checkpoint_id == cp3.checkpoint_id
    
    def test_validate_state(self, recovery_manager, initialized_state):
        """Test state validation."""
        # Valid state
        is_valid, warnings = recovery_manager.validate_state(initialized_state)
        assert is_valid
        assert len(warnings) == 0
        
        # Create invalid state
        invalid_state = dict(initialized_state)
        invalid_state["total_messages"] = 5
        invalid_state["messages"] = []  # Mismatch with total_messages
        
        is_valid, warnings = recovery_manager.validate_state(invalid_state)
        assert not is_valid
        assert len(warnings) > 0
        assert any("message count mismatch" in w.lower() for w in warnings)
    
    def test_repair_corrupted_state(self, recovery_manager):
        """Test state repair functionality."""
        # Create corrupted state
        corrupted_state = {
            "total_messages": 10,
            "messages": [
                {"speaker_id": "agent1"},
                {"speaker_id": "agent2"},
            ],
            "messages_by_agent": {"agent1": 5, "agent2": 5},  # Wrong counts
            "agents": {
                "agent1": {"message_count": 5},
                "agent2": {"message_count": 5},
            },
            "vote_history": [
                {"id": "vote1", "status": "active"},  # Should be completed
            ],
            "topics_info": {
                "topic1": {"status": "active"},
                "topic2": {"status": "active"},  # Only one should be active
            },
            "active_topic": "topic1",
        }
        
        warnings = [
            "Message count mismatch: recorded=10, actual=2",
            "Incomplete vote round vote1",
            "Topic 'topic2' marked as active but not current",
        ]
        
        repaired = recovery_manager.repair_corrupted_state(
            corrupted_state,
            warnings
        )
        
        assert repaired is not None
        assert repaired["total_messages"] == 2
        assert repaired["messages_by_agent"]["agent1"] == 1
        assert repaired["messages_by_agent"]["agent2"] == 1
        assert repaired["agents"]["agent1"]["message_count"] == 1
        assert repaired["vote_history"][0]["status"] == "failed"
        assert repaired["topics_info"]["topic2"]["status"] == "paused"
    
    def test_find_last_valid_state(self, recovery_manager, state_manager):
        """Test finding the last valid checkpoint."""
        # Create some checkpoints
        state1 = state_manager.initialize_state("session_1")
        cp1 = recovery_manager.create_checkpoint(state1, "valid1")
        
        # Create corrupted state
        state_manager.state["total_messages"] = 100
        state_manager.state["messages"] = []  # Corrupt
        cp2 = recovery_manager.create_checkpoint(state_manager.state, "corrupt")
        
        # Create another valid state
        state3 = state_manager.initialize_state("session_3")
        cp3 = recovery_manager.create_checkpoint(state3, "valid2")
        
        # Find last valid should return cp3
        last_valid = recovery_manager.find_last_valid_state()
        assert last_valid is not None
        assert last_valid.checkpoint_id == cp3.checkpoint_id
    
    def test_emergency_recovery(self, recovery_manager, state_manager):
        """Test emergency recovery functionality."""
        # Create initial valid state
        state1 = state_manager.initialize_state("session_1")
        cp1 = recovery_manager.create_checkpoint(state1, "valid", save_to_disk=True)
        
        # Corrupt the state
        state_manager.state["total_messages"] = 1000
        state_manager.state["messages"] = []
        
        # Simulate critical error
        error = StateCorruptionError(
            "State is corrupted",
            corrupted_fields=["messages"]
        )
        
        # Attempt emergency recovery
        success = recovery_manager.emergency_recovery(state_manager, error)
        
        assert success
        # State should be rolled back to last valid
        assert state_manager.state["total_messages"] == 0
        assert state_manager.state["session_id"] == "session_1"
    
    def test_emergency_recovery_reinitialize(self, recovery_manager):
        """Test emergency recovery with reinitialization."""
        # Create state manager with corrupted state
        state_manager = Mock()
        state_manager._state = {
            "session_id": "corrupted_session",
            "total_messages": "invalid",  # Invalid type
        }
        
        # Mock validation to fail
        with patch.object(recovery_manager, 'validate_state', return_value=(False, ["errors"])):
            with patch.object(recovery_manager, 'repair_corrupted_state', return_value=None):
                with patch.object(recovery_manager, 'find_last_valid_state', return_value=None):
                    # Mock initialize_state
                    state_manager.initialize_state = Mock(return_value={"session_id": "new_session"})
                    
                    error = StateCorruptionError("Critical corruption")
                    success = recovery_manager.emergency_recovery(state_manager, error)
                    
                    assert success
                    state_manager.initialize_state.assert_called_once()
    
    def test_checkpoint_file_operations(self, recovery_manager, initialized_state):
        """Test checkpoint file operations."""
        # Save checkpoint
        checkpoint = recovery_manager.create_checkpoint(
            initialized_state,
            operation="file_test",
            save_to_disk=True
        )
        
        checkpoint_file = recovery_manager.checkpoint_dir / f"{checkpoint.checkpoint_id}.pkl"
        assert checkpoint_file.exists()
        
        # Test loading
        loaded = recovery_manager._load_checkpoint_from_disk(checkpoint.checkpoint_id)
        assert loaded is not None
        assert loaded.checkpoint_id == checkpoint.checkpoint_id
        
        # Test deletion
        recovery_manager._delete_checkpoint_file(checkpoint.checkpoint_id)
        assert not checkpoint_file.exists()
    
    def test_recovery_statistics(self, recovery_manager, initialized_state):
        """Test recovery statistics generation."""
        # No checkpoints
        stats = recovery_manager.get_recovery_stats()
        assert stats["total_checkpoints"] == 0
        assert stats["checkpoint_rate"] == 0.0
        
        # Create checkpoints
        for i in range(3):
            recovery_manager.create_checkpoint(
                initialized_state,
                f"checkpoint_{i}",
                save_to_disk=(i == 0)  # Save first one to disk
            )
        
        stats = recovery_manager.get_recovery_stats()
        assert stats["total_checkpoints"] == 3
        assert stats["disk_checkpoints"] == 1
        assert stats["checkpoint_rate"] > 0
        assert stats["latest_checkpoint"] is not None
        assert stats["latest_checkpoint"]["checkpoint_id"] == "checkpoint_000003"


class TestStateRecoveryIntegration:
    """Test integration with state management."""
    
    def test_checkpoint_before_risky_operation(self, recovery_manager, state_manager):
        """Test creating checkpoint before risky operations."""
        # Initialize state
        state = state_manager.initialize_state("test_session")
        
        # Checkpoint before risky operation
        checkpoint = recovery_manager.create_checkpoint(
            state,
            operation="before_risky_op",
            save_to_disk=True
        )
        
        # Simulate risky operation that corrupts state
        try:
            state_manager.state["agents"] = "invalid"  # Corrupt
            state_manager.state["messages"] = None  # Corrupt
            raise StateCorruptionError("Operation failed")
        except StateCorruptionError:
            # Rollback on failure
            recovery_manager.rollback_to_checkpoint(
                checkpoint.checkpoint_id,
                state_manager
            )
        
        # State should be restored
        assert isinstance(state_manager.state["agents"], dict)
        assert isinstance(state_manager.state["messages"], list)
    
    def test_progressive_checkpointing(self, recovery_manager, state_manager):
        """Test progressive checkpointing during session."""
        # Initialize
        state = state_manager.initialize_state("test_session")
        cp1 = recovery_manager.create_checkpoint(state, "initialization")
        
        # Phase 1
        state_manager.transition_phase(1)
        cp2 = recovery_manager.create_checkpoint(state_manager.state, "phase_1")
        
        # Add messages
        state_manager.state["current_speaker_id"] = "moderator"
        state_manager.add_message("moderator", "Welcome!")
        cp3 = recovery_manager.create_checkpoint(state_manager.state, "first_message")
        
        # Verify we can rollback to any point
        recovery_manager.rollback_to_checkpoint(cp2.checkpoint_id, state_manager)
        assert state_manager.state["current_phase"] == 1
        assert state_manager.state["total_messages"] == 0
        
        recovery_manager.rollback_to_checkpoint(cp1.checkpoint_id, state_manager)
        assert state_manager.state["current_phase"] == 0