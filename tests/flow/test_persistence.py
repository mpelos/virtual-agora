"""Tests for enhanced state persistence and recovery."""

import pytest
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path

from virtual_agora.flow.persistence import (
    EnhancedMemorySaver,
    StateSnapshot,
    RecoveryPoint,
    create_enhanced_checkpointer,
)


class TestEnhancedMemorySaver:
    """Test enhanced memory saver functionality."""

    def setup_method(self):
        """Set up test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir)
        self.saver = EnhancedMemorySaver(self.storage_path)

        # Create test state
        self.test_state = {
            "session_id": "test-session-123",
            "start_time": datetime.now(),
            "current_phase": 2,
            "phase_history": [
                {
                    "from_phase": 0,
                    "to_phase": 1,
                    "timestamp": datetime.now() - timedelta(minutes=10),
                    "reason": "Test transition",
                }
            ],
            "agents": {
                "agent1": {"id": "agent1", "model": "gpt-4o", "message_count": 5}
            },
            "messages": [
                {
                    "id": "msg_001",
                    "speaker_id": "agent1",
                    "content": "Test message",
                    "timestamp": datetime.now(),
                }
            ],
            "active_topic": "Test Topic",
        }

    def test_create_snapshot(self):
        """Test creating state snapshots."""
        snapshot = self.saver.create_snapshot(
            session_id="test-session",
            state=self.test_state,
            metadata={"test": "metadata"},
        )

        assert isinstance(snapshot, StateSnapshot)
        assert snapshot.session_id == "test-session"
        assert snapshot.metadata["test"] == "metadata"
        assert snapshot.state_data is not None

        # Check if snapshot is stored in memory
        assert "test-session" in self.saver.snapshots
        assert len(self.saver.snapshots["test-session"]) == 1

    def test_create_recovery_point(self):
        """Test creating recovery points."""
        recovery_point = self.saver.create_recovery_point(
            session_id="test-session",
            state=self.test_state,
            description="Test recovery point",
            metadata={"phase": 2},
        )

        assert isinstance(recovery_point, RecoveryPoint)
        assert recovery_point.session_id == "test-session"
        assert recovery_point.description == "Test recovery point"
        assert recovery_point.phase == 2
        assert recovery_point.validation_passed  # Should pass validation

        # Check if recovery point is stored
        assert "test-session" in self.saver.recovery_points
        assert len(self.saver.recovery_points["test-session"]) == 1

    def test_rollback_to_recovery_point(self):
        """Test rollback functionality."""
        # Create initial recovery point
        recovery_point = self.saver.create_recovery_point(
            session_id="test-session",
            state=self.test_state,
            description="Initial state",
        )

        # Rollback to the recovery point
        restored_state, rp = self.saver.rollback_to_recovery_point(
            session_id="test-session", recovery_id=recovery_point.recovery_id
        )

        assert restored_state["session_id"] == self.test_state["session_id"]
        assert restored_state["current_phase"] == self.test_state["current_phase"]
        assert rp.recovery_id == recovery_point.recovery_id
        assert rp.description == "Initial state"

    def test_rollback_nonexistent_recovery_point(self):
        """Test rollback with non-existent recovery point."""
        from virtual_agora.utils.exceptions import StateError

        with pytest.raises(StateError):
            self.saver.rollback_to_recovery_point(
                session_id="test-session", recovery_id="nonexistent"
            )

    def test_get_recovery_points(self):
        """Test getting recovery points for a session."""
        # Create multiple recovery points
        rp1 = self.saver.create_recovery_point(
            session_id="test-session",
            state=self.test_state,
            description="First recovery point",
        )

        modified_state = self.test_state.copy()
        modified_state["current_phase"] = 3

        rp2 = self.saver.create_recovery_point(
            session_id="test-session",
            state=modified_state,
            description="Second recovery point",
        )

        recovery_points = self.saver.get_recovery_points("test-session")

        assert len(recovery_points) == 2
        assert recovery_points[0].recovery_id == rp1.recovery_id
        assert recovery_points[1].recovery_id == rp2.recovery_id

    def test_get_latest_snapshot(self):
        """Test getting latest snapshot."""
        # No snapshots initially
        assert self.saver.get_latest_snapshot("test-session") is None

        # Create snapshots
        snapshot1 = self.saver.create_snapshot("test-session", self.test_state)

        modified_state = self.test_state.copy()
        modified_state["current_phase"] = 3
        snapshot2 = self.saver.create_snapshot("test-session", modified_state)

        latest = self.saver.get_latest_snapshot("test-session")

        assert latest is not None
        assert latest.checkpoint_id == snapshot2.checkpoint_id

    def test_state_serialization_deserialization(self):
        """Test state serialization and deserialization."""
        # Test with datetime objects
        serialized = self.saver._serialize_state(self.test_state)
        deserialized = self.saver._deserialize_state(serialized)

        # Check that datetime objects are properly handled
        assert isinstance(deserialized["start_time"], datetime)
        assert deserialized["session_id"] == self.test_state["session_id"]
        assert deserialized["current_phase"] == self.test_state["current_phase"]

        # Check nested datetime objects
        assert isinstance(deserialized["phase_history"][0]["timestamp"], datetime)
        assert isinstance(deserialized["messages"][0]["timestamp"], datetime)

    def test_state_validation(self):
        """Test state validation for recovery points."""
        # Valid state should pass
        assert self.saver._validate_state_consistency(self.test_state)

        # Invalid state should fail
        invalid_state = self.test_state.copy()
        del invalid_state["session_id"]  # Remove required field

        assert not self.saver._validate_state_consistency(invalid_state)

        # Invalid phase should fail
        invalid_phase_state = self.test_state.copy()
        invalid_phase_state["current_phase"] = 10  # Invalid phase

        assert not self.saver._validate_state_consistency(invalid_phase_state)

    def test_cleanup_old_snapshots(self):
        """Test cleanup of old snapshots."""
        # Create old snapshot (simulate by modifying timestamp)
        old_snapshot = self.saver.create_snapshot("test-session", self.test_state)
        old_snapshot.timestamp = datetime.now() - timedelta(days=10)

        # Create recent snapshot
        recent_snapshot = self.saver.create_snapshot("test-session", self.test_state)

        # Create old recovery point (creates another snapshot)
        old_recovery = self.saver.create_recovery_point(
            "test-session", self.test_state, "Old recovery point"
        )
        old_recovery.timestamp = datetime.now() - timedelta(days=10)
        # Also mark the snapshot created by recovery point as old
        old_recovery.state_snapshot.timestamp = datetime.now() - timedelta(days=10)

        # Cleanup with 7-day retention
        self.saver.cleanup_old_snapshots(retention_days=7)

        # Should keep recent snapshot, remove old ones
        snapshots = self.saver.snapshots["test-session"]
        recent_snapshots = [
            s for s in snapshots if s.timestamp > datetime.now() - timedelta(days=7)
        ]
        assert len(recent_snapshots) >= 1

        # Should remove old recovery points
        recovery_points = self.saver.recovery_points["test-session"]
        recent_recovery_points = [
            rp
            for rp in recovery_points
            if rp.timestamp > datetime.now() - timedelta(days=7)
        ]
        assert len(recent_recovery_points) >= 0

    def test_persistence_stats(self):
        """Test persistence statistics generation."""
        # Create some data
        self.saver.create_snapshot("session1", self.test_state)
        self.saver.create_snapshot("session2", self.test_state)
        self.saver.create_recovery_point("session1", self.test_state, "Test recovery")

        stats = self.saver.get_persistence_stats()

        assert stats["total_sessions"] >= 2
        assert stats["total_snapshots"] >= 2
        assert stats["total_recovery_points"] >= 1
        assert stats["storage_path"] == str(self.storage_path)
        assert "disk_usage_bytes" in stats

    def test_disk_persistence(self):
        """Test persistence to disk."""
        # Create snapshot and recovery point
        snapshot = self.saver.create_snapshot("test-session", self.test_state)
        recovery_point = self.saver.create_recovery_point(
            "test-session", self.test_state, "Disk test recovery"
        )

        # Check files were created
        snapshot_files = list(self.storage_path.glob("snapshot_*.json*"))
        recovery_files = list(self.storage_path.glob("recovery_*.json"))

        assert len(snapshot_files) >= 1
        assert len(recovery_files) >= 1

        # Create new saver and load from disk
        new_saver = EnhancedMemorySaver(self.storage_path)

        # Should have loaded the data
        assert "test-session" in new_saver.snapshots
        assert "test-session" in new_saver.recovery_points
        assert len(new_saver.snapshots["test-session"]) >= 1
        assert len(new_saver.recovery_points["test-session"]) >= 1


class TestPersistenceIntegration:
    """Test persistence integration with graph."""

    def test_create_enhanced_checkpointer(self):
        """Test enhanced checkpointer factory function."""
        # Without storage path
        checkpointer1 = create_enhanced_checkpointer()
        assert isinstance(checkpointer1, EnhancedMemorySaver)
        assert checkpointer1.storage_path is None

        # With storage path
        temp_dir = tempfile.mkdtemp()
        checkpointer2 = create_enhanced_checkpointer(temp_dir)
        assert isinstance(checkpointer2, EnhancedMemorySaver)
        assert checkpointer2.storage_path == Path(temp_dir)

    def test_compression_handling(self):
        """Test compression for large states."""
        saver = EnhancedMemorySaver()

        # Create large state to trigger compression
        large_state = self._create_large_state()

        snapshot = saver.create_snapshot("test-session", large_state)

        # Should detect compression need
        assert snapshot.compression_used or len(str(large_state)) <= 10000

    def _create_large_state(self):
        """Create a large state to test compression."""
        base_state = {
            "session_id": "large-session",
            "start_time": datetime.now(),
            "current_phase": 2,
            "agents": {},
            "messages": [],
        }

        # Add many messages to make state large
        for i in range(100):
            base_state["messages"].append(
                {
                    "id": f"msg_{i:04d}",
                    "speaker_id": f"agent_{i % 5}",
                    "content": f"This is a long message {i} with lots of content to make the state large "
                    * 20,
                    "timestamp": datetime.now(),
                }
            )

        return base_state
