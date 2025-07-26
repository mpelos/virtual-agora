"""Enhanced State Persistence and Recovery for Virtual Agora.

This module provides comprehensive state persistence capabilities beyond
basic LangGraph checkpointing, including recovery strategies and state validation.
"""

import json
import pickle
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.utils.logging import get_logger
from virtual_agora.utils.exceptions import StateError

logger = get_logger(__name__)


@dataclass
class StateSnapshot:
    """A comprehensive state snapshot with metadata."""

    timestamp: datetime
    session_id: str
    checkpoint_id: str
    state_data: Dict[str, Any]
    metadata: Dict[str, Any]
    version: str = "1.0"
    compression_used: bool = False


@dataclass
class RecoveryPoint:
    """A recovery point with validation and rollback capability."""

    recovery_id: str
    timestamp: datetime
    session_id: str
    phase: int
    description: str
    state_snapshot: StateSnapshot
    validation_passed: bool
    recovery_metadata: Dict[str, Any]


class EnhancedMemorySaver(MemorySaver):
    """Enhanced memory saver with additional persistence features."""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize enhanced memory saver.

        Args:
            storage_path: Optional path for persistent storage
        """
        super().__init__()
        self.storage_path = storage_path
        self.snapshots: Dict[str, List[StateSnapshot]] = {}
        self.recovery_points: Dict[str, List[RecoveryPoint]] = {}

        if storage_path:
            storage_path.mkdir(parents=True, exist_ok=True)
            self._load_persistent_data()

    def create_snapshot(
        self,
        session_id: str,
        state: VirtualAgoraState,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StateSnapshot:
        """Create a comprehensive state snapshot.

        Args:
            session_id: Session ID
            state: Current state
            metadata: Additional metadata

        Returns:
            Created snapshot
        """
        checkpoint_id = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        snapshot = StateSnapshot(
            timestamp=datetime.now(),
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            state_data=self._serialize_state(state),
            metadata=metadata or {},
            compression_used=len(str(state)) > 10000,  # Compress large states
        )

        # Store in memory
        if session_id not in self.snapshots:
            self.snapshots[session_id] = []
        self.snapshots[session_id].append(snapshot)

        # Persist to disk if configured
        if self.storage_path:
            self._save_snapshot_to_disk(snapshot)

        logger.info(f"Created state snapshot {checkpoint_id} for session {session_id}")
        return snapshot

    def create_recovery_point(
        self,
        session_id: str,
        state: VirtualAgoraState,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RecoveryPoint:
        """Create a recovery point for rollback capability.

        Args:
            session_id: Session ID
            state: Current state
            description: Description of recovery point
            metadata: Additional metadata

        Returns:
            Created recovery point
        """
        recovery_id = f"recovery_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Create snapshot for this recovery point
        snapshot = self.create_snapshot(
            session_id,
            state,
            {"recovery_point": True, "description": description, **(metadata or {})},
        )

        # Validate state consistency
        validation_passed = self._validate_state_consistency(state)

        recovery_point = RecoveryPoint(
            recovery_id=recovery_id,
            timestamp=datetime.now(),
            session_id=session_id,
            phase=state["current_phase"],
            description=description,
            state_snapshot=snapshot,
            validation_passed=validation_passed,
            recovery_metadata=metadata or {},
        )

        # Store recovery point
        if session_id not in self.recovery_points:
            self.recovery_points[session_id] = []
        self.recovery_points[session_id].append(recovery_point)

        # Persist to disk if configured
        if self.storage_path:
            self._save_recovery_point_to_disk(recovery_point)

        logger.info(f"Created recovery point {recovery_id}: {description}")
        return recovery_point

    def get_recovery_points(self, session_id: str) -> List[RecoveryPoint]:
        """Get all recovery points for a session.

        Args:
            session_id: Session ID

        Returns:
            List of recovery points
        """
        return self.recovery_points.get(session_id, [])

    def rollback_to_recovery_point(
        self, session_id: str, recovery_id: str
    ) -> Tuple[VirtualAgoraState, RecoveryPoint]:
        """Rollback state to a specific recovery point.

        Args:
            session_id: Session ID
            recovery_id: Recovery point ID

        Returns:
            Tuple of (restored state, recovery point)

        Raises:
            StateError: If recovery point not found
        """
        recovery_points = self.recovery_points.get(session_id, [])

        recovery_point = None
        for rp in recovery_points:
            if rp.recovery_id == recovery_id:
                recovery_point = rp
                break

        if not recovery_point:
            raise StateError(
                f"Recovery point {recovery_id} not found for session {session_id}"
            )

        # Deserialize state from snapshot
        restored_state = self._deserialize_state(
            recovery_point.state_snapshot.state_data
        )

        logger.info(
            f"Rolled back session {session_id} to recovery point {recovery_id}: {recovery_point.description}"
        )

        return restored_state, recovery_point

    def get_latest_snapshot(self, session_id: str) -> Optional[StateSnapshot]:
        """Get the latest snapshot for a session.

        Args:
            session_id: Session ID

        Returns:
            Latest snapshot or None
        """
        snapshots = self.snapshots.get(session_id, [])
        return snapshots[-1] if snapshots else None

    def cleanup_old_snapshots(self, retention_days: int = 7):
        """Clean up old snapshots and recovery points.

        Args:
            retention_days: Number of days to retain snapshots
        """
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        for session_id in list(self.snapshots.keys()):
            # Clean snapshots
            self.snapshots[session_id] = [
                s for s in self.snapshots[session_id] if s.timestamp > cutoff_date
            ]

            # Clean recovery points
            if session_id in self.recovery_points:
                self.recovery_points[session_id] = [
                    rp
                    for rp in self.recovery_points[session_id]
                    if rp.timestamp > cutoff_date
                ]

        logger.info(f"Cleaned up snapshots older than {retention_days} days")

    def _serialize_state(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Serialize state for storage.

        Args:
            state: State to serialize

        Returns:
            Serialized state data
        """
        # Convert state to JSON-serializable format
        serialized = {}

        for key, value in state.items():
            if isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_dict_values(value)
            elif isinstance(value, list):
                serialized[key] = self._serialize_list_items(value)
            else:
                serialized[key] = value

        return serialized

    def _serialize_dict_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively serialize dictionary values."""
        result = {}
        for key, value in data.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, dict):
                result[key] = self._serialize_dict_values(value)
            elif isinstance(value, list):
                result[key] = self._serialize_list_items(value)
            else:
                result[key] = value
        return result

    def _serialize_list_items(self, data: List[Any]) -> List[Any]:
        """Recursively serialize list items."""
        result = []
        for item in data:
            if isinstance(item, datetime):
                result.append(item.isoformat())
            elif isinstance(item, dict):
                result.append(self._serialize_dict_values(item))
            elif isinstance(item, list):
                result.append(self._serialize_list_items(item))
            else:
                result.append(item)
        return result

    def _deserialize_state(self, data: Dict[str, Any]) -> VirtualAgoraState:
        """Deserialize state from storage.

        Args:
            data: Serialized state data

        Returns:
            Deserialized state
        """
        # Convert ISO format datetime strings back to datetime objects
        deserialized = {}

        for key, value in data.items():
            if isinstance(value, str) and self._is_iso_datetime(value):
                deserialized[key] = datetime.fromisoformat(value)
            elif isinstance(value, dict):
                deserialized[key] = self._deserialize_dict_values(value)
            elif isinstance(value, list):
                deserialized[key] = self._deserialize_list_items(value)
            else:
                deserialized[key] = value

        return deserialized

    def _deserialize_dict_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively deserialize dictionary values."""
        result = {}
        for key, value in data.items():
            if isinstance(value, str) and self._is_iso_datetime(value):
                result[key] = datetime.fromisoformat(value)
            elif isinstance(value, dict):
                result[key] = self._deserialize_dict_values(value)
            elif isinstance(value, list):
                result[key] = self._deserialize_list_items(value)
            else:
                result[key] = value
        return result

    def _deserialize_list_items(self, data: List[Any]) -> List[Any]:
        """Recursively deserialize list items."""
        result = []
        for item in data:
            if isinstance(item, str) and self._is_iso_datetime(item):
                result.append(datetime.fromisoformat(item))
            elif isinstance(item, dict):
                result.append(self._deserialize_dict_values(item))
            elif isinstance(item, list):
                result.append(self._deserialize_list_items(item))
            else:
                result.append(item)
        return result

    def _is_iso_datetime(self, value: str) -> bool:
        """Check if string is ISO datetime format."""
        try:
            datetime.fromisoformat(value)
            return True
        except (ValueError, TypeError):
            return False

    def _validate_state_consistency(self, state: VirtualAgoraState) -> bool:
        """Validate state consistency for recovery points.

        Args:
            state: State to validate

        Returns:
            True if state is consistent
        """
        try:
            # Basic structural validation
            required_fields = ["session_id", "current_phase", "agents", "messages"]

            for field in required_fields:
                if field not in state:
                    logger.warning(f"Missing required field: {field}")
                    return False

            # Phase validation
            phase = state["current_phase"]
            if not isinstance(phase, int) or phase < 0 or phase > 5:
                logger.warning(f"Invalid phase: {phase}")
                return False

            # Agent validation
            agents = state["agents"]
            if not isinstance(agents, dict) or len(agents) == 0:
                logger.warning("Invalid agents structure")
                return False

            # Message validation
            messages = state["messages"]
            if not isinstance(messages, list):
                logger.warning("Invalid messages structure")
                return False

            return True

        except Exception as e:
            logger.error(f"State validation error: {e}")
            return False

    def _save_snapshot_to_disk(self, snapshot: StateSnapshot):
        """Save snapshot to persistent storage."""
        if not self.storage_path:
            return

        try:
            snapshot_file = (
                self.storage_path / f"snapshot_{snapshot.checkpoint_id}.json"
            )

            # Use compression for large snapshots
            if snapshot.compression_used:
                snapshot_file = snapshot_file.with_suffix(".json.gz")
                with gzip.open(snapshot_file, "wt") as f:
                    json.dump(asdict(snapshot), f, default=str, indent=2)
            else:
                with open(snapshot_file, "w") as f:
                    json.dump(asdict(snapshot), f, default=str, indent=2)

            logger.debug(f"Saved snapshot to {snapshot_file}")

        except Exception as e:
            logger.error(f"Failed to save snapshot to disk: {e}")

    def _save_recovery_point_to_disk(self, recovery_point: RecoveryPoint):
        """Save recovery point to persistent storage."""
        if not self.storage_path:
            return

        try:
            recovery_file = (
                self.storage_path / f"recovery_{recovery_point.recovery_id}.json"
            )

            with open(recovery_file, "w") as f:
                json.dump(asdict(recovery_point), f, default=str, indent=2)

            logger.debug(f"Saved recovery point to {recovery_file}")

        except Exception as e:
            logger.error(f"Failed to save recovery point to disk: {e}")

    def _load_persistent_data(self):
        """Load snapshots and recovery points from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            # Load snapshots
            for snapshot_file in self.storage_path.glob("snapshot_*.json*"):
                self._load_snapshot_from_file(snapshot_file)

            # Load recovery points
            for recovery_file in self.storage_path.glob("recovery_*.json"):
                self._load_recovery_point_from_file(recovery_file)

            logger.info(f"Loaded persistent data from {self.storage_path}")

        except Exception as e:
            logger.error(f"Failed to load persistent data: {e}")

    def _load_snapshot_from_file(self, file_path: Path):
        """Load snapshot from file."""
        try:
            if file_path.suffix == ".gz":
                with gzip.open(file_path, "rt") as f:
                    data = json.load(f)
            else:
                with open(file_path, "r") as f:
                    data = json.load(f)

            # Convert dict back to StateSnapshot
            snapshot = StateSnapshot(**data)

            # Add to memory storage
            session_id = snapshot.session_id
            if session_id not in self.snapshots:
                self.snapshots[session_id] = []
            self.snapshots[session_id].append(snapshot)

        except Exception as e:
            logger.error(f"Failed to load snapshot from {file_path}: {e}")

    def _load_recovery_point_from_file(self, file_path: Path):
        """Load recovery point from file."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Convert dict back to RecoveryPoint
            recovery_point = RecoveryPoint(**data)

            # Add to memory storage
            session_id = recovery_point.session_id
            if session_id not in self.recovery_points:
                self.recovery_points[session_id] = []
            self.recovery_points[session_id].append(recovery_point)

        except Exception as e:
            logger.error(f"Failed to load recovery point from {file_path}: {e}")

    def get_persistence_stats(self) -> Dict[str, Any]:
        """Get statistics about persistent storage.

        Returns:
            Persistence statistics
        """
        total_snapshots = sum(len(snapshots) for snapshots in self.snapshots.values())
        total_recovery_points = sum(len(rps) for rps in self.recovery_points.values())

        sessions_with_data = set(self.snapshots.keys()) | set(
            self.recovery_points.keys()
        )

        stats = {
            "total_sessions": len(sessions_with_data),
            "total_snapshots": total_snapshots,
            "total_recovery_points": total_recovery_points,
            "storage_path": str(self.storage_path) if self.storage_path else None,
            "sessions_by_snapshots": {
                sid: len(snapshots) for sid, snapshots in self.snapshots.items()
            },
            "sessions_by_recovery_points": {
                sid: len(rps) for sid, rps in self.recovery_points.items()
            },
        }

        if self.storage_path and self.storage_path.exists():
            try:
                # Calculate disk usage
                total_size = sum(
                    f.stat().st_size for f in self.storage_path.iterdir() if f.is_file()
                )
                stats["disk_usage_bytes"] = total_size
                stats["disk_usage_mb"] = round(total_size / (1024 * 1024), 2)
            except Exception as e:
                logger.warning(f"Could not calculate disk usage: {e}")

        return stats


def create_enhanced_checkpointer(
    storage_path: Optional[Union[str, Path]] = None,
) -> EnhancedMemorySaver:
    """Create an enhanced checkpointer with persistence capabilities.

    Args:
        storage_path: Optional path for persistent storage

    Returns:
        Enhanced memory saver instance
    """
    if storage_path:
        storage_path = Path(storage_path)

    return EnhancedMemorySaver(storage_path)
