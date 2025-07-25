"""State recovery manager for Virtual Agora.

This module provides state checkpointing, rollback, and recovery
functionality to handle errors and maintain state consistency.
"""

import json
import pickle
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.state.validators import StateValidator
from virtual_agora.state.manager import StateManager
from virtual_agora.utils.logging import get_logger
from virtual_agora.utils.exceptions import (
    StateError,
    StateCorruptionError,
    RecoverableError,
)


logger = get_logger(__name__)


class StateCheckpoint:
    """Represents a state checkpoint."""
    
    def __init__(
        self,
        checkpoint_id: str,
        state_snapshot: Dict[str, Any],
        timestamp: datetime,
        operation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize checkpoint.
        
        Args:
            checkpoint_id: Unique checkpoint identifier
            state_snapshot: Complete state snapshot
            timestamp: When checkpoint was created
            operation: Operation that triggered checkpoint
            metadata: Additional checkpoint metadata
        """
        self.checkpoint_id = checkpoint_id
        self.state_snapshot = state_snapshot
        self.timestamp = timestamp
        self.operation = operation
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation,
            "metadata": self.metadata,
        }


class StateRecoveryManager:
    """Manages state recovery and checkpointing."""
    
    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        max_checkpoints: int = 10,
        auto_checkpoint: bool = True,
    ):
        """Initialize recovery manager.
        
        Args:
            checkpoint_dir: Directory for checkpoint files
            max_checkpoints: Maximum checkpoints to keep
            auto_checkpoint: Whether to auto-checkpoint on critical operations
        """
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints")
        self.max_checkpoints = max_checkpoints
        self.auto_checkpoint = auto_checkpoint
        
        self.checkpoints: List[StateCheckpoint] = []
        self.validator = StateValidator()
        self._checkpoint_counter = 0
        
        # Create checkpoint directory if needed
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(exist_ok=True)
    
    def create_checkpoint(
        self,
        state: VirtualAgoraState,
        operation: Optional[str] = None,
        save_to_disk: bool = False,
    ) -> StateCheckpoint:
        """Create a state checkpoint.
        
        Args:
            state: Current state to checkpoint
            operation: Operation triggering checkpoint
            save_to_disk: Whether to persist to disk
            
        Returns:
            Created checkpoint
        """
        # Generate checkpoint ID
        self._checkpoint_counter += 1
        checkpoint_id = f"checkpoint_{self._checkpoint_counter:06d}"
        
        # Deep copy state
        state_snapshot = deepcopy(dict(state))
        
        # Create checkpoint
        checkpoint = StateCheckpoint(
            checkpoint_id=checkpoint_id,
            state_snapshot=state_snapshot,
            timestamp=datetime.now(),
            operation=operation,
        )
        
        # Add to in-memory list
        self.checkpoints.append(checkpoint)
        
        # Maintain max checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            removed = self.checkpoints.pop(0)
            if save_to_disk:
                self._delete_checkpoint_file(removed.checkpoint_id)
        
        # Save to disk if requested
        if save_to_disk:
            self._save_checkpoint_to_disk(checkpoint)
        
        logger.info(
            f"Created checkpoint {checkpoint_id} "
            f"{'(persisted)' if save_to_disk else '(in-memory)'}"
        )
        
        return checkpoint
    
    def rollback_to_checkpoint(
        self,
        checkpoint_id: str,
        state_manager: StateManager,
    ) -> VirtualAgoraState:
        """Rollback state to a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to rollback to
            state_manager: State manager to update
            
        Returns:
            Restored state
            
        Raises:
            StateError: If checkpoint not found
        """
        # Find checkpoint
        checkpoint = self._find_checkpoint(checkpoint_id)
        if not checkpoint:
            raise StateError(f"Checkpoint not found: {checkpoint_id}")
        
        # Validate checkpoint state
        warnings = self.validator.validate_state_consistency(checkpoint.state_snapshot)
        if warnings:
            logger.warning(
                f"Checkpoint {checkpoint_id} has consistency warnings: {warnings}"
            )
        
        # Deep copy to avoid reference issues
        restored_state = deepcopy(checkpoint.state_snapshot)
        
        # Update state manager
        state_manager._state = restored_state
        
        logger.info(
            f"Rolled back to checkpoint {checkpoint_id} "
            f"from {checkpoint.timestamp}"
        )
        
        return restored_state
    
    def get_latest_checkpoint(self) -> Optional[StateCheckpoint]:
        """Get the most recent checkpoint.
        
        Returns:
            Latest checkpoint or None
        """
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]
    
    def validate_state(self, state: VirtualAgoraState) -> Tuple[bool, List[str]]:
        """Validate state consistency.
        
        Args:
            state: State to validate
            
        Returns:
            Tuple of (is_valid, warnings)
        """
        warnings = self.validator.validate_state_consistency(state)
        is_valid = len(warnings) == 0
        return is_valid, warnings
    
    def repair_corrupted_state(
        self,
        state: VirtualAgoraState,
        corruption_warnings: List[str],
    ) -> Optional[VirtualAgoraState]:
        """Attempt to repair corrupted state.
        
        Args:
            state: Corrupted state
            corruption_warnings: List of corruption warnings
            
        Returns:
            Repaired state or None if unrepairable
        """
        logger.warning(f"Attempting to repair {len(corruption_warnings)} issues")
        
        repaired_state = deepcopy(dict(state))
        repairs_made = []
        
        for warning in corruption_warnings:
            # Message count mismatch
            if "message count mismatch" in warning.lower():
                # Recalculate message counts
                actual_total = len(repaired_state.get("messages", []))
                repaired_state["total_messages"] = actual_total
                
                # Recalculate per-agent counts
                agent_counts = {}
                for msg in repaired_state.get("messages", []):
                    speaker = msg.get("speaker_id")
                    if speaker:
                        agent_counts[speaker] = agent_counts.get(speaker, 0) + 1
                
                repaired_state["messages_by_agent"] = agent_counts
                
                # Update agent message counts
                for agent_id, count in agent_counts.items():
                    if agent_id in repaired_state.get("agents", {}):
                        repaired_state["agents"][agent_id]["message_count"] = count
                
                repairs_made.append("Fixed message count inconsistencies")
            
            # Vote participation issues
            elif "incomplete vote round" in warning.lower():
                # Mark incomplete votes as failed
                for vote_round in repaired_state.get("vote_history", []):
                    if vote_round.get("status") == "active":
                        vote_round["status"] = "failed"
                        repairs_made.append(f"Marked vote {vote_round['id']} as failed")
            
            # Topic status issues
            elif "marked as active but not current" in warning.lower():
                # Fix topic status
                active_topic = repaired_state.get("active_topic")
                for topic, info in repaired_state.get("topics_info", {}).items():
                    if info.get("status") == "active" and topic != active_topic:
                        info["status"] = "paused"
                        repairs_made.append(f"Fixed status for topic: {topic}")
        
        # Validate repaired state
        is_valid, remaining_warnings = self.validate_state(repaired_state)
        
        if is_valid:
            logger.info(f"Successfully repaired state: {repairs_made}")
            return repaired_state
        else:
            logger.error(
                f"Could not fully repair state. Remaining issues: {remaining_warnings}"
            )
            return None
    
    def find_last_valid_state(self) -> Optional[StateCheckpoint]:
        """Find the most recent valid checkpoint.
        
        Returns:
            Last valid checkpoint or None
        """
        # Search backwards through checkpoints
        for checkpoint in reversed(self.checkpoints):
            is_valid, _ = self.validate_state(checkpoint.state_snapshot)
            if is_valid:
                return checkpoint
        
        # Check disk checkpoints if available
        if self.checkpoint_dir and self.checkpoint_dir.exists():
            checkpoint_files = sorted(
                self.checkpoint_dir.glob("checkpoint_*.pkl"),
                reverse=True
            )
            
            for checkpoint_file in checkpoint_files:
                try:
                    checkpoint = self._load_checkpoint_from_disk(
                        checkpoint_file.stem
                    )
                    if checkpoint:
                        is_valid, _ = self.validate_state(checkpoint.state_snapshot)
                        if is_valid:
                            return checkpoint
                except Exception as e:
                    logger.error(f"Error loading checkpoint {checkpoint_file}: {e}")
        
        return None
    
    def emergency_recovery(
        self,
        state_manager: StateManager,
        error: Exception,
    ) -> bool:
        """Perform emergency recovery after critical error.
        
        Args:
            state_manager: State manager
            error: Error that triggered recovery
            
        Returns:
            True if recovery successful
        """
        logger.error(f"Emergency recovery triggered by: {error}")
        
        # Try to repair current state
        if hasattr(state_manager, "_state") and state_manager._state:
            is_valid, warnings = self.validate_state(state_manager._state)
            
            if not is_valid:
                repaired = self.repair_corrupted_state(
                    state_manager._state,
                    warnings
                )
                if repaired:
                    state_manager._state = repaired
                    logger.info("Successfully repaired current state")
                    return True
        
        # Find last valid checkpoint
        last_valid = self.find_last_valid_state()
        if last_valid:
            try:
                self.rollback_to_checkpoint(
                    last_valid.checkpoint_id,
                    state_manager
                )
                logger.info(
                    f"Emergency recovery: rolled back to {last_valid.checkpoint_id}"
                )
                return True
            except Exception as e:
                logger.error(f"Rollback failed during emergency recovery: {e}")
        
        # Last resort: reinitialize state
        if hasattr(state_manager, "_state") and state_manager._state:
            session_id = state_manager._state.get("session_id", "emergency_recovery")
            try:
                new_state = state_manager.initialize_state(session_id)
                logger.warning("Emergency recovery: reinitialized state")
                return True
            except Exception as e:
                logger.critical(f"Failed to reinitialize state: {e}")
        
        return False
    
    def _find_checkpoint(self, checkpoint_id: str) -> Optional[StateCheckpoint]:
        """Find checkpoint by ID.
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            Checkpoint or None
        """
        # Check in-memory checkpoints
        for checkpoint in self.checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint
        
        # Try loading from disk
        if self.checkpoint_dir:
            return self._load_checkpoint_from_disk(checkpoint_id)
        
        return None
    
    def _save_checkpoint_to_disk(self, checkpoint: StateCheckpoint) -> None:
        """Save checkpoint to disk.
        
        Args:
            checkpoint: Checkpoint to save
        """
        if not self.checkpoint_dir:
            return
        
        file_path = self.checkpoint_dir / f"{checkpoint.checkpoint_id}.pkl"
        
        try:
            with open(file_path, "wb") as f:
                pickle.dump(checkpoint, f)
            logger.debug(f"Saved checkpoint to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint_from_disk(
        self,
        checkpoint_id: str
    ) -> Optional[StateCheckpoint]:
        """Load checkpoint from disk.
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            Checkpoint or None
        """
        if not self.checkpoint_dir:
            return None
        
        file_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, "rb") as f:
                checkpoint = pickle.load(f)
            logger.debug(f"Loaded checkpoint from {file_path}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def _delete_checkpoint_file(self, checkpoint_id: str) -> None:
        """Delete checkpoint file.
        
        Args:
            checkpoint_id: Checkpoint ID
        """
        if not self.checkpoint_dir:
            return
        
        file_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        
        if file_path.exists():
            try:
                file_path.unlink()
                logger.debug(f"Deleted checkpoint file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete checkpoint: {e}")
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics.
        
        Returns:
            Recovery statistics
        """
        total_checkpoints = len(self.checkpoints)
        
        if self.checkpoint_dir and self.checkpoint_dir.exists():
            disk_checkpoints = len(list(self.checkpoint_dir.glob("checkpoint_*.pkl")))
        else:
            disk_checkpoints = 0
        
        # Find oldest and newest checkpoints
        if self.checkpoints:
            oldest = self.checkpoints[0].timestamp
            newest = self.checkpoints[-1].timestamp
            time_span = (newest - oldest).total_seconds() / 60.0  # minutes
        else:
            time_span = 0.0
        
        return {
            "total_checkpoints": total_checkpoints,
            "disk_checkpoints": disk_checkpoints,
            "checkpoint_rate": (
                round(total_checkpoints / max(time_span, 1.0), 2)
                if total_checkpoints > 0 else 0.0
            ),
            "latest_checkpoint": (
                self.checkpoints[-1].to_dict()
                if self.checkpoints else None
            ),
        }