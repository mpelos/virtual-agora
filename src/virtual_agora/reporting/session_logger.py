"""Enhanced session logging for Virtual Agora.

This module extends the basic logging functionality to provide
comprehensive session logging with structured formats and analytics.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import gzip
import shutil

from ..utils.logging import SessionLogger as BaseSessionLogger, get_logger
from ..state.schema import VirtualAgoraState, Message, Vote, VoteRound

logger = get_logger(__name__)


class EventType(Enum):
    """Types of events that can be logged."""

    USER_INPUT = "USER_INPUT"
    AGENT_RESPONSE = "AGENT_RESPONSE"
    SYSTEM_EVENT = "SYSTEM_EVENT"
    VOTING_EVENT = "VOTING_EVENT"
    PHASE_TRANSITION = "PHASE_TRANSITION"
    TOPIC_CHANGE = "TOPIC_CHANGE"
    ERROR = "ERROR"
    WARNING = "WARNING"
    DEBUG = "DEBUG"


class LogLevel(Enum):
    """Log levels for filtering."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class EnhancedSessionLogger(BaseSessionLogger):
    """Enhanced session logger with structured logging and analytics."""

    def __init__(
        self,
        session_id: str,
        log_dir: Optional[Path] = None,
        enable_compression: bool = True,
        max_log_size_mb: int = 100,
    ):
        """Initialize EnhancedSessionLogger.

        Args:
            session_id: Unique session identifier.
            log_dir: Directory for log files.
            enable_compression: Whether to compress old logs.
            max_log_size_mb: Maximum log file size before rotation.
        """
        super().__init__(session_id, log_dir)
        self.enable_compression = enable_compression
        self.max_log_size_bytes = max_log_size_mb * 1024 * 1024
        self.event_counts = {event_type: 0 for event_type in EventType}

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.structured_log_path = (
            self.log_dir / f"session_{session_id}_structured.jsonl"
        )

        # Initialize structured log file
        self._init_structured_log()

    def _init_structured_log(self):
        """Initialize the structured log file with session metadata."""
        # Count session start as a system event first
        self.event_counts[EventType.SYSTEM_EVENT] += 1

        metadata = {
            "event_type": "SESSION_START",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "log_version": "1.0",
            "event_number": 1,  # Explicitly set to 1 for the first event
        }
        self._write_structured_event(metadata)

    def log_event(
        self,
        event_type: EventType,
        actor: str,
        content: Union[str, Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        level: LogLevel = LogLevel.INFO,
    ):
        """Log a structured event.

        Args:
            event_type: Type of event.
            actor: Actor involved (agent ID, system, user).
            content: Event content.
            metadata: Additional metadata.
            level: Log level.
        """
        # Update event counter
        self.event_counts[event_type] += 1

        # Create structured event
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type.value,
            "actor": actor,
            "content": content if isinstance(content, str) else json.dumps(content),
            "level": level.value,
            "metadata": metadata or {},
            "event_number": sum(self.event_counts.values()),
        }

        # Write to structured log
        self._write_structured_event(event)

        # Also log to standard logger
        self._log_standard(event_type, actor, content, level)

    def _write_structured_event(self, event: Dict[str, Any]):
        """Write event to structured log file."""
        with open(self.structured_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

        # Check for rotation
        self._check_rotation()

    def _log_standard(
        self,
        event_type: EventType,
        actor: str,
        content: Union[str, Dict[str, Any]],
        level: LogLevel,
    ):
        """Log to standard logger."""
        message = f"{event_type.value} - {actor}"
        if isinstance(content, str):
            message += f" - {content}"
        else:
            message += f" - {json.dumps(content)}"

        if level == LogLevel.DEBUG:
            self.logger.debug(message)
        elif level == LogLevel.INFO:
            self.logger.info(message)
        elif level == LogLevel.WARNING:
            self.logger.warning(message)
        elif level == LogLevel.ERROR:
            self.logger.error(message)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(message)

    def log_user_input(self, prompt: str, response: str):
        """Log user input with enhanced structure."""
        super().log_user_input(prompt, response)
        self.log_event(
            EventType.USER_INPUT,
            "user",
            {"prompt": prompt, "response": response},
        )

    def log_agent_response(self, agent_name: str, response: str):
        """Log agent response with enhanced structure."""
        super().log_agent_response(agent_name, response)
        self.log_event(
            EventType.AGENT_RESPONSE,
            agent_name,
            response,
        )

    def log_system_event(self, event: str, details: Optional[str] = None):
        """Log system event with enhanced structure."""
        super().log_system_event(event, details)
        self.log_event(
            EventType.SYSTEM_EVENT,
            "system",
            {"event": event, "details": details},
        )

    def log_voting_event(
        self,
        vote_type: str,
        voter: str,
        choice: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log a voting event.

        Args:
            vote_type: Type of vote.
            voter: Voter agent ID.
            choice: Vote choice.
            metadata: Additional vote metadata.
        """
        self.log_event(
            EventType.VOTING_EVENT,
            voter,
            {
                "vote_type": vote_type,
                "choice": choice,
            },
            metadata=metadata,
        )

    def log_phase_transition(
        self,
        from_phase: int,
        to_phase: int,
        reason: str,
    ):
        """Log a phase transition.

        Args:
            from_phase: Previous phase.
            to_phase: New phase.
            reason: Reason for transition.
        """
        self.log_event(
            EventType.PHASE_TRANSITION,
            "system",
            {
                "from_phase": from_phase,
                "to_phase": to_phase,
                "reason": reason,
            },
        )

    def log_topic_change(
        self,
        old_topic: Optional[str],
        new_topic: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log a topic change.

        Args:
            old_topic: Previous topic.
            new_topic: New topic.
            metadata: Additional metadata.
        """
        self.log_event(
            EventType.TOPIC_CHANGE,
            "system",
            {
                "old_topic": old_topic,
                "new_topic": new_topic,
            },
            metadata=metadata,
        )

    def log_error(self, error: str, exception: Optional[Exception] = None):
        """Log error with enhanced structure."""
        super().log_error(error, exception)

        exception_str = None
        if exception:
            # Format as "ExceptionType: message" to match expected test format
            exception_str = f"{type(exception).__name__}: {str(exception)}"

        self.log_event(
            EventType.ERROR,
            "system",
            {
                "error": error,
                "exception": exception_str,
                "traceback": str(exception.__traceback__) if exception else None,
            },
            level=LogLevel.ERROR,
        )

    def log_state_snapshot(self, state: VirtualAgoraState):
        """Log a snapshot of the current state.

        Args:
            state: Current Virtual Agora state.
        """
        snapshot = {
            "session_id": state.get("session_id"),
            "current_phase": state.get("current_phase"),
            "active_topic": state.get("active_topic"),
            "current_round": state.get("current_round"),
            "total_messages": state.get("total_messages"),
            "agent_count": len(state.get("agents", {})),
            "topics_completed": len(state.get("completed_topics", [])),
        }

        self.log_event(
            EventType.SYSTEM_EVENT,
            "system",
            {"type": "state_snapshot", "snapshot": snapshot},
            level=LogLevel.DEBUG,
        )

    def _check_rotation(self):
        """Check if log rotation is needed."""
        try:
            # Check structured log size
            if self.structured_log_path.stat().st_size > self.max_log_size_bytes:
                self._rotate_log(self.structured_log_path)

            # Check standard log size
            standard_log = self.log_dir / f"session_{self.session_id}.log"
            if (
                standard_log.exists()
                and standard_log.stat().st_size > self.max_log_size_bytes
            ):
                self._rotate_log(standard_log)

        except Exception as e:
            logger.error(f"Error checking log rotation: {e}")

    def _rotate_log(self, log_path: Path):
        """Rotate a log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_path = log_path.parent / f"{log_path.stem}_{timestamp}{log_path.suffix}"

        # Move current log
        shutil.move(str(log_path), str(rotated_path))

        # Compress if enabled
        if self.enable_compression:
            self._compress_log(rotated_path)

        logger.info(f"Rotated log: {log_path.name} -> {rotated_path.name}")

    def _compress_log(self, log_path: Path):
        """Compress a log file using gzip."""
        compressed_path = log_path.with_suffix(log_path.suffix + ".gz")

        with open(log_path, "rb") as f_in:
            with gzip.open(compressed_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove original
        log_path.unlink()
        logger.info(f"Compressed log: {log_path.name} -> {compressed_path.name}")

    def get_session_analytics(self) -> Dict[str, Any]:
        """Get analytics for the current session.

        Returns:
            Dictionary of session analytics.
        """
        # Parse structured log for analytics
        events = []
        try:
            with open(self.structured_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))
        except Exception as e:
            logger.error(f"Error reading structured log: {e}")
            return {}

        # Calculate analytics
        analytics = {
            "session_id": self.session_id,
            "total_events": len(events),
            "event_counts": {
                k.value: v for k, v in self.event_counts.items()
            },  # Convert EventType to string
            "start_time": events[0]["timestamp"] if events else None,
            "end_time": events[-1]["timestamp"] if events else None,
            "duration_seconds": None,
            "agents_active": set(),
            "topics_discussed": set(),
            "error_count": self.event_counts[EventType.ERROR],
            "warning_count": self.event_counts[EventType.WARNING],
        }

        # Calculate duration
        if analytics["start_time"] and analytics["end_time"]:
            start = datetime.fromisoformat(analytics["start_time"])
            end = datetime.fromisoformat(analytics["end_time"])
            analytics["duration_seconds"] = (end - start).total_seconds()

        # Extract unique agents and topics
        for event in events:
            if event.get("event_type") == EventType.AGENT_RESPONSE.value:
                analytics["agents_active"].add(event.get("actor"))

            if event.get("event_type") == EventType.TOPIC_CHANGE.value:
                content = json.loads(event.get("content", "{}"))
                if content.get("new_topic"):
                    analytics["topics_discussed"].add(content["new_topic"])

        # Convert sets to lists for JSON serialization
        analytics["agents_active"] = list(analytics["agents_active"])
        analytics["topics_discussed"] = list(analytics["topics_discussed"])

        return analytics

    def export_logs(self, output_dir: Path, format: str = "json") -> List[Path]:
        """Export logs in various formats.

        Args:
            output_dir: Directory to export logs to.
            format: Export format (json, csv, txt).

        Returns:
            List of exported file paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        exported_files = []

        if format == "json":
            # Copy structured log
            dest = output_dir / f"session_{self.session_id}_events.json"
            shutil.copy(self.structured_log_path, dest)
            exported_files.append(dest)

            # Export analytics
            analytics_path = output_dir / f"session_{self.session_id}_analytics.json"
            analytics_path.write_text(
                json.dumps(self.get_session_analytics(), indent=2), encoding="utf-8"
            )
            exported_files.append(analytics_path)

        elif format == "txt":
            # Copy standard log
            standard_log = self.log_dir / f"session_{self.session_id}.log"
            if standard_log.exists():
                dest = output_dir / standard_log.name
                shutil.copy(standard_log, dest)
                exported_files.append(dest)

        logger.info(f"Exported {len(exported_files)} log files to {output_dir}")
        return exported_files

    def close(self):
        """Close the logger and finalize session."""
        # Log session end
        self.log_event(
            EventType.SYSTEM_EVENT,
            "system",
            "SESSION_END",
            metadata=self.get_session_analytics(),
        )

        logger.info(f"Session {self.session_id} logging completed")
