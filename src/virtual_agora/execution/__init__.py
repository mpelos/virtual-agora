"""Clean execution architecture for Virtual Agora.

This package provides a clean, understandable architecture for session execution
that replaces the complex nested stream logic in the original main.py.

Core Components:
- SessionController: Single source of truth for session state and execution control
- StreamCoordinator: Handles stream lifecycle without premature breaks
- UnifiedStateManager: Centralizes all state management
- ExecutionTracker: Provides execution visibility and statistics

Key Benefits:
- No premature session termination
- Clear separation of concerns
- Single source of truth for state
- Comprehensive execution tracking
- Easy to understand and maintain
"""

from .session_controller import (
    SessionController,
    SessionState,
    EventType,
    ExecutionEvent,
    SessionStatistics,
    ExecutionContext,
)

from .stream_coordinator import (
    StreamCoordinator,
    ContinuationResult,
    InterruptContext,
    StreamHealth,
)

from .unified_state_manager import (
    UnifiedStateManager,
    StateLayer,
    SessionState as SessionStateData,
    FlowState,
    UIState,
    Statistics,
)

from .execution_tracker import (
    ExecutionTracker,
    ExecutionEventType,
    ExecutionEvent as TrackerExecutionEvent,
    PerformanceMetrics,
)

__all__ = [
    # Session Controller
    "SessionController",
    "SessionState",
    "EventType",
    "ExecutionEvent",
    "SessionStatistics",
    "ExecutionContext",
    # Stream Coordinator
    "StreamCoordinator",
    "ContinuationResult",
    "InterruptContext",
    "StreamHealth",
    # Unified State Manager
    "UnifiedStateManager",
    "StateLayer",
    "SessionStateData",
    "FlowState",
    "UIState",
    "Statistics",
    # Execution Tracker
    "ExecutionTracker",
    "ExecutionEventType",
    "TrackerExecutionEvent",
    "PerformanceMetrics",
]
