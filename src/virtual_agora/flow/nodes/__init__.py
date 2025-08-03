"""Virtual Agora flow nodes package.

This package provides the pluggable node architecture for Virtual Agora
discussion flows, introduced in Step 2.1 of the refactoring plan.

The architecture provides:
- Abstract base classes for all node types
- Standardized validation and error handling
- Human-in-the-loop (HITL) node patterns
- Agent orchestration capabilities
- Dependency injection system
"""

from .base import (
    FlowNode,
    HITLNode,
    AgentOrchestratorNode,
    NodeDependencies,
    NodeExecutionContext,
    NodeValidationError,
    NodeExecutionError,
)
from .discussion_round import DiscussionRoundNode

__all__ = [
    "FlowNode",
    "HITLNode",
    "AgentOrchestratorNode",
    "NodeDependencies",
    "NodeExecutionContext",
    "NodeValidationError",
    "NodeExecutionError",
    "DiscussionRoundNode",
]

# Version information for node interface compatibility
NODE_INTERFACE_VERSION = "2.1.0"
COMPATIBLE_VERSIONS = ["2.1.0"]


def get_node_interface_version() -> str:
    """Get the current node interface version.

    Returns:
        Node interface version string
    """
    return NODE_INTERFACE_VERSION


def is_compatible_version(version: str) -> bool:
    """Check if a version is compatible with current interface.

    Args:
        version: Version string to check

    Returns:
        True if version is compatible, False otherwise
    """
    return version in COMPATIBLE_VERSIONS
