"""Node registry system for Virtual Agora discussion flow.

This module implements a pluggable node architecture that allows dynamic
registration and management of flow nodes. It provides:

- Node registration and retrieval
- Node wrapper classes for legacy compatibility
- Validation and type checking
- Default registry creation for V13 nodes

This supports Step 2.3 of the architecture refactoring by enabling
a pluggable node architecture with clean separation of concerns.
"""

from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod

from langgraph.errors import GraphInterrupt
from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.flow.nodes.base import (
    FlowNode,
    NodeValidationError,
    NodeExecutionError,
)
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class NodeRegistry:
    """Registry for managing pluggable flow nodes.

    This class provides a centralized registry for all flow nodes,
    enabling dynamic node management and pluggable architecture.
    It supports both new FlowNode instances and legacy node functions
    through wrapper classes.

    Key features:
    - Dynamic node registration and retrieval
    - Type validation and safety checks
    - Legacy compatibility through node wrappers
    - Node metadata and documentation
    - Dependency validation between nodes
    """

    def __init__(self):
        """Initialize an empty node registry."""
        self._nodes: Dict[str, FlowNode] = {}
        self._node_metadata: Dict[str, Dict[str, Any]] = {}
        self._registration_order: List[str] = []

        logger.debug("Initialized empty NodeRegistry")

    def register_node(
        self, name: str, node: FlowNode, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a flow node in the registry.

        Args:
            name: Unique name for the node
            node: FlowNode instance to register
            metadata: Optional metadata about the node

        Raises:
            ValueError: If name is already registered or node is invalid
            TypeError: If node is not a FlowNode instance
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Node name must be a non-empty string")

        if not isinstance(node, FlowNode):
            raise TypeError(f"Node must be a FlowNode instance, got {type(node)}")

        if name in self._nodes:
            raise ValueError(f"Node '{name}' is already registered")

        # Validate the node has required methods
        required_methods = ["execute", "validate_preconditions", "get_node_name"]
        for method in required_methods:
            if not hasattr(node, method) or not callable(getattr(node, method)):
                raise TypeError(f"Node must implement '{method}' method")

        self._nodes[name] = node
        self._node_metadata[name] = metadata or {}
        self._registration_order.append(name)

        logger.debug(f"Registered node '{name}' of type {type(node).__name__}")

    def get_node(self, name: str) -> FlowNode:
        """Get registered node by name.

        Args:
            name: Name of the node to retrieve

        Returns:
            FlowNode instance

        Raises:
            KeyError: If node is not found
        """
        if name not in self._nodes:
            available_nodes = list(self._nodes.keys())
            raise KeyError(
                f"Node '{name}' not found. Available nodes: {available_nodes}"
            )

        return self._nodes[name]

    def get_all_nodes(self) -> Dict[str, FlowNode]:
        """Get all registered nodes.

        Returns:
            Dictionary mapping node names to FlowNode instances
        """
        return self._nodes.copy()

    def get_node_names(self) -> List[str]:
        """Get list of all registered node names.

        Returns:
            List of node names in registration order
        """
        return self._registration_order.copy()

    def has_node(self, name: str) -> bool:
        """Check if a node is registered.

        Args:
            name: Name of the node to check

        Returns:
            True if node is registered, False otherwise
        """
        return name in self._nodes

    def unregister_node(self, name: str) -> FlowNode:
        """Unregister a node from the registry.

        Args:
            name: Name of the node to unregister

        Returns:
            The unregistered FlowNode instance

        Raises:
            KeyError: If node is not found
        """
        if name not in self._nodes:
            raise KeyError(f"Node '{name}' not found")

        node = self._nodes.pop(name)
        self._node_metadata.pop(name, None)
        self._registration_order.remove(name)

        logger.debug(f"Unregistered node '{name}'")
        return node

    def get_node_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a registered node.

        Args:
            name: Name of the node

        Returns:
            Node metadata dictionary

        Raises:
            KeyError: If node is not found
        """
        if name not in self._nodes:
            raise KeyError(f"Node '{name}' not found")

        return self._node_metadata[name].copy()

    def update_node_metadata(self, name: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for a registered node.

        Args:
            name: Name of the node
            metadata: New metadata to set

        Raises:
            KeyError: If node is not found
        """
        if name not in self._nodes:
            raise KeyError(f"Node '{name}' not found")

        self._node_metadata[name].update(metadata)
        logger.debug(f"Updated metadata for node '{name}'")

    def validate_registry(self) -> Dict[str, Any]:
        """Validate the entire registry for consistency.

        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            "total_nodes": len(self._nodes),
            "valid_nodes": 0,
            "invalid_nodes": [],
            "missing_metadata": [],
            "duplicate_node_names": [],
            "validation_errors": [],
        }

        # Check for consistency between dictionaries
        nodes_set = set(self._nodes.keys())
        metadata_set = set(self._node_metadata.keys())
        order_set = set(self._registration_order)

        if nodes_set != order_set:
            validation_results["validation_errors"].append(
                "Registration order inconsistent with nodes"
            )

        # Validate each node
        for name, node in self._nodes.items():
            try:
                # Check if node is a valid FlowNode
                if not isinstance(node, FlowNode):
                    validation_results["invalid_nodes"].append(
                        f"{name}: not a FlowNode instance"
                    )
                    continue

                # Check required methods
                required_methods = [
                    "execute",
                    "validate_preconditions",
                    "get_node_name",
                ]
                for method in required_methods:
                    if not hasattr(node, method) or not callable(getattr(node, method)):
                        validation_results["invalid_nodes"].append(
                            f"{name}: missing method '{method}'"
                        )
                        continue

                validation_results["valid_nodes"] += 1

            except Exception as e:
                validation_results["invalid_nodes"].append(
                    f"{name}: validation error - {e}"
                )

        # Check for missing metadata
        for name in self._nodes.keys():
            if name not in self._node_metadata:
                validation_results["missing_metadata"].append(name)

        logger.debug(f"Registry validation results: {validation_results}")
        return validation_results

    def clear(self) -> None:
        """Clear all registered nodes."""
        cleared_count = len(self._nodes)
        self._nodes.clear()
        self._node_metadata.clear()
        self._registration_order.clear()

        logger.info(f"Cleared {cleared_count} nodes from registry")


class V13NodeWrapper(FlowNode):
    """Wrapper class to adapt legacy V13 node functions to FlowNode interface.

    This wrapper allows legacy node functions from nodes_v13.py to work
    with the new FlowNode interface without requiring immediate refactoring.
    It provides compatibility while enabling gradual migration to the new
    node architecture.
    """

    def __init__(
        self,
        node_function: Callable[[VirtualAgoraState], Dict[str, Any]],
        node_name: str,
        validation_function: Optional[Callable[[VirtualAgoraState], bool]] = None,
    ):
        """Initialize wrapper for legacy node function.

        Args:
            node_function: Legacy node function to wrap
            node_name: Name of the node for identification
            validation_function: Optional validation function for preconditions
        """
        super().__init__()
        self.node_function = node_function
        self.node_name = node_name
        self.validation_function = validation_function

        logger.debug(f"Created V13NodeWrapper for '{node_name}'")

    def execute(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Execute the wrapped legacy node function.

        Args:
            state: Current Virtual Agora state

        Returns:
            State updates from node execution

        Raises:
            NodeExecutionError: If execution fails
            GraphInterrupt: Propagated for HITL interactions
        """
        try:
            return self.node_function(state)
        except GraphInterrupt:
            # Allow GraphInterrupt to propagate for HITL functionality
            # This is critical for user input flow to work correctly
            raise
        except Exception as e:
            raise NodeExecutionError(
                f"Wrapped node '{self.node_name}' execution failed: {e}"
            ) from e

    def validate_preconditions(self, state: VirtualAgoraState) -> bool:
        """Validate preconditions using custom validation function if available.

        Args:
            state: Current Virtual Agora state

        Returns:
            True if preconditions are met, True by default if no validator
        """
        if self.validation_function:
            try:
                return self.validation_function(state)
            except Exception as e:
                self._validation_errors = [f"Validation function error: {e}"]
                return False

        # Default: assume preconditions are met for legacy nodes
        return True

    def get_node_name(self) -> str:
        """Get human-readable node name.

        Returns:
            Node name for identification
        """
        return f"V13Wrapper({self.node_name})"

    def __call__(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Make wrapper callable for direct LangGraph integration.

        This allows V13NodeWrapper to be used directly as a node function
        in LangGraph while maintaining the FlowNode interface.

        Args:
            state: Current Virtual Agora state

        Returns:
            State updates from node execution
        """
        return self.execute(state)


def create_default_v13_registry(
    v13_nodes, unified_discussion_node=None
) -> NodeRegistry:
    """Create node registry with all V13 nodes wrapped for compatibility.

    Args:
        v13_nodes: V13FlowNodes instance containing all legacy node functions
        unified_discussion_node: Optional DiscussionRoundNode to use instead of legacy

    Returns:
        NodeRegistry with all V13 nodes registered
    """
    registry = NodeRegistry()

    # Phase 0: Initialization nodes
    registry.register_node(
        "config_and_keys",
        V13NodeWrapper(v13_nodes.config_and_keys_node, "config_and_keys"),
        {"phase": 0, "type": "initialization"},
    )

    registry.register_node(
        "agent_instantiation",
        V13NodeWrapper(v13_nodes.agent_instantiation_node, "agent_instantiation"),
        {"phase": 0, "type": "initialization"},
    )

    registry.register_node(
        "get_theme",
        V13NodeWrapper(v13_nodes.get_theme_node, "get_theme"),
        {"phase": 0, "type": "initialization"},
    )

    # Phase 1: Agenda setting nodes
    registry.register_node(
        "agenda_proposal",
        V13NodeWrapper(v13_nodes.agenda_proposal_node, "agenda_proposal"),
        {"phase": 1, "type": "agenda_setting"},
    )

    registry.register_node(
        "topic_refinement",
        V13NodeWrapper(v13_nodes.topic_refinement_node, "topic_refinement"),
        {"phase": 1, "type": "agenda_setting"},
    )

    registry.register_node(
        "collate_proposals",
        V13NodeWrapper(v13_nodes.collate_proposals_node, "collate_proposals"),
        {"phase": 1, "type": "agenda_setting"},
    )

    registry.register_node(
        "agenda_voting",
        V13NodeWrapper(v13_nodes.agenda_voting_node, "agenda_voting"),
        {"phase": 1, "type": "agenda_setting"},
    )

    registry.register_node(
        "synthesize_agenda",
        V13NodeWrapper(v13_nodes.synthesize_agenda_node, "synthesize_agenda"),
        {"phase": 1, "type": "agenda_setting"},
    )

    registry.register_node(
        "agenda_approval",
        V13NodeWrapper(v13_nodes.agenda_approval_node, "agenda_approval"),
        {"phase": 1, "type": "agenda_setting"},
    )

    # Phase 2: Discussion nodes
    registry.register_node(
        "announce_item",
        V13NodeWrapper(v13_nodes.announce_item_node, "announce_item"),
        {"phase": 2, "type": "discussion"},
    )

    # Use unified discussion node if provided, otherwise legacy wrapper
    if unified_discussion_node:
        registry.register_node(
            "discussion_round",
            unified_discussion_node,
            {"phase": 2, "type": "discussion", "unified": True},
        )
    else:
        registry.register_node(
            "discussion_round",
            V13NodeWrapper(v13_nodes.discussion_round_node, "discussion_round"),
            {"phase": 2, "type": "discussion", "legacy": True},
        )

    registry.register_node(
        "round_summarization",
        V13NodeWrapper(v13_nodes.round_summarization_node, "round_summarization"),
        {"phase": 2, "type": "discussion"},
    )

    registry.register_node(
        "round_threshold_check",
        V13NodeWrapper(v13_nodes.round_threshold_check_node, "round_threshold_check"),
        {"phase": 2, "type": "discussion"},
    )

    registry.register_node(
        "end_topic_poll",
        V13NodeWrapper(v13_nodes.end_topic_poll_node, "end_topic_poll"),
        {"phase": 2, "type": "discussion"},
    )

    registry.register_node(
        "vote_evaluation",
        V13NodeWrapper(v13_nodes.vote_evaluation_node, "vote_evaluation"),
        {"phase": 2, "type": "discussion"},
    )

    registry.register_node(
        "periodic_user_stop",
        V13NodeWrapper(v13_nodes.periodic_user_stop_node, "periodic_user_stop"),
        {"phase": 2, "type": "user_interaction"},
    )

    registry.register_node(
        "user_topic_conclusion_confirmation",
        V13NodeWrapper(
            v13_nodes.user_topic_conclusion_confirmation_node,
            "user_topic_conclusion_confirmation",
        ),
        {"phase": 2, "type": "user_interaction"},
    )

    # Phase 3: Topic conclusion nodes
    registry.register_node(
        "final_considerations",
        V13NodeWrapper(v13_nodes.final_considerations_node, "final_considerations"),
        {"phase": 3, "type": "conclusion"},
    )

    registry.register_node(
        "topic_report_generation",
        V13NodeWrapper(
            v13_nodes.topic_report_generation_node, "topic_report_generation"
        ),
        {"phase": 3, "type": "reporting"},
    )

    registry.register_node(
        "topic_summary_generation",
        V13NodeWrapper(
            v13_nodes.topic_summary_generation_node, "topic_summary_generation"
        ),
        {"phase": 3, "type": "reporting"},
    )

    registry.register_node(
        "file_output",
        V13NodeWrapper(v13_nodes.file_output_node, "file_output"),
        {"phase": 3, "type": "output"},
    )

    # Phase 4: Continuation nodes
    registry.register_node(
        "agent_poll",
        V13NodeWrapper(v13_nodes.agent_poll_node, "agent_poll"),
        {"phase": 4, "type": "continuation"},
    )

    registry.register_node(
        "user_approval",
        V13NodeWrapper(v13_nodes.user_approval_node, "user_approval"),
        {"phase": 4, "type": "user_interaction"},
    )

    registry.register_node(
        "agenda_modification",
        V13NodeWrapper(v13_nodes.agenda_modification_node, "agenda_modification"),
        {"phase": 4, "type": "agenda_setting"},
    )

    # Phase 5: Final report nodes
    registry.register_node(
        "final_report_generation",
        V13NodeWrapper(v13_nodes.final_report_node, "final_report_generation"),
        {"phase": 5, "type": "reporting"},
    )

    registry.register_node(
        "multi_file_output",
        V13NodeWrapper(v13_nodes.multi_file_output_node, "multi_file_output"),
        {"phase": 5, "type": "output"},
    )

    # Legacy nodes for backward compatibility
    registry.register_node(
        "user_turn_participation",
        V13NodeWrapper(
            v13_nodes.user_turn_participation_node, "user_turn_participation"
        ),
        {"phase": 2, "type": "user_interaction", "legacy": True, "deprecated": True},
    )

    logger.info(
        f"Created default V13 registry with {len(registry.get_all_nodes())} nodes"
    )
    return registry


def create_hybrid_v13_registry(
    v13_nodes, agenda_node_factory=None, unified_discussion_node=None
) -> NodeRegistry:
    """Create hybrid registry with extracted nodes and V13 compatibility wrappers.

    This function creates a hybrid node registry that uses extracted FlowNode
    instances where available, and falls back to V13NodeWrapper for nodes
    that haven't been extracted yet. This enables gradual migration from
    the monolithic nodes_v13.py to focused individual nodes.

    Args:
        v13_nodes: V13FlowNodes instance for remaining legacy nodes
        agenda_node_factory: Factory for creating extracted agenda nodes (optional)
        unified_discussion_node: DiscussionRoundNode for discussion rounds (optional)

    Returns:
        NodeRegistry with hybrid extracted/wrapped nodes
    """
    registry = NodeRegistry()

    # Phase 0: Initialization nodes (still using V13 wrappers)
    registry.register_node(
        "config_and_keys",
        V13NodeWrapper(v13_nodes.config_and_keys_node, "config_and_keys"),
        {"phase": 0, "type": "initialization", "extracted": False},
    )

    registry.register_node(
        "agent_instantiation",
        V13NodeWrapper(v13_nodes.agent_instantiation_node, "agent_instantiation"),
        {"phase": 0, "type": "initialization", "extracted": False},
    )

    registry.register_node(
        "get_theme",
        V13NodeWrapper(v13_nodes.get_theme_node, "get_theme"),
        {"phase": 0, "type": "initialization", "extracted": False},
    )

    # Phase 1: Agenda nodes - USE EXTRACTED NODES WHERE AVAILABLE
    if agenda_node_factory:
        logger.info("Using AgendaNodeFactory for extracted agenda nodes")
        agenda_nodes = agenda_node_factory.create_all_agenda_nodes()

        for name, node in agenda_nodes.items():
            if node is not None:
                # Use extracted node
                registry.register_node(
                    name,
                    node,
                    {"phase": 1, "type": "agenda_setting", "extracted": True},
                )
                logger.info(f"Registered extracted agenda node: {name}")
            else:
                # Fallback to V13 wrapper
                logger.info(f"Using V13 wrapper fallback for agenda node: {name}")
                _register_v13_agenda_node(registry, v13_nodes, name)
    else:
        # No factory provided, use all V13 wrappers for agenda
        logger.info(
            "No AgendaNodeFactory provided, using V13 wrappers for all agenda nodes"
        )
        _register_all_v13_agenda_nodes(registry, v13_nodes)

    # Phase 2+: All other nodes still use V13 wrappers
    _register_remaining_v13_nodes(registry, v13_nodes, unified_discussion_node)

    # Log hybrid registry summary
    total_nodes = len(registry.get_all_nodes())
    extracted_count = len(
        [
            name
            for name, meta in registry._node_metadata.items()
            if meta.get("extracted", False)
        ]
    )
    v13_count = total_nodes - extracted_count

    logger.info(
        f"Created hybrid V13 registry with {total_nodes} total nodes: "
        f"{extracted_count} extracted, {v13_count} V13 wrappers"
    )

    return registry


def _register_v13_agenda_node(
    registry: NodeRegistry, v13_nodes, node_name: str
) -> None:
    """Register a single V13 agenda node wrapper.

    Args:
        registry: NodeRegistry to register with
        v13_nodes: V13FlowNodes instance
        node_name: Name of the agenda node
    """
    node_method_map = {
        "agenda_proposal": v13_nodes.agenda_proposal_node,
        "topic_refinement": v13_nodes.topic_refinement_node,
        "collate_proposals": v13_nodes.collate_proposals_node,
        "agenda_voting": v13_nodes.agenda_voting_node,
        "synthesize_agenda": v13_nodes.synthesize_agenda_node,
        "agenda_approval": v13_nodes.agenda_approval_node,
    }

    if node_name in node_method_map:
        registry.register_node(
            node_name,
            V13NodeWrapper(node_method_map[node_name], node_name),
            {"phase": 1, "type": "agenda_setting", "extracted": False},
        )
    else:
        logger.warning(f"Unknown agenda node: {node_name}")


def _register_all_v13_agenda_nodes(registry: NodeRegistry, v13_nodes) -> None:
    """Register all agenda nodes as V13 wrappers.

    Args:
        registry: NodeRegistry to register with
        v13_nodes: V13FlowNodes instance
    """
    agenda_nodes = [
        "agenda_proposal",
        "topic_refinement",
        "collate_proposals",
        "agenda_voting",
        "synthesize_agenda",
        "agenda_approval",
    ]

    for node_name in agenda_nodes:
        _register_v13_agenda_node(registry, v13_nodes, node_name)


def _register_remaining_v13_nodes(
    registry: NodeRegistry, v13_nodes, unified_discussion_node=None
) -> None:
    """Register all non-agenda nodes as V13 wrappers.

    Args:
        registry: NodeRegistry to register with
        v13_nodes: V13FlowNodes instance
        unified_discussion_node: Optional DiscussionRoundNode
    """
    # Phase 2: Discussion nodes
    registry.register_node(
        "announce_item",
        V13NodeWrapper(v13_nodes.announce_item_node, "announce_item"),
        {"phase": 2, "type": "discussion", "extracted": False},
    )

    # Use unified discussion node if provided, otherwise legacy wrapper
    if unified_discussion_node:
        registry.register_node(
            "discussion_round",
            unified_discussion_node,
            {"phase": 2, "type": "discussion", "unified": True, "extracted": True},
        )
    else:
        registry.register_node(
            "discussion_round",
            V13NodeWrapper(v13_nodes.discussion_round_node, "discussion_round"),
            {"phase": 2, "type": "discussion", "legacy": True, "extracted": False},
        )

    registry.register_node(
        "round_summarization",
        V13NodeWrapper(v13_nodes.round_summarization_node, "round_summarization"),
        {"phase": 2, "type": "discussion", "extracted": False},
    )

    registry.register_node(
        "round_threshold_check",
        V13NodeWrapper(v13_nodes.round_threshold_check_node, "round_threshold_check"),
        {"phase": 2, "type": "discussion", "extracted": False},
    )

    registry.register_node(
        "end_topic_poll",
        V13NodeWrapper(v13_nodes.end_topic_poll_node, "end_topic_poll"),
        {"phase": 2, "type": "discussion", "extracted": False},
    )

    registry.register_node(
        "vote_evaluation",
        V13NodeWrapper(v13_nodes.vote_evaluation_node, "vote_evaluation"),
        {"phase": 2, "type": "discussion", "extracted": False},
    )

    registry.register_node(
        "periodic_user_stop",
        V13NodeWrapper(v13_nodes.periodic_user_stop_node, "periodic_user_stop"),
        {"phase": 2, "type": "user_interaction", "extracted": False},
    )

    registry.register_node(
        "user_topic_conclusion_confirmation",
        V13NodeWrapper(
            v13_nodes.user_topic_conclusion_confirmation_node,
            "user_topic_conclusion_confirmation",
        ),
        {"phase": 2, "type": "user_interaction", "extracted": False},
    )

    # Phase 3: Topic conclusion nodes
    registry.register_node(
        "final_considerations",
        V13NodeWrapper(v13_nodes.final_considerations_node, "final_considerations"),
        {"phase": 3, "type": "conclusion", "extracted": False},
    )

    registry.register_node(
        "topic_report_generation",
        V13NodeWrapper(
            v13_nodes.topic_report_generation_node, "topic_report_generation"
        ),
        {"phase": 3, "type": "reporting", "extracted": False},
    )

    registry.register_node(
        "topic_summary_generation",
        V13NodeWrapper(
            v13_nodes.topic_summary_generation_node, "topic_summary_generation"
        ),
        {"phase": 3, "type": "reporting", "extracted": False},
    )

    registry.register_node(
        "file_output",
        V13NodeWrapper(v13_nodes.file_output_node, "file_output"),
        {"phase": 3, "type": "output", "extracted": False},
    )

    # Phase 4: Continuation nodes
    registry.register_node(
        "agent_poll",
        V13NodeWrapper(v13_nodes.agent_poll_node, "agent_poll"),
        {"phase": 4, "type": "continuation", "extracted": False},
    )

    registry.register_node(
        "user_approval",
        V13NodeWrapper(v13_nodes.user_approval_node, "user_approval"),
        {"phase": 4, "type": "user_interaction", "extracted": False},
    )

    registry.register_node(
        "agenda_modification",
        V13NodeWrapper(v13_nodes.agenda_modification_node, "agenda_modification"),
        {"phase": 4, "type": "agenda_setting", "extracted": False},
    )

    # Phase 5: Final report nodes
    registry.register_node(
        "final_report_generation",
        V13NodeWrapper(v13_nodes.final_report_node, "final_report_generation"),
        {"phase": 5, "type": "reporting", "extracted": False},
    )

    registry.register_node(
        "multi_file_output",
        V13NodeWrapper(v13_nodes.multi_file_output_node, "multi_file_output"),
        {"phase": 5, "type": "output", "extracted": False},
    )

    # Legacy nodes for backward compatibility
    registry.register_node(
        "user_turn_participation",
        V13NodeWrapper(
            v13_nodes.user_turn_participation_node, "user_turn_participation"
        ),
        {
            "phase": 2,
            "type": "user_interaction",
            "legacy": True,
            "deprecated": True,
            "extracted": False,
        },
    )
