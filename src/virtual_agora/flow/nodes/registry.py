"""Node registry system for Virtual Agora flow nodes.

This module provides dynamic node registration and management capabilities
for the pluggable node architecture. It allows nodes to be registered,
retrieved, and validated at runtime.
"""

from typing import Dict, Any, List, Optional, Type, Callable
from datetime import datetime
import logging
from copy import deepcopy

from .base import FlowNode, NodeDependencies, NodeExecutionContext
from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class NodeRegistryError(Exception):
    """Raised when node registry operations fail."""

    pass


class NodeRegistry:
    """Registry for managing flow nodes dynamically.

    This class provides a centralized registry for all flow nodes,
    allowing for dynamic registration, retrieval, and validation.
    It supports dependency injection and node lifecycle management.

    Key responsibilities:
    - Register and manage node instances
    - Provide node dependency injection
    - Validate node configurations
    - Track node execution statistics
    """

    def __init__(self, dependencies: Optional[NodeDependencies] = None):
        """Initialize node registry.

        Args:
            dependencies: Shared dependencies for all nodes
        """
        self.dependencies = dependencies
        self._nodes: Dict[str, FlowNode] = {}
        self._node_metadata: Dict[str, Dict[str, Any]] = {}
        self._execution_stats: Dict[str, Dict[str, Any]] = {}
        self._registration_history: List[Dict[str, Any]] = []

    def register_node(
        self,
        name: str,
        node_instance: FlowNode,
        metadata: Optional[Dict[str, Any]] = None,
        override: bool = False,
    ) -> None:
        """Register a node instance with the registry.

        Args:
            name: Unique name for the node
            node_instance: Node instance to register
            metadata: Optional metadata for the node
            override: Whether to override existing registration

        Raises:
            NodeRegistryError: If registration fails
        """
        if name in self._nodes and not override:
            raise NodeRegistryError(
                f"Node '{name}' is already registered. Use override=True to replace."
            )

        if not isinstance(node_instance, FlowNode):
            raise NodeRegistryError(
                f"Node must be an instance of FlowNode, got {type(node_instance)}"
            )

        # Inject dependencies if available
        if self.dependencies and hasattr(node_instance, "dependencies"):
            node_instance.dependencies = self.dependencies

        # Register the node
        self._nodes[name] = node_instance

        # Store metadata
        node_metadata = {
            "node_class": node_instance.__class__.__name__,
            "registration_time": datetime.now().isoformat(),
            "node_type": self._determine_node_type(node_instance),
            "custom_metadata": metadata or {},
        }
        self._node_metadata[name] = node_metadata

        # Initialize execution stats
        self._execution_stats[name] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0,
            "last_execution": None,
            "last_error": None,
        }

        # Record registration
        self._registration_history.append(
            {
                "action": "register",
                "node_name": name,
                "node_class": node_instance.__class__.__name__,
                "timestamp": datetime.now().isoformat(),
                "override": override,
            }
        )

        logger.info(
            f"Registered node '{name}' of type {node_instance.__class__.__name__}"
        )

    def register_node_class(
        self,
        name: str,
        node_class: Type[FlowNode],
        init_args: Optional[List[Any]] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        override: bool = False,
    ) -> None:
        """Register a node by instantiating its class.

        Args:
            name: Unique name for the node
            node_class: Node class to instantiate
            init_args: Arguments for node instantiation
            init_kwargs: Keyword arguments for node instantiation
            metadata: Optional metadata for the node
            override: Whether to override existing registration

        Raises:
            NodeRegistryError: If registration fails
        """
        try:
            # Prepare initialization arguments
            args = init_args or []
            kwargs = init_kwargs or {}

            # Add dependencies to kwargs if available
            if self.dependencies and "node_dependencies" not in kwargs:
                kwargs["node_dependencies"] = self.dependencies

            # Instantiate the node
            node_instance = node_class(*args, **kwargs)

            # Register the instance
            self.register_node(name, node_instance, metadata, override)

        except Exception as e:
            raise NodeRegistryError(
                f"Failed to register node class {node_class.__name__}: {e}"
            )

    def get_node(self, name: str) -> FlowNode:
        """Get a registered node by name.

        Args:
            name: Name of the node to retrieve

        Returns:
            Node instance

        Raises:
            NodeRegistryError: If node is not found
        """
        if name not in self._nodes:
            available_nodes = list(self._nodes.keys())
            raise NodeRegistryError(
                f"Node '{name}' not found. Available nodes: {available_nodes}"
            )

        return self._nodes[name]

    def execute_node(self, name: str, state: VirtualAgoraState) -> Dict[str, Any]:
        """Execute a registered node with tracking and statistics.

        Args:
            name: Name of the node to execute
            state: Current state

        Returns:
            State updates from node execution

        Raises:
            NodeRegistryError: If node is not found or execution fails
        """
        node = self.get_node(name)

        # Create execution context
        context = NodeExecutionContext(node, state)

        try:
            # Execute the node safely
            result = node.safe_execute(state)

            # Check if execution resulted in an error
            if "last_error" in result:
                # This indicates safe_execute caught an exception and returned error state
                error = RuntimeError(result["last_error"])
                context.mark_failed(error)
                self._update_execution_stats(name, context)
                logger.error(f"Node '{name}' execution failed: {result['last_error']}")
                raise NodeRegistryError(
                    f"Node '{name}' execution failed: {result['last_error']}"
                )

            # Mark as completed
            context.mark_completed(result)

            # Update statistics
            self._update_execution_stats(name, context)

            logger.debug(
                f"Node '{name}' executed successfully in {context.get_execution_time():.2f}s"
            )
            return result

        except NodeRegistryError:
            # Re-raise NodeRegistryError without wrapping
            raise
        except Exception as e:
            # Mark as failed
            context.mark_failed(e)

            # Update statistics
            self._update_execution_stats(name, context)

            logger.error(f"Node '{name}' execution failed: {e}")
            raise NodeRegistryError(f"Node '{name}' execution failed: {e}")

    def unregister_node(self, name: str) -> bool:
        """Unregister a node from the registry.

        Args:
            name: Name of the node to unregister

        Returns:
            True if node was unregistered, False if not found
        """
        if name not in self._nodes:
            return False

        # Remove node and metadata
        del self._nodes[name]
        del self._node_metadata[name]
        del self._execution_stats[name]

        # Record unregistration
        self._registration_history.append(
            {
                "action": "unregister",
                "node_name": name,
                "timestamp": datetime.now().isoformat(),
            }
        )

        logger.info(f"Unregistered node '{name}'")
        return True

    def list_nodes(self) -> List[str]:
        """Get list of all registered node names.

        Returns:
            List of registered node names
        """
        return list(self._nodes.keys())

    def get_node_info(self, name: str) -> Dict[str, Any]:
        """Get detailed information about a registered node.

        Args:
            name: Name of the node

        Returns:
            Dictionary with node information

        Raises:
            NodeRegistryError: If node is not found
        """
        if name not in self._nodes:
            raise NodeRegistryError(f"Node '{name}' not found")

        node = self._nodes[name]
        metadata = self._node_metadata[name]
        stats = self._execution_stats[name]

        return {
            "name": name,
            "class": node.__class__.__name__,
            "module": node.__class__.__module__,
            "metadata": metadata,
            "execution_stats": stats,
            "is_available": True,
            "dependencies_injected": hasattr(node, "dependencies")
            and node.dependencies is not None,
        }

    def validate_all_nodes(self, state: VirtualAgoraState) -> Dict[str, Dict[str, Any]]:
        """Validate all registered nodes against current state.

        Args:
            state: Current state to validate against

        Returns:
            Dictionary mapping node names to validation results
        """
        validation_results = {}

        for name, node in self._nodes.items():
            try:
                # Clear previous validation errors
                node._validation_errors.clear()

                # Validate preconditions
                is_valid = node.validate_preconditions(state)

                validation_results[name] = {
                    "valid": is_valid,
                    "errors": node.get_validation_errors(),
                    "node_class": node.__class__.__name__,
                    "validation_time": datetime.now().isoformat(),
                }

            except Exception as e:
                validation_results[name] = {
                    "valid": False,
                    "errors": [f"Validation exception: {e}"],
                    "node_class": node.__class__.__name__,
                    "validation_time": datetime.now().isoformat(),
                }

        return validation_results

    def get_nodes_by_type(self, node_type: str) -> Dict[str, FlowNode]:
        """Get all nodes of a specific type.

        Args:
            node_type: Type of nodes to retrieve ('flow', 'hitl', 'agent_orchestrator')

        Returns:
            Dictionary mapping node names to node instances
        """
        matching_nodes = {}

        for name, node in self._nodes.items():
            if self._node_metadata[name]["node_type"] == node_type:
                matching_nodes[name] = node

        return matching_nodes

    def get_execution_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get execution statistics for all nodes.

        Returns:
            Dictionary mapping node names to their execution statistics
        """
        return deepcopy(self._execution_stats)

    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary information about the registry.

        Returns:
            Dictionary with registry summary
        """
        node_types = {}
        for metadata in self._node_metadata.values():
            node_type = metadata["node_type"]
            node_types[node_type] = node_types.get(node_type, 0) + 1

        total_executions = sum(
            stats["total_executions"] for stats in self._execution_stats.values()
        )
        total_failures = sum(
            stats["failed_executions"] for stats in self._execution_stats.values()
        )

        return {
            "total_nodes": len(self._nodes),
            "node_types": node_types,
            "total_executions": total_executions,
            "total_failures": total_failures,
            "success_rate": (total_executions - total_failures)
            / max(1, total_executions),
            "dependencies_available": self.dependencies is not None,
            "registry_creation_time": min(
                (
                    metadata["registration_time"]
                    for metadata in self._node_metadata.values()
                ),
                default=None,
            ),
        }

    def clear_registry(self) -> None:
        """Clear all registered nodes from the registry.

        Warning: This removes all nodes and their statistics.
        """
        node_count = len(self._nodes)

        self._nodes.clear()
        self._node_metadata.clear()
        self._execution_stats.clear()

        # Record clearing
        self._registration_history.append(
            {
                "action": "clear_all",
                "nodes_removed": node_count,
                "timestamp": datetime.now().isoformat(),
            }
        )

        logger.warning(f"Cleared all {node_count} nodes from registry")

    def _determine_node_type(self, node: FlowNode) -> str:
        """Determine the type of a node based on its class hierarchy.

        Args:
            node: Node instance to analyze

        Returns:
            String indicating node type
        """
        # Import here to avoid circular imports
        from .base import HITLNode, AgentOrchestratorNode

        if isinstance(node, HITLNode):
            return "hitl"
        elif isinstance(node, AgentOrchestratorNode):
            return "agent_orchestrator"
        else:
            return "flow"

    def _update_execution_stats(self, name: str, context: NodeExecutionContext) -> None:
        """Update execution statistics for a node.

        Args:
            name: Node name
            context: Execution context with results
        """
        stats = self._execution_stats[name]

        # Update counters
        stats["total_executions"] += 1

        if context.success:
            stats["successful_executions"] += 1
        else:
            stats["failed_executions"] += 1
            stats["last_error"] = (
                str(context.error) if context.error else "Unknown error"
            )

        # Update timing
        execution_time = context.get_execution_time()
        stats["total_execution_time"] += execution_time
        stats["average_execution_time"] = (
            stats["total_execution_time"] / stats["total_executions"]
        )
        stats["last_execution"] = context.start_time.isoformat()


class CompatibilityNodeRegistry(NodeRegistry):
    """Node registry with backward compatibility for V13FlowNodes.

    This registry provides compatibility with the existing V13FlowNodes
    class while gradually transitioning to the new node architecture.
    """

    def __init__(
        self,
        dependencies: Optional[NodeDependencies] = None,
        v13_nodes: Optional[Any] = None,
    ):
        """Initialize compatibility registry.

        Args:
            dependencies: Shared dependencies for all nodes
            v13_nodes: Existing V13FlowNodes instance for compatibility
        """
        super().__init__(dependencies)
        self.v13_nodes = v13_nodes
        self._compatibility_mode = v13_nodes is not None

    def register_v13_compatibility_wrapper(self, name: str, method_name: str) -> None:
        """Register a compatibility wrapper for V13FlowNodes methods.

        Args:
            name: Node name
            method_name: Method name in V13FlowNodes
        """
        if not self._compatibility_mode:
            raise NodeRegistryError("V13 compatibility mode not enabled")

        if not hasattr(self.v13_nodes, method_name):
            raise NodeRegistryError(f"V13FlowNodes has no method '{method_name}'")

        # Create wrapper node
        wrapper = V13CompatibilityWrapper(self.v13_nodes, method_name)

        # Register the wrapper
        self.register_node(
            name,
            wrapper,
            {
                "compatibility_wrapper": True,
                "v13_method": method_name,
                "wrapper_type": "v13_compatibility",
            },
        )

        logger.info(
            f"Registered V13 compatibility wrapper for '{name}' -> '{method_name}'"
        )

    def migrate_from_v13(self, node_mappings: Dict[str, str]) -> None:
        """Migrate all V13 nodes to compatibility wrappers.

        Args:
            node_mappings: Dictionary mapping node names to V13FlowNodes method names
        """
        if not self._compatibility_mode:
            raise NodeRegistryError("V13 compatibility mode not enabled")

        for node_name, method_name in node_mappings.items():
            try:
                self.register_v13_compatibility_wrapper(node_name, method_name)
            except Exception as e:
                logger.error(f"Failed to migrate V13 node '{node_name}': {e}")

        logger.info(
            f"Migrated {len(node_mappings)} V13 nodes to compatibility wrappers"
        )


class V13CompatibilityWrapper(FlowNode):
    """Wrapper node for V13FlowNodes method compatibility.

    This wrapper allows V13FlowNodes methods to be used as FlowNode
    instances while maintaining the new interface.
    """

    def __init__(self, v13_nodes: Any, method_name: str):
        """Initialize compatibility wrapper.

        Args:
            v13_nodes: V13FlowNodes instance
            method_name: Method name to wrap
        """
        super().__init__()
        self.v13_nodes = v13_nodes
        self.method_name = method_name
        self.method = getattr(v13_nodes, method_name)

    def execute(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Execute the wrapped V13 method.

        Args:
            state: Current state

        Returns:
            State updates from V13 method
        """
        return self.method(state)

    def validate_preconditions(self, state: VirtualAgoraState) -> bool:
        """Basic validation for V13 compatibility.

        Args:
            state: Current state

        Returns:
            True if state is valid dict, False otherwise
        """
        if not isinstance(state, dict):
            self._validation_errors.append("State must be a dictionary")
            return False

        return True

    def get_node_name(self) -> str:
        """Get node name for V13 compatibility wrapper.

        Returns:
            Method name as node name
        """
        return f"V13_{self.method_name}"
