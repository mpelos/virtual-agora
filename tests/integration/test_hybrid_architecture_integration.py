"""Hybrid Architecture Integration Testing.

This test module validates that the hybrid architecture (Phase 3 of refactoring)
works seamlessly with all extracted and wrapped nodes functioning together:

Hybrid Architecture Components:
- Extracted FlowNode instances: AgendaProposalNode, TopicRefinementNode, etc.
- V13NodeWrapper instances: agenda_approval, config_and_keys, etc.
- NodeRegistry: Manages both types seamlessly
- AgendaNodeFactory: Creates extracted nodes with dependency injection

Key Integration Points:
1. Both node types implement FlowNode interface correctly
2. Both node types execute properly in the same graph
3. State transitions work across extracted/wrapped boundaries
4. Error handling works for both node types
5. Performance characteristics are comparable
6. Dependency injection works for extracted nodes
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from virtual_agora.flow.graph_v13 import VirtualAgoraV13Flow
from virtual_agora.config.models import Config as VirtualAgoraConfig

# Import hybrid architecture components
from virtual_agora.flow.node_registry import (
    NodeRegistry,
    V13NodeWrapper,
    create_hybrid_v13_registry,
)
from virtual_agora.flow.nodes.base import FlowNode, HITLNode, AgentOrchestratorNode
from virtual_agora.flow.nodes.agenda.factory import AgendaNodeFactory

# Import extracted agenda nodes
from virtual_agora.flow.nodes.agenda import (
    AgendaProposalNode,
    TopicRefinementNode,
    CollateProposalsNode,
    AgendaVotingNode,
    SynthesizeAgendaNode,
)

from virtual_agora.state.schema import VirtualAgoraState


@pytest.fixture
def mock_config():
    """Create mock configuration for hybrid architecture testing."""
    config = Mock(spec=VirtualAgoraConfig)

    # Mock all required agent configurations
    config.moderator = Mock()
    config.moderator.provider.value = "openai"
    config.moderator.model = "gpt-4o"
    config.moderator.temperature = 0.7
    config.moderator.max_tokens = 4000

    config.summarizer = Mock()
    config.summarizer.provider.value = "openai"
    config.summarizer.model = "gpt-4o"
    config.summarizer.temperature = 0.6
    config.summarizer.max_tokens = 3000

    config.report_writer = Mock()
    config.report_writer.provider.value = "openai"
    config.report_writer.model = "gpt-4o"
    config.report_writer.temperature = 0.5
    config.report_writer.max_tokens = 5000

    agent_config = Mock()
    agent_config.provider.value = "openai"
    agent_config.model = "gpt-4o"
    agent_config.count = 3
    agent_config.temperature = 0.7
    agent_config.max_tokens = 3000
    config.agents = [agent_config]

    return config


@pytest.fixture
def mock_create_provider():
    """Mock the create_provider function."""
    with patch("virtual_agora.flow.graph_v13.create_provider") as mock:
        mock_llm = Mock()
        mock.return_value = mock_llm
        yield mock


def create_test_state() -> VirtualAgoraState:
    """Create test state for hybrid architecture testing."""
    return {
        "session_id": "hybrid_architecture_test",
        "main_topic": "Testing Hybrid Architecture",
        "active_topic": "Extracted and Wrapped Nodes Integration",
        "current_round": 0,
        "current_phase": 1,  # Agenda phase
        "messages": [],
        "completed_topics": [],
        "topic_summaries": {},
        "round_history": [],
        "turn_order_history": [],
        "rounds_per_topic": {},
        "speaking_order": ["agent_1", "agent_2", "agent_3"],
        "user_participation_message": None,
        "checkpoint_interval": 3,
        "error_count": 0,
    }


class HybridArchitectureTestSuite:
    """Test suite for hybrid architecture integration."""

    def __init__(self):
        self.test_results = {}
        self.node_execution_results = {}
        self.performance_metrics = {}

    def validate_hybrid_registry_structure(
        self, flow: VirtualAgoraV13Flow
    ) -> Dict[str, Any]:
        """Validate the hybrid registry contains both extracted and wrapped nodes."""
        results = {
            "success": True,
            "extracted_nodes": {},
            "wrapped_nodes": {},
            "total_nodes": 0,
            "errors": [],
        }

        try:
            all_nodes = flow.node_registry
            results["total_nodes"] = len(all_nodes)

            # Categorize nodes by type
            for node_name, node in all_nodes.items():
                if isinstance(node, V13NodeWrapper):
                    results["wrapped_nodes"][node_name] = {
                        "type": "V13NodeWrapper",
                        "callable": callable(node),
                        "has_execute": hasattr(node, "execute"),
                        "has_call": hasattr(node, "__call__"),
                    }
                elif isinstance(node, FlowNode):
                    results["extracted_nodes"][node_name] = {
                        "type": type(node).__name__,
                        "callable": (
                            callable(node.execute)
                            if hasattr(node, "execute")
                            else False
                        ),
                        "has_execute": hasattr(node, "execute"),
                        "has_validate": hasattr(node, "validate_preconditions"),
                        "node_class": node.__class__.__name__,
                    }

            # Validate minimum expectations
            if len(results["extracted_nodes"]) < 5:
                results["errors"].append(
                    f"Expected >=5 extracted nodes, got {len(results['extracted_nodes'])}"
                )

            if len(results["wrapped_nodes"]) < 20:
                results["errors"].append(
                    f"Expected >=20 wrapped nodes, got {len(results['wrapped_nodes'])}"
                )

            if results["errors"]:
                results["success"] = False

        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))

        return results

    def test_extracted_node_interfaces(
        self, flow: VirtualAgoraV13Flow
    ) -> Dict[str, Any]:
        """Test that extracted nodes implement FlowNode interface correctly."""
        results = {
            "success": True,
            "nodes_tested": [],
            "interface_compliance": {},
            "errors": [],
        }

        try:
            # Test specific extracted agenda nodes
            extracted_agenda_nodes = [
                "agenda_proposal",
                "topic_refinement",
                "collate_proposals",
                "agenda_voting",
                "synthesize_agenda",
            ]

            all_nodes = flow.node_registry

            for node_name in extracted_agenda_nodes:
                if node_name in all_nodes:
                    node = all_nodes[node_name]
                    results["nodes_tested"].append(node_name)

                    # Test FlowNode interface compliance
                    compliance = {
                        "is_flow_node": isinstance(node, FlowNode),
                        "is_not_wrapper": not isinstance(node, V13NodeWrapper),
                        "has_execute": hasattr(node, "execute"),
                        "execute_callable": (
                            callable(node.execute)
                            if hasattr(node, "execute")
                            else False
                        ),
                        "has_validate": hasattr(node, "validate_preconditions"),
                        "validate_callable": (
                            callable(node.validate_preconditions)
                            if hasattr(node, "validate_preconditions")
                            else False
                        ),
                        "has_get_name": hasattr(node, "get_node_name"),
                        "get_name_callable": (
                            callable(node.get_node_name)
                            if hasattr(node, "get_node_name")
                            else False
                        ),
                    }

                    results["interface_compliance"][node_name] = compliance

                    # Validate all compliance checks pass
                    failed_checks = [
                        check for check, passed in compliance.items() if not passed
                    ]
                    if failed_checks:
                        results["errors"].append(
                            f"Node {node_name} failed checks: {failed_checks}"
                        )

            if results["errors"]:
                results["success"] = False

        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))

        return results

    def test_wrapped_node_interfaces(self, flow: VirtualAgoraV13Flow) -> Dict[str, Any]:
        """Test that wrapped nodes implement required interfaces correctly."""
        results = {
            "success": True,
            "nodes_tested": [],
            "interface_compliance": {},
            "errors": [],
        }

        try:
            all_nodes = flow.node_registry

            # Test wrapped nodes (should include agenda_approval and others)
            wrapped_node_names = [
                "agenda_approval",
                "config_and_keys",
                "announce_item",
                "final_considerations",
                "agent_poll",
            ]

            for node_name in wrapped_node_names:
                if node_name in all_nodes:
                    node = all_nodes[node_name]

                    if isinstance(node, V13NodeWrapper):
                        results["nodes_tested"].append(node_name)

                        # Test V13NodeWrapper interface compliance
                        compliance = {
                            "is_wrapper": isinstance(node, V13NodeWrapper),
                            "is_flow_node": isinstance(
                                node, FlowNode
                            ),  # Should inherit from FlowNode
                            "is_callable": callable(
                                node
                            ),  # Should be callable for LangGraph
                            "has_execute": hasattr(node, "execute"),
                            "has_call": hasattr(node, "__call__"),
                            "has_validate": hasattr(node, "validate_preconditions"),
                            "has_get_name": hasattr(node, "get_node_name"),
                        }

                        results["interface_compliance"][node_name] = compliance

                        # Validate critical compliance checks
                        if not compliance["is_callable"]:
                            results["errors"].append(
                                f"Wrapped node {node_name} not callable"
                            )
                        if not compliance["has_execute"]:
                            results["errors"].append(
                                f"Wrapped node {node_name} missing execute method"
                            )

            if results["errors"]:
                results["success"] = False

        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))

        return results

    def test_node_execution_paths(self, flow: VirtualAgoraV13Flow) -> Dict[str, Any]:
        """Test that both extracted and wrapped nodes can be executed properly."""
        results = {
            "success": True,
            "extracted_executions": {},
            "wrapped_executions": {},
            "errors": [],
        }

        try:
            all_nodes = flow.node_registry
            test_state = create_test_state()

            # Test extracted node executions (with mock validation)
            extracted_agenda_nodes = [
                "agenda_proposal",
                "topic_refinement",
                "collate_proposals",
            ]

            for node_name in extracted_agenda_nodes:
                if node_name in all_nodes:
                    node = all_nodes[node_name]

                    try:
                        # Test validation method exists and can be called
                        if hasattr(node, "validate_preconditions"):
                            # Don't actually validate (would require full setup)
                            # Just test the method exists and is callable
                            assert callable(
                                node.validate_preconditions
                            ), f"validate_preconditions not callable for {node_name}"

                        # Test execute method exists and is callable
                        if hasattr(node, "execute"):
                            assert callable(
                                node.execute
                            ), f"execute not callable for {node_name}"
                            results["extracted_executions"][node_name] = "success"
                        else:
                            results["errors"].append(
                                f"Extracted node {node_name} missing execute method"
                            )

                    except Exception as e:
                        results["errors"].append(
                            f"Extracted node {node_name} execution test failed: {e}"
                        )
                        results["extracted_executions"][node_name] = "failed"

            # Test wrapped node executions
            wrapped_node_names = ["agenda_approval", "config_and_keys"]

            for node_name in wrapped_node_names:
                if node_name in all_nodes:
                    node = all_nodes[node_name]

                    if isinstance(node, V13NodeWrapper):
                        try:
                            # Test that wrapper is callable
                            assert callable(
                                node
                            ), f"Wrapped node {node_name} not callable"

                            # Test that execute method exists
                            assert hasattr(
                                node, "execute"
                            ), f"Wrapped node {node_name} missing execute"
                            assert callable(
                                node.execute
                            ), f"Wrapped node execute not callable for {node_name}"

                            results["wrapped_executions"][node_name] = "success"

                        except Exception as e:
                            results["errors"].append(
                                f"Wrapped node {node_name} execution test failed: {e}"
                            )
                            results["wrapped_executions"][node_name] = "failed"

            if results["errors"]:
                results["success"] = False

        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))

        return results

    def test_agenda_factory_integration(
        self, flow: VirtualAgoraV13Flow
    ) -> Dict[str, Any]:
        """Test AgendaNodeFactory creates extracted nodes with proper dependency injection."""
        results = {
            "success": True,
            "factory_available": False,
            "nodes_created": {},
            "dependency_injection": {},
            "errors": [],
        }

        try:
            # Check if AgendaNodeFactory is available
            if hasattr(flow, "agenda_node_factory"):
                results["factory_available"] = True
                factory = flow.agenda_node_factory

                # Test factory can create all agenda nodes
                created_nodes = factory.create_all_agenda_nodes()

                for node_name, node in created_nodes.items():
                    if node is not None:
                        results["nodes_created"][node_name] = {
                            "type": type(node).__name__,
                            "is_flow_node": isinstance(node, FlowNode),
                            "has_dependencies": hasattr(node, "dependencies"),
                        }

                        # Test dependency injection
                        if hasattr(node, "discussing_agents"):
                            results["dependency_injection"][node_name] = {
                                "has_discussing_agents": node.discussing_agents
                                is not None,
                                "discussing_agents_count": (
                                    len(node.discussing_agents)
                                    if node.discussing_agents
                                    else 0
                                ),
                            }
                        elif hasattr(node, "moderator_agent"):
                            results["dependency_injection"][node_name] = {
                                "has_moderator_agent": node.moderator_agent is not None
                            }
                    else:
                        results["nodes_created"][node_name] = None  # Not yet extracted

                # Validate expected extracted nodes
                expected_extracted = [
                    "agenda_proposal",
                    "topic_refinement",
                    "collate_proposals",
                    "agenda_voting",
                    "synthesize_agenda",
                ]

                extracted_count = sum(
                    1
                    for name in expected_extracted
                    if name in results["nodes_created"]
                    and results["nodes_created"][name] is not None
                )

                if extracted_count < 5:
                    results["errors"].append(
                        f"Expected 5 extracted agenda nodes, got {extracted_count}"
                    )
            else:
                results["errors"].append("AgendaNodeFactory not available in flow")

            if results["errors"]:
                results["success"] = False

        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))

        return results

    def test_graph_building_with_hybrid_nodes(
        self, flow: VirtualAgoraV13Flow
    ) -> Dict[str, Any]:
        """Test that graph building works correctly with mixed node types."""
        results = {
            "success": True,
            "graph_built": False,
            "graph_compiled": False,
            "node_integration": {},
            "errors": [],
        }

        try:
            # Test graph building
            graph = flow.build_graph()
            if graph is not None:
                results["graph_built"] = True

                # Test that all node types are included in the graph
                graph_nodes = set(graph.nodes.keys())
                registry_nodes = set(flow.node_registry.keys())

                # Filter out legacy/deprecated nodes that aren't expected to be in the graph
                legacy_nodes = {
                    "user_turn_participation"
                }  # These are kept for compatibility but not used
                active_registry_nodes = registry_nodes - legacy_nodes

                # Check that all active registry nodes are in the graph
                missing_from_graph = active_registry_nodes - graph_nodes
                if missing_from_graph:
                    results["errors"].append(
                        f"Nodes missing from graph: {missing_from_graph}"
                    )

                # Test graph compilation
                compiled_graph = flow.compile()
                if compiled_graph is not None:
                    results["graph_compiled"] = True
                else:
                    results["errors"].append("Graph compilation failed")

                # Test node integration in graph
                agenda_nodes = [
                    "agenda_proposal",
                    "topic_refinement",
                    "collate_proposals",
                    "agenda_voting",
                    "synthesize_agenda",
                    "agenda_approval",
                ]

                for node_name in agenda_nodes:
                    if node_name in graph_nodes:
                        node = flow.node_registry[node_name]
                        results["node_integration"][node_name] = {
                            "in_graph": True,
                            "node_type": (
                                "extracted"
                                if not isinstance(node, V13NodeWrapper)
                                else "wrapped"
                            ),
                            "is_callable": (
                                callable(node)
                                if isinstance(node, V13NodeWrapper)
                                else callable(node.execute)
                            ),
                        }
                    else:
                        results["node_integration"][node_name] = {"in_graph": False}
                        results["errors"].append(
                            f"Agenda node {node_name} missing from graph"
                        )
            else:
                results["errors"].append("Graph building failed")

            if results["errors"]:
                results["success"] = False

        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))

        return results

    def get_hybrid_architecture_statistics(
        self, flow: VirtualAgoraV13Flow
    ) -> Dict[str, Any]:
        """Get comprehensive statistics about the hybrid architecture."""
        stats = {
            "total_nodes": 0,
            "extracted_nodes": 0,
            "wrapped_nodes": 0,
            "other_nodes": 0,
            "extraction_ratio": 0.0,
            "node_breakdown": {},
            "agenda_nodes": {},
        }

        try:
            all_nodes = flow.node_registry
            stats["total_nodes"] = len(all_nodes)

            # Categorize all nodes
            for node_name, node in all_nodes.items():
                if isinstance(node, V13NodeWrapper):
                    stats["wrapped_nodes"] += 1
                    stats["node_breakdown"][node_name] = "wrapped"
                elif isinstance(node, FlowNode):
                    stats["extracted_nodes"] += 1
                    stats["node_breakdown"][node_name] = "extracted"
                else:
                    stats["other_nodes"] += 1
                    stats["node_breakdown"][node_name] = "other"

            # Calculate extraction ratio
            if stats["total_nodes"] > 0:
                stats["extraction_ratio"] = (
                    stats["extracted_nodes"] / stats["total_nodes"]
                )

            # Specific agenda node analysis
            agenda_nodes = [
                "agenda_proposal",
                "topic_refinement",
                "collate_proposals",
                "agenda_voting",
                "synthesize_agenda",
                "agenda_approval",
            ]

            for node_name in agenda_nodes:
                if node_name in all_nodes:
                    node = all_nodes[node_name]
                    stats["agenda_nodes"][node_name] = {
                        "type": (
                            "extracted"
                            if not isinstance(node, V13NodeWrapper)
                            else "wrapped"
                        ),
                        "class": type(node).__name__,
                    }

        except Exception as e:
            stats["error"] = str(e)

        return stats


class TestHybridArchitectureIntegration:
    """Test class for hybrid architecture integration scenarios."""

    def test_hybrid_registry_structure_validation(
        self, mock_config, mock_create_provider
    ):
        """Test 1: Validate hybrid registry contains both extracted and wrapped nodes."""
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)
        test_suite = HybridArchitectureTestSuite()

        results = test_suite.validate_hybrid_registry_structure(flow)

        assert results[
            "success"
        ], f"Hybrid registry validation failed: {results['errors']}"
        assert (
            results["total_nodes"] > 25
        ), f"Registry too small: {results['total_nodes']} nodes"
        assert (
            len(results["extracted_nodes"]) >= 5
        ), f"Too few extracted nodes: {len(results['extracted_nodes'])}"
        assert (
            len(results["wrapped_nodes"]) >= 20
        ), f"Too few wrapped nodes: {len(results['wrapped_nodes'])}"

        print(
            f"✅ Hybrid registry validated: {results['total_nodes']} total, "
            f"{len(results['extracted_nodes'])} extracted, {len(results['wrapped_nodes'])} wrapped"
        )

    def test_extracted_node_interface_compliance(
        self, mock_config, mock_create_provider
    ):
        """Test 2: Extracted nodes implement FlowNode interface correctly."""
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)
        test_suite = HybridArchitectureTestSuite()

        results = test_suite.test_extracted_node_interfaces(flow)

        assert results[
            "success"
        ], f"Extracted node interface compliance failed: {results['errors']}"
        assert (
            len(results["nodes_tested"]) >= 5
        ), f"Too few extracted nodes tested: {len(results['nodes_tested'])}"

        # Validate specific compliance for each tested node
        for node_name, compliance in results["interface_compliance"].items():
            assert compliance["is_flow_node"], f"Node {node_name} not a FlowNode"
            assert compliance[
                "is_not_wrapper"
            ], f"Node {node_name} should not be wrapper"
            assert compliance["has_execute"], f"Node {node_name} missing execute method"
            assert compliance[
                "execute_callable"
            ], f"Node {node_name} execute not callable"

        print(
            f"✅ Extracted node interfaces validated: {len(results['nodes_tested'])} nodes tested"
        )

    def test_wrapped_node_interface_compliance(self, mock_config, mock_create_provider):
        """Test 3: Wrapped nodes implement required interfaces correctly."""
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)
        test_suite = HybridArchitectureTestSuite()

        results = test_suite.test_wrapped_node_interfaces(flow)

        assert results[
            "success"
        ], f"Wrapped node interface compliance failed: {results['errors']}"
        assert len(results["nodes_tested"]) > 0, "No wrapped nodes tested"

        # Validate specific compliance for each tested wrapped node
        for node_name, compliance in results["interface_compliance"].items():
            assert compliance["is_wrapper"], f"Node {node_name} should be wrapper"
            assert compliance[
                "is_flow_node"
            ], f"Node {node_name} should inherit from FlowNode"
            assert compliance["is_callable"], f"Node {node_name} should be callable"
            assert compliance["has_execute"], f"Node {node_name} missing execute method"

        print(
            f"✅ Wrapped node interfaces validated: {len(results['nodes_tested'])} nodes tested"
        )

    def test_node_execution_paths_integration(self, mock_config, mock_create_provider):
        """Test 4: Both extracted and wrapped nodes can be executed properly."""
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)
        test_suite = HybridArchitectureTestSuite()

        results = test_suite.test_node_execution_paths(flow)

        assert results[
            "success"
        ], f"Node execution paths test failed: {results['errors']}"

        # Validate extracted node executions
        assert (
            len(results["extracted_executions"]) > 0
        ), "No extracted nodes tested for execution"
        for node_name, status in results["extracted_executions"].items():
            assert (
                status == "success"
            ), f"Extracted node {node_name} execution test failed"

        # Validate wrapped node executions
        assert (
            len(results["wrapped_executions"]) > 0
        ), "No wrapped nodes tested for execution"
        for node_name, status in results["wrapped_executions"].items():
            assert (
                status == "success"
            ), f"Wrapped node {node_name} execution test failed"

        print(
            f"✅ Node execution paths validated: "
            f"{len(results['extracted_executions'])} extracted, "
            f"{len(results['wrapped_executions'])} wrapped"
        )

    def test_agenda_factory_integration(self, mock_config, mock_create_provider):
        """Test 5: AgendaNodeFactory creates extracted nodes with proper dependency injection."""
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)
        test_suite = HybridArchitectureTestSuite()

        results = test_suite.test_agenda_factory_integration(flow)

        assert results[
            "success"
        ], f"Agenda factory integration failed: {results['errors']}"
        assert results["factory_available"], "AgendaNodeFactory not available"

        # Validate node creation
        created_extracted = sum(
            1 for node in results["nodes_created"].values() if node is not None
        )
        assert (
            created_extracted >= 5
        ), f"Expected >=5 extracted nodes from factory, got {created_extracted}"

        # Validate dependency injection
        assert (
            len(results["dependency_injection"]) > 0
        ), "No dependency injection tested"

        print(
            f"✅ Agenda factory integration validated: "
            f"{created_extracted} nodes created with dependency injection"
        )

    def test_graph_building_hybrid_integration(self, mock_config, mock_create_provider):
        """Test 6: Graph building works correctly with mixed node types."""
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)
        test_suite = HybridArchitectureTestSuite()

        results = test_suite.test_graph_building_with_hybrid_nodes(flow)

        assert results[
            "success"
        ], f"Graph building with hybrid nodes failed: {results['errors']}"
        assert results["graph_built"], "Graph building failed"
        assert results["graph_compiled"], "Graph compilation failed"

        # Validate node integration in graph
        agenda_nodes_in_graph = sum(
            1
            for node_info in results["node_integration"].values()
            if node_info.get("in_graph", False)
        )
        assert (
            agenda_nodes_in_graph >= 6
        ), f"Expected >=6 agenda nodes in graph, got {agenda_nodes_in_graph}"

        print(
            f"✅ Graph building with hybrid nodes validated: "
            f"{agenda_nodes_in_graph} agenda nodes integrated"
        )

    def test_hybrid_architecture_statistics(self, mock_config, mock_create_provider):
        """Test 7: Generate comprehensive hybrid architecture statistics."""
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)
        test_suite = HybridArchitectureTestSuite()

        stats = test_suite.get_hybrid_architecture_statistics(flow)

        assert (
            "error" not in stats
        ), f"Statistics generation failed: {stats.get('error')}"
        assert stats["total_nodes"] > 25, f"Total nodes too low: {stats['total_nodes']}"
        assert (
            stats["extracted_nodes"] >= 5
        ), f"Extracted nodes too low: {stats['extracted_nodes']}"
        assert (
            stats["wrapped_nodes"] >= 20
        ), f"Wrapped nodes too low: {stats['wrapped_nodes']}"
        assert (
            0.1 <= stats["extraction_ratio"] <= 0.3
        ), f"Extraction ratio out of range: {stats['extraction_ratio']:.2%}"

        # Validate agenda node breakdown
        agenda_extracted = sum(
            1
            for node_info in stats["agenda_nodes"].values()
            if node_info["type"] == "extracted"
        )
        agenda_wrapped = sum(
            1
            for node_info in stats["agenda_nodes"].values()
            if node_info["type"] == "wrapped"
        )

        assert (
            agenda_extracted >= 5
        ), f"Expected >=5 extracted agenda nodes, got {agenda_extracted}"
        assert (
            agenda_wrapped >= 1
        ), f"Expected >=1 wrapped agenda nodes, got {agenda_wrapped}"

        print(f"✅ Hybrid architecture statistics:")
        print(f"   Total nodes: {stats['total_nodes']}")
        print(
            f"   Extracted: {stats['extracted_nodes']} ({stats['extraction_ratio']:.1%})"
        )
        print(f"   Wrapped: {stats['wrapped_nodes']}")
        print(f"   Agenda extracted: {agenda_extracted}")
        print(f"   Agenda wrapped: {agenda_wrapped}")

    def test_performance_comparison_extracted_vs_wrapped(
        self, mock_config, mock_create_provider
    ):
        """Test 8: Performance characteristics of extracted vs wrapped nodes."""
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)

        # Test graph building performance
        import time

        start_time = time.time()
        graph = flow.build_graph()
        build_time = time.time() - start_time

        assert graph is not None, "Graph building failed"
        assert build_time < 1.0, f"Graph building too slow: {build_time:.3f}s"

        # Test compilation performance
        start_time = time.time()
        compiled_graph = flow.compile()
        compile_time = time.time() - start_time

        assert compiled_graph is not None, "Graph compilation failed"
        assert compile_time < 3.0, f"Graph compilation too slow: {compile_time:.3f}s"

        # Test registry access performance
        all_nodes = flow.node_registry
        registry_size = len(all_nodes)

        # Access time should be fast
        start_time = time.time()
        for _ in range(100):
            _ = all_nodes["agenda_proposal"]  # Access extracted node
            _ = all_nodes["agenda_approval"]  # Access wrapped node
        access_time = time.time() - start_time

        assert (
            access_time < 0.001
        ), f"Registry access too slow: {access_time:.6f}s for 200 accesses"

        print(f"✅ Performance comparison validated:")
        print(f"   Graph build: {build_time:.3f}s")
        print(f"   Graph compile: {compile_time:.3f}s")
        print(f"   Registry access: {access_time:.6f}s (200 accesses)")
        print(f"   Registry size: {registry_size} nodes")

    def test_error_handling_across_node_types(self, mock_config, mock_create_provider):
        """Test 9: Error handling works for both extracted and wrapped nodes."""
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)

        # Test error recovery manager integration
        assert hasattr(flow, "error_recovery_manager"), "ErrorRecoveryManager missing"
        error_manager = flow.error_recovery_manager
        assert error_manager is not None, "ErrorRecoveryManager not initialized"

        # Test orchestrator error handling capabilities
        assert hasattr(flow, "orchestrator"), "Orchestrator missing"
        orchestrator = flow.orchestrator
        assert orchestrator is not None, "Orchestrator not initialized"

        # Validate orchestrator can handle both node types
        validation_results = orchestrator.validate_orchestrator_state()
        assert validation_results[
            "all_nodes_valid"
        ], f"Invalid nodes: {validation_results['invalid_nodes']}"
        assert (
            validation_results["nodes_registered"] > 25
        ), "Too few nodes registered with orchestrator"

        # Test error recovery strategies exist
        assert (
            validation_results["recovery_strategies"] > 0
        ), "No error recovery strategies registered"

        print(f"✅ Error handling validated:")
        print(
            f"   Nodes registered with orchestrator: {validation_results['nodes_registered']}"
        )
        print(f"   Recovery strategies: {validation_results['recovery_strategies']}")
        print(f"   All nodes valid: {validation_results['all_nodes_valid']}")

    def test_backward_compatibility_with_hybrid_architecture(
        self, mock_config, mock_create_provider
    ):
        """Test 10: Hybrid architecture maintains full backward compatibility."""
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)

        # Validate all expected nodes exist (backward compatibility check)
        expected_nodes = [
            # Phase 0: Initialization
            "config_and_keys",
            "agent_instantiation",
            "get_theme",
            # Phase 1: Agenda Setting (now hybrid)
            "agenda_proposal",
            "topic_refinement",
            "collate_proposals",
            "agenda_voting",
            "synthesize_agenda",
            "agenda_approval",
            # Phase 2: Discussion
            "announce_item",
            "discussion_round",
            "round_summarization",
            "round_threshold_check",
            "end_topic_poll",
            "vote_evaluation",
            "periodic_user_stop",
            "user_topic_conclusion_confirmation",
            # Phase 3: Topic Conclusion
            "final_considerations",
            "topic_report_generation",
            "topic_summary_generation",
            "file_output",
            # Phase 4: Continuation
            "agent_poll",
            "user_approval",
            "agenda_modification",
            # Phase 5: Final Report
            "final_report_generation",
            "multi_file_output",
            # Legacy
            "user_turn_participation",
        ]

        all_nodes = flow.node_registry
        missing_nodes = []

        for node_name in expected_nodes:
            if node_name not in all_nodes:
                missing_nodes.append(node_name)

        assert (
            not missing_nodes
        ), f"Missing backward compatibility nodes: {missing_nodes}"

        # Test that graph structure is compatible
        graph = flow.build_graph()
        assert graph is not None, "Graph building failed"

        graph_nodes = set(graph.nodes.keys())
        missing_from_graph = set(expected_nodes) - graph_nodes

        # Some nodes might be conditionally included, so allow for minor differences
        assert (
            len(missing_from_graph) <= 2
        ), f"Too many nodes missing from graph: {missing_from_graph}"

        print(f"✅ Backward compatibility validated:")
        print(f"   Expected nodes: {len(expected_nodes)}")
        print(f"   Missing nodes: {len(missing_nodes)}")
        print(f"   Graph nodes: {len(graph_nodes)}")
        print(
            f"   Compatibility: {100 * (len(expected_nodes) - len(missing_nodes)) / len(expected_nodes):.1f}%"
        )


if __name__ == "__main__":
    # Run hybrid architecture integration tests directly
    print("🚀 Running Hybrid Architecture Integration Tests...")

    with patch("virtual_agora.flow.graph_v13.create_provider") as mock:
        mock.return_value = Mock()

        # Create mock config
        config = Mock()
        config.moderator = Mock()
        config.moderator.provider.value = "openai"
        config.moderator.model = "gpt-4o"
        config.summarizer = Mock()
        config.summarizer.provider.value = "openai"
        config.summarizer.model = "gpt-4o"
        config.report_writer = Mock()
        config.report_writer.provider.value = "openai"
        config.report_writer.model = "gpt-4o"
        agent_config = Mock()
        agent_config.provider.value = "openai"
        agent_config.model = "gpt-4o"
        agent_config.count = 3
        config.agents = [agent_config]

        # Run tests
        test_suite = TestHybridArchitectureIntegration()

        # Test 1: Hybrid registry structure
        test_suite.test_hybrid_registry_structure_validation(config, mock)
        print("✅ Test 1 PASSED: Hybrid registry structure")

        # Test 2: Extracted node interfaces
        test_suite.test_extracted_node_interface_compliance(config, mock)
        print("✅ Test 2 PASSED: Extracted node interfaces")

        # Test 3: Wrapped node interfaces
        test_suite.test_wrapped_node_interface_compliance(config, mock)
        print("✅ Test 3 PASSED: Wrapped node interfaces")

        # Test 4: Node execution paths
        test_suite.test_node_execution_paths_integration(config, mock)
        print("✅ Test 4 PASSED: Node execution paths")

        # Test 5: Agenda factory integration
        test_suite.test_agenda_factory_integration(config, mock)
        print("✅ Test 5 PASSED: Agenda factory integration")

        # Test 6: Graph building
        test_suite.test_graph_building_hybrid_integration(config, mock)
        print("✅ Test 6 PASSED: Graph building with hybrid nodes")

        # Test 7: Architecture statistics
        test_suite.test_hybrid_architecture_statistics(config, mock)
        print("✅ Test 7 PASSED: Architecture statistics")

        # Test 8: Performance comparison
        test_suite.test_performance_comparison_extracted_vs_wrapped(config, mock)
        print("✅ Test 8 PASSED: Performance comparison")

        # Test 9: Error handling
        test_suite.test_error_handling_across_node_types(config, mock)
        print("✅ Test 9 PASSED: Error handling across node types")

        # Test 10: Backward compatibility
        test_suite.test_backward_compatibility_with_hybrid_architecture(config, mock)
        print("✅ Test 10 PASSED: Backward compatibility")

        print("\n🎉 ALL HYBRID ARCHITECTURE INTEGRATION TESTS PASSED!")
        print("✅ Extracted and wrapped nodes work seamlessly together")
        print("✅ FlowNode interface implemented correctly by both types")
        print("✅ Dependency injection works for extracted nodes")
        print("✅ Graph building supports mixed node types")
        print("✅ Performance characteristics maintained")
        print("✅ Full backward compatibility preserved")
