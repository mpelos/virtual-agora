#!/usr/bin/env python
"""Debug the graph execution issue."""

import sys
import logging
from unittest.mock import patch, Mock
from virtual_agora.flow.graph_v13 import VirtualAgoraV13Flow
from virtual_agora.config.models import Config as VirtualAgoraConfig
from virtual_agora.state.manager import StateManager

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create minimal config with all required fields
config = VirtualAgoraConfig(
    moderator={"provider": "google", "model": "gemini-2.5-flash-lite"},
    summarizer={"provider": "google", "model": "gemini-2.5-flash-lite"},
    topic_report={"provider": "google", "model": "gemini-2.5-flash-lite"},
    ecclesia_report={"provider": "google", "model": "gemini-2.5-flash-lite"},
    agents=[{"provider": "google", "model": "gemini-2.5-flash-lite", "count": 2}],
)

print("Creating flow...")

# Mock the provider creation to avoid API key issues
mock_llm = Mock()
mock_llm.invoke.return_value = Mock(content="Mock response")

with patch(
    "virtual_agora.providers.factory.ProviderFactory.create_provider",
    return_value=mock_llm,
):
    flow = VirtualAgoraV13Flow(config, enable_monitoring=False)

    print("Compiling graph...")
    flow.compile()

    print("Creating session...")
    session_id = flow.create_session(main_topic="Test Topic")

    print(f"Session created: {session_id}")

    # Try with a simple session ID without underscores
    simple_session_id = "testsession123"
    flow.state_manager.state["session_id"] = simple_session_id

    print(f"\nChanged session_id to simple format: {simple_session_id}")
    print("\nTrying to stream graph...")
    config_dict = {"configurable": {"thread_id": simple_session_id}}

    # Get initial state
    print(f"\nInitial state keys: {list(flow.state_manager.state.keys())}")

    # Check if checkpointer has any state
    print("\nChecking checkpointer state...")
    try:
        checkpoint = flow.checkpointer.get(config_dict["configurable"])
        if checkpoint:
            print(f"Found checkpoint: {type(checkpoint)}")
            if hasattr(checkpoint, "channel_values"):
                print(f"Channel values keys: {list(checkpoint.channel_values.keys())}")
        else:
            print("No checkpoint found in checkpointer")
    except Exception as e:
        print(f"Error checking checkpointer: {e}")

    try:
        update_count = 0
        for update in flow.stream(config_dict):
            update_count += 1
            print(f"\nUpdate {update_count}: {update}")

            # Stop after first update to debug
            if update_count >= 1:
                break

    except Exception as e:
        print(f"\nError during streaming: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    print("\nChecking if empty dict is the issue...")
    # Test the node directly
    state = flow.state_manager.state
    result = flow.nodes.config_and_keys_node(state)
    print(f"config_and_keys_node returned: {result}")
    print(f"Type: {type(result)}")
    print(f"Is empty: {len(result) == 0}")

    print("\nInspecting the compiled graph structure...")
    # Check what nodes are in the graph
    if hasattr(flow.compiled_graph, "nodes"):
        print(f"Graph nodes: {flow.compiled_graph.nodes}")
    if hasattr(flow.compiled_graph, "graph"):
        if hasattr(flow.compiled_graph.graph, "nodes"):
            print(f"Inner graph nodes: {list(flow.compiled_graph.graph.nodes.keys())}")
        if hasattr(flow.compiled_graph.graph, "_nodes"):
            print(f"Private nodes: {list(flow.compiled_graph.graph._nodes.keys())}")

    print("\nTrying to get the graph structure...")
    # Check the StateGraph structure
    if flow.graph:
        print(f"StateGraph nodes: {list(flow.graph.nodes.keys())}")
        print(f"StateGraph edges: {flow.graph.edges}")

    print("\nLooking for 'session_id' in the graph...")
    # Search for any reference to 'session_id' in the graph
    import inspect

    for name, node in flow.graph.nodes.items():
        print(f"\nNode '{name}':")
        if callable(node):
            # Get source if possible
            try:
                source = inspect.getsource(node)
                if "session_id" in source:
                    print(f"  Contains 'session_id' in source!")
            except:
                pass
