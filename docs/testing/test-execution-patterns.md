# Test Execution Patterns for Production Parity

## Overview

This document analyzes the exact execution patterns used in production and provides specifications for replicating them in tests. Understanding these patterns is critical for creating tests that catch real production issues.

## Production Execution Analysis

### Main Application Flow

```python
# From main.py:948
for update in stream_coordinator.coordinate_stream_execution(config_dict):
    # Process stream updates
    if isinstance(update, dict):
        # Extract state updates from flow update
        state_updates = {}
        for node_name, node_data in update.items():
            # Update unified state manager
```

### StreamCoordinator Pattern

```python
# From execution/stream_coordinator.py:100
stream_iter = self.flow.stream(config_dict)
for update in stream_iter:
    update_count += 1
    update_keys = list(update.keys()) if isinstance(update, dict) else [str(type(update))]
    # Process each update with error handling
```

### VirtualAgoraV13Flow Stream Pattern

```python
# From flow/graph_v13.py:803
for update in self.compiled_graph.stream(input_data, config):
    # CRITICAL: State manager sync after each update
    if isinstance(update, dict) and update:
        all_state_updates = {}
        for node_name, node_result in update.items():
            if isinstance(node_result, dict):
                all_state_updates.update(node_result)
        if all_state_updates:
            self.state_manager.update_state(all_state_updates)
```

## Key Production Patterns to Replicate

### 1. Streaming Execution Pattern

**Production Behavior:**
- Uses `compiled_graph.stream(input_data, config)` 
- Processes updates incrementally
- Syncs state manager after each update
- Handles GraphInterrupt exceptions

**Test Requirements:**
```python
def test_streaming_execution(self):
    """Test must use streaming execution pattern."""
    flow = self.create_test_flow()
    config_dict = {"configurable": {"thread_id": "test_session"}}
    
    # MUST use stream() not invoke()
    updates = []
    for update in flow.stream(config_dict):
        updates.append(update)
        # Validate state sync happens
        assert isinstance(update, dict)
        
    # Validate incremental processing
    assert len(updates) > 0
```

### 2. GraphInterrupt Handling Pattern

**Production Behavior:**
- GraphInterrupt raised during user input nodes
- Exception propagates through V13NodeWrapper
- StreamCoordinator catches and handles interrupts
- State preserved for resumption

**Test Requirements:**
```python
def test_graphinterrupt_handling(self):
    """Test must simulate real GraphInterrupt scenarios."""
    from langgraph.errors import GraphInterrupt
    
    # Mock interrupt during agenda approval
    with patch('virtual_agora.flow.nodes_v13.interrupt') as mock_interrupt:
        mock_interrupt.side_effect = GraphInterrupt(...)
        
        # Execute stream and catch interrupt
        try:
            list(flow.stream(config_dict))
        except GraphInterrupt as e:
            # Validate interrupt data
            assert e.value['type'] == 'agenda_approval'
            assert 'proposed_agenda' in e.value
```

### 3. State Synchronization Pattern

**Production Behavior:**
- State manager updates after each graph update
- Reducer fields handled by LangGraph
- Manual state validation during updates
- Error recovery updates state directly

**Test Requirements:**
```python
def test_state_synchronization(self):
    """Test must validate state sync pattern."""
    flow = self.create_test_flow()
    initial_state = flow.state_manager.get_snapshot()
    
    # Execute one stream update
    stream_iter = flow.stream(config_dict)
    first_update = next(stream_iter)
    
    # Validate state was synchronized
    updated_state = flow.state_manager.get_snapshot()
    assert updated_state != initial_state
    
    # Validate specific fields updated
    if 'messages' in first_update.get(list(first_update.keys())[0], {}):
        assert len(updated_state['messages']) > len(initial_state['messages'])
```

### 4. Error Recovery Pattern

**Production Behavior:**
- Errors caught by stream coordinator
- Recovery manager validates and repairs state
- Emergency recovery protocols activate
- State consistency maintained

**Test Requirements:**
```python
def test_error_recovery_pattern(self):
    """Test must replicate error recovery flow."""
    flow = self.create_test_flow()
    
    # Inject error during node execution
    with patch.object(flow.nodes, 'agenda_proposal_node') as mock_node:
        mock_node.side_effect = Exception("Test error")
        
        # Verify error recovery activates
        with pytest.raises(Exception):
            list(flow.stream(config_dict))
            
        # Validate recovery manager was called
        assert flow.error_recovery_manager.emergency_recovery.called
```

## Configuration Patterns

### Session Creation Pattern

```python
# Production pattern from graph_v13.py:641
session_id = flow.create_session(
    main_topic="Test Topic",
    user_defines_topics=False
)
config_dict = {"configurable": {"thread_id": session_id}}
```

### Graph Compilation Pattern

```python
# Production pattern from graph_v13.py:590
if self.compiled_graph is None:
    self.compile()  # Must compile before streaming

# Streaming with proper configuration
result = self.compiled_graph.stream(input_data, config)
```

### State Initialization Pattern

```python
# Production pattern from state/manager.py:99
self._state = VirtualAgoraState(
    # Session metadata
    session_id=session_id,
    start_time=now,
    config_hash=config_hash,
    # CRITICAL: Reducer fields NOT initialized as empty lists
    # They are managed by LangGraph reducers
    # phase_history=[], # WRONG - causes schema errors
    # vote_history=[], # WRONG - causes KeyError
)
```

## Common Anti-Patterns in Existing Tests

### ❌ Anti-Pattern 1: Using invoke() Instead of stream()
```python
# WRONG - doesn't match production
result = flow.invoke(config_dict)

# CORRECT - matches production
for update in flow.stream(config_dict):
    process_update(update)
```

### ❌ Anti-Pattern 2: Mocking Internal Components
```python
# WRONG - bypasses production code paths
with patch('virtual_agora.flow.graph_v13.VirtualAgoraV13Flow.stream'):
    test_something()

# CORRECT - mock external dependencies only
with patch('virtual_agora.providers.create_provider'):
    test_with_real_stream()
```

### ❌ Anti-Pattern 3: Incomplete State Validation
```python
# WRONG - only checks high-level state
assert flow.state_manager.state['current_phase'] == 1

# CORRECT - validates complete state consistency
state_snapshot = flow.state_manager.get_snapshot()
validation_result = state_validator.validate_state_consistency(state_snapshot)
assert validation_result['is_valid'], f"State errors: {validation_result['errors']}"
```

### ❌ Anti-Pattern 4: Ignoring GraphInterrupt
```python
# WRONG - tests don't handle user input scenarios
def test_agenda_approval():
    # Test logic that avoids interrupt handling

# CORRECT - explicitly test interrupt scenarios
def test_agenda_approval_with_interrupt():
    with mock_user_input('approve'):
        updates = list(flow.stream(config_dict))
        assert any('agenda_approval' in update for update in updates)
```

## Test Environment Requirements

### Required Mocking

1. **LLM Providers**: All `create_provider()` calls
2. **File Operations**: All file I/O in nodes
3. **Time Operations**: Deterministic timestamps
4. **User Input**: GraphInterrupt simulation

### Required Real Components

1. **Graph Structure**: Complete LangGraph compilation
2. **State Management**: Full StateManager operation
3. **Node Execution**: All FlowNode and V13NodeWrapper execution
4. **Error Recovery**: Complete ErrorRecoveryManager operation

### Memory and Performance Monitoring

```python
import psutil
import time

def test_with_performance_monitoring():
    """Template for performance-aware testing."""
    process = psutil.Process()
    start_memory = process.memory_info().rss
    start_time = time.time()
    
    # Execute test
    run_test_scenario()
    
    # Validate performance
    end_memory = process.memory_info().rss
    end_time = time.time()
    
    memory_increase = end_memory - start_memory
    execution_time = end_time - start_time
    
    assert memory_increase < 100 * 1024 * 1024  # 100MB limit
    assert execution_time < 30.0  # 30 second limit
```

## Validation Checkpoints

### After Each Test

1. **State Consistency**: No schema violations
2. **Memory Usage**: Within acceptable bounds
3. **Error State**: Proper cleanup and recovery
4. **Graph State**: Compilation integrity maintained

### After Test Suite

1. **Coverage Analysis**: All production paths tested
2. **Performance Regression**: No significant degradation
3. **Error Scenario Coverage**: All failure modes tested
4. **Configuration Coverage**: All variants tested

This pattern analysis ensures that tests accurately reflect production execution and catch the types of issues that occur in real deployments.