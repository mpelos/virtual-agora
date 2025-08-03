# Comprehensive Mocking Strategies for Production-Quality Testing

## Overview

This document provides detailed mocking strategies to ensure tests have production-quality behavior while remaining deterministic and fast. The key is to mock external dependencies while preserving all internal execution paths.

## Mocking Philosophy

### What to Mock (External Dependencies)
- ✅ LLM Provider calls (`create_provider()`, agent responses)
- ✅ File I/O operations (reading/writing files)
- ✅ Network requests (if any)
- ✅ Time-based operations (for deterministic tests)
- ✅ User input (GraphInterrupt simulation)

### What NOT to Mock (Internal Logic)
- ❌ Graph execution (`flow.stream()`)
- ❌ State management (`StateManager`)
- ❌ Node execution (`FlowNode.execute()`)
- ❌ Error recovery (`ErrorRecoveryManager`)
- ❌ Orchestrator components

## LLM Provider Mocking

### Base Provider Mock Setup

```python
# tests/framework/provider_mocking.py
from unittest.mock import Mock, patch
from typing import Dict, Any, List

class LLMProviderMock:
    """Comprehensive LLM provider mocking for realistic agent behavior."""
    
    def __init__(self):
        self.response_templates = self._load_response_templates()
        self.call_count = 0
        
    def create_mock_llm(self) -> Mock:
        """Create mock LLM with realistic response patterns."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = self._generate_response
        mock_llm.call.side_effect = self._generate_response
        return mock_llm
    
    def _generate_response(self, input_data: Any) -> Dict[str, Any]:
        """Generate realistic responses based on input context."""
        self.call_count += 1
        
        # Analyze input to determine response type
        if isinstance(input_data, list):  # Message format
            last_message = input_data[-1].content if input_data else ""
        else:
            last_message = str(input_data)
        
        # Route to appropriate response generator
        if "propose topics" in last_message.lower():
            return self._generate_topic_proposal()
        elif "vote" in last_message.lower():
            return self._generate_vote_response()
        elif "discuss" in last_message.lower():
            return self._generate_discussion_response()
        else:
            return self._generate_generic_response()
    
    def _generate_topic_proposal(self) -> Dict[str, Any]:
        """Generate realistic topic proposals."""
        topics = [
            "The impact of artificial intelligence on society",
            "Climate change adaptation strategies", 
            "Future of remote work and digital collaboration",
            "Ethical considerations in biotechnology",
            "Sustainable urban development approaches"
        ]
        selected_topics = topics[:3]  # Realistic number
        
        return {
            "content": f"I propose the following topics:\n" + 
                      "\n".join(f"{i+1}. {topic}" for i, topic in enumerate(selected_topics)),
            "response_metadata": {"model": "mock-model"}
        }
    
    def _generate_vote_response(self) -> Dict[str, Any]:
        """Generate realistic voting responses."""
        choices = [
            "I vote for option 1 as it provides the most comprehensive coverage",
            "Option 2 seems most relevant to current discussions",
            "I support option 3 for its practical implications"
        ]
        
        return {
            "content": choices[self.call_count % len(choices)],
            "response_metadata": {"model": "mock-model"}
        }
    
    def _generate_discussion_response(self) -> Dict[str, Any]:
        """Generate realistic discussion contributions."""
        responses = [
            "This is a fascinating topic that requires careful consideration. Based on current research...",
            "I'd like to build on the previous point by adding that we should also consider...",
            "While I agree with the general direction, I think we need to examine the potential drawbacks...",
            "From a different perspective, we might want to explore how this relates to..."
        ]
        
        return {
            "content": responses[self.call_count % len(responses)],
            "response_metadata": {"model": "mock-model"}
        }

# Usage in tests
@pytest.fixture
def mock_llm_provider():
    provider_mock = LLMProviderMock()
    with patch('virtual_agora.providers.create_provider') as mock_create:
        mock_create.return_value = provider_mock.create_mock_llm()
        yield provider_mock
```

### Agent-Specific Mocking

```python
# Agent behavior patterns for different roles
class AgentMockingStrategies:
    
    @staticmethod
    def mock_moderator_agent(flow):
        """Mock moderator with realistic facilitation behavior."""
        moderator = flow.specialized_agents.get("moderator")
        if moderator:
            moderator.collect_proposals = Mock(return_value=[
                "AI and machine learning applications",
                "Sustainable technology solutions", 
                "Digital privacy and security"
            ])
            moderator.generate_response = Mock(return_value=
                '{"proposed_agenda": ["AI applications", "Sustainable tech", "Digital privacy"]}'
            )
            moderator.facilitate_discussion = Mock(return_value=
                "Let's begin our discussion on this important topic..."
            )
    
    @staticmethod  
    def mock_discussing_agents(flow):
        """Mock discussing agents with varied response patterns."""
        response_patterns = [
            lambda topic: f"I believe {topic} is crucial for our future development...",
            lambda topic: f"While {topic} is important, we should consider the challenges...",
            lambda topic: f"Building on previous points about {topic}, I'd add that..."
        ]
        
        for i, agent in enumerate(flow.discussing_agents):
            pattern = response_patterns[i % len(response_patterns)]
            agent.__call__ = Mock(side_effect=lambda x, p=pattern: {
                "messages": [Mock(content=p("the current topic"))]
            })
```

## GraphInterrupt Simulation

### User Input Mocking Framework

```python
# tests/framework/hitl_mocking.py
from langgraph.errors import GraphInterrupt
from langgraph.types import Interrupt
from contextlib import contextmanager
from typing import Dict, Any, Generator

class HITLMockingFramework:
    """Framework for mocking human-in-the-loop interactions."""
    
    def __init__(self):
        self.pending_responses = {}
        self.interaction_history = []
    
    @contextmanager
    def mock_user_input(self, responses: Dict[str, Any]) -> Generator:
        """Context manager for mocking user input responses.
        
        Args:
            responses: Dict mapping interaction types to user responses
            Example: {
                'agenda_approval': 'approve',
                'topic_conclusion': 'continue',
                'session_continuation': 'next_topic'
            }
        """
        self.pending_responses.update(responses)
        
        with patch('virtual_agora.flow.nodes_v13.interrupt') as mock_interrupt:
            mock_interrupt.side_effect = self._handle_interrupt
            yield mock_interrupt
    
    def _handle_interrupt(self, value: Dict[str, Any]) -> str:
        """Handle interrupt based on type and return appropriate response."""
        interrupt_type = value.get('type')
        
        # Record interaction for validation
        self.interaction_history.append({
            'type': interrupt_type,
            'value': value,
            'timestamp': datetime.now()
        })
        
        # Return pre-configured response or default
        if interrupt_type in self.pending_responses:
            response = self.pending_responses[interrupt_type]
            del self.pending_responses[interrupt_type]  # Use once
            return response
        
        # Default responses for common interactions
        defaults = {
            'agenda_approval': 'approve',
            'topic_conclusion': 'continue', 
            'session_continuation': 'next_topic',
            'periodic_stop': 'continue',
            'user_approval': 'approve'
        }
        
        return defaults.get(interrupt_type, 'continue')
    
    def simulate_interrupt_scenario(self, interrupt_type: str, **kwargs) -> GraphInterrupt:
        """Simulate specific interrupt scenario for testing."""
        interrupt_data = {
            'type': interrupt_type,
            **kwargs
        }
        
        if interrupt_type == 'agenda_approval':
            interrupt_data.update({
                'proposed_agenda': ["Topic 1", "Topic 2", "Topic 3"],
                'message': 'Please review and approve the proposed discussion agenda.',
                'options': ['approve', 'edit', 'reorder', 'reject']
            })
        elif interrupt_type == 'topic_conclusion':
            interrupt_data.update({
                'current_topic': 'Test Topic',
                'message': 'Would you like to conclude this topic?',
                'options': ['conclude', 'continue', 'modify']
            })
        
        return GraphInterrupt((Interrupt(
            value=interrupt_data,
            resumable=True,
            ns=[f'{interrupt_type}:test-namespace'],
            when='during'
        ),))

# Usage example
def test_agenda_approval_flow():
    hitl_mock = HITLMockingFramework()
    
    with hitl_mock.mock_user_input({'agenda_approval': 'approve'}):
        flow = create_test_flow()
        updates = list(flow.stream(config_dict))
        
        # Validate interaction occurred
        assert len(hitl_mock.interaction_history) == 1
        assert hitl_mock.interaction_history[0]['type'] == 'agenda_approval'
```

## State Mocking and Validation

### State Fixture Creation

```python
# tests/framework/state_mocking.py
from virtual_agora.state.schema import VirtualAgoraState, AgentInfo, HITLState, FlowControl
from datetime import datetime
from typing import Dict, Any

class StateMockingFramework:
    """Framework for creating realistic test states."""
    
    @staticmethod
    def create_minimal_state(session_id: str = "test_session") -> VirtualAgoraState:
        """Create minimal valid state for testing."""
        now = datetime.now()
        
        return VirtualAgoraState(
            # Session metadata
            session_id=session_id,
            start_time=now,
            config_hash="test_hash",
            
            # UI state management
            ui_state={
                "console_initialized": False,
                "theme_applied": False,
                "accessibility_enabled": False,
                "dashboard_active": False,
                "current_display_mode": "full",
                "progress_operations": {},
                "last_ui_update": now,
            },
            
            # Phase management
            current_phase=0,
            phase_start_time=now,
            
            # Round management
            current_round=0,
            rounds_per_topic={},
            
            # HITL state
            hitl_state=HITLState(
                awaiting_approval=False,
                approval_type=None,
                prompt_message=None,
                options=None,
                approval_history=[],
            ),
            
            # Flow control
            flow_control=FlowControl(
                max_rounds_per_topic=10,
                auto_conclude_threshold=3,
                context_window_limit=8000,
                cycle_detection_enabled=True,
                max_iterations_per_phase=5,
            ),
            
            # Topic management
            user_defines_topics=False,
            user_defined_agenda=False,
            active_topic=None,
            topic_queue=[],
            proposed_topics=[],
            topics_info={},
            
            # Agent management
            agents={
                "moderator": AgentInfo(
                    id="moderator",
                    model="test-model",
                    provider="test",
                    role="moderator",
                    message_count=0,
                    created_at=now,
                ),
                "test_agent_1": AgentInfo(
                    id="test_agent_1",
                    model="test-model",
                    provider="test",
                    role="participant",
                    message_count=0,
                    created_at=now,
                )
            },
            moderator_id="moderator",
            current_speaker_id="moderator",
            speaking_order=["test_agent_1"],
            next_speaker_index=0,
            
            # Discussion history
            messages=[],
            last_message_id="0",
            
            # Voting system
            active_vote=None,
            # Note: vote_history and votes use reducers - not initialized
            
            # Consensus tracking
            consensus_proposals={},
            consensus_reached={},
            
            # Generated content
            phase_summaries={},
            topic_summaries={},
            consensus_summaries={},
            final_report=None,
            
            # Runtime statistics
            total_messages=0,
            messages_by_phase={i: 0 for i in range(5)},
            messages_by_agent={"moderator": 0, "test_agent_1": 0},
            messages_by_topic={},
            vote_participation_rate={},
            
            # Tool execution
            active_tool_calls={},
            tool_metrics={
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "average_execution_time_ms": 0.0,
                "calls_by_tool": {},
                "calls_by_agent": {},
                "errors_by_type": {},
            },
            tools_enabled_agents=[],
            
            # v1.3 specific fields
            specialized_agents={
                "moderator": "moderator",
                "summarizer": "summarizer",
                "report_writer": "report_writer"
            },
            periodic_stop_counter=0,
            checkpoint_interval=3,
            
            # Error tracking
            last_error=None,
            error_count=0,
        )
    
    @staticmethod
    def create_discussion_state(topic: str, round_num: int = 1) -> Dict[str, Any]:
        """Create state configured for discussion phase."""
        state = StateMockingFramework.create_minimal_state()
        
        state.update({
            'current_phase': 2,  # Discussion phase
            'current_round': round_num,
            'active_topic': topic,
            'topic_queue': [topic],
            'rounds_per_topic': {topic: round_num}
        })
        
        return state
```

## File I/O Mocking

### File Operation Mocking

```python
# tests/framework/file_mocking.py
from unittest.mock import patch, mock_open
from contextlib import contextmanager
from typing import Dict, Any

@contextmanager
def mock_file_operations():
    """Mock all file I/O operations used in the application."""
    
    # Mock file writing operations
    with patch('builtins.open', mock_open()) as mock_file:
        # Mock file existence checks
        with patch('os.path.exists', return_value=True):
            # Mock directory creation
            with patch('os.makedirs'):
                # Mock file reading with realistic content
                mock_file.return_value.read.return_value = '{"test": "data"}'
                yield mock_file

# Usage in tests
def test_with_file_mocking():
    with mock_file_operations():
        # Test code that performs file operations
        flow = create_test_flow()
        # File operations will be mocked automatically
```

## Performance Monitoring Integration

### Memory and Time Tracking

```python
# tests/framework/performance_monitoring.py
import psutil
import time
from contextlib import contextmanager
from typing import Dict, Any

@contextmanager
def performance_monitoring():
    """Monitor performance metrics during test execution."""
    process = psutil.Process()
    start_memory = process.memory_info().rss
    start_time = time.perf_counter()
    
    metrics = {
        'start_memory_mb': start_memory / (1024 * 1024),
        'start_time': start_time,
        'peak_memory_mb': start_memory / (1024 * 1024),
    }
    
    try:
        yield metrics
    finally:
        end_memory = process.memory_info().rss
        end_time = time.perf_counter()
        
        metrics.update({
            'end_memory_mb': end_memory / (1024 * 1024),
            'peak_memory_mb': max(metrics['peak_memory_mb'], end_memory / (1024 * 1024)),
            'execution_time_s': end_time - start_time,
            'memory_increase_mb': (end_memory - start_memory) / (1024 * 1024)
        })

# Usage
def test_with_performance_monitoring():
    with performance_monitoring() as metrics:
        # Execute test
        run_heavy_operation()
        
    # Validate performance
    assert metrics['memory_increase_mb'] < 100  # 100MB limit
    assert metrics['execution_time_s'] < 30     # 30 second limit
```

## Complete Test Example

```python
# Example combining all mocking strategies
def test_complete_session_with_comprehensive_mocking():
    """Example test using all mocking strategies."""
    
    # Setup comprehensive mocking
    hitl_mock = HITLMockingFramework()
    
    with mock_llm_provider() as llm_mock:
        with hitl_mock.mock_user_input({
            'agenda_approval': 'approve',
            'topic_conclusion': 'continue',
            'session_continuation': 'next_topic'
        }):
            with mock_file_operations():
                with performance_monitoring() as metrics:
                    
                    # Create and execute flow
                    flow = VirtualAgoraV13Flow(test_config, enable_monitoring=False)
                    AgentMockingStrategies.mock_moderator_agent(flow)
                    AgentMockingStrategies.mock_discussing_agents(flow)
                    
                    session_id = flow.create_session(main_topic="Test Topic")
                    config_dict = {"configurable": {"thread_id": session_id}}
                    
                    # Execute with streaming (production pattern)
                    updates = []
                    for update in flow.stream(config_dict):
                        updates.append(update)
                        # Break after reasonable progress to avoid infinite loops
                        if len(updates) >= 10:
                            break
                    
                    # Validate results
                    assert len(updates) > 0
                    assert len(hitl_mock.interaction_history) > 0
                    assert metrics['memory_increase_mb'] < 50
                    assert llm_mock.call_count > 0
```

This comprehensive mocking strategy ensures tests are both realistic and deterministic, catching production issues while remaining fast and reliable.