# Phase 5: Testing & Validation Guide

## Purpose
This document provides a comprehensive testing strategy for validating the v1.3 migration. It covers unit testing for specialized agents, integration testing for the new graph flow, and end-to-end validation of the complete system.

## Prerequisites
- Completed Phases 1-4 (all implementation work)
- Existing test infrastructure understanding
- Access to test data and scenarios
- Testing tools: pytest, pytest-asyncio, pytest-mock

## Testing Strategy Overview

### Testing Layers

1. **Unit Tests**: Individual component validation
   - Specialized agent behavior
   - Node functions
   - State management
   - HITL components

2. **Integration Tests**: Component interaction validation
   - Agent coordination
   - Graph flow paths
   - State transitions
   - HITL gates

3. **End-to-End Tests**: Complete workflow validation
   - Full session simulation
   - Performance benchmarks
   - User experience validation
   - Report generation

4. **Migration Tests**: Backward compatibility
   - Config migration
   - State migration
   - API compatibility

## Unit Testing Implementation

### Specialized Agent Tests

```python
# tests/agents/test_summarizer.py

import pytest
from unittest.mock import Mock, patch
from virtual_agora.agents.summarizer import SummarizerAgent
from tests.helpers.fake_llm import FakeLLM

class TestSummarizerAgent:
    """Unit tests for SummarizerAgent."""
    
    @pytest.fixture
    def summarizer(self):
        """Create test summarizer instance."""
        return SummarizerAgent(
            agent_id="test_summarizer",
            llm=FakeLLM(
                responses=["This is a test summary of the discussion."]
            ),
            compression_ratio=0.3,
            max_summary_tokens=500
        )
    
    def test_initialization(self, summarizer):
        """Test agent initialization with v1.3 prompt."""
        assert summarizer.agent_id == "test_summarizer"
        assert summarizer.compression_ratio == 0.3
        assert "text compression tool" in summarizer.system_prompt
        assert "agent-agnostic summary" in summarizer.system_prompt
    
    def test_summarize_round(self, summarizer):
        """Test round summarization functionality."""
        messages = [
            {"speaker": "agent1", "content": "First point about AI safety"},
            {"speaker": "agent2", "content": "I agree and add this point"},
            {"speaker": "agent3", "content": "Different perspective here"}
        ]
        
        summary = summarizer.summarize_round(
            messages=messages,
            topic="AI Safety",
            round_number=1
        )
        
        assert isinstance(summary, str)
        assert len(summary) < len(str(messages))  # Compressed
        assert "test summary" in summary
    
    def test_compression_metrics(self, summarizer):
        """Test compression metric tracking."""
        # Create long messages
        messages = [
            {"content": "Long discussion " * 100}
            for _ in range(5)
        ]
        
        with patch.object(summarizer, '_estimate_tokens') as mock_tokens:
            mock_tokens.side_effect = [1000, 300]  # Original, compressed
            
            summary = summarizer.summarize_round(messages, "Test", 1)
            
            # Verify compression tracking
            assert mock_tokens.call_count == 2
            # Check logging would show 30% compression
    
    def test_max_token_limit(self, summarizer):
        """Test max token enforcement."""
        summarizer.max_summary_tokens = 10  # Very low limit
        
        messages = [{"content": "Extremely long " * 1000}]
        
        summary = summarizer.summarize_round(messages, "Test", 1)
        
        # Should respect token limit
        assert len(summary.split()) <= 15  # Some buffer
```

```python
# tests/agents/test_topic_report_agent.py

class TestTopicReportAgent:
    """Unit tests for TopicReportAgent."""
    
    @pytest.fixture
    def topic_agent(self):
        """Create test topic report agent."""
        return TopicReportAgent(
            agent_id="test_topic_report",
            llm=FakeLLM(responses=[
                "Comprehensive report with all required sections..."
            ])
        )
    
    def test_synthesize_topic(self, topic_agent):
        """Test topic synthesis functionality."""
        round_summaries = [
            "Round 1: Initial arguments presented",
            "Round 2: Debate intensified", 
            "Round 3: Some consensus emerging"
        ]
        
        final_considerations = [
            "Agent 1: Important point to consider",
            "Agent 2: Alternative perspective"
        ]
        
        report = topic_agent.synthesize_topic(
            round_summaries=round_summaries,
            final_considerations=final_considerations,
            topic="AI Alignment",
            discussion_theme="AI Safety"
        )
        
        assert isinstance(report, str)
        assert "Comprehensive report" in report
    
    def test_report_structure(self, topic_agent):
        """Test report follows required structure."""
        # Mock LLM to return structured report
        structured_response = """
        1. Topic Overview: AI Alignment challenges
        2. Major Themes: Technical and philosophical
        3. Points of Consensus: Need for research
        4. Areas of Disagreement: Timeline urgency
        5. Key Insights: Novel approaches
        6. Implications: Further research needed
        """
        
        topic_agent.llm = FakeLLM(responses=[structured_response])
        
        report = topic_agent.synthesize_topic([], [], "Test", "Theme")
        
        # Verify all sections present
        assert "Topic Overview" in report
        assert "Major Themes" in report
        assert "Points of Consensus" in report
        assert "Key Insights" in report
```

### Node Function Tests

```python
# tests/flow/test_nodes_v3.py

import pytest
from unittest.mock import Mock, MagicMock
from virtual_agora.flow.nodes import (
    AgendaNodes, DiscussionNodes, ConclusionNodes
)
from virtual_agora.state.schema import VirtualAgoraState

class TestAgendaNodes:
    """Test agenda-related nodes."""
    
    def test_collate_proposals_node(self):
        """Test proposal collation with moderator."""
        state = {
            "proposed_topics": [
                {"agent_id": "agent1", "proposals": "Topic A, Topic B"},
                {"agent_id": "agent2", "proposals": "Topic B, Topic C"}
            ]
        }
        
        mock_moderator = Mock()
        mock_moderator.collect_proposals.return_value = [
            "Topic A", "Topic B", "Topic C"
        ]
        
        result = AgendaNodes.collate_proposals_node(state, mock_moderator)
        
        assert result["collated_topics"] == ["Topic A", "Topic B", "Topic C"]
        mock_moderator.collect_proposals.assert_called_once()
    
    def test_synthesize_agenda_node(self):
        """Test agenda synthesis with JSON output."""
        state = {
            "agenda_votes": [
                {"agent_id": "agent1", "vote": "I prefer B then A"},
                {"agent_id": "agent2", "vote": "A is most important"}
            ]
        }
        
        mock_moderator = Mock()
        mock_moderator.synthesize_agenda.return_value = {
            "proposed_agenda": ["Topic A", "Topic B"]
        }
        
        result = AgendaNodes.synthesize_agenda_node(state, mock_moderator)
        
        assert result["proposed_agenda"] == ["Topic A", "Topic B"]

class TestDiscussionNodes:
    """Test discussion loop nodes."""
    
    def test_round_summarization_node(self):
        """Test summarizer invocation."""
        state = {
            "messages": [
                {"speaker": "agent1", "content": "Point 1"},
                {"speaker": "agent2", "content": "Point 2"}
            ],
            "active_topic": "Test Topic",
            "current_round": 3,
            "agents_count": 2
        }
        
        mock_summarizer = Mock()
        mock_summarizer.summarize_round.return_value = "Round summary"
        
        result = DiscussionNodes.round_summarization_node(
            state, mock_summarizer
        )
        
        assert result["round_summaries"] == ["Round summary"]
        assert result["last_round_summary"] == "Round summary"
        
        # Verify correct messages passed
        call_args = mock_summarizer.summarize_round.call_args
        assert len(call_args[1]["messages"]) == 2
    
    def test_periodic_user_stop_node(self):
        """Test 5-round periodic stop."""
        state = {
            "current_round": 5,
            "active_topic": "AI Ethics"
        }
        
        result = DiscussionNodes.periodic_user_stop_node(state)
        
        assert result["hitl_state"]["awaiting_approval"] == True
        assert result["hitl_state"]["approval_type"] == "periodic_stop"
        assert "Do you wish to end" in result["hitl_state"]["prompt_message"]
        assert result["periodic_stop_counter"] == 0  # Reset
```

### HITL Component Tests

```python
# tests/ui/test_hitl_manager_v3.py

import pytest
from unittest.mock import patch, MagicMock
from io import StringIO
from rich.console import Console
from virtual_agora.ui.hitl_manager import (
    EnhancedHITLManager, HITLInteraction, HITLApprovalType
)

class TestEnhancedHITLManager:
    """Test enhanced HITL manager."""
    
    @pytest.fixture
    def manager(self):
        """Create test HITL manager."""
        console = Console(file=StringIO())
        return EnhancedHITLManager(console)
    
    def test_periodic_stop_handling(self, manager):
        """Test 5-round periodic stop."""
        interaction = HITLInteraction(
            approval_type=HITLApprovalType.PERIODIC_STOP,
            prompt_message="End topic discussion?",
            context={
                "current_round": 5,
                "active_topic": "AI Safety"
            }
        )
        
        # Mock user choosing to end
        with patch("rich.prompt.Confirm.ask", return_value=True):
            with patch("rich.prompt.Prompt.ask", return_value="Testing done"):
                response = manager.process_interaction(interaction)
        
        assert response["force_topic_end"] == True
        assert response["reason"] == "Testing done"
        assert response["checkpoint_round"] == 5
    
    def test_agenda_editing(self, manager):
        """Test agenda editing capability."""
        interaction = HITLInteraction(
            approval_type=HITLApprovalType.AGENDA_APPROVAL,
            prompt_message="Approve agenda?",
            context={
                "proposed_agenda": ["Topic A", "Topic B"]
            }
        )
        
        # Mock user choosing to edit
        with patch("rich.prompt.Prompt.ask") as mock_prompt:
            mock_prompt.side_effect = [
                "Edit",  # Choose edit action
                "Modified Topic A",  # Edit first topic
                "Topic B",  # Keep second topic
                "",  # No new topics
            ]
            
            response = manager.process_interaction(interaction)
        
        assert response["approved"] == True
        assert response["agenda"] == ["Modified Topic A", "Topic B"]
        assert response["edited"] == True
```

## Integration Testing Implementation

### Agent Coordination Tests

```python
# tests/integration/test_agent_coordination_v3.py

import pytest
from virtual_agora.agents import (
    create_specialized_agents, create_discussing_agents
)
from virtual_agora.flow.nodes import FlowNodes

class TestAgentCoordination:
    """Test specialized agents working together."""
    
    @pytest.fixture
    def agents(self, test_config):
        """Create all agents."""
        specialized = create_specialized_agents(test_config)
        discussing = create_discussing_agents(test_config)
        return specialized, discussing
    
    def test_round_to_summary_flow(self, agents):
        """Test discussion round to summary flow."""
        specialized, discussing = agents
        
        # Simulate discussion round
        messages = []
        for agent in discussing[:3]:
            response = agent.generate_response(
                "Discuss AI safety concerns"
            )
            messages.append({
                "speaker": agent.agent_id,
                "content": response
            })
        
        # Summarize round
        summary = specialized["summarizer"].summarize_round(
            messages=messages,
            topic="AI Safety",
            round_number=1
        )
        
        assert len(summary) < len(str(messages))
        assert isinstance(summary, str)
    
    def test_summaries_to_topic_report(self, agents):
        """Test round summaries to topic report flow."""
        specialized, _ = agents
        
        # Create sample summaries
        round_summaries = [
            "Round 1: Initial concerns raised",
            "Round 2: Technical solutions proposed",
            "Round 3: Consensus on research priorities"
        ]
        
        # Generate topic report
        report = specialized["topic_report"].synthesize_topic(
            round_summaries=round_summaries,
            final_considerations=["Final thought"],
            topic="AI Safety",
            discussion_theme="Future of AI"
        )
        
        assert len(report) > len(" ".join(round_summaries))
        assert isinstance(report, str)
    
    def test_topic_reports_to_final_report(self, agents):
        """Test topic reports to final report flow."""
        specialized, _ = agents
        
        # Create sample topic reports
        topic_reports = {
            "AI Safety": "Comprehensive safety discussion...",
            "AI Ethics": "Ethical considerations explored...",
            "AI Governance": "Governance frameworks discussed..."
        }
        
        # Generate report structure
        structure = specialized["ecclesia_report"].generate_report_structure(
            topic_reports=topic_reports,
            discussion_theme="Future of AI"
        )
        
        assert isinstance(structure, list)
        assert len(structure) > 0
        
        # Write a section
        section = specialized["ecclesia_report"].write_section(
            section_title=structure[0],
            topic_reports=topic_reports,
            discussion_theme="Future of AI",
            previous_sections={}
        )
        
        assert isinstance(section, str)
        assert len(section) > 0
```

### Graph Flow Tests

```python
# tests/integration/test_graph_flow_v3.py

import pytest
from langgraph.graph import StateGraph
from virtual_agora.flow.graph import VirtualAgoraFlow
from virtual_agora.state.schema import VirtualAgoraState

class TestGraphFlow:
    """Test complete graph flow paths."""
    
    @pytest.fixture
    def flow(self, test_config):
        """Create test flow."""
        return VirtualAgoraFlow(test_config)
    
    def test_agenda_setting_path(self, flow):
        """Test complete agenda setting phase."""
        initial_state = {
            "main_topic": "AI Safety",
            "session_id": "test_session",
            "current_phase": 1
        }
        
        # Run through agenda setting
        graph = flow.build_graph()
        
        # Mock HITL approval
        with patch("virtual_agora.flow.nodes.get_user_approval") as mock:
            mock.return_value = {
                "approved": True,
                "agenda": ["Topic A", "Topic B"]
            }
            
            result = graph.invoke(
                initial_state,
                {"recursion_limit": 10}
            )
        
        assert "agenda" in result
        assert result["agenda"]["topics"] == ["Topic A", "Topic B"]
    
    def test_discussion_with_periodic_stop(self, flow):
        """Test discussion with 5-round checkpoint."""
        state = {
            "main_topic": "AI Safety",
            "active_topic": "AI Alignment",
            "current_round": 4,  # Next round triggers stop
            "agents": {"agent1": {}, "agent2": {}},
            "speaking_order": ["agent1", "agent2"]
        }
        
        graph = flow.build_graph()
        
        # Mock user not forcing end
        with patch("virtual_agora.flow.nodes.get_periodic_stop") as mock:
            mock.return_value = {"force_topic_end": False}
            
            # Run one round
            result = graph.invoke(state, {"recursion_limit": 5})
            
            # Should have prompted user at round 5
            assert mock.called
            assert result["current_round"] == 5
    
    def test_topic_conclusion_flow(self, flow):
        """Test topic conclusion with report generation."""
        state = {
            "active_topic": "AI Safety",
            "main_topic": "Future of AI",
            "current_round": 5,
            "round_summaries": [
                {"topic": "AI Safety", "content": "Summary 1"},
                {"topic": "AI Safety", "content": "Summary 2"}
            ],
            "topic_conclusion_votes": [
                {"agent_id": "a1", "vote": "yes"},
                {"agent_id": "a2", "vote": "yes"},
                {"agent_id": "a3", "vote": "no"}
            ]
        }
        
        graph = flow.build_graph()
        
        # Run conclusion flow
        result = graph.invoke(state, {"recursion_limit": 10})
        
        assert "topic_summaries" in result
        assert "AI Safety" in result["topic_summaries"]
        assert "topic_report_saved" in result
```

## End-to-End Testing

### Complete Session Simulation

```python
# tests/e2e/test_complete_session_v3.py

import pytest
from virtual_agora.main import VirtualAgoraApplication

class TestCompleteSession:
    """End-to-end session tests."""
    
    @pytest.fixture
    def app(self, test_config):
        """Create test application."""
        return VirtualAgoraApplication(test_config)
    
    def test_minimal_session(self, app):
        """Test minimal complete session."""
        # Initial theme
        with patch("rich.prompt.Prompt.ask") as mock_prompt:
            mock_prompt.return_value = "Future of AI"
            
            # Start session
            app.start_session()
        
        # Verify initialization
        assert app.state["main_topic"] == "Future of AI"
        assert len(app.specialized_agents) == 4
        assert len(app.discussing_agents) >= 2
    
    def test_full_session_with_reports(self, app, tmp_path):
        """Test complete session with report generation."""
        # Configure output directory
        app.report_dir = tmp_path
        
        # Mock all user interactions
        user_responses = [
            "Future of AI",  # Theme
            "Approve",  # Agenda
            "no",  # Don't end at checkpoint
            "yes",  # End topic after votes
            "yes",  # Continue session
            "no",  # Don't modify agenda
            "no",  # End session
        ]
        
        with patch("rich.prompt.Prompt.ask") as mock_prompt:
            mock_prompt.side_effect = user_responses
            
            # Run complete session
            app.run()
        
        # Verify outputs
        report_files = list(tmp_path.glob("*.md"))
        assert len(report_files) >= 2  # Topic + final reports
        
        # Check topic report
        topic_reports = [f for f in report_files if "agenda_summary" in f.name]
        assert len(topic_reports) >= 1
        
        # Check final report
        final_reports = [f for f in report_files if "final_report" in f.name]
        assert len(final_reports) >= 1
```

### Performance Benchmarks

```python
# tests/e2e/test_performance_v3.py

import pytest
import time
from statistics import mean, stdev

class TestPerformance:
    """Performance validation tests."""
    
    def test_agent_response_times(self, agents):
        """Benchmark agent response times."""
        specialized, discussing = agents
        
        # Test each specialized agent
        timings = {}
        
        for agent_type, agent in specialized.items():
            times = []
            
            for _ in range(5):  # Multiple runs
                start = time.time()
                
                if agent_type == "summarizer":
                    agent.summarize_round(
                        [{"content": "Test"}], "Topic", 1
                    )
                elif agent_type == "moderator":
                    agent.synthesize_agenda([{"vote": "A then B"}])
                # ... other agent types
                
                times.append(time.time() - start)
            
            timings[agent_type] = {
                "mean": mean(times),
                "stdev": stdev(times) if len(times) > 1 else 0
            }
        
        # Verify performance targets
        for agent_type, stats in timings.items():
            assert stats["mean"] < 2.0  # Under 2 seconds
            print(f"{agent_type}: {stats['mean']:.3f}s ± {stats['stdev']:.3f}s")
    
    def test_memory_usage(self, app):
        """Test memory usage during session."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run session simulation
        for round_num in range(20):  # 20 rounds
            app.state["current_round"] = round_num
            
            # Simulate round
            messages = [
                {"content": f"Message {i}"} for i in range(5)
            ]
            app.state["messages"].extend(messages)
            
            # Check memory growth
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory
            
            assert memory_growth < 500  # Less than 500MB growth
```

## Migration Testing

### Configuration Migration Tests

```python
# tests/migration/test_config_migration.py

def test_v1_to_v3_config_migration():
    """Test configuration migration."""
    v1_config = {
        "moderator": {
            "provider": "Google",
            "model": "gemini-2.5-pro"
        },
        "agents": [
            {
                "provider": "OpenAI",
                "model": "gpt-4o",
                "count": 2
            }
        ]
    }
    
    v3_config = migrate_config_v1_to_v3(v1_config)
    
    # Verify all required fields
    assert "summarizer" in v3_config
    assert "topic_report" in v3_config
    assert "ecclesia_report" in v3_config
    
    # Verify existing fields preserved
    assert v3_config["moderator"] == v1_config["moderator"]
    assert v3_config["agents"] == v1_config["agents"]
```

### API Compatibility Tests

```python
# tests/migration/test_api_compatibility.py

def test_moderator_api_compatibility():
    """Test ModeratorAgent API changes."""
    
    # v1.1 style (should fail)
    with pytest.raises(TypeError):
        moderator = ModeratorAgent(
            agent_id="test",
            llm=FakeLLM(),
            mode="synthesis"  # No longer accepted
        )
    
    # v1.3 style (should work)
    moderator = ModeratorAgent(
        agent_id="test",
        llm=FakeLLM()
        # No mode parameter
    )
    
    assert moderator is not None
```

## Test Organization

### Directory Structure
```
tests/
├── agents/
│   ├── test_summarizer.py
│   ├── test_topic_report_agent.py
│   ├── test_ecclesia_report_agent.py
│   └── test_moderator_v3.py
├── flow/
│   ├── test_nodes_v3.py
│   ├── test_edges_v3.py
│   └── test_graph_v3.py
├── ui/
│   ├── test_hitl_manager_v3.py
│   ├── test_components_v3.py
│   └── test_dashboard_v3.py
├── integration/
│   ├── test_agent_coordination_v3.py
│   ├── test_graph_flow_v3.py
│   └── test_hitl_integration_v3.py
├── e2e/
│   ├── test_complete_session_v3.py
│   ├── test_performance_v3.py
│   └── test_report_generation_v3.py
└── migration/
    ├── test_config_migration.py
    ├── test_state_migration.py
    └── test_api_compatibility.py
```

### Test Execution Strategy

```bash
# Run all v1.3 tests
pytest tests/ -m v3

# Run by phase
pytest tests/agents/  # Phase 2 validation
pytest tests/flow/    # Phase 3 validation
pytest tests/ui/      # Phase 4 validation

# Run integration tests
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=virtual_agora --cov-report=html

# Performance tests (slower)
pytest tests/e2e/test_performance_v3.py -v --benchmark
```

## Validation Checklist

### Unit Test Coverage
- [ ] All specialized agents have unit tests
- [ ] Each node function tested independently
- [ ] HITL components tested with mocks
- [ ] State management tested
- [ ] Edge conditions tested

### Integration Test Coverage
- [ ] Agent coordination validated
- [ ] Graph flow paths tested
- [ ] HITL integration verified
- [ ] State transitions tested
- [ ] Error scenarios handled

### E2E Test Coverage
- [ ] Complete session runnable
- [ ] Reports generated correctly
- [ ] Performance targets met
- [ ] Memory usage acceptable
- [ ] User experience smooth

### Migration Validation
- [ ] Config migration working
- [ ] State migration tested
- [ ] API compatibility verified
- [ ] Backward compatibility checked

## Common Testing Issues

### Issue 1: Async Test Complexity
**Problem**: Graph execution is async
**Solution**: Use pytest-asyncio fixtures

### Issue 2: LLM Mocking
**Problem**: Real LLMs slow and expensive
**Solution**: Comprehensive FakeLLM implementation

### Issue 3: HITL Testing
**Problem**: User interaction hard to test
**Solution**: Mock prompt functions consistently

### Issue 4: State Complexity
**Problem**: Large state objects
**Solution**: Builder pattern for test states

## Success Criteria

1. **Test Coverage**: >95% code coverage
2. **Performance**: All operations <2s
3. **Reliability**: No flaky tests
4. **Compatibility**: v1.1 configs work
5. **Quality**: Reports match specification

## References

- Testing Framework: `tests/conftest.py`
- Test Helpers: `tests/helpers/`
- v1.3 Specification: `docs/project_spec_2.md`
- Current Tests: `tests/` directory

## Migration Complete

Once all tests pass, the v1.3 migration is complete. Create final documentation and prepare for release.

---

**Document Version**: 1.0
**Phase**: 5 of 5
**Status**: Implementation Guide