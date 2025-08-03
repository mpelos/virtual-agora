# Virtual Agora Architecture Refactoring: Implementation Steps

## Phase 1: Extract Core Abstractions (Week 1)

### Step 1.1: Create Round Management Abstraction (Day 1)

**Objective**: Centralize round state and transitions in a dedicated component

**Files to Create**:

- `src/virtual_agora/flow/round_manager.py`
- `tests/flow/test_round_manager.py`

**Implementation**:

```python
class RoundManager:
    """Centralized round state and transition management."""

    def get_current_round(self, state: VirtualAgoraState) -> int:
        """Get current round number with consistent logic."""

    def start_new_round(self, state: VirtualAgoraState) -> int:
        """Increment and return new round number."""

    def can_start_round(self, state: VirtualAgoraState) -> bool:
        """Determine if conditions allow starting new round."""

    def get_round_metadata(self, state: VirtualAgoraState, round_num: int) -> Dict:
        """Get round-specific metadata and context."""
```

**Acceptance Criteria**:

- [ ] Single source of truth for round numbers
- [ ] Consistent logic across all round operations
- [ ] 100% test coverage for round transitions
- [ ] No breaking changes to existing functionality

**Migration Strategy**:

1. Create `RoundManager` with current logic
2. Add comprehensive tests
3. Replace scattered round logic one location at a time
4. Validate behavior matches existing system

---

### Step 1.2: Extract Message Coordination (Day 2)

**Objective**: Create dedicated component for message assembly and routing

**Files to Create**:

- `src/virtual_agora/flow/message_coordinator.py`
- `tests/flow/test_message_coordinator.py`

**Implementation**:

```python
class MessageCoordinator:
    """Coordinates message assembly and routing for agents."""

    def assemble_agent_context(
        self,
        agent_id: str,
        round_num: int,
        state: VirtualAgoraState
    ) -> Tuple[str, List[BaseMessage]]:
        """Assemble complete context for agent."""

    def store_user_message(
        self,
        content: str,
        round_num: int,
        state: VirtualAgoraState
    ) -> VirtualAgoraState:
        """Store user participation message with correct round."""

    def get_messages_for_round(
        self,
        round_num: int,
        state: VirtualAgoraState
    ) -> List[ProcessedMessage]:
        """Get all messages for specific round."""
```

**Acceptance Criteria**:

- [ ] Centralized message assembly logic
- [ ] Consistent round numbering for user messages
- [ ] Clean interface for context building
- [ ] Backward compatible with existing message handling

---

### Step 1.3: Create Flow State Manager (Day 3)

**Objective**: Establish clear state boundaries and transitions

**Files to Create**:

- `src/virtual_agora/flow/state_manager.py`
- `tests/flow/test_state_manager.py`

**Implementation**:

```python
class FlowStateManager:
    """Manages discussion flow state and transitions."""

    def __init__(self, round_manager: RoundManager, message_coordinator: MessageCoordinator):
        self.round_manager = round_manager
        self.message_coordinator = message_coordinator

    def prepare_round_state(self, state: VirtualAgoraState) -> RoundState:
        """Prepare state for new discussion round."""

    def apply_user_participation(
        self,
        user_input: str,
        state: VirtualAgoraState
    ) -> VirtualAgoraState:
        """Apply user participation to current state."""

    def finalize_round(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Clean up and finalize round state."""
```

**Acceptance Criteria**:

- [ ] Clear state transition boundaries
- [ ] Composed from round and message managers
- [ ] Immutable state operations
- [ ] Full test coverage of state transitions

---

## Phase 2: Restructure Flow Control (Week 2)

### Step 2.1: Create Node Interface (Day 4)

**Objective**: Establish pluggable node architecture

**Files to Create**:

- `src/virtual_agora/flow/nodes/base.py`
- `src/virtual_agora/flow/nodes/__init__.py`
- `tests/flow/nodes/test_base.py`

**Implementation**:

```python
from abc import ABC, abstractmethod

class FlowNode(ABC):
    """Base interface for all flow nodes."""

    @abstractmethod
    def execute(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Execute node logic and return state updates."""

    @abstractmethod
    def validate_preconditions(self, state: VirtualAgoraState) -> bool:
        """Validate state before execution."""

    def get_node_name(self) -> str:
        """Get human-readable node name."""
        return self.__class__.__name__

class HITLNode(FlowNode):
    """Base class for human-in-the-loop nodes."""

    @abstractmethod
    def create_interrupt_payload(self, state: VirtualAgoraState) -> Dict:
        """Create interrupt payload for user interaction."""

    @abstractmethod
    def process_user_input(self, user_input: Dict, state: VirtualAgoraState) -> Dict[str, Any]:
        """Process user input and return state updates."""
```

**Acceptance Criteria**:

- [ ] Clear interface for all nodes
- [ ] HITL-specific base class
- [ ] Validation framework
- [ ] Documentation and examples

---

### Step 2.2: Create Discussion Round Node (Day 5)

**Objective**: Implement focused discussion round node with **configurable** user participation timing

**Files to Create**:

- `src/virtual_agora/flow/nodes/discussion_round.py`
- `src/virtual_agora/flow/participation_strategies.py`
- `tests/flow/nodes/test_discussion_round.py`
- `tests/flow/test_participation_strategies.py`

**Implementation**:

```python
from enum import Enum
from abc import ABC, abstractmethod

class ParticipationTiming(Enum):
    """When user participation should occur in discussion flow."""
    START_OF_ROUND = "start_of_round"  # User input before agents speak
    END_OF_ROUND = "end_of_round"      # User input after agents speak
    DISABLED = "disabled"              # No user participation

class UserParticipationStrategy(ABC):
    """Strategy pattern for user participation timing."""

    @abstractmethod
    def should_request_participation_before_agents(self, state: VirtualAgoraState) -> bool:
        """Check if user participation needed before agents speak."""

    @abstractmethod
    def should_request_participation_after_agents(self, state: VirtualAgoraState) -> bool:
        """Check if user participation needed after agents speak."""

    @abstractmethod
    def get_participation_context(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Get context for user participation prompt."""

class StartOfRoundParticipation(UserParticipationStrategy):
    """User participates at the beginning of each round."""

    def should_request_participation_before_agents(self, state: VirtualAgoraState) -> bool:
        return state.get("current_round", 0) >= 1

    def should_request_participation_after_agents(self, state: VirtualAgoraState) -> bool:
        return False

    def get_participation_context(self, state: VirtualAgoraState) -> Dict[str, Any]:
        return {
            "timing": "round_start",
            "message": f"Round {state.get('current_round', 0)} is about to begin. Provide guidance for the agents.",
            "show_previous_summary": True
        }

class EndOfRoundParticipation(UserParticipationStrategy):
    """User participates at the end of each round (current behavior)."""

    def should_request_participation_before_agents(self, state: VirtualAgoraState) -> bool:
        return False

    def should_request_participation_after_agents(self, state: VirtualAgoraState) -> bool:
        return state.get("current_round", 0) >= 1

    def get_participation_context(self, state: VirtualAgoraState) -> Dict[str, Any]:
        return {
            "timing": "round_end",
            "message": f"Round {state.get('current_round', 0)} has completed. What would you like to do next?",
            "show_round_summary": True
        }

class DiscussionRoundNode(HITLNode):
    """Handles complete discussion round with configurable user participation timing."""

    def __init__(self,
                 flow_state_manager: FlowStateManager,
                 discussing_agents: List[DiscussionAgent],
                 specialized_agents: Dict[str, Any],
                 participation_strategy: UserParticipationStrategy):
        self.flow_state_manager = flow_state_manager
        self.discussing_agents = discussing_agents
        self.specialized_agents = specialized_agents
        self.participation_strategy = participation_strategy

    def execute(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Execute complete discussion round with configurable user participation."""
        updates = {}

        # Phase 1: Pre-agent user participation (if strategy allows)
        if self.participation_strategy.should_request_participation_before_agents(state):
            participation_updates = self._handle_user_participation(state, "before_agents")
            updates.update(participation_updates)
            # Update state with user input for agent context
            state = {**state, **participation_updates}

        # Phase 2: Execute agent discussions
        discussion_updates = self._execute_agent_discussions(state)
        updates.update(discussion_updates)

        # Phase 3: Post-agent user participation (if strategy allows)
        if self.participation_strategy.should_request_participation_after_agents(state):
            participation_updates = self._handle_user_participation(state, "after_agents")
            updates.update(participation_updates)

        return updates

    def _handle_user_participation(self, state: VirtualAgoraState, timing: str) -> Dict[str, Any]:
        """Handle user participation with proper context and timing."""
        context = self.participation_strategy.get_participation_context(state)
        context["timing_phase"] = timing

        # Use LangGraph interrupt mechanism
        user_input = interrupt({
            "type": "user_turn_participation",
            "current_round": state.get("current_round", 0),
            "current_topic": state.get("active_topic", "Unknown Topic"),
            "context": context,
            "message": context["message"],
            "options": ["continue", "participate", "finalize"],
        })

        return self._process_user_participation_input(user_input, state, timing)

    def _execute_agent_discussions(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Execute agent discussions with proper context."""
        # Existing agent discussion logic here
        pass

    @classmethod
    def create_with_timing(cls, timing: ParticipationTiming, *args, **kwargs):
        """Factory method to create node with specific timing strategy."""
        strategies = {
            ParticipationTiming.START_OF_ROUND: StartOfRoundParticipation(),
            ParticipationTiming.END_OF_ROUND: EndOfRoundParticipation(),
            ParticipationTiming.DISABLED: DisabledParticipation(),
        }
        return cls(*args, participation_strategy=strategies[timing], **kwargs)
```

**Configuration Interface**:

```python
# In configuration or flow setup
class DiscussionFlowConfig:
    """Configuration for discussion flow behavior."""

    def __init__(self):
        # EASY TO CHANGE: Single line configuration change
        self.user_participation_timing = ParticipationTiming.START_OF_ROUND
        # Alternative: ParticipationTiming.END_OF_ROUND
        # Alternative: ParticipationTiming.DISABLED

    def create_discussion_node(self, flow_state_manager, agents, specialized_agents):
        return DiscussionRoundNode.create_with_timing(
            self.user_participation_timing,
            flow_state_manager,
            agents,
            specialized_agents
        )
```

**Acceptance Criteria**:

- [ ] **Configurable user participation timing** (start/end/disabled)
- [ ] **Single configuration change** switches timing behavior
- [ ] **Strategy pattern** enables easy extension of participation modes
- [ ] **Zero graph changes** required when switching timing
- [ ] Proper round state management for both timing modes
- [ ] Clean separation of user input and agent execution
- [ ] Full test coverage for all timing strategies
- [ ] **Migration path** from current end-of-round to start-of-round

---

### Step 2.3: Create Flow Orchestrator (Day 6)

**Objective**: Separate high-level orchestration from implementation details

**Files to Create**:

- `src/virtual_agora/flow/orchestrator.py`
- `tests/flow/test_orchestrator.py`

**Implementation**:

```python
class DiscussionFlowOrchestrator:
    """High-level orchestration of discussion flow."""

    def __init__(self,
                 node_registry: Dict[str, FlowNode],
                 conditions: V13FlowConditions):
        self.nodes = node_registry
        self.conditions = conditions

    def execute_discussion_flow(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Execute complete discussion flow with proper orchestration."""

    def handle_node_execution(self, node_name: str, state: VirtualAgoraState) -> Dict[str, Any]:
        """Execute single node with error handling and validation."""

    def determine_next_node(self, current_node: str, state: VirtualAgoraState) -> str:
        """Determine next node based on conditions and state."""
```

**Acceptance Criteria**:

- [ ] Clear separation of orchestration and implementation
- [ ] Error handling and recovery
- [ ] Pluggable node architecture
- [ ] Comprehensive logging and monitoring

---

## Phase 3: Consolidate Node Logic (Week 3)

### Step 3.1: Extract Individual Nodes (Days 7-8)

**Objective**: Break down monolithic `nodes_v13.py` into focused components

**Files to Create**:

- `src/virtual_agora/flow/nodes/agenda_proposal.py`
- `src/virtual_agora/flow/nodes/topic_conclusion.py`
- `src/virtual_agora/flow/nodes/vote_evaluation.py`
- `src/virtual_agora/flow/nodes/report_generation.py`
- Individual test files for each node

**Implementation Strategy**:

1. **Day 7**: Extract agenda and voting nodes
2. **Day 8**: Extract conclusion and reporting nodes

**Example Structure**:

```python
class AgendaProposalNode(HITLNode):
    """Handles agenda proposal and approval process."""

    def execute(self, state: VirtualAgoraState) -> Dict[str, Any]:
        # Focused agenda logic only

    def create_interrupt_payload(self, state: VirtualAgoraState) -> Dict:
        # Agenda-specific interrupt payload

    def process_user_input(self, user_input: Dict, state: VirtualAgoraState) -> Dict[str, Any]:
        # Agenda-specific user input processing
```

**Acceptance Criteria**:

- [ ] Each node has single responsibility
- [ ] Clean interfaces between nodes
- [ ] Complete test coverage
- [ ] No functional regression

---

### Step 3.2: Update Graph Definition (Day 9)

**Objective**: Modify graph to use new node architecture

**Files to Modify**:

- `src/virtual_agora/flow/graph_v13.py`

**Implementation**:

```python
class VirtualAgoraV13Flow:
    def __init__(self, ...):
        # Initialize new components
        self.round_manager = RoundManager()
        self.message_coordinator = MessageCoordinator()
        self.flow_state_manager = FlowStateManager(self.round_manager, self.message_coordinator)

        # Create node registry
        self.nodes = {
            "discussion_round": DiscussionRoundNode(
                self.flow_state_manager,
                discussing_agents,
                specialized_agents
            ),
            "agenda_proposal": AgendaProposalNode(...),
            # ... other nodes
        }

    def create_graph(self):
        # Simplified graph with new architecture
        graph.add_node("discussion_round", self.nodes["discussion_round"].execute)
        # Remove separate user_participation_check node
        # Direct edges from threshold check to discussion_round
```

**Acceptance Criteria**:

- [ ] Simplified graph structure
- [ ] User participation integrated into discussion round
- [ ] No separate user_participation_check node needed
- [ ] All existing routes preserved

---

## Phase 4: Integration & Validation (Week 4)

### Step 4.1: Integration Testing (Days 10-11)

**Objective**: Ensure new architecture works end-to-end

**Testing Strategy**:

- **Day 10**: Component integration tests
- **Day 11**: Full flow integration tests

**Files to Create**:

- `tests/integration/test_refactored_flow.py`
- `tests/integration/test_user_participation_integration.py`

**Test Scenarios**:

```python
def test_user_participation_at_round_start():
    """Test that user messages appear in agent context when provided at round start."""

def test_user_participation_at_round_end():
    """Test that user messages appear in agent context when provided at round end (legacy)."""

def test_participation_timing_configuration_change():
    """Test that changing participation timing requires only configuration change."""
    # This is the KEY test for the requirement
    config = DiscussionFlowConfig()

    # Test 1: Start of round configuration
    config.user_participation_timing = ParticipationTiming.START_OF_ROUND
    node_start = config.create_discussion_node(flow_manager, agents, specialized)

    # Test 2: End of round configuration
    config.user_participation_timing = ParticipationTiming.END_OF_ROUND
    node_end = config.create_discussion_node(flow_manager, agents, specialized)

    # Verify different behavior with same graph structure
    assert_different_participation_timing(node_start, node_end)

def test_round_numbering_consistency():
    """Test that round numbering is consistent across all components."""

def test_flow_state_transitions():
    """Test that state transitions work correctly with new architecture."""

def test_backward_compatibility():
    """Test that existing functionality is preserved."""

def test_zero_graph_changes_for_timing_switch():
    """Test that graph definition doesn't change when switching participation timing."""
    # Verify same graph structure works for both timing strategies
    graph_with_start_timing = create_graph(ParticipationTiming.START_OF_ROUND)
    graph_with_end_timing = create_graph(ParticipationTiming.END_OF_ROUND)

    assert graph_structure_identical(graph_with_start_timing, graph_with_end_timing)
```

**Acceptance Criteria**:

- [ ] All existing functionality preserved
- [ ] User participation works at round start
- [ ] User participation works at round end (backward compatibility)
- [ ] **Single configuration change switches participation timing**
- [ ] **Zero graph structure changes required for timing switch**
- [ ] Round numbering is consistent for both timing modes
- [ ] No performance regression

---

### Step 4.2: Performance Validation (Day 12)

**Objective**: Ensure refactoring doesn't impact performance

**Validation Areas**:

- Memory usage during discussion rounds
- Response time for user interactions
- Context building performance
- Overall flow execution time

**Files to Create**:

- `tests/performance/test_refactored_performance.py`
- `docs/performance-comparison.md`

**Acceptance Criteria**:

- [ ] No more than 5% performance regression
- [ ] Memory usage within acceptable limits
- [ ] User interaction response time < 2 seconds
- [ ] Documented performance comparison

---

### Step 4.3: Feature Parity Verification (Day 13)

**Objective**: Verify all features work identically to original implementation

**Verification Process**:

1. Complete user journey testing
2. Edge case testing
3. Error condition testing
4. State recovery testing

**Files to Create**:

- `tests/verification/test_feature_parity.py`
- `docs/feature-parity-checklist.md`

**Acceptance Criteria**:

- [ ] All user journeys work identically
- [ ] Error handling behavior preserved
- [ ] State recovery works correctly
- [ ] Documentation updated

---

### Step 4.4: Documentation & Cleanup (Day 14)

**Objective**: Complete refactoring with proper documentation

**Tasks**:

1. Update architecture documentation
2. Create migration guide
3. Update developer onboarding docs
4. Clean up deprecated code

**Files to Update/Create**:

- `docs/architecture-overview.md`
- `docs/developer-guide.md`
- `docs/migration-from-v13.md`
- Remove deprecated files

**Acceptance Criteria**:

- [ ] Complete architecture documentation
- [ ] Developer guide updated
- [ ] Migration path documented
- [ ] Deprecated code removed

---

## Success Metrics

### Quantitative Metrics

- **Lines of code**: Reduce `nodes_v13.py` from 1,185 to <200 lines
- **Cyclomatic complexity**: Reduce average complexity by 50%
- **Test coverage**: Maintain >90% coverage throughout refactoring
- **Change amplification**: Simple changes should touch <3 files
- **Configuration flexibility**: User participation timing change = 1 line change

### Qualitative Metrics

- **Developer feedback**: Survey team on architecture improvement
- **Feature velocity**: Time to implement new features should decrease
- **Bug rate**: Post-refactoring bug rate should not increase
- **Code review time**: Review time should decrease due to clearer structure

---

## Risk Management

### High-Risk Areas

1. **State management**: Changes to round/message state handling
2. **Context building**: Integration with existing context system
3. **HITL integration**: User interaction and interrupt handling

### Mitigation Strategies

1. **Incremental approach**: Each phase can be rolled back independently
2. **Feature flags**: Use flags to enable/disable new architecture
3. **Parallel implementation**: Run old and new systems in parallel during transition
4. **Comprehensive testing**: Test-driven refactoring approach

### Rollback Plan

Each phase includes rollback instructions and criteria for when to rollback. If any phase fails acceptance criteria, rollback to previous stable state and reassess approach.

---

**Status**: Implementation Ready
**Last Updated**: August 2025
**Next Review**: After Phase 1 completion
