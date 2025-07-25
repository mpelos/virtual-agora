# Epic 6: Discussion Flow & State Management

## Epic Overview
Implement the core discussion flow using LangGraph, managing the complex state transitions and multi-phase workflow of the Virtual Agora application.

## Technical Context
- **Framework:** LangGraph for state machine implementation
- **Flow:** 5 phases with cycles and conditional transitions
- **State Complexity:** Multi-level state with discussion history
- **Key Pattern:** Stateful graph with conditional edges

---

## User Stories

### Story 6.1: LangGraph Application Architecture
**As a** developer  
**I want** to implement the application using LangGraph  
**So that** complex workflows are manageable and maintainable

**Acceptance Criteria:**
- Define LangGraph state schema
- Implement node functions for each phase
- Configure conditional edges based on state
- Set up graph compilation and execution
- Implement state persistence between nodes
- Support graph visualization for debugging
- Handle graph execution errors

**Technical Notes:**
- Follow LangGraph best practices
- Consider implementing custom graph nodes
- Plan for graph testing strategies

---

### Story 6.2: Phase Flow Implementation
**As a** system  
**I want** to implement all 5 phases of discussion  
**So that** the complete workflow functions correctly

**Acceptance Criteria:**
- Phase 0: Initialization and setup
- Phase 1: Agenda setting with voting
- Phase 2-3: Discussion rounds with conclusion polling
- Phase 4: Agenda re-evaluation
- Phase 5: Final report generation
- Proper transitions between phases
- State consistency across phases

**Technical Notes:**
- Implement phases as graph nodes
- Use conditional edges for phase transitions
- Consider phase timeout handling

---

### Story 6.3: Discussion Round Orchestration
**As a** system  
**I want** to orchestrate discussion rounds  
**So that** agents participate in order

**Acceptance Criteria:**
- Implement rotating turn order algorithm
- Manage round counter
- Coordinate agent responses
- Trigger round summarization
- Handle round 3+ conclusion polling
- Support dynamic round counts
- Track round metrics

**Technical Notes:**
- Implement efficient turn rotation
- Consider round parallelization options
- Plan for round interruption handling

---

### Story 6.4: State Schema and Management
**As a** developer  
**I want** comprehensive state management  
**So that** all application data is tracked properly

**Acceptance Criteria:**
- Define complete state schema:
  - Current phase
  - Agenda (approved and remaining)
  - Agent list and states
  - Discussion history
  - Vote records
  - Round summaries
  - Topic summaries
- Implement state updates atomically
- Validate state transitions
- Support state inspection

**Technical Notes:**
- Use TypedDict for state definition
- Implement state validation middleware
- Consider state compression for large discussions

---

### Story 6.5: Conditional Logic Implementation
**As a** system  
**I want** to implement conditional transitions  
**So that** the flow adapts based on decisions

**Acceptance Criteria:**
- Implement conditional edges for:
  - Round number checks (>2 for polling)
  - Vote tallying (majority + 1)
  - Topic exhaustion
  - User approvals (HITL gates)
  - Agenda modifications
- Ensure all conditions are exhaustive
- Handle edge cases in conditions

**Technical Notes:**
- Use LangGraph conditional edges
- Implement condition functions clearly
- Test all conditional paths

---

### Story 6.6: Context Window Management
**As a** system  
**I want** to manage context effectively  
**So that** discussions stay within token limits

**Acceptance Criteria:**
- Track token usage across rounds
- Implement context summarization strategy
- Prioritize recent and relevant context
- Use Moderator summaries for compression
- Maintain discussion coherence
- Warn when approaching limits
- Implement context pruning algorithms

**Technical Notes:**
- Calculate tokens accurately per provider
- Implement sliding window approach
- Consider implementing context caching

---

### Story 6.7: Cycle Detection and Prevention
**As a** system  
**I want** to prevent infinite cycles  
**So that** discussions eventually conclude

**Acceptance Criteria:**
- Detect potential infinite loops:
  - Agenda modification cycles
  - Failed vote cycles
  - State transition loops
- Implement maximum iteration limits
- Add circuit breakers
- Provide escape mechanisms
- Log cycle detection events

**Technical Notes:**
- Implement cycle detection algorithms
- Set reasonable iteration limits
- Design graceful termination

---

### Story 6.8: State Persistence and Recovery
**As a** system  
**I want** to handle state persistence  
**So that** sessions can be analyzed or resumed

**Acceptance Criteria:**
- Serialize state at checkpoints
- Implement state recovery mechanism
- Support session pause/resume (future)
- Export state for analysis
- Handle corruption gracefully
- Version state schema
- Compress large state objects

**Technical Notes:**
- Choose appropriate serialization format
- Implement state migration strategy
- Consider using state snapshots

---

### Story 6.9: Flow Monitoring and Debugging
**As a** developer  
**I want** to monitor flow execution  
**So that** issues can be diagnosed quickly

**Acceptance Criteria:**
- Log all state transitions
- Track execution time per node
- Monitor token usage per phase
- Visualize current flow position
- Export execution traces
- Implement debug mode
- Support flow replay for testing

**Technical Notes:**
- Use LangGraph debugging features
- Implement custom monitoring hooks
- Consider implementing flow visualization

---

## Dependencies
- Epic 1: Core Infrastructure (for state foundation)
- Epic 3: Moderator Agent (for phase orchestration)
- Epic 4: Discussion Agent Framework (for agent coordination)
- Epic 5: Agenda Management (for agenda state)

## Definition of Done
- Complete flow executes end-to-end successfully
- All phase transitions work correctly
- Conditional logic handles all cases
- State remains consistent throughout
- Context management keeps discussions coherent
- No infinite loops possible
- Performance meets requirements
- Debugging tools are functional