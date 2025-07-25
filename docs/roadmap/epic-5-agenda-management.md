# Epic 5: Agenda Management System

## Epic Overview

Implement the democratic agenda setting and modification system that allows agents to propose, vote on, and dynamically update discussion topics throughout the session.

## Technical Context

- **Process:** Propose → Vote → Synthesize → Approve → Modify → Re-vote
- **Democratic Principle:** All agents have equal voice
- **Human Control:** Final approval always with user
- **Dynamic Updates:** Agenda can be modified between topics

---

## User Stories

### Story 5.1: Topic Proposal Collection System

**As a** system
**I want** to collect topic proposals from all agents
**So that** diverse perspectives shape the agenda

**Acceptance Criteria:**

- Request proposals from all agents simultaneously
- Collect 3-5 proposals per agent
- Handle proposal timeouts gracefully
- Aggregate all proposals into single list
- Remove exact duplicates
- Maintain proposal attribution for analytics
- Format proposals for presentation

**Technical Notes:**

- Consider parallel proposal collection
- Implement proposal similarity detection
- Set reasonable timeout limits

---

### Story 5.2: Voting Orchestration Engine

**As a** system
**I want** to orchestrate voting rounds
**So that** agents can democratically set priorities

**Acceptance Criteria:**

- Present complete proposal list to all agents
- Request ranking/preferences in natural language
- Collect all votes within timeout period
- Handle missing or invalid votes
- Pass votes to Moderator for synthesis
- Support both initial and modification voting
- Log all voting data

**Technical Notes:**

- Implement vote collection parallelization
- Consider vote validation preprocessing
- Design for vote audit trail

---

### Story 5.3: Agenda Synthesis and Ranking

**As a** system
**I want** to synthesize votes into ranked agenda
**So that** discussions follow democratic priorities

**Acceptance Criteria:**

- Aggregate natural language votes
- Use Moderator to synthesize rankings
- Produce JSON format: {"proposed_agenda": [...]}
- Handle tie-breaking consistently
- Validate JSON output structure
- Support agenda modification synthesis
- Provide synthesis explanation

**Technical Notes:**

- Implement robust JSON parsing
- Consider alternative ranking algorithms
- Plan for ranking transparency

---

### Story 5.4: Agenda State Management

**As a** system
**I want** to maintain agenda state
**So that** progress can be tracked and modified

**Acceptance Criteria:**

- Store approved agenda in application state
- Track current topic index
- Mark completed topics
- Support agenda modifications
- Maintain agenda history
- Calculate remaining topics
- Handle empty agenda gracefully

**Technical Notes:**

- Use immutable state updates
- Implement agenda versioning
- Consider agenda persistence

---

### Story 5.5: Agenda Modification Flow

**As a** system
**I want** to facilitate agenda modifications
**So that** discussions can adapt to new insights

**Acceptance Criteria:**

- Trigger modification after topic completion
- Present remaining topics to agents
- Request add/remove suggestions
- Collect modification proposals
- Return to voting flow with new items
- Handle edge cases:
  - All topics removed
  - No modifications suggested
  - Duplicate additions

**Technical Notes:**

- Reuse existing voting infrastructure
- Track modification frequency
- Implement modification limits

---

### Story 5.6: Topic Transition Management

**As a** system
**I want** to manage transitions between topics
**So that** discussions flow smoothly

**Acceptance Criteria:**

- Clear announcement of topic completion
- Save topic summary before transition
- Update agenda state
- Reset agent states (warnings)
- Announce new topic clearly
- Provide transition context
- Handle last topic specially

**Technical Notes:**

- Implement transition hooks
- Consider transition animations for UI
- Plan for transition metrics

---

### Story 5.7: Agenda Analytics and Reporting

**As a** system
**I want** to track agenda metrics
**So that** discussion patterns can be analyzed

**Acceptance Criteria:**

- Track proposal acceptance rates
- Monitor topic discussion duration
- Record modification patterns
- Calculate agent participation by topic
- Generate agenda statistics
- Export agenda evolution data
- Support post-session analysis

**Technical Notes:**

- Design metrics schema
- Implement efficient data collection
- Plan for analytics export

---

### Story 5.8: Edge Case Handling

**As a** system
**I want** to handle agenda edge cases
**So that** the system remains stable

**Acceptance Criteria:**

- Handle empty proposal lists
- Manage single-agent scenarios
- Deal with all agents abstaining
- Process invalid vote formats
- Handle agenda exhaustion
- Manage cyclic modifications
- Implement safeguards and limits

**Technical Notes:**

- Implement comprehensive validation
- Design fallback behaviors
- Add circuit breakers for infinite loops

---

## Dependencies

- Epic 3: Moderator Agent (for synthesis and ranking)
- Epic 4: Discussion Agent Framework (for proposals and voting)
- Epic 7: Human-in-the-Loop Controls (for approval flow)

## Definition of Done

- Complete agenda lifecycle tested end-to-end
- Democratic voting produces consistent results
- Modifications work smoothly between topics
- Edge cases don't crash the system
- State management is reliable
- Analytics data is accurately collected
- Performance meets requirements for large agent pools
