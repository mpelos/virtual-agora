# Epic 4: Discussion Agent Framework

## Epic Overview
Create the framework for Discussion Agents that participate in structured debates, including agent instantiation, identity management, and interaction protocols.

## Technical Context
- **Agent Naming:** Based on model + index (e.g., gpt-4o-1, gpt-4o-2)
- **Core Behavior:** Thoughtful participation, topic proposals, voting
- **Multiple Instances:** Support multiple agents of same model type
- **Interaction Pattern:** Turn-based with context awareness

---

## User Stories

### Story 4.1: Base Discussion Agent Implementation
**As a** developer  
**I want** a base discussion agent class  
**So that** all agents share common functionality

**Acceptance Criteria:**
- Create base DiscussionAgent class
- Implement core prompt from section 5.B
- Support conversation history management
- Include methods for:
  - Generating responses
  - Proposing topics
  - Voting on agendas
  - Voting on topic conclusion
- Maintain agent state (warnings, mute status)

**Technical Notes:**
- Use composition over inheritance where possible
- Consider implementing agent memory limits
- Support future persona extensions

---

### Story 4.2: Agent Factory and Instantiation
**As a** system  
**I want** to create agents based on configuration  
**So that** the agent pool matches config.yml specifications

**Acceptance Criteria:**
- Parse agent configuration from config.yml
- Create correct number of each agent type
- Generate unique names (model-index format)
- Assign appropriate LLM provider instances
- Validate agent creation
- Handle provider initialization failures

**Technical Notes:**
- Implement factory pattern
- Consider agent pooling for efficiency
- Support dynamic agent addition (future feature)

---

### Story 4.3: Agent Response Generation
**As a** discussion agent  
**I want** to generate thoughtful responses  
**So that** I contribute meaningfully to discussions

**Acceptance Criteria:**
- Accept topic and conversation context
- Generate relevant, substantive responses
- Stay within token limits
- Build upon previous discussion points
- Maintain consistent agent perspective
- Handle various discussion phases appropriately

**Technical Notes:**
- Implement response length targets
- Consider implementing response quality checks
- Add response timing metrics

---

### Story 4.4: Topic Proposal Mechanism
**As a** discussion agent  
**I want** to propose discussion topics  
**So that** the agenda reflects diverse perspectives

**Acceptance Criteria:**
- Generate 3-5 sub-topics based on main topic
- Ensure proposals are:
  - Specific and actionable
  - Relevant to main topic
  - Diverse in scope
- Format proposals clearly
- Avoid duplicate proposals within same agent
- Consider different angles/perspectives

**Technical Notes:**
- Implement proposal validation
- Consider proposal quality scoring
- Track proposal acceptance rates

---

### Story 4.5: Voting Behavior Implementation
**As a** discussion agent  
**I want** to vote on agendas and topic conclusions  
**So that** I participate in democratic decisions

**Acceptance Criteria:**
- Parse voting requests from Moderator
- For agenda voting:
  - Review all proposed topics
  - Express preferences in natural language
  - Provide reasoning for choices
- For conclusion voting:
  - Vote "Yes" or "No" clearly
  - Provide justification
- Handle voting edge cases gracefully

**Technical Notes:**
- Implement vote parsing validation
- Consider implementing strategic voting
- Track voting patterns for analysis

---

### Story 4.6: Context and Memory Management
**As a** discussion agent  
**I want** to maintain conversation context  
**So that** my responses are coherent and build on prior discussion

**Acceptance Criteria:**
- Maintain conversation history within token limits
- Prioritize recent and relevant context
- Include:
  - Current topic
  - Recent agent responses
  - Round summaries from Moderator
  - Own previous contributions
- Implement context pruning strategies
- Preserve key discussion points

**Technical Notes:**
- Implement sliding window for context
- Consider implementing importance scoring
- Balance context size with response quality

---

### Story 4.7: Agent State Management
**As a** system  
**I want** to track agent state  
**So that** rules can be enforced consistently

**Acceptance Criteria:**
- Track per agent:
  - Warning count per topic
  - Mute status
  - Turn participation
  - Vote history
- Reset warnings on topic change
- Persist state across rounds
- Support state inspection for debugging

**Technical Notes:**
- Implement state as separate concern
- Consider implementing state persistence
- Add state validation logic

---

### Story 4.8: Multi-Agent Coordination
**As a** system  
**I want** agents to work together effectively  
**So that** discussions are productive and orderly

**Acceptance Criteria:**
- Implement turn-based coordination
- Ensure agents wait for their turn
- Handle agent response timeouts
- Support parallel proposal generation
- Coordinate voting phases
- Prevent response collisions

**Technical Notes:**
- Consider implementing agent queuing
- Add concurrency controls where needed
- Monitor agent performance metrics

---

### Story 4.9: Agent Response Validation
**As a** system  
**I want** to validate agent outputs  
**So that** the discussion proceeds smoothly

**Acceptance Criteria:**
- Validate response format and content
- Check response length limits
- Ensure voting responses are parseable
- Handle malformed responses gracefully
- Implement retry logic for failures
- Log validation issues

**Technical Notes:**
- Implement validation as middleware
- Consider implementing response repair
- Track validation failure patterns

---

## Dependencies
- Epic 1: Core Infrastructure (for configuration and state)
- Epic 2: LLM Provider Integration (for model access)

## Definition of Done
- Agents successfully participate in mock discussions
- Multiple agents of same type work correctly
- Voting mechanisms function reliably
- Context management keeps discussions coherent
- Agent state tracking is accurate
- Performance meets response time targets
- Edge cases handled gracefully