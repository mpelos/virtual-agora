# Epic 3: Moderator Agent Implementation

## Epic Overview

Implement the Moderator Agent, a specialized non-participating AI that facilitates the discussion process, manages voting, enforces relevance, and produces summaries and reports.

## Technical Context

- **Role:** Process facilitator, not content participant
- **Key Responsibilities:** Agenda synthesis, turn management, relevance enforcement, summarization, report generation
- **Special Mode:** Acts as "The Writer" during final report generation
- **Output Requirements:** Structured JSON for specific operations

---

## User Stories

### Story 3.1: Moderator Agent Core Implementation

**As a** system
**I want** a specialized Moderator agent
**So that** discussions are facilitated impartially and efficiently

**Acceptance Criteria:**

- Implement Moderator class inheriting from base agent
- Configure with specified model (gemini-2.5-pro)
- Implement core prompt as specified in section 5.A
- Ensure Moderator never expresses opinions on topics
- Support multiple operational modes:
  - Facilitation mode
  - Synthesis mode
  - Writer mode
- Maintain conversation context

**Technical Notes:**

- Moderator uses more complex prompts than regular agents
- Must handle structured output generation (JSON)
- Consider prompt versioning for improvements

---

### Story 3.2: Agenda Synthesis and Voting Management

**As a** Moderator
**I want** to synthesize agent proposals into a ranked agenda
**So that** discussions follow a democratic process

**Acceptance Criteria:**

- Request 3-5 sub-topic proposals from each agent
- Collate proposals into unique list
- Present list to agents for voting
- Analyze natural language votes
- Produce ranked agenda as JSON: {"proposed_agenda": ["Topic1", "Topic2", ...]}
- Implement tie-breaking logic
- Validate JSON output format

**Technical Notes:**

- Use structured output parsing
- Handle edge cases (duplicate proposals, unclear votes)
- Log all voting data for transparency

---

### Story 3.3: Discussion Round Management

**As a** Moderator
**I want** to manage discussion rounds
**So that** all agents participate fairly

**Acceptance Criteria:**

- Announce current topic clearly
- Manage rotating turn order:
  - Round 1: [A, B, C]
  - Round 2: [B, C, A]
  - Round 3: [C, A, B]
- Track current speaker
- Provide appropriate context to each agent
- Signal round completion
- Handle agent timeouts gracefully

**Technical Notes:**

- Implement turn order rotation algorithm
- Consider implementing time limits per turn
- Track participation metrics

---

### Story 3.4: Relevance Enforcement System

**As a** Moderator
**I want** to ensure discussions stay on topic
**So that** conversations remain productive

**Acceptance Criteria:**

- Review each agent response for relevance
- Implement "warn-then-mute" protocol:
  - First off-topic: Issue warning
  - Second off-topic: Skip turn and mute for current topic
- Track warnings per agent per topic
- Provide clear feedback about relevance violations
- Log all enforcement actions

**Technical Notes:**

- Define clear relevance criteria
- Consider implementing relevance scoring
- Allow for some topic evolution within bounds

---

### Story 3.5: Round and Topic Summarization

**As a** Moderator
**I want** to create concise summaries
**So that** context remains manageable and focused

**Acceptance Criteria:**

- Generate round summaries after each complete round
- Create comprehensive topic summaries when topics conclude
- Ensure summaries are:
  - Agent-agnostic (no attribution)
  - Concise but complete
  - Focused on key insights
- Manage context window effectively
- Format summaries for both system use and human readability

**Technical Notes:**

- Implement summary length targets
- Consider key point extraction
- Preserve critical information while reducing tokens

---

### Story 3.6: Topic Conclusion Polling

**As a** Moderator
**I want** to conduct democratic polls for topic closure
**So that** discussions end at appropriate times

**Acceptance Criteria:**

- Start polling from round 3 onwards
- Ask clear yes/no question with justification request
- Tally votes accurately
- Determine if majority + 1 achieved
- Announce results clearly
- Trigger minority round if vote passes
- Handle invalid responses appropriately

**Technical Notes:**

- Implement vote parsing logic
- Track voting history
- Consider implementing vote prediction

---

### Story 3.7: Minority Considerations Management

**As a** Moderator
**I want** to give dissenting voters final say
**So that** all perspectives are captured before topic closure

**Acceptance Criteria:**

- Identify agents who voted "No"
- Prompt them specifically for final considerations
- Ensure responses are captured
- Include minority views in topic summary
- Manage this as a special round type
- Handle case where all agents voted "Yes"

**Technical Notes:**

- Track vote-to-agent mapping
- Implement special prompting for minority round
- Ensure fair representation in summaries

---

### Story 3.8: Report Writer Mode Implementation

**As a** Moderator (The Writer)
**I want** to generate structured final reports
**So that** session insights are preserved professionally

**Acceptance Criteria:**

- Switch to "Writer" mode when session ends
- Review all topic summaries
- Define report structure as JSON list of sections
- Generate content for each section
- Ensure professional, comprehensive output
- Maintain consistent tone and style
- Reference specific discussions appropriately

**Technical Notes:**

- Implement mode switching logic
- Consider report templates
- Ensure citations to specific topics

---

### Story 3.9: Agenda Modification Facilitation

**As a** Moderator
**I want** to facilitate agenda updates between topics
**So that** discussions can adapt based on insights

**Acceptance Criteria:**

- Present remaining topics to agents
- Request modification suggestions (add/remove)
- Collate all suggestions
- Facilitate new voting round
- Produce updated agenda
- Handle edge cases (empty agenda, all topics removed)

**Technical Notes:**

- Reuse voting logic from initial agenda setting
- Track agenda evolution history
- Handle agenda validation

---

## Dependencies

- Epic 1: Core Infrastructure (for state management)
- Epic 2: LLM Provider Integration (for Gemini model)
- Epic 4: Discussion Agent Framework (for agent interaction)

## Definition of Done

- Moderator successfully facilitates complete mock session
- All output formats validated (especially JSON)
- Relevance enforcement tested with various scenarios
- Summary quality meets requirements
- Report generation produces professional output
- Mode transitions work smoothly
- All edge cases handled gracefully
