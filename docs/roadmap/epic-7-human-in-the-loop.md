# Epic 7: Human-in-the-Loop (HITL) Controls

## Epic Overview

Implement human control points throughout the application, ensuring users maintain ultimate authority over the discussion flow while providing smooth interaction experiences.

## Technical Context

- **Control Points:** Topic input, agenda approval, continuation decisions
- **User Authority:** Human can override or edit any AI decision
- **Interface:** Terminal-based prompts with rich formatting
- **Validation:** Robust input handling and error recovery

---

## User Stories

### Story 7.1: Initial Topic Input Interface

**As a** user
**I want** to specify the discussion topic
**So that** agents discuss what interests me

**Acceptance Criteria:**

- Display clear welcome message
- Prompt for discussion topic
- Validate input (non-empty, reasonable length)
- Allow multi-line topic descriptions
- Confirm topic before proceeding
- Support topic modification before start
- Handle special characters properly

**Technical Notes:**

- Use rich library for formatting
- Implement input validation
- Consider topic templates for guidance

---

### Story 7.2: Agenda Approval Workflow

**As a** user
**I want** to review and approve the proposed agenda
**So that** I control what gets discussed

**Acceptance Criteria:**

- Display proposed agenda clearly
- Show vote tallies and rankings
- Allow three options:
  - Approve as-is
  - Edit agenda items
  - Reject and request new proposals
- Support agenda reordering
- Validate edited agenda
- Confirm final agenda

**Technical Notes:**

- Implement intuitive editing interface
- Support keyboard shortcuts
- Show agenda modification preview

---

### Story 7.3: Topic Continuation Gate

**As a** user
**I want** to control session continuation
**So that** I can end discussions when appropriate

**Acceptance Criteria:**

- Display topic completion summary
- Show remaining agenda items
- Provide clear options:
  - Continue to next topic
  - End session
  - Modify agenda first
- Display session statistics
- Confirm session termination
- Handle accidental inputs

**Technical Notes:**

- Show time elapsed and estimates
- Implement confirmation dialogs
- Support quick session end option

---

### Story 7.4: Interactive Agenda Editing

**As a** user
**I want** to edit the agenda interactively
**So that** I can shape the discussion precisely

**Acceptance Criteria:**

- Display numbered agenda items
- Support operations:
  - Add new items
  - Remove items
  - Reorder items
  - Edit item text
  - Merge similar items
- Show edit preview
- Validate modifications
- Support undo capability

**Technical Notes:**

- Consider implementing menu-based interface
- Use arrow keys for navigation
- Support batch operations

---

### Story 7.5: Emergency Controls

**As a** user
**I want** emergency control options
**So that** I can intervene when needed

**Acceptance Criteria:**

- Implement interrupt mechanism (Ctrl+C handling)
- Provide pause/resume functionality
- Allow mid-discussion interventions:
  - Skip current speaker
  - End topic early
  - Mute disruptive agent
- Graceful emergency shutdown
- State preservation on interrupt

**Technical Notes:**

- Implement signal handlers
- Design interrupt menu
- Ensure data integrity

---

### Story 7.6: Input Validation and Error Recovery

**As a** user
**I want** robust input handling
**So that** mistakes don't crash the system

**Acceptance Criteria:**

- Validate all user inputs
- Provide clear error messages
- Re-prompt on invalid input
- Support input cancellation
- Handle edge cases:
  - Empty inputs
  - Extremely long inputs
  - Special characters
  - Timeout scenarios
- Implement input history

**Technical Notes:**

- Use validation decorators
- Implement retry logic
- Consider input autocomplete

---

### Story 7.7: User Preference Management

**As a** user
**I want** to set preferences
**So that** the system behaves as I expect

**Acceptance Criteria:**

- Support preference settings:
  - Auto-approve unanimous votes
  - Default continuation choice
  - Timeout preferences
  - Display verbosity
- Save preferences between sessions
- Allow runtime preference changes
- Show current preferences

**Technical Notes:**

- Implement preference file
- Support command-line overrides
- Design preference UI

---

### Story 7.8: Help and Guidance System

**As a** user
**I want** contextual help
**So that** I understand my options

**Acceptance Criteria:**

- Provide help at each decision point
- Show keyboard shortcuts
- Explain available options
- Provide examples where relevant
- Support detailed help mode
- Include tooltips for complex features
- Quick reference card

**Technical Notes:**

- Implement context-aware help
- Support help command
- Design help formatting

---

### Story 7.9: Session Control Dashboard

**As a** user
**I want** to see session status
**So that** I can make informed decisions

**Acceptance Criteria:**

- Display current status:
  - Active topic
  - Round number
  - Speaking agent
  - Time elapsed
  - Topics completed
- Show progress indicators
- Display agent participation metrics
- Update in real-time
- Support minimal/detailed views

**Technical Notes:**

- Implement status bar
- Use rich layouts
- Consider progress animations

---

## Dependencies

- Epic 5: Agenda Management (for agenda interaction)
- Epic 6: Discussion Flow (for control point integration)
- Epic 9: Terminal UI (for interface components)

## Definition of Done

- All control points tested with various inputs
- Input validation prevents all crashes
- Emergency controls work reliably
- User experience is smooth and intuitive
- Help system provides adequate guidance
- Preferences persist correctly
- Interface is responsive
- Accessibility considerations addressed
