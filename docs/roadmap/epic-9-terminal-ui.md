# Epic 9: Terminal User Interface

## Epic Overview
Create a rich, color-coded terminal interface using the `rich` library to provide clear visual distinction between different actors and enhance the user experience.

## Technical Context
- **Library:** Python rich for terminal formatting
- **Design Goals:** Clarity, readability, accessibility
- **Color Scheme:** Distinct colors per agent group
- **Features:** Progress bars, tables, formatted text

---

## User Stories

### Story 9.1: Terminal UI Framework Setup
**As a** developer  
**I want** to establish the UI framework  
**So that** all output is consistently formatted

**Acceptance Criteria:**
- Initialize rich console properly
- Configure color themes
- Set up layout managers
- Implement output buffering
- Support different terminal types
- Handle terminal resize
- Configure emoji support

**Technical Notes:**
- Use rich.console.Console
- Implement singleton pattern
- Consider terminal capability detection

---

### Story 9.2: Color-Coded Agent Display
**As a** user  
**I want** agents to have distinct colors  
**So that** I can easily follow the conversation

**Acceptance Criteria:**
- Assign colors by provider:
  - Moderator: Bold White
  - OpenAI agents: Unique color
  - Google agents: Unique color
  - Anthropic agents: Unique color
  - Grok agents: Unique color
- Ensure color contrast
- Support color-blind friendly options
- Maintain consistency throughout session
- Handle many agents gracefully

**Technical Notes:**
- Implement color assignment algorithm
- Use rich.style for formatting
- Consider accessibility standards

---

### Story 9.3: System Message Formatting
**As a** user  
**I want** clear system messages  
**So that** I understand what's happening

**Acceptance Criteria:**
- Format message types:
  - User prompts: Yellow
  - System info: Default
  - Errors: Red
  - Warnings: Orange
  - Success: Green
- Use icons/symbols for clarity
- Implement message boxing
- Support message timestamps
- Handle multi-line messages

**Technical Notes:**
- Use rich.panel for important messages
- Implement message type enum
- Consider message history display

---

### Story 9.4: Progress Indicators
**As a** user  
**I want** to see operation progress  
**So that** I know the system is working

**Acceptance Criteria:**
- Show progress for:
  - Agent response generation
  - Voting collection
  - Report generation
  - File operations
- Use appropriate indicators:
  - Spinners for indeterminate
  - Progress bars for determinate
- Display operation context
- Handle nested progress
- Show time estimates

**Technical Notes:**
- Use rich.progress
- Implement progress context manager
- Consider progress persistence

---

### Story 9.5: Discussion Display Layout
**As a** user  
**I want** organized discussion display  
**So that** conversations are easy to follow

**Acceptance Criteria:**
- Display elements:
  - Agent name and identifier
  - Timestamp
  - Response content
  - Round markers
- Use consistent formatting:
  - Indentation for responses
  - Separators between turns
  - Headers for new rounds
- Support long responses
- Implement text wrapping
- Show context indicators

**Technical Notes:**
- Use rich.table for structured display
- Implement smart text wrapping
- Consider response truncation options

---

### Story 9.6: Interactive Menus and Prompts
**As a** user  
**I want** intuitive interactive elements  
**So that** I can easily make choices

**Acceptance Criteria:**
- Implement menu types:
  - Single choice
  - Multiple choice
  - Text input
  - Confirmation dialogs
- Show keyboard shortcuts
- Highlight current selection
- Support arrow key navigation
- Validate selections
- Provide input hints

**Technical Notes:**
- Use rich.prompt
- Implement custom prompt classes
- Consider menu animations

---

### Story 9.7: Status Dashboard
**As a** user  
**I want** a status dashboard  
**So that** I can monitor session state

**Acceptance Criteria:**
- Display components:
  - Current phase
  - Active topic
  - Round number
  - Agent states
  - Time elapsed
  - Memory usage
- Use live updating
- Support minimized view
- Position appropriately
- Handle terminal constraints

**Technical Notes:**
- Use rich.live
- Implement dashboard layout
- Consider split-screen display

---

### Story 9.8: Output Formatting Utilities
**As a** developer  
**I want** formatting utilities  
**So that** output is consistent

**Acceptance Criteria:**
- Implement formatters for:
  - Agent responses
  - Voting results
  - Summaries
  - Tables
  - Lists
  - Code blocks
- Support markdown rendering
- Handle special characters
- Implement truncation
- Support export formatting

**Technical Notes:**
- Create formatting module
- Use rich.markdown
- Implement formatter registry

---

### Story 9.9: Accessibility Features
**As a** user with accessibility needs  
**I want** accessible output options  
**So that** I can use the system effectively

**Acceptance Criteria:**
- Support features:
  - High contrast mode
  - Screen reader friendly output
  - Configurable font sizes
  - Reduced animations
  - Alternative to colors
- Implement accessibility commands
- Test with screen readers
- Document accessibility options
- Follow WCAG guidelines

**Technical Notes:**
- Implement accessibility mode
- Use semantic markup
- Test with accessibility tools

---

## Dependencies
- Epic 1: Core Infrastructure (for configuration)
- Epic 7: HITL Controls (for user interaction)
- All other epics (for display requirements)

## Definition of Done
- UI displays correctly on various terminals
- Colors are distinct and accessible
- All text is properly formatted
- Progress indicators work smoothly
- Interactive elements are intuitive
- Performance is responsive
- Accessibility features function correctly
- Documentation includes UI examples