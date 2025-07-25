# Epic 2: LLM Provider Integration Layer

## Epic Overview

Create a modernized LLM provider integration layer that leverages LangGraph's recommended patterns for initializing and managing multiple LLM providers (Google, OpenAI, Anthropic, Grok) with seamless StateGraph integration.

## Technical Context

- **Framework:** LangGraph with LangChain's `init_chat_model` pattern
- **Providers:** Google Gemini, OpenAI, Anthropic, Grok
- **Key Pattern:** LangGraph StateGraph integration with unified provider initialization
- **Error Handling:** LangGraph-style fallbacks and error recovery
- **Integration:** Optimized for LangGraph workflows and tool calling

---

## User Stories

### Story 2.1: Migrate to LangChain's init_chat_model Pattern

**As a** developer
**I want** to use LangGraph's recommended `init_chat_model()` pattern for provider initialization
**So that** the system follows LangGraph best practices and simplifies provider management

**Acceptance Criteria:**

- Update ProviderFactory to use `init_chat_model("provider:model")` pattern
- Maintain support for provider-specific configurations
- Ensure backward compatibility with existing config.yml format
- Support all current providers through the unified interface
- Preserve caching and performance optimizations
- Add proper fallback handling using `.with_fallbacks()`

**Technical Notes:**

- Use LangChain's `init_chat_model` for consistent provider initialization
- Map Virtual Agora provider types to LangChain provider strings
- Maintain provider-specific parameter support through extra_kwargs

---

### Story 2.2: Enhanced LangGraph State Integration

**As a** developer
**I want** the LLMAgent to be optimized for LangGraph StateGraph workflows
**So that** agents can seamlessly participate in LangGraph-based discussion flows

**Acceptance Criteria:**

- Optimize LLMAgent for use as StateGraph nodes
- Improve message formatting for LangGraph's message passing patterns
- Add support for streaming responses in StateGraph context
- Ensure thread-safe operation in concurrent StateGraph execution
- Support LangGraph's state management patterns
- Add integration with LangGraph's memory and checkpointing

**Technical Notes:**

- Leverage LangGraph's `add_messages` helper for message handling
- Support LangGraph's streaming patterns with proper message chunking
- Integrate with LangGraph's state schema requirements

---

### Story 2.3: Unified Error Handling with LangGraph Patterns

**As a** developer
**I want** to leverage LangGraph's error handling and fallback mechanisms
**So that** the system can gracefully handle provider failures and maintain conversation flow

**Acceptance Criteria:**

- Implement LangGraph-style model fallbacks using `.with_fallbacks()`
- Integrate with LangGraph's retry mechanisms
- Support circuit breaker patterns for persistent failures
- Maintain existing provider-specific error classification
- Add proper error recovery for StateGraph workflows
- Implement graceful degradation strategies

**Technical Notes:**

- Use LangChain's model fallback patterns
- Integrate with LangGraph's error handling pipeline
- Maintain backward compatibility with existing error handling

---

### Story 2.4: Tool Integration Optimization

**As a** developer
**I want** to optimize LLMAgent for LangGraph's tool calling patterns
**So that** agents can seamlessly use tools in discussion workflows

**Acceptance Criteria:**

- Add support for `.bind_tools()` pattern in LLMAgent
- Integrate with LangGraph's ToolNode for tool execution
- Support both individual tool calls and batch tool operations
- Handle tool call responses in StateGraph context
- Add tool call validation and error handling
- Support streaming tool calls where applicable

**Technical Notes:**

- Use LangChain's tool binding patterns
- Integrate with LangGraph's ToolNode workflows
- Support tool call serialization for state management

---

### Story 2.5: Provider Registry Modernization

**As a** developer
**I want** an updated provider registry that aligns with LangGraph's ecosystem
**So that** the system can leverage the latest models and capabilities

**Acceptance Criteria:**

- Update model registry with latest model versions and capabilities
- Add support for new LangGraph-compatible provider features
- Improve model capability detection (streaming, tools, function calling)
- Add provider-specific feature flags and limitations
- Update model metadata with current context windows and capabilities
- Add support for emerging providers in the LangGraph ecosystem

**Technical Notes:**

- Research latest model capabilities and versions
- Align model names with LangChain's provider naming conventions
- Update provider feature detection logic

---

## Dependencies

- Epic 1: Core Infrastructure (for configuration and logging)

## Definition of Done

- Migration to `init_chat_model` pattern completed for all providers
- LLMAgent optimized for LangGraph StateGraph workflows
- Model fallbacks and error recovery implemented using LangGraph patterns
- Tool integration with ToolNode functionality working
- Provider registry updated with latest models and capabilities
- All existing functionality preserved with enhanced LangGraph integration
- Comprehensive testing with LangGraph StateGraph workflows
- Documentation updated with LangGraph integration examples
