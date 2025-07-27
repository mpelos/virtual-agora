# Phase 1: Configuration & State Migration Guide

## Purpose
This document provides detailed instructions for updating Virtual Agora's configuration models and state schema to support the v1.3 node-centric architecture with specialized agents. This is the foundation phase that enables all subsequent migration work.

## Prerequisites
- Understanding of the v1.1 architecture (monolithic ModeratorAgent)
- Familiarity with Pydantic models and YAML configuration
- Access to both v1.1 and v1.3 specifications
- Python 3.10+ development environment

## Configuration Model Updates

### Step 1: Analyze Current Configuration Structure

The current v1.1 configuration (`src/virtual_agora/config/models.py`) contains:
- `ModeratorConfig`: Single moderator with multiple internal modes
- `AgentConfig`: Discussion participants only
- `Config`: Root model with moderator + agents list

### Step 2: Define New Specialized Agent Models

Create new Pydantic models for each specialized agent. Each model should follow this pattern:

```python
class SummarizerConfig(BaseModel):
    """Configuration for the Summarizer agent.
    
    The summarizer compresses round discussions into compacted context.
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    
    provider: Provider = Field(
        ..., 
        description="LLM provider for the summarizer"
    )
    model: str = Field(
        ...,
        description="Model name/ID for the summarizer",
        min_length=1,
    )
    # Optional fields for future extensibility
    temperature: Optional[float] = Field(
        default=0.3,  # Lower temperature for consistent summarization
        ge=0.0,
        le=1.0,
        description="Temperature for summarization"
    )
    max_tokens: Optional[int] = Field(
        default=500,
        description="Maximum tokens for summaries"
    )
```

Repeat this pattern for:
- `TopicReportConfig`: For agenda item synthesis
- `EcclesiaReportConfig`: For final report generation

### Step 3: Update Root Configuration Model

Modify the `Config` class to include all specialized agents:

```python
class Config(BaseModel):
    """Root configuration for Virtual Agora v1.3."""
    
    # Required specialized agents
    moderator: ModeratorConfig = Field(
        ..., 
        description="Process facilitation agent"
    )
    summarizer: SummarizerConfig = Field(
        ..., 
        description="Round compression agent"
    )
    topic_report: TopicReportConfig = Field(
        ..., 
        description="Topic synthesis agent"
    )
    ecclesia_report: EcclesiaReportConfig = Field(
        ..., 
        description="Final report generation agent"
    )
    
    # Discussion participants
    agents: list[AgentConfig] = Field(
        ...,
        description="List of discussing agents",
        min_length=1,
    )
```

### Step 4: Create Configuration Migration Helper

Implement a migration utility to convert v1.1 configs to v1.3:

```python
# src/virtual_agora/config/migration.py
def migrate_config_v1_to_v3(old_config: dict) -> dict:
    """Convert v1.1 config to v1.3 format."""
    
    # Start with the existing config
    new_config = old_config.copy()
    
    # Add specialized agents using sensible defaults
    if 'summarizer' not in new_config:
        # Use moderator's provider as default
        moderator_provider = old_config['moderator']['provider']
        new_config['summarizer'] = {
            'provider': moderator_provider,
            'model': _get_default_model(moderator_provider, 'summarizer')
        }
    
    # Similar logic for topic_report and ecclesia_report
    # ...
    
    return new_config
```

### Step 5: Update Configuration Validation

Enhance `src/virtual_agora/config/validators.py`:

```python
def validate_specialized_agents(config: Config) -> None:
    """Validate specialized agent configurations."""
    
    # Ensure all specialized agents use compatible models
    providers_models = {
        'moderator': (config.moderator.provider, config.moderator.model),
        'summarizer': (config.summarizer.provider, config.summarizer.model),
        'topic_report': (config.topic_report.provider, config.topic_report.model),
        'ecclesia_report': (config.ecclesia_report.provider, config.ecclesia_report.model),
    }
    
    # Validate each agent's model is available
    for agent_type, (provider, model) in providers_models.items():
        if not is_model_available(provider, model):
            raise ValueError(
                f"{agent_type} model '{model}' not available for {provider}"
            )
```

## State Schema Evolution

### Step 1: Analyze Current State Structure

The v1.1 state (`src/virtual_agora/state/schema.py`) tracks:
- Single moderator with mode switching
- Basic HITL approval
- Simple topic progression

### Step 2: Add Specialized Agent Tracking

Update `VirtualAgoraState` to track specialized agents:

```python
class VirtualAgoraState(TypedDict):
    # ... existing fields ...
    
    # Specialized agent tracking
    specialized_agents: Dict[str, str]  # agent_type -> agent_id
    agent_invocations: Annotated[
        List[AgentInvocation], list.append
    ]  # Track which agents were called when
    
    # Enhanced context flow
    round_summaries: Annotated[
        List[RoundSummary], list.append
    ]  # Compacted summaries per round
    
    # Periodic HITL stops
    periodic_stop_counter: int  # Tracks rounds for 5-round stops
    user_stop_history: Annotated[
        List[UserStop], list.append
    ]  # When user was asked to stop
```

### Step 3: Enhanced HITL State

Expand the `HITLState` for new interaction patterns:

```python
class HITLState(TypedDict):
    """Enhanced Human-in-the-Loop state."""
    
    awaiting_approval: bool
    approval_type: Optional[str]  # Extended set of types
    
    # New approval types for v1.3
    # - 'periodic_stop': Every 5 rounds
    # - 'topic_conclusion_override': User can force topic end
    # - 'session_continuation': After each topic
    # - 'final_report_generation': Before final report
    
    # Periodic stop tracking
    last_periodic_stop_round: Optional[int]
    periodic_stop_responses: List[Dict[str, Any]]
```

### Step 4: Context Flow Tracking

Add structures to track context flow between specialized agents:

```python
class AgentContext(TypedDict):
    """Context provided to a specialized agent."""
    
    agent_type: str  # 'summarizer', 'topic_report', etc.
    input_context: Dict[str, Any]
    timestamp: datetime
    source_node: str  # Which graph node called this agent
    
class ContextFlow(TypedDict):
    """Tracks how context flows between agents."""
    
    round_context: List[str]  # Current round's messages
    compacted_summaries: List[str]  # Previous round summaries
    active_topic: str
    discussion_theme: str
```

### Step 5: Create State Migration Utilities

Implement state migration helpers:

```python
# src/virtual_agora/state/migration.py
def migrate_state_v1_to_v3(old_state: dict) -> dict:
    """Migrate v1.1 state to v1.3 format."""
    
    new_state = old_state.copy()
    
    # Initialize new fields
    new_state['specialized_agents'] = {}
    new_state['agent_invocations'] = []
    new_state['round_summaries'] = []
    new_state['periodic_stop_counter'] = 0
    new_state['user_stop_history'] = []
    
    # Migrate HITL state
    if 'hitl_state' in new_state:
        new_state['hitl_state']['last_periodic_stop_round'] = None
        new_state['hitl_state']['periodic_stop_responses'] = []
    
    return new_state
```

## Implementation Instructions

### Development Workflow

1. **Branch Setup**
   ```bash
   git checkout -b migration/phase-1-config-state
   ```

2. **Incremental Updates**
   - Update configuration models first
   - Test configuration loading
   - Update state schema
   - Test state initialization
   - Implement migration utilities
   - Test migrations with sample data

3. **Validation Checkpoints**
   - After each model update, run: `pytest tests/unit/test_config.py`
   - Validate YAML parsing: `pytest tests/config/test_validators.py`
   - Check state initialization: `pytest tests/unit/test_state.py`

### Testing Requirements

1. **Configuration Tests**
   ```python
   def test_v3_config_structure():
       """Verify v1.3 config loads correctly."""
       config_dict = {
           'moderator': {'provider': 'Google', 'model': 'gemini-2.5-pro'},
           'summarizer': {'provider': 'OpenAI', 'model': 'gpt-4o'},
           'topic_report': {'provider': 'Anthropic', 'model': 'claude-3'},
           'ecclesia_report': {'provider': 'Google', 'model': 'gemini-2.5-pro'},
           'agents': [{'provider': 'OpenAI', 'model': 'gpt-4o', 'count': 2}]
       }
       config = Config(**config_dict)
       assert config.summarizer.provider == Provider.OPENAI
   ```

2. **Migration Tests**
   ```python
   def test_config_migration():
       """Test v1.1 to v1.3 migration."""
       old_config = {
           'moderator': {'provider': 'Google', 'model': 'gemini-2.5-pro'},
           'agents': [{'provider': 'OpenAI', 'model': 'gpt-4o', 'count': 2}]
       }
       new_config = migrate_config_v1_to_v3(old_config)
       assert 'summarizer' in new_config
       assert 'topic_report' in new_config
   ```

## Common Issues and Solutions

### Issue 1: Missing Required Fields
**Problem**: Existing configs lack new required fields
**Solution**: Use migration utility with sensible defaults

### Issue 2: Provider Compatibility
**Problem**: Not all providers support all agent types
**Solution**: Implement provider capability matrix

### Issue 3: State Size Growth
**Problem**: New state fields increase memory usage
**Solution**: Implement state pruning for old rounds

## Validation Checklist

- [ ] All new configuration models have proper validation
- [ ] Root Config model includes all specialized agents
- [ ] Configuration migration preserves existing settings
- [ ] State schema supports new agent invocation tracking
- [ ] HITL state handles periodic stops
- [ ] Migration utilities handle edge cases
- [ ] All tests pass with new structures
- [ ] Documentation is updated

## References

- v1.1 Specification: `docs/project_spec.md`
- v1.3 Specification: `docs/project_spec_2.md`
- Configuration Models: `src/virtual_agora/config/models.py`
- State Schema: `src/virtual_agora/state/schema.py`

## Next Phase

Once configuration and state updates are complete and tested, proceed to Phase 2: Specialized Agent Architecture.

---

**Document Version**: 1.0
**Phase**: 1 of 5
**Status**: Implementation Guide